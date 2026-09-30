"""Experiment-integrity regressions; every provider call is mocked."""
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from pydantic import BaseModel, ValidationError
from typesafe_sdk import RetryPolicy, TypeSafeClient

import decision_optimizer.benchmark as benchmark
import decision_optimizer.parsing.dayplan as parsing
from decision_optimizer.benchmark import ALL_EXPERIMENT_CONFIGS, BenchmarkRunner, ExtractionStage, SolveStage, LiveBenchmarkBackend
from decision_optimizer.dayplan import DayPlan, TaskAssignment, SolveStatus, materialize_day_plan_solution, solve_day_plan
from decision_optimizer.direct_solver import DAYPLAN_DIRECT_INSTRUCTIONS, DirectSolverOutputError, solve_day_plan_direct
from decision_optimizer.evaluation import align_names, dayplan_formulation_matches, shift_formulation_matches, evaluate_dayplan, evaluate_shift_schedule
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.failures import failure_category
from decision_optimizer.jev import JevOutputError, JevResult, apply_jev, build_jev_questions
from decision_optimizer.shift_schedule import SolveStatus as ShiftStatus, ShiftAssignment, materialize_shift_schedule_solution, solve_shift_schedule
from decision_optimizer.telemetry import estimate_openai_cost, RunTelemetry
from scripts.label_jev_fixtures import load_selected_cases, build_draft_answer_key, attach_benchmark_observations
from scripts.generate_workforce_cases import generate_cases
from scripts.run_benchmark import load_benchmark_cases, DEFAULT_RESULTS_DIR
from tests.test_benchmark import FakeBenchmarkBackend, benchmark_case, read_jsonl, small_plan
from tests.test_jev import sample_plan, sample_schedule, FakeJevClient, _custom_answers


class FailingBackend(FakeBenchmarkBackend):
    def __init__(self, stage, error):
        super().__init__()
        self.stage, self.error = stage, error
        self.failed_attempts = 0

    def _fail(self, stage):
        if stage == self.stage:
            self.failed_attempts += 1
            raise self.error

    def extract(self, **kwargs):
        self._fail('extraction')
        return super().extract(**kwargs)

    def decide(self, **kwargs):
        self._fail('jev')
        return super().decide(**kwargs)

    def solve(self, **kwargs):
        self._fail('solve')
        return super().solve(**kwargs)


@pytest.mark.parametrize('stage,error', [
    ('extraction', parsing.DayPlanOutputError('malformed extraction')),
    ('extraction', parsing.DayPlanRefusalError('refused')),
    ('extraction', TimeoutError('deadline')),
    ('jev', JevOutputError('invalid answer')),
    ('jev', TimeoutError('deadline')),
    ('solve', DirectSolverOutputError('malformed direct output')),
    ('solve', TimeoutError('deadline')),
])
def test_terminal_failures_are_scored_and_never_retried_on_repeat_or_resume(tmp_path, stage, error):
    bad = FailingBackend(stage, error)
    runner = BenchmarkRunner(tmp_path, bad)
    first = runner.run([benchmark_case()], ALL_EXPERIMENT_CONFIGS)
    records = read_jsonl(tmp_path / 'latest.jsonl')
    terminal = [r for r in records if r['status'] == 'terminal_failure']
    assert terminal
    assert first['failed'] == len(terminal)
    assert all(not r['retryable'] and r['canonical_evaluation']['valid_output'] is False for r in terminal)
    assert all(r['failure_category'] == ('timeout' if isinstance(error, TimeoutError) else 'invalid_model_output') for r in terminal)
    attempts = bad.failed_attempts
    assert runner.run([benchmark_case()], ALL_EXPERIMENT_CONFIGS)['skipped'] == 8
    assert bad.failed_attempts == attempts
    healthy = FakeBenchmarkBackend()
    assert BenchmarkRunner(tmp_path, healthy).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS) == {'succeeded': 0, 'failed': 0, 'skipped': 8}
    assert not healthy.extract_calls and not healthy.solve_calls and not healthy.decide_calls


@pytest.mark.parametrize('error', [parsing.DayPlanOutputError('invalid'), TimeoutError('timeout')])
def test_terminal_extraction_cache_is_shared_when_more_arms_are_added(tmp_path, error):
    config = ALL_EXPERIMENT_CONFIGS[0]
    bad = FailingBackend('extraction', error)
    BenchmarkRunner(tmp_path, bad).run([benchmark_case()], [config])
    healthy = FakeBenchmarkBackend()
    counts = BenchmarkRunner(tmp_path, healthy).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])
    assert counts == {'succeeded': 0, 'failed': 3, 'skipped': 1}
    assert healthy.extract_calls == []
    rows = read_jsonl(tmp_path / 'latest.jsonl')
    assert sum(r['actual_paid_call_attempts_created_for_this_row']['openai_call_attempts'] for r in rows) == 1


def test_transport_failure_remains_distinct_and_can_resume(tmp_path):
    bad = FailingBackend('extraction', httpx.ConnectError('network down'))
    BenchmarkRunner(tmp_path, bad).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])
    rows = read_jsonl(tmp_path / 'latest.jsonl')
    assert all(r['failure_category'] == 'transport_failure' and r['retryable'] for r in rows)
    assert all(r['canonical_evaluation'] is None for r in rows)
    assert BenchmarkRunner(tmp_path, FakeBenchmarkBackend()).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])['succeeded'] == 4


def test_arbitrary_pydantic_error_is_not_a_model_output_failure():
    class LocalConfig(BaseModel):
        count: int
    with pytest.raises(ValidationError) as caught:
        LocalConfig(count='bad')
    class Client:
        responses = None
        def __init__(self): self.responses = self
        def parse(self, **kwargs): raise caught.value
    with pytest.raises(parsing.DayPlanAPIError) as boundary:
        parsing.parse_dayplan('Plan this', client=Client())
    assert failure_category(boundary.value) == 'implementation_or_configuration_error'


@pytest.mark.parametrize('malformed', [None, {'plan': {'bad': 'shape'}, 'missing_info': []}])
def test_actual_extraction_output_boundary_is_terminal_and_cached(tmp_path, malformed):
    class Client:
        def __init__(self): self.responses = self; self.calls = 0
        def parse(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(output_parsed=malformed, refusal=None, usage=SimpleNamespace(input_tokens=7, output_tokens=3), model='gpt-6-luna')
    client = Client()
    counts = BenchmarkRunner(tmp_path, LiveBenchmarkBackend(openai_client=client)).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])
    assert counts['failed'] == 4 and client.calls == 1
    assert BenchmarkRunner(tmp_path, LiveBenchmarkBackend(openai_client=client)).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])['skipped'] == 4
    assert client.calls == 1
    assert all(r['resolved_openai_models'] == ['gpt-6-luna'] for r in read_jsonl(tmp_path / 'latest.jsonl'))


@pytest.mark.parametrize('case_id', ['impossible_hard_workout_window', 'infeasible_staffing_is_extracted_not_repaired'])
def test_infeasible_metrics_are_na_and_partial_schedule_cannot_beat_correct_report(case_id):
    case = next(c for c in load_benchmark_cases() if c.case_id == case_id)
    if case.domain == 'dayplan':
        correct = solve_day_plan(case.canonical)
        partial = materialize_day_plan_solution(case.canonical, [])
        evaluate = evaluate_dayplan
    else:
        correct = solve_shift_schedule(case.canonical)
        partial = materialize_shift_schedule_solution(case.canonical, [ShiftAssignment(employee_name='Alice', shift_id='front_morning')])
        evaluate = evaluate_shift_schedule
    good, bad, missing = [evaluate(case.canonical, s, arm_spec=case.canonical) for s in (correct, partial, None)]
    for result in (good, bad, missing):
        assert result.hard_constraints_satisfied is None
        assert result.hard_constraints_total is None
        assert result.required_completed is None
    assert good.feasible_correctly_reported and good.canonical_validation_valid
    assert not bad.feasible_correctly_reported and not bad.canonical_validation_valid
    assert not missing.valid_output and not missing.feasible_correctly_reported
    unknown = correct.model_copy(update={'status': SolveStatus.UNKNOWN if case.domain == 'dayplan' else ShiftStatus.UNKNOWN})
    assert not evaluate(case.canonical, unknown).feasible_correctly_reported


def test_empty_workforce_schedule_has_binary_invalid_headline_despite_vacuous_checks():
    canonical = sample_schedule()
    result = evaluate_shift_schedule(canonical, materialize_shift_schedule_solution(canonical, []), arm_spec=canonical)
    assert result.valid_output and not result.canonical_validation_valid
    assert result.required_completed is False
    assert result.hard_constraints_satisfied > 0  # diagnostics only
    assert result.constraint_violations


@pytest.mark.parametrize('probabilities,expected,weight,confidence,applied', [
    ({0: .75, 1: 0, 2: 0, 3: 0, 4: .25}, 1.0, 1, .75, True),
    ({0: .5, 1: 0, 2: 0, 3: 0, 4: .5}, 2.0, 1, .5, False),
    ({0: .2, 1: .2, 2: .2, 3: .2, 4: .2}, 2.0, 1, .2, False),
    ({0: 0, 1: 0, 2: .25, 3: 0, 4: .75}, 3.5, 5, .75, True),
])
def test_score_uses_modal_level_probability_not_expected_value_or_sdk_confidence(probabilities, expected, weight, confidence, applied):
    result = apply_jev('Plan it', sample_plan(), threshold=.7, client=FakeJevClient(_custom_answers({('pref_weight', 'preference:1:minimize_work_interruptions'): (expected, .99, probabilities)})))
    decision = next(d for d in result.decisions if d.item == 'preference:1:minimize_work_interruptions')
    assert decision.jev_value == weight and decision.confidence == confidence
    assert decision.applied is applied
    assert decision.expected_score == expected
    assert len(decision.probabilities) == 5
    assert result.problem.preferences[1].weight == (weight if applied else 2)


@pytest.mark.parametrize('problem,prompt', [(sample_plan(), 'Arrange these activities.'), (sample_schedule(), 'Staff these shifts.')])
def test_actual_typesafe_serialized_http_request_contains_no_gpt_answer_leakage(problem, prompt):
    captured = []
    def transport_call(req):
        payload = json.loads(req.content)
        captured.append(payload)
        answers = {}
        for name, question in payload['questions'].items():
            if question['type'] == 'choice':
                levels = list(question['criteria'])
                answers[name] = {'type': 'choice', 'choice': levels[0], 'confidence': .9, 'probabilities': {level: .9 if i == 0 else .1 for i, level in enumerate(levels)}}
            elif question['type'] == 'score':
                answers[name] = {'type': 'score', 'score': 0.0, 'confidence': .9, 'legend': {str(i): str(v) for i, v in enumerate(question['criteria'])}, 'probabilities': {str(i): 1.0 if i == 0 else 0.0 for i in range(5)}}
            else:
                answers[name] = {'type': 'noul', 'noul': .9}
        return httpx2.Response(200, json={'model': 'jev-mock', 'usage': {'input_tokens': 12, 'output_tokens': 0}, 'answers': answers})
    with TypeSafeClient(api_key='mock-test-key', transport=httpx2.MockTransport(transport_call), retry=RetryPolicy(max_retries=0)) as client:
        result = apply_jev(prompt, problem, client=client)
    assert captured and result.decisions
    for payload in captured:
        text = json.dumps(payload['state'])
        for secret in ('gpt_value', '"required":', '"optional":', '"mode":', 'passive', 'hard_latest', 'hard_window', 'soft_preference', 'selection_role', 'negative_control', 'unavailable'):
            assert secret not in text
        assert '"weight":' not in text
        for question in payload['questions'].values():
            assert 'hard_latest' not in question['instructions'] and 'soft_preference' not in question['instructions']
    assert all(d.locator and d.context and d.semantic_question_id for d in result.decisions)


def test_final_weight_decision_reports_target_removed_by_hardness_rewrite():
    result = apply_jev('Firm deadline', sample_plan(), client=FakeJevClient(_custom_answers({('constraint_hardness', 'preference:0:report'): ('hard', .99)})))
    weight = next(d for d in result.decisions if d.item == 'preference:0:finish_before')
    assert weight.applied and weight.final_value is None and weight.final_value_status == 'not_represented'


def test_dayplan_formulation_match_is_stable_across_timezone_serialization():
    from datetime import timezone

    canonical = sample_plan()
    arm = canonical.model_copy(deep=True)
    arm.horizon.start = arm.horizon.start.replace(tzinfo=timezone.utc)
    arm.horizon.end = arm.horizon.end.replace(tzinfo=timezone.utc)
    arm.preferences[0].time = arm.preferences[0].time.replace(tzinfo=timezone.utc)
    persisted = DayPlan.model_validate(arm.model_dump(mode="json"))
    for structural in (False, True):
        assert dayplan_formulation_matches(canonical, arm, structural=structural)
        assert dayplan_formulation_matches(canonical, persisted, structural=structural)
    arm.preferences[0].weight += 1
    assert dayplan_formulation_matches(canonical, arm, structural=True)
    assert not dayplan_formulation_matches(canonical, arm)
    arm.horizon.end = arm.horizon.end.replace(hour=17)
    assert not dayplan_formulation_matches(canonical, arm, structural=True)


def test_structural_match_ignores_weights_but_preserves_hard_soft_and_requirements():
    canonical = sample_plan()
    arm = canonical.model_copy(deep=True)
    arm.preferences[0].weight = 3
    assert not dayplan_formulation_matches(canonical, arm)
    assert dayplan_formulation_matches(canonical, arm, structural=True)
    record = evaluate_dayplan(canonical, solve_day_plan(arm), arm_spec=arm)
    assert record.formulation_match_structural and not record.formulation_match
    arm.tasks[1].latest_end = arm.preferences[0].time
    arm.preferences.pop(0)
    assert not dayplan_formulation_matches(canonical, arm, structural=True)
    workforce = sample_schedule()
    reweighted = workforce.model_copy(deep=True)
    reweighted.objective_weights.preference_penalty = 3
    assert not shift_formulation_matches(workforce, reweighted)
    assert shift_formulation_matches(workforce, reweighted, structural=True)
    reweighted.employees[0].unavailable = []
    assert not shift_formulation_matches(workforce, reweighted, structural=True)


def test_duplicate_unavailability_is_semantically_idempotent_and_generator_deduplicates():
    schedule = sample_schedule()
    duplicate = schedule.model_copy(deep=True)
    duplicate.employees[0].unavailable *= 2
    assert shift_formulation_matches(schedule, duplicate)
    for case in generate_cases():
        for employee in case['canonical']['employees']:
            ids = [u['shift_id'] for u in employee['unavailable']]
            assert len(ids) == len(set(ids))


def test_benchmark_infeasible_staffing_override_is_complete_and_original_fixture_unchanged():
    case = next(c for c in load_benchmark_cases() if c.case_id == 'infeasible_staffing_is_extracted_not_repaired')
    for shift in case.canonical.shifts:
        assert shift.id in case.prompt and shift.day.isoformat() in case.prompt
    original = next(c for c in json.loads((benchmark.Path(__file__).parent / 'fixtures/shift_schedule_nl_eval.json').read_text()) if c['name'] == case.case_id)
    assert '2026-10-05' not in original['prompt']


@pytest.mark.parametrize('produced,canonical,matched', [
    ('20261012_morning', '20261011_morning', False),
    ('s2', 's1', False), ('shift_12', 'shift_1_2', False),
    ('2026-10-12 morning shift', '20261012_morning', True),
    ('Morning groceries', 'grocery', True),
])
def test_identifier_alignment_preserves_meaningful_numbers(produced, canonical, matched):
    assert bool(align_names([produced], [canonical]).mapping) is matched


def test_direct_prompt_excludes_fixed_events_and_invalid_extras_remain_visible():
    assert 'task assignments only' in DAYPLAN_DIRECT_INSTRUCTIONS and 'exclude fixed events' in DAYPLAN_DIRECT_INSTRUCTIONS
    plan = small_plan()
    plan.fixed_events = []
    class Client:
        def __init__(self): self.responses = self
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed={'status': 'feasible', 'assignments': [{'task': 'fixed meeting', 'start': '09:00', 'end': '09:30'}], 'unscheduled_tasks': ['report'], 'explanation': ''})
    solution = solve_day_plan_direct(plan, client=Client()).solution
    result = evaluate_dayplan(plan, solution, arm_spec=plan)
    assert result.alignment_errors and not result.canonical_validation_valid


def test_models_pricing_resolved_response_and_isolated_pilot_directory():
    assert {c.model for c in ALL_EXPERIMENT_CONFIGS} == {'gpt-6-luna', 'gpt-6.1-sol'}
    assert len(ALL_EXPERIMENT_CONFIGS) == 8
    assert ExperimentConfig().label().startswith('Sol 6.1')
    assert estimate_openai_cost('gpt-6.1-sol', 100, 10) > 0
    with pytest.raises(ValueError, match='No OpenAI pricing'):
        estimate_openai_cost('unknown-model', 100, 10)
    telemetry = RunTelemetry.start(ExperimentConfig(), 'dayplan')
    telemetry.record_openai_response(SimpleNamespace(model='gpt-6.1-sol-resolved', usage={'input_tokens': 5, 'output_tokens': 3}))
    assert telemetry.resolved_openai_models == ['gpt-6.1-sol-resolved']
    assert DEFAULT_RESULTS_DIR.name == 'pilot_sol61_v2'


def test_cache_fingerprints_cover_prompt_configuration_implementation_and_actual_problem(monkeypatch):
    case = benchmark_case()
    first = benchmark._extraction_fingerprint(case, 'gpt-6-luna')
    monkeypatch.setattr(parsing, 'DAYPLAN_EXTRACTION_INSTRUCTIONS', parsing.DAYPLAN_EXTRACTION_INSTRUCTIONS + '\nnew instruction')
    assert benchmark._extraction_fingerprint(case, 'gpt-6-luna') != first
    assert benchmark._extraction_fingerprint(replace(case, prompt='changed'), 'gpt-6-luna') != first
    assert benchmark._extraction_fingerprint(case, 'gpt-6.1-sol') != first
    j = benchmark._jev_fingerprint(first, 64, .7, 'jev-latest', problem_hash='one')
    assert benchmark._jev_fingerprint(first, 64, .7, 'jev-latest', problem_hash='two') != j
    assert benchmark._jev_fingerprint(first, 64, .8, 'jev-latest', problem_hash='one') != j
    monkeypatch.setattr(benchmark, '_implementation_hash', lambda: 'new-version')
    assert benchmark._jev_fingerprint(first, 64, .7, 'jev-latest', problem_hash='one') != j


def test_changed_prompt_and_corrupt_extraction_cannot_resume_as_compatible(tmp_path):
    backend = FakeBenchmarkBackend()
    BenchmarkRunner(tmp_path, backend).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:1])
    changed = replace(benchmark_case(), prompt='A different prompt')
    healthy = FakeBenchmarkBackend()
    assert BenchmarkRunner(tmp_path, healthy).run([changed], ALL_EXPERIMENT_CONFIGS[:1])['succeeded'] == 1
    assert len(healthy.extract_calls) == 1
    path = next((tmp_path / 'cache').glob('*extraction.json'))
    data = json.loads(path.read_text())
    data['problem']['tasks'][0]['duration_min'] = 99
    path.write_text(json.dumps(data))
    newer_arm = BenchmarkRunner(tmp_path, FakeBenchmarkBackend()).run([changed], ALL_EXPERIMENT_CONFIGS[:2])
    assert newer_arm['failed'] == 1  # hash corruption is explicit, never a new sample


def test_logging_and_labels_join_actual_benchmark_cache_with_missing_unaligned_questions(tmp_path):
    raw = {'name': 'tiny', 'domain': 'dayplan', 'source': 'unit-test', 'prompt': benchmark_case().prompt, 'canonical': small_plan().model_dump(mode='json')}
    backend = FakeBenchmarkBackend()
    backend.plan.tasks[0].name = 'report draft'
    backend.plan.tasks.append(backend.plan.tasks[0].model_copy(update={'name': 'extra item', 'required': False}))
    BenchmarkRunner(tmp_path, backend).run([benchmark_case()], ALL_EXPERIMENT_CONFIGS[:4])
    logs = read_jsonl(tmp_path / 'jev_decisions.jsonl')
    # The mock's report decision cannot join an actual report-draft question; retained explicitly.
    assert logs and 'answer_key_alignment' in logs[0]
    draft = build_draft_answer_key([raw])
    attach_benchmark_observations(draft, [raw], tmp_path, ['gpt-6-luna'])
    coverage = draft['cases'][0]['observation_coverage']['gpt-6-luna']
    assert coverage['observation_source'] == 'benchmark_cache'
    assert coverage['unaligned_generated'] == 2
    assert draft['cases'][0]['unaligned_observed_questions']['gpt-6-luna']
    assert all(label['observations']['gpt-6-luna']['gpt_baseline_answer'] is not None for label in draft['cases'][0]['expected_jev'])
    # Reading caches invokes neither provider.
    assert len(backend.extract_calls) == 1 and len(backend.decide_calls) == 1


def test_missing_cache_is_not_reported_as_a_model_omission_and_labels_remain_pending(tmp_path):
    draft = build_draft_answer_key(load_selected_cases())
    attach_benchmark_observations(draft, load_selected_cases(), tmp_path, ['gpt-6-luna'])
    excluded = [(c['name'], l) for c in draft['cases'] for l in c['expected_jev'] if not l['primary_accuracy_calibration_eligible']]
    assert {c for c, _ in excluded} == {'workout_before_four_soft', 'soft_finish_preference', 'preferences_and_availability'}
    assert len(excluded) == 3
    assert all(l['review_status'] != 'human_reviewed' for c in draft['cases'] for l in c['expected_jev'])
    assert all(c['observation_coverage']['gpt-6-luna']['cache_status'] == 'missing_extraction_cache' for c in draft['cases'])


def test_all_fourteen_cases_run_all_eight_architectures_with_mocks_and_matching_extraction_hashes(tmp_path):
    cases = load_benchmark_cases()
    class FixtureBackend:
        def __init__(self): self.extract_count = self.decide_count = self.solve_count = 0
        def extract(self, **kwargs):
            self.extract_count += 1
            case = next(c for c in cases if c.prompt == kwargs['prompt'])
            telemetry = kwargs['telemetry']; telemetry.n_model_calls = 1; telemetry.finish()
            return ExtractionStage(case.canonical.model_copy(deep=True), [], telemetry)
        def decide(self, **kwargs):
            self.decide_count += 1
            telemetry = kwargs['telemetry']
            result = apply_jev(kwargs['prompt'], kwargs['problem'], client=CanonicalJevClient(kwargs['prompt'], kwargs['problem']), telemetry=telemetry)
            telemetry.finish()
            return result
        def solve(self, **kwargs):
            self.solve_count += 1
            problem = kwargs['problem']; telemetry = kwargs['telemetry']
            if kwargs['config'].solution_engine == 'direct_llm': telemetry.n_model_calls = 1
            solution = solve_day_plan(problem) if isinstance(problem, DayPlan) else solve_shift_schedule(problem)
            telemetry.finish()
            return SolveStage(solution, None, telemetry)
    class CanonicalJevClient:
        def __init__(self, request, problem):
            self.problem = problem
            self.questions = {q.name: q for q in build_jev_questions(request, problem)}
        def system_one(self, **kwargs):
            answers = {}
            for name in kwargs['questions']:
                question = self.questions[name]
                value = question.question_type.gpt_value(self.problem, question.target)
                primitive = question.question_type.primitive
                if primitive == 'choice': answers[name] = {'choice': value, 'confidence': 1., 'probabilities': {key: float(key == value) for key in question.sdk_question.criteria}}
                elif primitive == 'score': answers[name] = {'score': float(value - 1), 'confidence': 1., 'probabilities': {i: float(i == value - 1) for i in range(5)}}
                else: answers[name] = {'noul': float(value)}
            return SimpleNamespace(answers=answers, model='jev-mock', usage=None)
    backend = FixtureBackend()
    runner = BenchmarkRunner(tmp_path, backend)
    assert runner.preview(cases, ALL_EXPERIMENT_CONFIGS)['total_cells'] == 112
    assert runner.run(cases, ALL_EXPERIMENT_CONFIGS) == {'succeeded': 112, 'failed': 0, 'skipped': 0}
    assert (backend.extract_count, backend.decide_count, backend.solve_count) == (28, 28, 112)
    rows = read_jsonl(tmp_path / 'latest.jsonl')
    for case in cases:
        for model in ('gpt-6-luna', 'gpt-6.1-sol'):
            paired = [r for r in rows if r['case_id'] == case.case_id and r['model'] == model]
            assert len(paired) == 4 and len({r['extraction_problem_hash'] for r in paired}) == 1
    assert all(r['canonical_evaluation']['canonical_validation_valid'] for r in rows)
    assert all(r['canonical_evaluation']['formulation_match_structural'] for r in rows)
    assert runner.run(cases, ALL_EXPERIMENT_CONFIGS)['skipped'] == 112


def test_partial_jev_observations_survive_failed_chunk_cache_resume_and_label_join(tmp_path):
    class ChunkClient:
        def __init__(self): self.calls = 0
        def system_one(self, **kwargs):
            self.calls += 1
            answers = {}
            if self.calls == 1:
                name, q = next(iter(kwargs['questions'].items()))
                value = next(iter(q.criteria))
                answers[name] = {'choice': value, 'confidence': .9, 'probabilities': {k: .9 if k == value else .1 for k in q.criteria}}
            return SimpleNamespace(answers=answers, model='jev-partial', usage=None)
    client = ChunkClient()
    class Backend(FakeBenchmarkBackend):
        def decide(self, **kwargs):
            return apply_jev(kwargs['prompt'], kwargs['problem'], client=client, max_questions_per_request=1, telemetry=kwargs['telemetry'])
    runner = BenchmarkRunner(tmp_path, Backend(), jev_batch_size=1)
    assert runner.run([benchmark_case()], [ALL_EXPERIMENT_CONFIGS[3]])['failed'] == 1
    logs = read_jsonl(tmp_path / 'jev_decisions.jsonl')
    assert len(logs) == 1 and logs[0]['semantic_question_id']
    assert logs[0]['answer_key_alignment']['alignment_status'] == 'aligned'
    assert not logs[0]['applied'] and logs[0]['final_value_status'] == 'stage_failed_not_applied'
    row = read_jsonl(tmp_path / 'latest.jsonl')[0]
    assert row['jev_missing_question_names'] and row['jev_decisions']
    assert row['actual_paid_call_attempts_created_for_this_row']['jev_call_attempts'] == 2
    raw = {'name': 'tiny', 'domain': 'dayplan', 'source': 'unit-test', 'prompt': benchmark_case().prompt, 'canonical': small_plan().model_dump(mode='json')}
    draft = build_draft_answer_key([raw])
    attach_benchmark_observations(draft, [raw], tmp_path, ['gpt-6-luna'])
    assert draft['cases'][0]['observation_coverage']['gpt-6-luna']['jev_cache_status'] == 'terminal_failure'
    assert draft['jev_observed']
    # A new paired engine reads the same terminal cache and creates no new sample.
    assert BenchmarkRunner(tmp_path, Backend(), jev_batch_size=1).run([benchmark_case()], [ALL_EXPERIMENT_CONFIGS[2]])['failed'] == 1
    assert client.calls == 2
    assert len(read_jsonl(tmp_path / 'jev_decisions.jsonl')) == 1


def test_canonical_gpt_omissions_and_numeric_unaligned_questions_remain_explicit():
    canonical = sample_plan()
    extracted = canonical.model_copy(deep=True)
    extracted.tasks = [task for task in extracted.tasks if task.name != 'dishwasher']
    case = replace(benchmark_case(), canonical=canonical)
    observation = benchmark.question_observations(case, extracted, [])
    assert observation['not_generated'] == 2
    assert len(observation['missing_canonical_question_ids']) == 2
    workforce = sample_schedule()
    renamed = workforce.model_copy(deep=True)
    renamed.shifts[0].id = 's9'
    renamed.employees[0].unavailable[0].shift_id = 's9'
    shift_case = replace(case, domain='shift_schedule', canonical=workforce)
    observation = benchmark.question_observations(shift_case, renamed, [])
    assert observation['unaligned_generated'] and observation['not_generated']


@pytest.mark.parametrize('field_path,expected', [('answers.q000_task_requirement.choice', 'invalid_model_output'), ('usage.input_tokens', 'transport_failure')])
def test_typesafe_sdk_validation_classified_at_answer_boundary(field_path, expected):
    from typesafe_sdk import TypeSafeAPIResponseValidationError
    class Client:
        def system_one(self, **kwargs):
            raise TypeSafeAPIResponseValidationError(200, {}, httpx2.Headers(), field_path)
    with pytest.raises(Exception) as caught:
        apply_jev('Plan it', small_plan(), client=Client())
    assert failure_category(caught.value) == expected


@pytest.mark.parametrize('body,status,expected', [
    ({'plan': None, 'missing_info': []}, 'completed', 'invalid_model_output'),
    ({'plan': 'invalid', 'missing_info': []}, 'completed', 'invalid_model_output'),
])
def test_real_openai_sdk_invalid_structured_output_is_terminal_without_sdk_retry(body, status, expected):
    from openai import OpenAI
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            'id': 'resp_mock', 'object': 'response', 'created_at': 0, 'status': status,
            'model': 'gpt-6-luna', 'usage': {'input_tokens': 9, 'output_tokens': 2, 'total_tokens': 11}, 'output': [{'id': 'msg_mock', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': json.dumps(body), 'annotations': []}]}],
        })
    with OpenAI(api_key='mock-test-key', max_retries=3, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        telemetry = RunTelemetry.start(ExperimentConfig(model='gpt-6-luna'), 'dayplan')
        with pytest.raises(parsing.DayPlanOutputError) as caught:
            parsing.parse_dayplan('Plan it', client=client, model='gpt-6-luna', request_timeout_seconds=1, telemetry=telemetry)
    assert failure_category(caught.value) == expected and len(calls) == 1
    assert telemetry.resolved_openai_models == ['gpt-6-luna']
    assert telemetry.openai_input_tokens == 9 and telemetry.estimated_cost > 0


def test_real_openai_sdk_timeouts_disable_automatic_retries():
    from openai import OpenAI
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('deadline', request=request)
    with OpenAI(api_key='mock-test-key', max_retries=3, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        with pytest.raises(parsing.DayPlanAPIError) as caught:
            parsing.parse_dayplan('Plan it', client=client, model='gpt-6-luna', request_timeout_seconds=1)
    assert failure_category(caught.value) == 'timeout' and len(calls) == 1


def test_canonical_unknown_does_not_masquerade_as_infeasible(monkeypatch):
    import decision_optimizer.evaluation as evaluation
    canonical = small_plan()
    unknown = solve_day_plan(canonical).model_copy(update={'status': SolveStatus.UNKNOWN})
    monkeypatch.setattr(evaluation, 'solve_day_plan', lambda _: unknown)
    with pytest.raises(RuntimeError, match='Canonical feasibility was not established'):
        evaluate_dayplan(canonical, None)


def test_typesafe_sdk_invalid_answer_retains_other_valid_answers_and_response_usage():
    captured = []
    def handler(req):
        payload = json.loads(req.content)
        captured.append(payload)
        names = list(payload['questions'])
        return httpx2.Response(200, json={
            'model': 'jev-invalid-mock', 'usage': {'input_tokens': 17, 'output_tokens': 0},
            'answers': {
                names[0]: {'type': 'choice', 'choice': 'required', 'confidence': .9, 'probabilities': {'required': .9, 'optional': .1}},
                names[1]: {'type': 'choice', 'confidence': .9, 'probabilities': {'active': .9, 'passive': .1}},
            },
        })
    telemetry = RunTelemetry.start(ExperimentConfig(use_jev=True), 'dayplan')
    with TypeSafeClient(api_key='mock-test-key', transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=3)) as client:
        with pytest.raises(JevOutputError) as caught:
            apply_jev('Plan it', small_plan(), client=client, telemetry=telemetry, max_retries=0)
    assert len(captured) == 1
    assert telemetry.jev_input_tokens == 17
    assert caught.value.resolved_jev_model == 'jev-invalid-mock'
    assert len(caught.value.observed_decisions) == 1
    assert len(caught.value.invalid_questions) == 1
