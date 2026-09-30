"""Offline reconciliation and prospective failure-persistence regressions."""
from copy import deepcopy
from datetime import timedelta, timezone
import json
import socket
from types import SimpleNamespace

import pytest

import decision_optimizer.benchmark as benchmark
from decision_optimizer.evaluation import shift_formulation_matches
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.parsing.dayplan import DayPlanOutputError
from decision_optimizer.shift_schedule import ShiftSchedule, Unavailability
from decision_optimizer.telemetry import estimate_openai_cost
import scripts.reconcile_benchmark as reconciliation
from tests.test_benchmark import FakeBenchmarkBackend, benchmark_case, read_jsonl
from tests.test_jev import sample_schedule


@pytest.mark.parametrize('tz', [timezone.utc, timezone(timedelta(hours=-4)), timezone(timedelta(hours=5, minutes=30))])
def test_shift_comparison_normalizes_timezone_without_loosening_semantics(tz):
    canonical = sample_schedule()
    canonical.employees[0].unavailable = [
        # A real local-clock interval, not only an unavailable shift identifier.
        Unavailability(
            day=canonical.shifts[0].day, start='09:00', end='13:00')
    ]
    arm = canonical.model_copy(deep=True)
    for shift in arm.shifts:
        shift.start = shift.start.replace(tzinfo=tz)
        shift.end = shift.end.replace(tzinfo=tz)
    interval = arm.employees[0].unavailable[0]
    interval.start = interval.start.replace(tzinfo=tz)
    interval.end = interval.end.replace(tzinfo=tz)
    assert interval.start != canonical.employees[0].unavailable[0].start
    for structural in [False, True]:
        assert shift_formulation_matches(canonical, arm, structural=structural)
        assert shift_formulation_matches(arm, canonical, structural=structural)
        roundtrip = ShiftSchedule.model_validate_json(arm.model_dump_json())
        assert shift_formulation_matches(canonical, roundtrip, structural=structural)
        changed = arm.model_copy(deep=True)
        changed.employees[0].unavailable[0].start = interval.start.replace(hour=10)
        assert not shift_formulation_matches(canonical, changed, structural=structural)
        changed = arm.model_copy(deep=True)
        changed.shifts[0].required_staff += 1
        assert not shift_formulation_matches(canonical, changed, structural=structural)
    weighted = arm.model_copy(deep=True)
    weighted.objective_weights.fairness += 1
    assert not shift_formulation_matches(canonical, weighted)
    assert shift_formulation_matches(canonical, weighted, structural=True)


class RecordedFailureBackend(FakeBenchmarkBackend):
    def __init__(self, stage, timeout=False):
        super().__init__()
        self.stage = stage
        self.timeout = timeout

    def extract(self, **kwargs):
        t = kwargs['telemetry']
        t.n_model_call_attempts += 1
        t.record_openai_response(SimpleNamespace(model=t.model, usage={'input_tokens': 10, 'output_tokens': 2}))
        if self.stage == 'extraction':
            raise DayPlanOutputError('invalid completed extraction')
        result = super().extract(**kwargs)
        t.finish()
        return result

    def decide(self, **kwargs):
        t = kwargs['telemetry']
        t.n_jev_call_attempts += 1
        result = super().decide(**kwargs)
        t.finish()
        if self.stage == 'jev':
            from decision_optimizer.jev import JevOutputError
            error = JevOutputError('invalid completed answer')
            error.observed_decisions = result.decisions
            error.resolved_jev_model = result.resolved_model
            raise error
        return result

    def solve(self, **kwargs):
        t = kwargs['telemetry']
        t.n_model_call_attempts += 1
        if self.timeout:
            raise TimeoutError('no response metadata')
        t.record_openai_response(SimpleNamespace(model=t.model, usage={'input_tokens': 7, 'output_tokens': 3}))
        from decision_optimizer.direct_solver import DirectSolverOutputError
        raise DirectSolverOutputError('invalid completed solve')


@pytest.mark.parametrize('stage', ['extraction', 'jev', 'solve'])
def test_failure_persistence_retains_completed_usage_and_charges_shared_stage_once(tmp_path, stage):
    config = ExperimentConfig(model='gpt-6-luna', use_jev=True, solution_engine='direct_llm')
    configs = [config, config.model_copy(update={'solution_engine': 'cp_sat'})] if stage != 'solve' else [config]
    runner = benchmark.BenchmarkRunner(tmp_path, RecordedFailureBackend(stage))
    runner.run([benchmark_case()], configs)
    rows = read_jsonl(runner.results_path)
    paid = rows[0]['actual_paid_call_attempts_created_for_this_row']
    assert paid['openai_input_tokens'] == (17 if stage == 'solve' else 10)
    assert paid['openai_output_tokens'] == (5 if stage == 'solve' else 2)
    assert paid['openai_estimated_cost_usd'] == pytest.approx(estimate_openai_cost(config.model, paid['openai_input_tokens'], paid['openai_output_tokens']))
    assert paid['openai_responses_with_metadata'] == (2 if stage == 'solve' else 1)
    assert paid['openai_usage_unknown_calls'] == 0
    assert paid['jev_input_tokens'] == (0 if stage == 'extraction' else 5)
    assert paid['jev_questions'] == (0 if stage == 'extraction' else 1)
    assert paid['jev_completed_calls'] == (0 if stage == 'extraction' else 1)
    if len(rows) > 1:
        assert rows[1]['actual_paid_call_attempts_created_for_this_row']['openai_input_tokens'] == 0
        assert rows[1]['actual_paid_call_attempts_created_for_this_row']['jev_input_tokens'] == 0
    # Cached terminal failures on resume retain evidence but never charge again.
    assert runner.run([benchmark_case()], configs)['skipped'] == len(configs)


def test_downstream_timeout_retains_completed_jev_evidence_and_unknown_solve_usage(tmp_path):
    config = ExperimentConfig(model='gpt-6-luna', use_jev=True, solution_engine='direct_llm')
    runner = benchmark.BenchmarkRunner(tmp_path, RecordedFailureBackend('solve', timeout=True))
    runner.run([benchmark_case()], [config])
    row = read_jsonl(runner.results_path)[0]
    paid = row['actual_paid_call_attempts_created_for_this_row']
    assert row['status'] == 'terminal_failure' and row['failure_category'] == 'timeout'
    assert row['canonical_evaluation']['valid_output'] is False
    assert paid['openai_call_attempts'] == 2 and paid['openai_responses_with_metadata'] == 1
    assert paid['openai_usage_unknown_calls'] == 1 and paid['unknown_usage_billing_usd'] is None
    assert paid['openai_input_tokens'] == 10 and paid['openai_output_tokens'] == 2
    assert paid['jev_input_tokens'] == 5 and paid['jev_questions'] == 1 and paid['jev_completed_calls'] == 1
    cache = next(runner.cache_dir.glob('*_jev.json'))
    upstream = json.loads(cache.read_text())
    assert row['jev_decisions'] == upstream['decisions']
    assert row['resolved_jev_model'] == upstream['resolved_jev_model'] == 'jev-test-1.0'
    assert row['final_problem'] == upstream['problem']
    assert row['intermediate_reuse']['jev_fingerprint'] == upstream['fingerprint']
    assert row['question_alignment']['observed_questions'][0]['jev_observation_status'] == 'observed'


@pytest.fixture(scope='module')
def reconciled(tmp_path_factory):
    output = tmp_path_factory.mktemp('reconciled')
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError('network or benchmark execution forbidden')
        patch.setattr(socket.socket, 'connect', forbidden)
        patch.setattr(benchmark.BenchmarkRunner, 'run', forbidden)
        patch.setattr(benchmark.LiveBenchmarkBackend, 'extract', forbidden)
        patch.setattr(benchmark.LiveBenchmarkBackend, 'decide', forbidden)
        patch.setattr(benchmark.LiveBenchmarkBackend, 'solve', forbidden)
        manifest = reconciliation.reconcile(output_dir=output)
        first = {p.name: p.read_bytes() for p in output.iterdir()}
        assert reconciliation.reconcile(output_dir=output) == manifest
        assert {p.name: p.read_bytes() for p in output.iterdir()} == first
    return output, manifest, reconciliation.read_jsonl(output / 'reconciled.jsonl')


def test_reconciliation_is_idempotent_and_preserves_all_audited_raw_bytes(reconciled):
    output, manifest, rows = reconciled
    audit = json.loads(reconciliation.AUDIT_PATH.read_text())
    assert reconciliation.verify_raw(reconciliation.RAW_DIR, audit) == {k: v for k, v in manifest['input_sha256'].items() if k not in {'audit_details', 'audit_report'}}
    assert manifest['validation']['raw_hashes_verified_before_and_after'] == 55
    for path, digest in manifest['derived_sha256'].items():
        assert reconciliation.sha256(output / path) == digest
    assert not manifest['paid_observations_generated_or_replaced']


def test_reconciled_112_rows_only_explicit_fields_change(reconciled):
    _, manifest, rows = reconciled
    assert manifest['before'] == manifest['after']
    assert manifest['after']['rows'] == 112
    assert manifest['after']['status'] == {'success': 103, 'terminal_failure': 9}
    assert manifest['after']['execution_success_without_usable_output'] == 8
    assert manifest['validation']['offline_replayed_rows'] == 112
    raw = reconciliation.read_jsonl(reconciliation.RAW_DIR / 'latest.jsonl')
    corrected = set(manifest['correction_rules'][0]['affected_result_keys'])
    assert len(corrected) == 8
    for before, after in zip(raw, rows):
        restored = deepcopy(after)
        changes = restored.pop('reconciliation')['raw_values_superseded_for_analysis']
        for field, old in changes.items():
            if old['present']:
                restored[field] = old['value']
            else:
                del restored[field]
        assert restored == before
        if after['result_key'] in corrected:
            for field in reconciliation.FORMULATION_FIELDS:
                assert before['canonical_evaluation'][field] is False
                assert after['canonical_evaluation'][field] is True
        else:
            assert after['canonical_evaluation'] == before['canonical_evaluation']
    recovered = next(r for r in rows if r['result_key'] in manifest['correction_rules'][2]['affected_result_keys'])
    assert len(recovered['jev_decisions']) == 3 and recovered['resolved_jev_model'] == 'jev-1.13.0'
    assert recovered['status'] == 'terminal_failure'
    assert recovered['canonical_evaluation']['valid_output'] is False
    assert 'solution' not in recovered
    assert all(q['jev_observation_status'] == 'observed' for q in recovered['question_alignment']['observed_questions'])


def test_reconciled_label_coverage_is_frozen(reconciled):
    _, manifest, _ = reconciled
    assert manifest['label_counts'] == {'human_reviewed_approved': 54, 'generator_derived_scored': 71, 'subjective_excluded': 3, 'primary_scored_total': 125, 'labels_total': 128}
    assert manifest['coverage_totals'] == {'canonical': 256, 'aligned': 228, 'missing': 28, 'unaligned_generated': 2, 'unaligned_decisions': 0, 'primary': 250, 'primary_aligned': 223, 'primary_missing': 27}
    unaligned = [q for p in manifest['coverage_by_pair'] for q in p['unaligned_questions']]
    assert len(unaligned) == 2 and all(q['canonical_question_id'] is None for q in unaligned)


def test_reconciled_expenditure_counts_unique_stages_and_keeps_timeouts_unknown(reconciled):
    _, manifest, rows = reconciled
    assert manifest['experiment_totals'] == {'openai_call_attempts': 76, 'openai_responses_with_metadata': 74, 'openai_input_tokens': 86583, 'openai_output_tokens': 46461, 'openai_estimated_cost_usd': .2874103, 'jev_call_attempts': 24, 'jev_completed_calls': 24, 'jev_input_tokens': 50946, 'jev_questions': 230, 'openai_usage_unknown_calls': 2, 'jev_usage_unknown_calls': 0, 'timeout_billing_usd': None, 'jev_cost_usd': None}
    assert sum(r['actual_paid_call_attempts_created_for_this_row']['openai_input_tokens'] for r in rows) == 86583
    assert sum(r['actual_paid_call_attempts_created_for_this_row']['jev_input_tokens'] for r in rows) == 50946
    assert len(manifest['unique_stage_ledger']) == 100  # 28 extraction + 24 Jev + 48 Direct


def test_additional_discrepancy_stops_before_writing_derived_outputs(tmp_path, monkeypatch):
    original = reconciliation._evaluate
    def discrepant(*args, **kwargs):
        record = original(*args, **kwargs)
        return record.model_copy(update={'required_completed': not record.required_completed})
    monkeypatch.setattr(reconciliation, '_evaluate', discrepant)
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError, match='additional evaluation discrepancy'):
        reconciliation.reconcile(output_dir=output)
    assert not output.exists()


def test_raw_hash_discrepancy_stops_without_reconciliation(tmp_path, monkeypatch):
    monkeypatch.setattr(reconciliation, 'sha256', lambda path: 'changed')
    with pytest.raises(ValueError, match='immutable artifact hashes'):
        reconciliation.reconcile(output_dir=tmp_path / 'must_not_exist')
    assert not (tmp_path / 'must_not_exist').exists()
