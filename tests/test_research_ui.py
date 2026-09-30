"""Presentation-only regression tests. Provider calls are forbidden in offline UI."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import app
from decision_optimizer.application import solve_confirmed_dayplan, solve_confirmed_shift_schedule
from decision_optimizer.dayplan import DayPlan
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.jev import JevDecision, JevResult, MissingTypeSafeAPIKeyError
from decision_optimizer.parsing.dayplan import DayPlanExtraction
from decision_optimizer.research_presentation import (
    FINAL, ROOT, CUSTOM_VALIDATION_NOTE, architecture_summary_rows, case_outcome_rows,
    jev_decision_rows, load_final_benchmark, research_findings, runtime_summary,
)
from decision_optimizer.shift_schedule import ShiftSchedule
from decision_optimizer.telemetry import RunTelemetry


@pytest.fixture
def view():
    return load_final_benchmark()


def _run_app():
    import app
    app.main()


def _forbidden(*args, **kwargs):
    raise AssertionError("Offline presentation attempted provider execution")


def _offline_app(monkeypatch):
    for name in ('parse_dayplan_request', 'parse_shift_schedule_request', 'apply_jev',
                 'solve_confirmed_dayplan', 'solve_confirmed_shift_schedule'):
        monkeypatch.setattr(app, name, _forbidden)
    import openai
    import decision_optimizer.jev as jev
    monkeypatch.setattr(openai, 'OpenAI', _forbidden)
    monkeypatch.setattr(jev, 'TypeSafeClient', _forbidden)
    return AppTest.from_function(_run_app).run()


def test_eight_labels_and_defaults():
    labels = [c.label() for c in app.architecture_configurations()]
    assert labels == [
        'Luna 6 · Direct LLM', 'Luna 6 · CP-SAT',
        'Luna 6 · Jev · Direct LLM', 'Luna 6 · Jev · CP-SAT',
        'Sol 6.1 · Direct LLM', 'Sol 6.1 · CP-SAT',
        'Sol 6.1 · Jev · Direct LLM', 'Sol 6.1 · Jev · CP-SAT',
    ]
    assert ExperimentConfig().model == 'gpt-6.1-sol'
    assert ExperimentConfig().solution_engine == 'cp_sat'


def test_loader_reads_only_frozen_presentation_artifacts(monkeypatch):
    read = []
    original = Path.open
    def track(path, *args, **kwargs):
        read.append(path.resolve())
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', track)
    loaded = load_final_benchmark()
    assert len(loaded.cells) == 112 and len(loaded.case_ids) == 14
    assert set(read) == {
        FINAL / 'reconciled.jsonl', FINAL / 'reconciliation_manifest.json',
        FINAL / 'analysis/analysis_summary.json', FINAL / 'analysis/objective_gaps.csv',
        FINAL / 'analysis/calibration_metrics.json',
    }


def test_loader_rejects_changed_reconciled_artifact(monkeypatch):
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'changed' if p.name == 'reconciled.jsonl' else original(p))
    with pytest.raises(ValueError, match='provenance mismatch'):
        load_final_benchmark()


def test_summary_is_exact_phase10_presentation(view):
    csv_rows = list(csv.DictReader((FINAL / 'analysis/architecture_summary.csv').read_text().splitlines()))
    rows = architecture_summary_rows(view)
    assert len(rows) == 8
    for shown, committed in zip(rows, csv_rows):
        assert shown['Architecture'] == committed['architecture']
        for label, num, den in (
            ('Correct outcome', 'outcome_correct', 'outcome_denominator'),
            ('Usable output', 'usable_output', 'usable_output_denominator'),
            ('Canonical valid (feasible)', 'feasible_canonical_valid', 'feasible_denominator'),
            ('Correct infeasible', 'infeasible_correct', 'infeasible_denominator'),
            ('Structural match', 'structural_formulation_match', 'formulation_denominator'),
            ('Full match', 'full_formulation_match', 'formulation_denominator'),
            ('Optimum / valid feasible', 'optimal', 'optimal_denominator'),
        ):
            assert shown[label] == f'{committed[num]}/{committed[den]}'
        assert shown['Median latency (s)'] == round(float(committed['latency_median_s']), 3)
        assert shown['Median OpenAI cost ($)'] == f"{float(committed['openai_cost_median_usd']):.7f}"
        assert shown['Terminal failures'] == int(committed['terminal_failure'])
        assert shown['Abstentions'] == int(committed['abstention_no_usable_output'])


@pytest.mark.parametrize('case_index', range(14))
def test_each_case_has_all_eight_recorded_outcomes(view, case_index):
    rows = case_outcome_rows(view, view.case_ids[case_index])
    assert {r['Architecture'] for r in rows} == {c.label() for c in app.architecture_configurations()}
    cells = {c['architecture']: c for c in view.cells if c['case_id'] == view.case_ids[case_index]}
    for row in rows:
        cell = cells[row['Architecture']]
        assert ('Terminal failure' in row['Status']) == (cell['status'] == 'terminal_failure')
        assert ('Abstention' in row['Status']) == (cell['status'] == 'success' and not cell['canonical_evaluation']['valid_output'])
        if not cell['canonical_evaluation']['canonical_feasible']:
            assert row['Canonical optimum attained'] == row['Objective gap'] == 'N/A'


def test_compare_and_findings_cannot_call_providers(monkeypatch, view):
    at = _offline_app(monkeypatch)
    assert not at.exception
    at.radio(key='research_mode').set_value('Benchmark / Compare').run()
    assert not at.exception
    assert not at.button  # no live execution action exists
    for case in view.case_ids:
        at.selectbox[0].set_value(case).run()
        assert not at.exception
        assert len(at.dataframe[0].value) == 8
        assert at.dataframe[1].value.to_dict('records') == architecture_summary_rows(view)
    at.radio(key='research_mode').set_value('Research findings').run()
    assert not at.exception
    assert [m.value for m in at.metric] == ['0.005284', '0.075577', '0.291500']


@pytest.mark.parametrize('available', [False, True])
def test_jev_defaults_to_key_availability(monkeypatch, available):
    monkeypatch.setattr(app._config, 'typesafe_api_key', lambda: 'test-only' if available else None)
    at = _offline_app(monkeypatch)
    assert not at.exception
    assert at.toggle(key='use_jev').value is available
    assert at.toggle(key='use_jev').disabled is not available
    assert at.selectbox[1].value == 'gpt-6.1-sol'
    assert at.radio[1].value == 'cp_sat'
    if not available:
        assert any('not configured' in c.value for c in at.caption)


def test_actual_jev_records_render_without_inferred_final_value(view):
    decisions = next(c['jev_decisions'] for c in view.cells if c['jev_decisions'])
    typed = [JevDecision.model_validate(d) for d in decisions]
    rows = jev_decision_rows(typed)
    for row, record in zip(rows, decisions):
        assert row['GPT baseline'] == str(record['gpt_value'])
        assert row['Jev proposal'] == str(record['jev_value'])
        assert row['Selected probability'] == record['confidence']
        assert row['Applied?'] == ('Yes' if record['applied'] else 'No')
        assert row['Changed formulation?'] == ('Yes' if record['changed'] else 'No')
        assert row['Final represented value'] == str(record['final_value'])
    typed[0].final_value = None
    assert jev_decision_rows(typed)[0]['Final represented value'] == 'Not represented / not recorded'


@pytest.mark.parametrize('engine', ['cp_sat', 'direct_llm'])
def test_custom_header_never_claims_canonical_scoring(engine):
    plan = DayPlan.model_validate_json((ROOT / 'tests/cases/simple_active.json').read_text())
    run = solve_confirmed_dayplan(plan)
    run.telemetry.solution_engine = engine
    summary = runtime_summary(run)
    assert not any('canonical' in key.lower() for key in summary)
    assert 'no canonical ground truth' in CUSTOM_VALIDATION_NOTE
    assert summary['Optimal for interpreted problem'] == ('Yes (CP-SAT proven)' if engine == 'cp_sat' else 'Not established')
    assert summary['Runtime hard constraints'] == 'Passed'


def test_nonfeasible_empty_output_does_not_claim_valid_hard_constraints():
    plan = DayPlan.model_validate_json((ROOT / 'tests/cases/infeasible_precedence.json').read_text())
    summary = runtime_summary(solve_confirmed_dayplan(plan))
    assert summary['Objective'] == 'N/A'
    assert summary['Runtime hard constraints'] == 'N/A (no feasible schedule)'
    assert summary['Optimal for interpreted problem'] == 'Not established'


def test_both_domains_keep_usable_runtime_headers():
    schedule = ShiftSchedule.model_validate_json((ROOT / 'tests/cases/shift_schedule_basic.json').read_text())
    summary = runtime_summary(solve_confirmed_shift_schedule(schedule))
    assert summary['Usable result'] == 'Yes'
    assert summary['Runtime hard constraints'] == 'Passed'


def test_jev_enabled_interpret_confirm_solve_uses_real_records_and_pipeline_telemetry(monkeypatch):
    plan = DayPlan.model_validate_json((ROOT / 'tests/cases/simple_active.json').read_text())
    monkeypatch.setattr(app._config, 'typesafe_api_key', lambda: 'test-only')
    def parse(request, *, config, telemetry):
        telemetry.record_openai_response(SimpleNamespace(usage={'input_tokens': 100, 'output_tokens': 20}))
        return DayPlanExtraction(plan=plan, missing_info=[])
    decision = JevDecision(question_type='task_mode', item=plan.tasks[0].name, primitive='choice',
        gpt_value='active', jev_value='active', confidence=.95, applied=True, changed=False,
        final_value='active', final_value_status='represented', probabilities={'active': .95, 'passive': .05})
    def review(request, problem, *, telemetry, **kwargs):
        telemetry.record_jev_response(SimpleNamespace(usage={}), 1)
        return JevResult(problem=problem, decisions=[decision])
    monkeypatch.setattr(app, 'parse_dayplan_request', parse)
    monkeypatch.setattr(app, 'apply_jev', review)
    at = AppTest.from_function(_run_app).run()
    at.text_area[0].set_value('Schedule two activities').run()
    at.button(key='dayplan_interpret').click().run()
    assert not at.exception
    assert any(frame.value.to_dict('records') == jev_decision_rows([decision]) for frame in at.dataframe)
    at.button(key='dayplan_solve').click().run()
    assert not at.exception
    run = at.session_state['dayplan_run']
    assert run.jev_decisions == [decision]
    assert run.telemetry.n_model_calls == 1
    assert run.telemetry.n_jev_questions == run.telemetry.n_jev_calls == 1
    assert run.telemetry.estimated_cost > 0
    assert run.telemetry.latency_total_s >= at.session_state['dayplan_interpretation_telemetry'].latency_total_s
    assert any(CUSTOM_VALIDATION_NOTE == c.value for c in at.caption)
    at.toggle(key='use_jev').set_value(False).run()
    assert 'dayplan_run' not in at.session_state
    assert 'dayplan_extraction' not in at.session_state


def test_jev_configuration_failure_is_safe_and_actionable(monkeypatch):
    plan = DayPlan.model_validate_json((ROOT / 'tests/cases/simple_active.json').read_text())
    monkeypatch.setattr(app._config, 'typesafe_api_key', lambda: 'test-only')
    monkeypatch.setattr(app, 'parse_dayplan_request', lambda *a, **kw: DayPlanExtraction(plan=plan, missing_info=[]))
    def fail(*args, **kwargs):
        raise MissingTypeSafeAPIKeyError('secret-value-must-not-appear')
    monkeypatch.setattr(app, 'apply_jev', fail)
    at = AppTest.from_function(_run_app).run()
    at.button(key='dayplan_interpret').click().run()
    assert not at.exception
    assert 'Jev review failed' in at.error[0].value
    assert 'secret-value' not in at.error[0].value


def test_historical_results_not_presented_as_current_benchmark(view):
    readme = (ROOT / 'README.md').read_text()
    assert 'historical development artifacts' in readme
    assert 'not current Sol 6.1 benchmark evidence' in readme
    assert 'No paid benchmark results are claimed yet' not in readme
    assert '13 / 13' not in readme  # removed old MVP evaluation table
    for row in view.summary['architecture_summary']:
        assert row['architecture'] in readme
        assert f"| {row['architecture']} | {row['outcome_correct']}/14 |" in readme
    findings = research_findings(view)
    assert '52/56' in findings['Base model'] and '39/56' in findings['Base model']
    assert '223/223' in findings['Jev: negative / null result']
    assert 'Eight genuine formulation changes' in findings['Jev: negative / null result']


@pytest.mark.parametrize('domain', ['dayplan', 'shift_schedule'])
def test_clarification_retains_both_domains_and_accumulates_interpretation_calls(monkeypatch, domain):
    from decision_optimizer.parsing.shift_schedule import ShiftScheduleExtraction
    day = domain == 'dayplan'
    schema = DayPlan if day else ShiftSchedule
    field = 'plan' if day else 'schedule'
    extraction_type = DayPlanExtraction if day else ShiftScheduleExtraction
    problem = schema.model_validate_json((ROOT / 'tests/cases' / ('simple_active.json' if day else 'shift_schedule_basic.json')).read_text())
    monkeypatch.setattr(app._config, 'typesafe_api_key', lambda: 'test-only')
    def parse(*args, telemetry, **kwargs):
        telemetry.record_openai_response(SimpleNamespace(usage={'input_tokens': 100, 'output_tokens': 20}))
        return extraction_type(**{field: None, 'missing_info': ['Please specify the horizon.']})
    def clarify(*args, telemetry, **kwargs):
        telemetry.record_openai_response(SimpleNamespace(usage={'input_tokens': 100, 'output_tokens': 20}))
        return extraction_type(**{field: problem, 'missing_info': []})
    requests = []
    def review(request, confirmed, **kwargs):
        requests.append(request)
        return JevResult(problem=confirmed, decisions=[])
    monkeypatch.setattr(app, 'parse_' + domain + '_request', parse)
    monkeypatch.setattr(app, 'clarify_' + domain + '_request', clarify)
    monkeypatch.setattr(app, 'apply_jev', review)
    at = AppTest.from_function(_run_app).run()
    if not day:
        at.selectbox(key='problem_type').set_value('Workforce Scheduler').run()
    at.text_area[0].set_value('Please schedule these activities.').run()
    at.button(key=domain + '_interpret').click().run()
    assert not at.exception
    assert not requests  # incomplete extraction must not be reviewed
    at.text_input(key=domain + '_clarification').set_value('09:00 to 17:00').run()
    at.button(key=domain + '_clarify').click().run()
    assert not at.exception
    assert requests == ['Please schedule these activities.\nClarification: 09:00 to 17:00']
    at.button(key=domain + '_solve').click().run()
    assert not at.exception
    run = at.session_state[domain + '_run']
    assert run.telemetry.n_model_calls == 2
    assert runtime_summary(run)['Usable result'] == 'Yes'
