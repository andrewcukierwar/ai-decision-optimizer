import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import decision_optimizer.application as application
from decision_optimizer.application import (
    EditedInterpretationError,
    clarify_dayplan_request,
    clarify_shift_schedule_request,
    parse_dayplan_request,
    parse_shift_schedule_request,
    solve_confirmed_dayplan,
    solve_confirmed_shift_schedule,
    validate_edited_dayplan_json,
    validate_edited_shift_schedule_json,
)
from decision_optimizer.dayplan import DayPlan, SolveStatus as DayPlanSolveStatus, solve_day_plan
from decision_optimizer.diagnostics import (
    diagnose_dayplan_infeasibility,
    diagnose_shift_schedule_infeasibility,
)
from decision_optimizer.explanations import (
    build_dayplan_explanation_payload,
    build_shift_schedule_explanation_payload,
)
from decision_optimizer.parsing.dayplan import DayPlanExtraction
from decision_optimizer.parsing.shift_schedule import ShiftScheduleExtraction
from decision_optimizer.shift_schedule import ShiftSchedule, SolveStatus as ShiftSolveStatus
from decision_optimizer.presentation import dayplan_infeasibility_summary
from app import _store_extraction


CASES = Path(__file__).parent / "cases"


class FakeStreamlit:
    def __init__(self, session_state=None):
        self.session_state = session_state or {}


class FakeClient:
    def __init__(self, responses):
        self._queue = list(responses)

        class Responses:
            def __init__(self, outer):
                self.outer = outer

            def parse(self, **kwargs):
                return self.outer._queue.pop(0)

        self.responses_api = Responses(self)
        self.responses = self.responses_api


def parsed_response(output):
    return SimpleNamespace(output_parsed=output, refusal=None)


def load_dayplan(name):
    return DayPlan.model_validate(json.loads((CASES / name).read_text()))


def load_schedule(name):
    return ShiftSchedule.model_validate(json.loads((CASES / name).read_text()))


def test_dayplan_parse_confirm_solve_validate_flow():
    plan = load_dayplan("simple_active.json")
    extraction = DayPlanExtraction(plan=plan, missing_info=[])
    parsed = parse_dayplan_request("Plan my day.", client=FakeClient([parsed_response(extraction)]))

    confirmed = validate_edited_dayplan_json(
        json.dumps(parsed.plan.model_dump(mode="json"))
    )
    run = solve_confirmed_dayplan(confirmed)

    assert run.solution.status == DayPlanSolveStatus.OPTIMAL
    assert run.validation.valid, run.validation.errors


def test_shift_schedule_parse_confirm_solve_validate_flow():
    schedule = load_schedule("shift_schedule_basic.json")
    extraction = ShiftScheduleExtraction(schedule=schedule, missing_info=[])
    parsed = parse_shift_schedule_request(
        "Staff the store.", client=FakeClient([parsed_response(extraction)])
    )

    confirmed = validate_edited_shift_schedule_json(
        json.dumps(parsed.schedule.model_dump(mode="json"))
    )
    run = solve_confirmed_shift_schedule(confirmed)

    assert run.solution.status == ShiftSolveStatus.OPTIMAL
    assert run.validation.valid, run.validation.errors


def test_invalid_edited_json_is_rejected_before_solving():
    invalid = json.dumps(
        {
            "horizon": {"start": "09:00", "end": "13:00"},
            "tasks": [
                {"name": "focus", "mode": "active", "duration_min": 0}
            ],
        }
    )

    with pytest.raises(EditedInterpretationError, match="tasks.0.duration_min"):
        validate_edited_dayplan_json(invalid)
    with pytest.raises(EditedInterpretationError, match="Invalid JSON"):
        validate_edited_shift_schedule_json("{not json")


def test_incomplete_extraction_enters_clarification_state_and_one_round_completes():
    incomplete = DayPlanExtraction(
        plan=None, missing_info=["What is the planning horizon?"]
    )
    complete = DayPlanExtraction(plan=load_dayplan("simple_active.json"), missing_info=[])
    first = parse_dayplan_request(
        "Plan my day.", client=FakeClient([parsed_response(incomplete)])
    )
    final = clarify_dayplan_request(
        "Plan my day.",
        first,
        "Use 09:00-13:00 and schedule the existing tasks.",
        client=FakeClient([parsed_response(complete)]),
    )

    assert first.plan is None
    assert first.missing_info
    assert final.plan is not None
    assert final.missing_info == []


def test_incomplete_shift_schedule_supports_one_clarification_round():
    incomplete = ShiftScheduleExtraction(
        schedule=None, missing_info=["Which shifts and dates should be staffed?"]
    )
    complete = ShiftScheduleExtraction(
        schedule=load_schedule("shift_schedule_basic.json"), missing_info=[]
    )
    first = parse_shift_schedule_request(
        "Build next week's rota.", client=FakeClient([parsed_response(incomplete)])
    )
    final = clarify_shift_schedule_request(
        "Build next week's rota.",
        first,
        "Use the shifts in the confirmed schedule.",
        client=FakeClient([parsed_response(complete)]),
    )

    assert first.schedule is None
    assert final.schedule is not None


def test_store_incomplete_dayplan_extraction_keeps_clarification_state_available():
    state = FakeStreamlit({"dayplan_edited_json": "stale"})
    extraction = DayPlanExtraction(
        plan=None, missing_info=["What is the planning horizon?"]
    )

    _store_extraction(state, "dayplan", extraction, clarification_used=False)

    assert state.session_state["dayplan_extraction"] == extraction
    assert state.session_state["dayplan_clarification_used"] is False
    assert "dayplan_edited_json" not in state.session_state


def test_store_incomplete_shift_schedule_extraction_keeps_clarification_state_available():
    state = FakeStreamlit({"shift_schedule_edited_json": "stale"})
    extraction = ShiftScheduleExtraction(
        schedule=None, missing_info=["Which shifts and dates should be staffed?"]
    )

    _store_extraction(state, "shift_schedule", extraction, clarification_used=False)

    assert state.session_state["shift_schedule_extraction"] == extraction
    assert state.session_state["shift_schedule_clarification_used"] is False
    assert "shift_schedule_edited_json" not in state.session_state


def test_store_complete_extractions_keeps_editable_json_for_both_problem_types():
    dayplan = DayPlanExtraction(
        plan=load_dayplan("simple_active.json"), missing_info=[]
    )
    day_state = FakeStreamlit()
    _store_extraction(day_state, "dayplan", dayplan, clarification_used=False)

    schedule = ShiftScheduleExtraction(
        schedule=load_schedule("shift_schedule_basic.json"), missing_info=[]
    )
    shift_state = FakeStreamlit()
    _store_extraction(
        shift_state, "shift_schedule", schedule, clarification_used=False
    )

    assert json.loads(day_state.session_state["dayplan_edited_json"]) == dayplan.plan.model_dump(mode="json")
    assert json.loads(shift_state.session_state["shift_schedule_edited_json"]) == schedule.schedule.model_dump(mode="json")


def test_infeasible_dayplan_has_structured_diagnosis():
    plan = load_dayplan("infeasible_precedence.json")
    diagnostic = diagnose_dayplan_infeasibility(plan)
    run = solve_confirmed_dayplan(plan)

    assert diagnostic.problem_type == "dayplan"
    assert diagnostic.findings
    assert any(
        finding.family == "precedence" for finding in diagnostic.findings
    )
    assert diagnostic.suggestions
    assert run.solution.status == DayPlanSolveStatus.INFEASIBLE
    assert run.diagnosis is not None


def test_infeasible_dayplan_summary_shows_capacity_shortfall_without_penalty_metric():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "fixed_events": [
                {"name": "Meeting", "start": "10:00", "end": "11:00"}
            ],
            "tasks": [
                {"name": "task 1", "duration_min": 70, "mode": "active"},
                {"name": "task 2", "duration_min": 70, "mode": "active"},
                {"name": "task 3", "duration_min": 70, "mode": "active"},
            ],
        }
    )
    diagnostic = diagnose_dayplan_infeasibility(plan)

    assert diagnostic.required_active_minutes == 210
    assert diagnostic.available_person_minutes == 120
    assert diagnostic.capacity_shortfall_minutes == 90
    assert dayplan_infeasibility_summary(diagnostic) == [
        "No feasible schedule",
        "Deterministic diagnosis completed",
        "Required active work: 210 minutes",
        "Available person-time after fixed events: 120 minutes",
        "Capacity shortfall: 90 minutes",
    ]
    assert all("Preference penalty" not in line for line in dayplan_infeasibility_summary(diagnostic))


def test_infeasible_shift_schedule_has_structured_diagnosis():
    schedule = load_schedule("shift_schedule_infeasible.json")
    diagnostic = diagnose_shift_schedule_infeasibility(schedule)
    run = solve_confirmed_shift_schedule(schedule)

    assert diagnostic.problem_type == "shift_schedule"
    assert diagnostic.findings
    assert any(
        finding.code == "coverage_eligibility_or_availability"
        for finding in diagnostic.findings
    )
    assert diagnostic.suggestions
    assert run.solution.status == ShiftSolveStatus.INFEASIBLE
    assert run.diagnosis is not None


def test_explanation_payloads_are_grounded_structured_facts_only():
    dayplan = load_dayplan("simple_active.json")
    day_result = solve_day_plan(dayplan)
    day_validation = application.validate_dayplan(dayplan, day_result)
    day_payload = build_dayplan_explanation_payload(
        dayplan, day_result, day_validation
    )

    schedule = load_schedule("shift_schedule_basic.json")
    shift_result = application.solve_shift_schedule(schedule)
    shift_validation = application.validate_shift_schedule(schedule, shift_result)
    shift_payload = build_shift_schedule_explanation_payload(
        schedule, shift_result, shift_validation
    )

    assert day_payload is not None
    assert shift_payload is not None
    assert "raw_request" not in day_payload.model_dump()
    assert "raw_request" not in shift_payload.model_dump()
    assert day_payload.validation_valid is True
    assert shift_payload.validation_valid is True
    assert day_payload.objective_value == day_result.objective_value
    assert shift_payload.objective_breakdown == shift_result.objective_breakdown


def test_validation_failure_prevents_normal_explanation_path(monkeypatch):
    plan = load_dayplan("simple_active.json")
    valid_result = solve_day_plan(plan)
    valid_result.assignments[0].end = valid_result.assignments[0].start

    monkeypatch.setattr(application, "solve_day_plan", lambda *args, **kwargs: valid_result)
    run = solve_confirmed_dayplan(plan)

    assert run.validation.valid is False
    assert run.explanation_facts is None
