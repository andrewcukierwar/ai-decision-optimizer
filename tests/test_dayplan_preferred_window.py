import json
from pathlib import Path
from types import SimpleNamespace

from decision_optimizer.dayplan import DayPlan, solve_day_plan, validate_solution
from decision_optimizer.explanations import (
    build_dayplan_explanation_payload,
    render_dayplan_explanation,
)
from decision_optimizer.parsing.dayplan import DayPlanExtraction, parse_dayplan
from scripts.evaluate_dayplan_nl import formulation_differences


FIXTURES = Path(__file__).parent / "fixtures"


def _preferred_window_plan(
    *, fixed_events=None, horizon_end="17:00", preferred_start="10:00", preferred_end="10:15"
):
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": horizon_end},
            "fixed_events": fixed_events or [],
            "tasks": [
                {
                    "name": "lunch",
                    "duration_min": 30,
                    "mode": "active",
                    "required": True,
                }
            ],
            "precedences": [],
            "preferences": [
                {
                    "type": "preferred_window",
                    "task": "lunch",
                    "start": preferred_start,
                    "end": preferred_end,
                    "weight": 1,
                }
            ],
        }
    )


def test_preferred_window_can_have_zero_penalty():
    plan = _preferred_window_plan(preferred_end="11:00")
    result = solve_day_plan(plan)

    assert result.objective_value == 0
    assert result.preference_penalties[0].amount == 0
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def test_preferred_window_nonzero_optimum_is_materialized():
    plan = _preferred_window_plan(
        fixed_events=[{"name": "meeting", "start": "09:00", "end": "10:00"}]
    )
    result = solve_day_plan(plan)

    assert result.objective_value == 15
    assert result.preference_penalties[0].amount == 15
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def test_preferred_window_starting_before_window_has_known_penalty():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {
                    "name": "early_task",
                    "duration_min": 30,
                    "mode": "active",
                    "latest_end": "09:30",
                    "required": True,
                }
            ],
            "preferences": [
                {
                    "type": "preferred_window",
                    "task": "early_task",
                    "start": "10:00",
                    "end": "11:00",
                    "weight": 1,
                }
            ],
        }
    )
    result = solve_day_plan(plan)

    assert result.objective_value == 60
    assert result.preference_penalties[0].amount == 60
    assert validate_solution(plan, result).valid


def test_validator_catches_corrupted_preferred_window_penalty():
    plan = _preferred_window_plan()
    result = solve_day_plan(plan)
    result.preference_penalties[0].amount += 1

    report = validate_solution(plan, result)

    assert report.valid is False
    assert any("preference penalty" in error for error in report.errors)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.responses = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _real_request_plan():
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "18:00"},
            "work_window": {"start": "09:00", "end": "18:00"},
            "fixed_events": [
                {"name": "meeting", "start": "10:30", "end": "11:00"}
            ],
            "tasks": [
                {"name": "groceries", "duration_min": 30, "mode": "active", "required": True},
                {"name": "wash", "duration_min": 30, "mode": "passive", "required": True},
                {"name": "transfer", "duration_min": 5, "mode": "active", "required": True},
                {"name": "dry", "duration_min": 45, "mode": "passive", "required": True},
                {"name": "lift", "duration_min": 60, "mode": "active", "required": True},
                {"name": "lunch", "duration_min": 30, "mode": "active", "required": True},
            ],
            "precedences": [
                {
                    "before": "wash",
                    "after": "transfer",
                    "min_gap_min": 0,
                    "max_gap_min": 0,
                },
                {
                    "before": "transfer",
                    "after": "dry",
                    "min_gap_min": 0,
                    "max_gap_min": 0,
                },
            ],
            "preferences": [
                {
                    "type": "finish_before",
                    "task": "groceries",
                    "time": "12:00",
                    "weight": 1,
                },
                {
                    "type": "preferred_window",
                    "task": "lunch",
                    "start": "12:00",
                    "end": "13:00",
                    "weight": 1,
                },
                {
                    "type": "preferred_window",
                    "task": "lift",
                    "start": "12:00",
                    "end": "15:00",
                    "weight": 1,
                },
                {
                    "type": "finish_before",
                    "task": "lift",
                    "time": "16:00",
                    "weight": 1,
                },
            ],
        }
    )


def test_real_clarification_reextracts_full_dayplan_and_fixture_checks_completeness():
    case = next(
        case
        for case in json.loads((FIXTURES / "dayplan_nl_eval.json").read_text())
        if case["name"] == "clarified_monday_wfh_full_problem"
    )
    first = DayPlanExtraction(
        plan=None,
        missing_info=["What planning horizon should bound the day?", "How long should lunch take?"],
    )
    final = DayPlanExtraction(plan=_real_request_plan(), missing_info=[])
    client = FakeClient(SimpleNamespace(output_parsed=final, refusal=None))

    result = parse_dayplan(
        case["prompt"],
        previous_extraction=first,
        clarification=case["clarification"],
        client=client,
    )

    clarification_request = client.calls[0]["input"][1]["content"]
    assert case["prompt"] in clarification_request
    assert case["clarification"] in clarification_request
    assert "Re-extract the full original scheduling problem" in clarification_request
    assert "Do not return only the clarified subset" in clarification_request
    assert set(case["expected"]["required_tasks"]) == set(case["expected"]["tasks"])
    assert formulation_differences(result, case["expected"]) == []


def test_repeated_meeting_nl_fixture_accepts_duplicate_fixed_event_names():
    case = next(
        case
        for case in json.loads((FIXTURES / "dayplan_nl_eval.json").read_text())
        if case["name"] == "repeated_generic_meetings"
    )
    expected = case["expected"]
    plan = DayPlan.model_validate(
        {
            "horizon": expected["horizon"],
            "work_window": expected["work_window"],
            "fixed_events": expected["fixed_events"],
            "tasks": [],
            "precedences": [],
            "preferences": [],
        }
    )

    extraction = DayPlanExtraction(plan=plan, missing_info=[])

    assert formulation_differences(extraction, expected) == []


def test_real_laundry_process_chain_requires_immediate_handoffs():
    plan = _real_request_plan()
    result = solve_day_plan(plan)
    assignments = {item.name: item for item in result.assignments}

    assert assignments["transfer"].start == assignments["wash"].end
    assert assignments["dry"].start == assignments["transfer"].end
    assert validate_solution(plan, result).valid


def test_dayplan_explanation_describes_preference_metadata_and_outcomes():
    plan = _real_request_plan()
    result = solve_day_plan(plan)
    validation = validate_solution(plan, result)
    facts = build_dayplan_explanation_payload(plan, result, validation)

    assert facts is not None
    lunch = next(
        item
        for item in facts.preference_penalties
        if item.task == "lunch"
    )
    assert lunch.preferred_start == "12:00"
    assert lunch.preferred_end == "13:00"
    explanation = render_dayplan_explanation(facts)
    assert "lunch was scheduled within its preferred 12:00-13:00 window" in explanation
    assert "groceries finished by the preferred 12:00 target" in explanation
