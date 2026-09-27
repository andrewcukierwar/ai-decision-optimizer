from types import SimpleNamespace

import pytest

from decision_optimizer.dayplan import DayPlan, SolveStatus, TimeWindow
from decision_optimizer.parsing.dayplan import (
    DEFAULT_MODEL,
    DAYPLAN_EXTRACTION_INSTRUCTIONS,
    DayPlanAPIError,
    DayPlanExtraction,
    DayPlanOutputError,
    DayPlanRefusalError,
    MissingAPIKeyError,
    parse_dayplan,
    solve_from_text,
)
from scripts.evaluate_dayplan_nl import formulation_differences, matches_expected


def plan_with_one_task():
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "13:00"},
            "work_window": None,
            "fixed_events": [],
            "tasks": [
                {"name": "focus", "duration_min": 60, "mode": "active", "required": True}
            ],
            "precedences": [],
            "preferences": [],
        }
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def parsed_response(plan):
    return SimpleNamespace(output_parsed=plan, refusal=None)


def complete_extraction(plan):
    return DayPlanExtraction(plan=plan, missing_info=[])


def incomplete_extraction(*questions):
    return DayPlanExtraction(plan=None, missing_info=list(questions))


def test_parser_uses_responses_pydantic_parsing_and_configurable_model(monkeypatch):
    plan = plan_with_one_task()
    extraction = complete_extraction(plan)
    client = FakeClient([parsed_response(extraction)])
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    result = parse_dayplan("Plan a focus block.", client=client)

    assert result == extraction
    call = client.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is DayPlanExtraction
    assert call["input"][0]["role"] == "system"
    assert call["input"][1]["content"] == "Plan a focus block."


def test_parser_uses_default_model_when_environment_is_unset(monkeypatch):
    plan = plan_with_one_task()
    client = FakeClient([parsed_response(complete_extraction(plan))])
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    parse_dayplan("Plan a focus block.", client=client)

    assert client.responses.calls[0]["model"] == DEFAULT_MODEL


def test_one_round_clarification_includes_original_plan_and_answer():
    first = incomplete_extraction("What time does the day start and end?")
    final = plan_with_one_task()
    client = FakeClient([parsed_response(complete_extraction(final))])

    result = parse_dayplan(
        "I need a focus block today.",
        previous_extraction=first,
        clarification="Use a 09:00-13:00 day and make focus 60 minutes.",
        client=client,
    )

    assert result.plan == final
    request = client.responses.calls[0]["input"][1]["content"]
    assert "I need a focus block today." in request
    assert "What time does the day start and end?" in request
    assert "Use a 09:00-13:00 day" in request
    assert "Re-extract the full original scheduling problem" in request
    assert "Do not return only the clarified subset" in request


def test_missing_info_stops_orchestration_before_solving():
    incomplete = incomplete_extraction("What is the planning horizon?")
    result = solve_from_text(
        "Please plan my day.", client=FakeClient([parsed_response(incomplete)])
    )

    assert result.extraction == incomplete
    assert result.plan is None
    assert result.solution is None
    assert result.validation is None


def test_parse_solve_validate_orchestration_is_deterministic():
    plan = plan_with_one_task()
    result = solve_from_text(
        "Schedule one focus block.",
        client=FakeClient([parsed_response(complete_extraction(plan))]),
    )

    assert result.solution is not None
    assert result.solution.status == SolveStatus.OPTIMAL
    assert result.validation is not None
    assert result.validation.valid, result.validation.errors


def test_missing_api_key_is_a_typed_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        parse_dayplan("Plan my day.")


def test_api_errors_are_wrapped_as_application_errors():
    client = FakeClient([RuntimeError("service unavailable")])

    with pytest.raises(DayPlanAPIError, match="service unavailable"):
        parse_dayplan("Plan my day.", client=client)


def test_refusal_is_reported_as_a_typed_error():
    client = FakeClient([SimpleNamespace(output_parsed=None, refusal="not allowed")])

    with pytest.raises(DayPlanRefusalError, match="not allowed"):
        parse_dayplan("Plan my day.", client=client)


def test_missing_or_invalid_structured_output_is_reported():
    no_output = FakeClient([SimpleNamespace(output_parsed=None)])
    with pytest.raises(DayPlanOutputError, match="no parsed"):
        parse_dayplan("Plan my day.", client=no_output)

    invalid = FakeClient([SimpleNamespace(output_parsed={"plan": {"horizon": {"start": "bad"}}})])
    with pytest.raises(DayPlanOutputError, match="valid DayPlanExtraction"):
        parse_dayplan("Plan my day.", client=invalid)


def test_incomplete_extraction_does_not_construct_a_partial_dayplan():
    extraction = incomplete_extraction("How long should the workout take?")
    client = FakeClient([parsed_response(extraction)])

    result = parse_dayplan("Schedule a workout today.", client=client)

    assert result.plan is None
    assert result.missing_info == ["How long should the workout take?"]


def test_core_dayplan_still_requires_a_horizon():
    with pytest.raises(ValueError):
        DayPlan.model_validate(
            {
                "horizon": None,
                "work_window": None,
                "fixed_events": [],
                "tasks": [],
                "precedences": [],
                "preferences": [],
            }
        )


def test_dayplan_has_no_redundant_missing_info_channel():
    assert "missing_info" not in DayPlan.model_fields
    assert "only when the user explicitly gives work hours" in DAYPLAN_EXTRACTION_INSTRUCTIONS
    assert "early afternoon=12:00-15:00" in DAYPLAN_EXTRACTION_INSTRUCTIONS
    assert "max_gap_min=0" in DAYPLAN_EXTRACTION_INSTRUCTIONS


def test_structured_validation_error_preserves_location_and_reason():
    invalid = FakeClient(
        [
            SimpleNamespace(
                output_parsed={
                    "plan": {
                        "horizon": {"start": "09:00", "end": "13:00"},
                        "tasks": [{"name": "focus", "mode": "active"}],
                    },
                    "missing_info": [],
                }
            )
        ]
    )

    with pytest.raises(DayPlanOutputError) as error:
        parse_dayplan("Schedule focus.", client=invalid)

    assert "plan.tasks.0.duration_min" in str(error.value)
    assert "Field required" in str(error.value)


def test_evaluator_reports_consequential_field_differences():
    extraction = complete_extraction(plan_with_one_task())
    expected = {
        "horizon": {"start": "09:00", "end": "13:00"},
        "work_window": None,
        "tasks": {
            "focus": {"duration_min": 30, "mode": "passive", "required": True}
        },
        "fixed_events": {},
        "precedences": [],
        "preferences": [],
    }

    differences = formulation_differences(extraction, expected)

    assert "task focus duration expected 30, actual 60" in differences
    assert "task focus mode expected passive, actual active" in differences
    assert not matches_expected(extraction, expected)


def test_evaluator_accepts_harmless_task_name_variation():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "13:00"},
            "work_window": None,
            "fixed_events": [],
            "tasks": [
                {
                    "name": "grocery trip",
                    "duration_min": 30,
                    "mode": "active",
                    "required": True,
                }
            ],
            "precedences": [],
            "preferences": [],
        }
    )
    extraction = complete_extraction(plan)
    expected = {
        "horizon": {"start": "09:00", "end": "13:00"},
        "work_window": None,
        "fixed_events": {},
        "tasks": {
            "groceries": {"duration_min": 30, "mode": "active", "required": True}
        },
        "precedences": [],
        "preferences": [],
    }

    assert matches_expected(extraction, expected)


def test_evaluator_catches_inferred_work_window():
    plan = plan_with_one_task().model_copy(
        update={"work_window": TimeWindow(start="09:00", end="13:00")}
    )
    expected = {
        "horizon": {"start": "09:00", "end": "13:00"},
        "work_window": None,
        "fixed_events": {},
        "tasks": {"focus": {"duration_min": 60, "mode": "active", "required": True}},
        "precedences": [],
        "preferences": [],
    }

    differences = formulation_differences(complete_extraction(plan), expected)

    assert "expected work window null, actual 09:00-13:00" in differences
