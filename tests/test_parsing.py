from types import SimpleNamespace

import pytest

from decision_optimizer.dayplan import DayPlan, SolveStatus
from decision_optimizer.parsing.dayplan import (
    DEFAULT_MODEL,
    DayPlanAPIError,
    DayPlanOutputError,
    DayPlanRefusalError,
    MissingAPIKeyError,
    parse_dayplan,
    solve_from_text,
)


def plan_with_one_task(*, missing_info=None):
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
            "missing_info": missing_info or [],
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


def test_parser_uses_responses_pydantic_parsing_and_configurable_model(monkeypatch):
    plan = plan_with_one_task()
    client = FakeClient([parsed_response(plan)])
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    result = parse_dayplan("Plan a focus block.", client=client)

    assert result == plan
    call = client.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is DayPlan
    assert call["input"][0]["role"] == "system"
    assert call["input"][1]["content"] == "Plan a focus block."


def test_parser_uses_default_model_when_environment_is_unset(monkeypatch):
    plan = plan_with_one_task()
    client = FakeClient([parsed_response(plan)])
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    parse_dayplan("Plan a focus block.", client=client)

    assert client.responses.calls[0]["model"] == DEFAULT_MODEL


def test_one_round_clarification_includes_original_plan_and_answer():
    first = DayPlan.model_validate(
        {
            "horizon": None,
            "work_window": None,
            "fixed_events": [],
            "tasks": [],
            "precedences": [],
            "preferences": [],
            "missing_info": ["What time does the day start and end?"],
        }
    )
    final = plan_with_one_task()
    client = FakeClient([parsed_response(final)])

    result = parse_dayplan(
        "I need a focus block today.",
        first_plan=first,
        clarification="Use a 09:00-13:00 day and make focus 60 minutes.",
        client=client,
    )

    assert result == final
    request = client.responses.calls[0]["input"][1]["content"]
    assert "I need a focus block today." in request
    assert "What time does the day start and end?" in request
    assert "Use a 09:00-13:00 day" in request


def test_missing_info_stops_orchestration_before_solving():
    incomplete = DayPlan.model_validate(
        {
            "horizon": None,
            "work_window": None,
            "fixed_events": [],
            "tasks": [],
            "precedences": [],
            "preferences": [],
            "missing_info": ["What is the planning horizon?"],
        }
    )
    result = solve_from_text(
        "Please plan my day.", client=FakeClient([parsed_response(incomplete)])
    )

    assert result.plan == incomplete
    assert result.solution is None
    assert result.validation is None


def test_parse_solve_validate_orchestration_is_deterministic():
    plan = plan_with_one_task()
    result = solve_from_text(
        "Schedule one focus block.", client=FakeClient([parsed_response(plan)])
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

    invalid = FakeClient([SimpleNamespace(output_parsed={"horizon": {"start": "bad"}})])
    with pytest.raises(DayPlanOutputError, match="valid DayPlan"):
        parse_dayplan("Plan my day.", client=invalid)
