import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision_optimizer.shift_schedule import (
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    RequiredDaysOffRule,
    ShiftSchedule,
    SolveStatus,
)
from decision_optimizer.parsing.shift_schedule import (
    DEFAULT_MODEL,
    SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS,
    ShiftScheduleAPIError,
    ShiftScheduleExtraction,
    ShiftScheduleMissingAPIKeyError,
    ShiftScheduleOutputError,
    ShiftScheduleRefusalError,
    parse_shift_schedule,
    solve_from_text,
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


def sample_schedule():
    return ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "front", "day": "2026-10-05", "start": "09:00", "end": "13:00", "location": "store", "required_staff": 1}
            ],
            "employees": [
                {"name": "Ava", "max_hours": 8, "eligible_locations": ["store"]}
            ],
            "rules": [],
        }
    )


def parsed_response(extraction):
    return SimpleNamespace(output_parsed=extraction, refusal=None)


def complete_extraction(schedule):
    return ShiftScheduleExtraction(schedule=schedule, missing_info=[])


def incomplete_extraction(*questions):
    return ShiftScheduleExtraction(schedule=None, missing_info=list(questions))


def test_parser_uses_structured_shift_schedule_output_and_configurable_model(monkeypatch):
    extraction = complete_extraction(sample_schedule())
    client = FakeClient([parsed_response(extraction)])
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    result = parse_shift_schedule("Staff the front desk.", client=client)

    assert result == extraction
    call = client.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["text_format"] is ShiftScheduleExtraction
    assert call["input"][0]["content"] == SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS


def test_structured_output_schema_uses_anyof_not_oneof_for_rules():
    extraction_schema = ShiftScheduleExtraction.model_json_schema()
    schedule_schema = ShiftSchedule.model_json_schema()
    rules_schema = schedule_schema["properties"]["rules"]

    assert "anyOf" in rules_schema["items"]
    assert "oneOf" not in rules_schema["items"]
    assert "oneOf" not in json.dumps(rules_schema)
    assert "anyOf" in extraction_schema["properties"]["schedule"]
    assert "oneOf" not in json.dumps(extraction_schema)


@pytest.mark.parametrize(
    ("rule_payload", "rule_class"),
    [
        ({"type": "minimum_rest", "min_rest_hours": 8}, MinimumRestRule),
        ({"type": "maximum_consecutive_days", "max_days": 3}, MaximumConsecutiveDaysRule),
        (
            {
                "type": "required_days_off",
                "employee_name": "Ava",
                "days": ["2026-10-06"],
            },
            RequiredDaysOffRule,
        ),
    ],
)
def test_each_closed_rule_variant_validates_to_its_concrete_model(rule_payload, rule_class):
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "front",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                }
            ],
            "employees": [
                {"name": "Ava", "max_hours": 8, "eligible_locations": ["store"]}
            ],
            "rules": [rule_payload],
        }
    )

    assert isinstance(schedule.rules[0], rule_class)


@pytest.mark.parametrize(
    "rule_payload",
    [
        {"type": "minimum_rest", "min_rest_hours": -1},
        {"type": "maximum_consecutive_days"},
        {
            "type": "required_days_off",
            "employee_name": "Ava",
            "days": ["2026-10-06", "2026-10-06"],
        },
        {"type": "unsupported_rule", "value": 1},
    ],
)
def test_malformed_closed_rule_payloads_fail_validation(rule_payload):
    with pytest.raises(ValueError):
        ShiftSchedule.model_validate(
            {
                "shifts": [
                    {
                        "id": "front",
                        "day": "2026-10-05",
                        "start": "09:00",
                        "end": "13:00",
                        "location": "store",
                        "required_staff": 1,
                    }
                ],
                "employees": [
                    {"name": "Ava", "max_hours": 8, "eligible_locations": ["store"]}
                ],
                "rules": [rule_payload],
            }
        )


def test_parser_uses_default_model_when_environment_is_unset(monkeypatch):
    client = FakeClient([parsed_response(complete_extraction(sample_schedule()))])
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    parse_shift_schedule("Staff the front desk.", client=client)

    assert client.responses.calls[0]["model"] == DEFAULT_MODEL


def test_one_round_clarification_includes_original_extraction_and_answer():
    first = incomplete_extraction("What date and time are the shifts?")
    final = sample_schedule()
    client = FakeClient([parsed_response(complete_extraction(final))])

    result = parse_shift_schedule(
        "Please staff the store.",
        previous_extraction=first,
        clarification="Use the front shift on 2026-10-05 from 09:00 to 13:00.",
        client=client,
    )

    assert result.schedule == final
    request = client.responses.calls[0]["input"][1]["content"]
    assert "Please staff the store." in request
    assert "What date and time are the shifts?" in request
    assert "2026-10-05" in request


def test_missing_info_stops_orchestration_before_solving():
    extraction = incomplete_extraction("Which dates, times, and staffing levels apply?")
    result = solve_from_text(
        "Build next week's rota.", client=FakeClient([parsed_response(extraction)])
    )

    assert result.schedule is None
    assert result.solution is None
    assert result.validation is None


def test_parse_solve_validate_orchestration_is_deterministic():
    result = solve_from_text(
        "Staff one front shift.",
        client=FakeClient([parsed_response(complete_extraction(sample_schedule()))]),
    )

    assert result.solution is not None
    assert result.solution.status == SolveStatus.OPTIMAL
    assert result.validation is not None
    assert result.validation.valid, result.validation.errors


def test_missing_api_key_is_a_typed_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ShiftScheduleMissingAPIKeyError, match="OPENAI_API_KEY"):
        parse_shift_schedule("Staff the store.")


def test_api_errors_are_wrapped_as_application_errors():
    client = FakeClient([RuntimeError("service unavailable")])

    with pytest.raises(ShiftScheduleAPIError, match="service unavailable"):
        parse_shift_schedule("Staff the store.", client=client)


def test_refusal_and_invalid_structured_output_are_typed_errors():
    refusal = FakeClient([SimpleNamespace(output_parsed=None, refusal="not allowed")])
    with pytest.raises(ShiftScheduleRefusalError, match="not allowed"):
        parse_shift_schedule("Staff the store.", client=refusal)

    no_output = FakeClient([SimpleNamespace(output_parsed=None)])
    with pytest.raises(ShiftScheduleOutputError, match="no parsed"):
        parse_shift_schedule("Staff the store.", client=no_output)

    invalid = FakeClient([SimpleNamespace(output_parsed={"schedule": None, "missing_info": []})])
    with pytest.raises(ShiftScheduleOutputError, match="valid ShiftScheduleExtraction"):
        parse_shift_schedule("Staff the store.", client=invalid)


def test_nl_fixture_has_three_cases_including_clarification_case():
    fixture = Path(__file__).parent / "fixtures" / "shift_schedule_nl_eval.json"
    cases = json.loads(fixture.read_text())

    assert len(cases) == 3
    assert any(case["expected"].get("missing_info") for case in cases)
