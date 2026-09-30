"""Translate workforce natural language into the typed ShiftSchedule schema.

The model is used only to populate the typed extraction boundary.  It never
selects employees, creates solver constraints, or explains a result.
"""

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import BaseModel, ValidationError, model_validator

from decision_optimizer.shift_schedule import (
    ShiftSchedule,
    ShiftScheduleSolution,
    ValidationReport,
    solve_shift_schedule,
    validate_solution,
)
from decision_optimizer.config import DEFAULT_OPENAI_MODEL, openai_api_key, openai_model
from decision_optimizer.telemetry import RunTelemetry


DEFAULT_MODEL = DEFAULT_OPENAI_MODEL

SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS = """You extract one small workforce scheduling problem into the provided ShiftSchedule schema.

Translate only facts supported by the user's text. Do not generate solver code,
choose assignments, or write optimization reasoning.

- Every shift needs an id, ISO calendar date, HH:MM start and end, location, and required_staff.
- Every employee needs a name, max_hours, eligible_locations, and any stated unavailable windows or shifts.
- Use unavailable.shift_id for a named unavailable shift, day alone for a full unavailable day, and day plus start/end for a time window.
- preferred_shifts is a soft preference list of shift ids. Do not invent preferences.
- Supported hard rules are minimum_rest (whole hours), maximum_consecutive_days, and required_days_off for a named employee.
- If the user gives no numeric objective weights, preserve the schema defaults exactly: preference_penalty=1 and fairness=1.
- Do not disable an objective by setting its weight to 0 unless the user explicitly asks to ignore or disable that objective.
- If the user expresses a preference or fairness goal without a numeric weight, use the default weight 1.
- Do not invent dates, shift times, staffing requirements, eligibility, availability, max hours, or employee names.
- Use missing_info only for genuinely missing or ambiguous factual information required by the schema.
- Do not return a partial ShiftSchedule. If the schedule cannot be formulated safely, return schedule=null.
- Do not assess whether the schedule can be staffed. If all schema facts are known, return the complete ShiftSchedule and let the deterministic optimizer determine feasibility.
- Do not ask the user to add employees, increase capacity, relax constraints, change requirements, or otherwise repair likely infeasibility during extraction.
- Once a clarification supplies the missing facts, reconstruct the complete original problem, incorporate the clarification, and return the complete ShiftSchedule.
- If all consequential factual information is present, return missing_info=[] even when the hard constraints appear contradictory or impossible.
"""


class ShiftScheduleError(Exception):
    """Base class for application-level ShiftSchedule parsing errors."""


class ShiftScheduleInputError(ShiftScheduleError):
    """The application was given an invalid parse or clarification request."""


class ShiftScheduleMissingAPIKeyError(ShiftScheduleError):
    """The OpenAI client cannot be created because the API key is absent."""


class ShiftScheduleAPIError(ShiftScheduleError):
    """The model request failed before a usable result was returned."""


class ShiftScheduleRefusalError(ShiftScheduleError):
    """The model explicitly refused the extraction request."""


class ShiftScheduleOutputError(ShiftScheduleError):
    """The model response was absent or invalid for the extraction boundary."""


class ShiftScheduleExtraction(BaseModel):
    """Structured boundary result for a complete schedule or clarification."""

    schedule: Optional[ShiftSchedule]
    missing_info: List[str]

    @model_validator(mode="after")
    def validate_completeness(self) -> "ShiftScheduleExtraction":
        if self.schedule is None and not self.missing_info:
            raise ValueError("missing_info is required when schedule is null")
        if self.schedule is not None and self.missing_info:
            raise ValueError("schedule must be null when missing_info is populated")
        return self


@dataclass
class ShiftScheduleTextSolveResult:
    """Typed extraction plus deterministic solve and validation when complete."""

    extraction: ShiftScheduleExtraction
    solution: Optional[ShiftScheduleSolution] = None
    validation: Optional[ValidationReport] = None

    @property
    def schedule(self) -> Optional[ShiftSchedule]:
        return self.extraction.schedule


def parse_shift_schedule(
    user_text: str,
    *,
    previous_extraction: Optional[ShiftScheduleExtraction] = None,
    clarification: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
    request_timeout_seconds: Optional[float] = None,
) -> ShiftScheduleExtraction:
    """Parse natural language into a complete schedule or one clarification request."""

    if not isinstance(user_text, str) or not user_text.strip():
        raise ShiftScheduleInputError("user_text must be a non-empty string")
    if (previous_extraction is None) != (clarification is None):
        raise ShiftScheduleInputError(
            "previous_extraction and clarification must be supplied together for a clarification round"
        )
    if clarification is not None and not clarification.strip():
        raise ShiftScheduleInputError("clarification must be a non-empty string")

    request = _build_user_request(user_text, previous_extraction, clarification)
    if client is None:
        client = _create_openai_client()

    try:
        request_kwargs = {
            "model": model or openai_model(),
            "input": [
                {"role": "system", "content": SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS},
                {"role": "user", "content": request},
            ],
            "text_format": ShiftScheduleExtraction,
        }
        if request_timeout_seconds is not None:
            request_kwargs["timeout"] = request_timeout_seconds
        if telemetry is None:
            response = client.responses.parse(**request_kwargs)
        else:
            with telemetry.track("llm"):
                response = client.responses.parse(**request_kwargs)
            telemetry.record_openai_response(response)
    except ValidationError as exc:
        raise ShiftScheduleOutputError(
            "OpenAI returned structured data that failed ShiftScheduleExtraction validation: %s"
            % _format_validation_error(exc)
        ) from exc
    except ShiftScheduleError:
        raise
    except Exception as exc:
        raise ShiftScheduleAPIError(
            "ShiftSchedule parsing request failed: %s" % exc
        ) from exc

    refusal = _response_refusal(response)
    if refusal:
        raise ShiftScheduleRefusalError(
            "The model refused ShiftSchedule extraction: " + refusal
        )

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise ShiftScheduleOutputError(
            "The model returned no parsed ShiftScheduleExtraction output"
        )

    try:
        return (
            parsed
            if isinstance(parsed, ShiftScheduleExtraction)
            else ShiftScheduleExtraction.model_validate(parsed)
        )
    except ValidationError as exc:
        raise ShiftScheduleOutputError(
            "The parsed model output is not a valid ShiftScheduleExtraction: %s"
            % _format_validation_error(exc)
        ) from exc


def solve_from_text(
    user_text: str,
    *,
    previous_extraction: Optional[ShiftScheduleExtraction] = None,
    clarification: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
    time_limit_seconds: Optional[float] = 10.0,
) -> ShiftScheduleTextSolveResult:
    """Parse, solve, and independently validate a complete extraction."""

    extraction = parse_shift_schedule(
        user_text,
        previous_extraction=previous_extraction,
        clarification=clarification,
        client=client,
        model=model,
        telemetry=telemetry,
    )
    if extraction.schedule is None:
        return ShiftScheduleTextSolveResult(extraction=extraction)

    solution = solve_shift_schedule(
        extraction.schedule, time_limit_seconds=time_limit_seconds
    )
    validation = validate_solution(extraction.schedule, solution)
    return ShiftScheduleTextSolveResult(
        extraction=extraction, solution=solution, validation=validation
    )


def _create_openai_client() -> Any:
    if not openai_api_key():
        raise ShiftScheduleMissingAPIKeyError(
            "OPENAI_API_KEY is required for live ShiftSchedule parsing"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ShiftScheduleAPIError(
            "The openai package is not installed; install the project dependencies"
        ) from exc
    try:
        return OpenAI()
    except Exception as exc:
        raise ShiftScheduleAPIError(
            "Could not create the OpenAI client: %s" % exc
        ) from exc


def _build_user_request(
    original: str,
    previous_extraction: Optional[ShiftScheduleExtraction],
    clarification: Optional[str],
) -> str:
    if previous_extraction is None:
        return original
    previous_json = json.dumps(previous_extraction.model_dump(mode="json"), indent=2)
    return (
        "Original user request:\n"
        + original
        + "\n\nPrior extraction:\n"
        + previous_json
        + "\n\nUser clarification:\n"
        + clarification
        + "\n\nReturn a corrected ShiftScheduleExtraction, preserving supported facts from the original request."
    )


def _response_refusal(response: Any) -> Optional[str]:
    direct_refusal = getattr(response, "refusal", None)
    if direct_refusal:
        return str(direct_refusal)

    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def _format_validation_error(error: ValidationError) -> str:
    details = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "$"
        details.append("%s: %s" % (location, item.get("msg", "validation error")))
    return "; ".join(details) or str(error)
