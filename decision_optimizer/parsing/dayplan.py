"""Translate DayPlan natural language into the existing typed schema.

The model is used only at this boundary.  It never sees solver variables and
never selects or explains a schedule; the deterministic DayPlan core remains
responsible for solving and validation.
"""

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import BaseModel, ValidationError, model_validator

from decision_optimizer.dayplan import (
    DayPlan,
    DayPlanSolution,
    ValidationReport,
    solve_day_plan,
    validate_solution,
)
from decision_optimizer.config import DEFAULT_OPENAI_MODEL, openai_api_key, openai_model
from decision_optimizer.telemetry import RunTelemetry


DEFAULT_MODEL = DEFAULT_OPENAI_MODEL

DAYPLAN_EXTRACTION_INSTRUCTIONS = """You extract one personal-day scheduling problem into the provided DayPlan schema.

Translate only facts supported by the user's text. Preserve DayPlan semantics exactly; do not generate solver code, choose a schedule, or write optimization reasoning.

- Use HH:MM for times and integer minutes for durations.
- An active task requires the person's attention. A passive task can run while the person does something else.
- Fixed meetings/events belong in fixed_events, not tasks.
- Fixed event names are display labels and may repeat; preserve repeated labels and do not invent numbering solely to make them unique.
- Explicitly required activities have required=true. Do not invent optional activities.
- Clear hard language (must, need to, no later than, by, cannot overlap) becomes hard constraints.
- Clear soft language (prefer, ideally, would like) becomes a supported Preference. Use weight=1 when no numeric priority is stated; preserve a stated priority.
- Supported soft timing preferences include finish_before, minimize_work_interruptions, and preferred_window. Use preferred_window with task, start, end, and weight when the user supports a time range; its penalty is the minutes outside that range.
- Phrases such as "around 12 to 1" can support preferred_window 12:00-13:00. "Before lunch" can support finish_before at the explicitly stated lunch start when the lunch window anchors noon. When a daypart convention is needed for a soft preference, use morning=08:00-12:00, early afternoon=12:00-15:00, afternoon=12:00-17:00, and evening=17:00-21:00. These are preferred windows only, never hard constraints; do not invent exact times for vague language outside these conventions.
- An explicit nearby boundary such as "crowded around 16:00" can support finish_before at 16:00 in addition to an early-afternoon preferred_window when both are grounded by the user's words.
- Populate work_window only when the user explicitly gives work hours or a work window. Do not infer it from the general horizon or phrases such as "keep the day open".
- Sequential statements such as laundry wash -> transfer -> dry become precedences. When wording clearly describes a continuous physical or process chain (for example prep -> bake, load -> run machine, wash -> transfer -> dry, or cook -> remove from oven), use max_gap_min=0 between every adjacent stage, including active -> passive and passive -> active handoffs. Do not apply max_gap_min=0 to generic "A before B" statements when immediate continuation is not clearly implied.
- Before returning a complete DayPlan, re-read the entire user request and account for every explicitly stated required activity, duration, fixed event, active/passive distinction, ordering/dependency, hard time constraint, and supported soft preference. Do not silently drop a consequential statement. If it cannot be represented safely using this schema, return a concise missing_info question instead.
- Context such as a weekday or working from home is not a modeled DayPlan constraint unless it changes a supported timing or resource fact; do not invent unsupported schema fields.
- Do not invent durations, deadlines, work windows, horizons, or other consequential facts.
- If information genuinely needed to formulate the problem is missing or ambiguous, set plan=null and put concise questions/issues in missing_info instead of guessing. Do not put stylistic or nonessential uncertainty there.
- If all consequential information is present, set missing_info=[] and put the complete valid DayPlan in plan.
- Do not return a partial DayPlan. If a required task duration, time bound, horizon, or other consequential fact is missing, return plan=null.
"""


class DayPlanError(Exception):
    """Base class for application-level natural-language DayPlan errors."""


class DayPlanInputError(DayPlanError):
    """The application was given an invalid parse or clarification request."""


class MissingAPIKeyError(DayPlanError):
    """The OpenAI client cannot be created because the API key is absent."""


class DayPlanAPIError(DayPlanError):
    """The model request failed before a usable structured result was returned."""


class DayPlanRefusalError(DayPlanError):
    """The model explicitly refused the extraction request."""


class DayPlanOutputError(DayPlanError):
    """The model response was absent or could not be validated as a DayPlan."""


class DayPlanExtraction(BaseModel):
    """Structured boundary result for either a complete plan or clarification."""

    plan: Optional[DayPlan]
    missing_info: List[str]

    @model_validator(mode="after")
    def validate_completeness(self) -> "DayPlanExtraction":
        if self.plan is None and not self.missing_info:
            raise ValueError("missing_info is required when plan is null")
        if self.plan is not None and self.missing_info:
            raise ValueError("plan must be null when missing_info is populated")
        return self


@dataclass
class TextSolveResult:
    """The typed interpretation and, when possible, deterministic solve result."""

    extraction: DayPlanExtraction
    solution: Optional[DayPlanSolution] = None
    validation: Optional[ValidationReport] = None

    @property
    def plan(self) -> Optional[DayPlan]:
        """Expose the complete plan for callers that do not need wrapper details."""

        return self.extraction.plan


def parse_dayplan(
    user_text: str,
    *,
    previous_extraction: Optional[DayPlanExtraction] = None,
    clarification: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
) -> DayPlanExtraction:
    """Parse natural language into a complete plan or one clarification request.

    For clarification, pass the original request again together with the first
    extraction and the user's answer::

        parse_dayplan(
            original,
            previous_extraction=first_extraction,
            clarification=answer,
        )
    """

    if not isinstance(user_text, str) or not user_text.strip():
        raise DayPlanInputError("user_text must be a non-empty string")
    if (previous_extraction is None) != (clarification is None):
        raise DayPlanInputError(
            "previous_extraction and clarification must be supplied together for a clarification round"
        )
    if clarification is not None and not clarification.strip():
        raise DayPlanInputError("clarification must be a non-empty string")

    request = _build_user_request(user_text, previous_extraction, clarification)
    if client is None:
        client = _create_openai_client()

    try:
        if telemetry is None:
            response = client.responses.parse(
                model=model or openai_model(),
                input=[
                    {"role": "system", "content": DAYPLAN_EXTRACTION_INSTRUCTIONS},
                    {"role": "user", "content": request},
                ],
                text_format=DayPlanExtraction,
            )
        else:
            with telemetry.track("llm"):
                response = client.responses.parse(
                    model=model or openai_model(),
                    input=[
                        {"role": "system", "content": DAYPLAN_EXTRACTION_INSTRUCTIONS},
                        {"role": "user", "content": request},
                    ],
                    text_format=DayPlanExtraction,
                )
            telemetry.record_openai_response(response)
    except ValidationError as exc:
        raise DayPlanOutputError(
            "OpenAI returned structured data that failed DayPlanExtraction validation: %s"
            % _format_validation_error(exc)
        ) from exc
    except DayPlanError:
        raise
    except Exception as exc:
        raise DayPlanAPIError("DayPlan parsing request failed: %s" % exc) from exc

    refusal = _response_refusal(response)
    if refusal:
        raise DayPlanRefusalError("The model refused DayPlan extraction: " + refusal)

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise DayPlanOutputError(
            "The model returned no parsed DayPlanExtraction output"
        )

    try:
        return (
            parsed
            if isinstance(parsed, DayPlanExtraction)
            else DayPlanExtraction.model_validate(parsed)
        )
    except ValidationError as exc:
        raise DayPlanOutputError(
            "The parsed model output is not a valid DayPlanExtraction: %s"
            % _format_validation_error(exc)
        ) from exc


def solve_from_text(
    user_text: str,
    *,
    previous_extraction: Optional[DayPlanExtraction] = None,
    clarification: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
    time_limit_seconds: Optional[float] = 10.0,
) -> TextSolveResult:
    """Parse, then solve and independently validate when interpretation is complete."""

    extraction = parse_dayplan(
        user_text,
        previous_extraction=previous_extraction,
        clarification=clarification,
        client=client,
        model=model,
        telemetry=telemetry,
    )
    if extraction.plan is None:
        return TextSolveResult(extraction=extraction)

    solution = solve_day_plan(extraction.plan, time_limit_seconds=time_limit_seconds)
    validation = validate_solution(extraction.plan, solution)
    return TextSolveResult(
        extraction=extraction, solution=solution, validation=validation
    )


def _create_openai_client() -> Any:
    if not openai_api_key():
        raise MissingAPIKeyError("OPENAI_API_KEY is required for live DayPlan parsing")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DayPlanAPIError(
            "The openai package is not installed; install the project dependencies"
        ) from exc
    try:
        return OpenAI()
    except Exception as exc:
        raise DayPlanAPIError("Could not create the OpenAI client: %s" % exc) from exc


def _build_user_request(
    original: str,
    previous_extraction: Optional[DayPlanExtraction],
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
        + "\n\nRe-extract the full original scheduling problem using the clarification. Do not return only the clarified subset. Re-read the original request and preserve every supported task, duration, fixed event, mode, dependency, hard constraint, and soft preference before returning the corrected DayPlanExtraction."
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
    """Keep Pydantic locations and reasons while omitting input values."""

    details = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "$"
        details.append("%s: %s" % (location, item.get("msg", "validation error")))
    return "; ".join(details) or str(error)
