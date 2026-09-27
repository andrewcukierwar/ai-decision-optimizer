"""Translate DayPlan natural language into the existing typed schema.

The model is used only at this boundary.  It never sees solver variables and
never selects or explains a schedule; the deterministic DayPlan core remains
responsible for solving and validation.
"""

import json
import os
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


DEFAULT_MODEL = "gpt-6-luna"

DAYPLAN_EXTRACTION_INSTRUCTIONS = """You extract one personal-day scheduling problem into the provided DayPlan schema.

Translate only facts supported by the user's text. Preserve DayPlan semantics exactly; do not generate solver code, choose a schedule, or write optimization reasoning.

- Use HH:MM for times and integer minutes for durations.
- An active task requires the person's attention. A passive task can run while the person does something else.
- Fixed meetings/events belong in fixed_events, not tasks.
- Explicitly required activities have required=true. Do not invent optional activities.
- Clear hard language (must, need to, no later than, by, cannot overlap) becomes hard constraints.
- Clear soft language (prefer, ideally, would like) becomes a supported Preference. Use weight=1 when no numeric priority is stated; preserve a stated priority.
- Sequential statements such as laundry wash -> transfer -> dry become precedences.
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
        if self.plan is not None and self.plan.missing_info:
            raise ValueError("complete plan must not contain DayPlan.missing_info")
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
        response = client.responses.parse(
            model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            input=[
                {"role": "system", "content": DAYPLAN_EXTRACTION_INSTRUCTIONS},
                {"role": "user", "content": request},
            ],
            text_format=DayPlanExtraction,
        )
    except ValidationError as exc:
        raise DayPlanOutputError(
            "OpenAI returned structured data that failed DayPlanExtraction validation"
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
        raise DayPlanOutputError("The model returned no parsed DayPlanExtraction output")

    try:
        return (
            parsed
            if isinstance(parsed, DayPlanExtraction)
            else DayPlanExtraction.model_validate(parsed)
        )
    except ValidationError as exc:
        raise DayPlanOutputError(
            "The parsed model output is not a valid DayPlanExtraction: %s" % exc
        ) from exc


def solve_from_text(
    user_text: str,
    *,
    previous_extraction: Optional[DayPlanExtraction] = None,
    clarification: Optional[str] = None,
    client: Any = None,
    model: Optional[str] = None,
    time_limit_seconds: Optional[float] = 10.0,
) -> TextSolveResult:
    """Parse, then solve and independently validate when interpretation is complete."""

    extraction = parse_dayplan(
        user_text,
        previous_extraction=previous_extraction,
        clarification=clarification,
        client=client,
        model=model,
    )
    if extraction.plan is None:
        return TextSolveResult(extraction=extraction)

    solution = solve_day_plan(extraction.plan, time_limit_seconds=time_limit_seconds)
    validation = validate_solution(extraction.plan, solution)
    return TextSolveResult(extraction=extraction, solution=solution, validation=validation)


def _create_openai_client() -> Any:
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is required for live DayPlan parsing"
        )
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
        + "\n\nReturn a corrected DayPlanExtraction, preserving supported facts from the original request."
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
