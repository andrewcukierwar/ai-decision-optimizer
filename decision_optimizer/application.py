"""UI-facing orchestration kept separate from Streamlit widgets."""

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Optional

from pydantic import ValidationError

from .dayplan import DayPlan, DayPlanSolution, solve_day_plan, validate_solution as validate_dayplan
from .diagnostics import InfeasibilityDiagnostic, diagnose_dayplan_infeasibility, diagnose_shift_schedule_infeasibility
from .explanations import (
    DayPlanExplanationFacts,
    ShiftScheduleExplanationFacts,
    build_dayplan_explanation_payload,
    build_shift_schedule_explanation_payload,
)
from .parsing.dayplan import DayPlanExtraction, parse_dayplan
from .parsing.shift_schedule import (
    ShiftScheduleExtraction,
    parse_shift_schedule,
)
from .shift_schedule import ShiftSchedule, ShiftScheduleSolution, solve_shift_schedule
from .shift_schedule import validate_solution as validate_shift_schedule


class EditedInterpretationError(ValueError):
    """The user-edited JSON did not validate against the typed schema."""


@dataclass
class DayPlanRun:
    plan: DayPlan
    solution: DayPlanSolution
    validation: Any
    diagnosis: Optional[InfeasibilityDiagnostic] = None
    explanation_facts: Optional[DayPlanExplanationFacts] = None


@dataclass
class ShiftScheduleRun:
    schedule: ShiftSchedule
    solution: ShiftScheduleSolution
    validation: Any
    diagnosis: Optional[InfeasibilityDiagnostic] = None
    explanation_facts: Optional[ShiftScheduleExplanationFacts] = None


def validate_edited_dayplan_json(raw_json: str) -> DayPlan:
    """Parse and validate edited DayPlan JSON before it can reach the solver."""

    return _validate_json(raw_json, DayPlan)


def validate_edited_shift_schedule_json(raw_json: str) -> ShiftSchedule:
    """Parse and validate edited ShiftSchedule JSON before it can reach the solver."""

    return _validate_json(raw_json, ShiftSchedule)


def solve_confirmed_dayplan(
    plan: DayPlan, *, time_limit_seconds: Optional[float] = 10.0
) -> DayPlanRun:
    """Solve, independently validate, diagnose if needed, and gate explanation facts."""

    solution = solve_day_plan(plan, time_limit_seconds=time_limit_seconds)
    validation = validate_dayplan(plan, solution)
    diagnosis = None
    if solution.status.value == "infeasible":
        diagnosis = diagnose_dayplan_infeasibility(
            plan, time_limit_seconds=min(time_limit_seconds or 2.0, 2.0)
        )
    explanation_facts = build_dayplan_explanation_payload(plan, solution, validation)
    return DayPlanRun(
        plan=plan,
        solution=solution,
        validation=validation,
        diagnosis=diagnosis,
        explanation_facts=explanation_facts,
    )


def solve_confirmed_shift_schedule(
    schedule: ShiftSchedule, *, time_limit_seconds: Optional[float] = 10.0
) -> ShiftScheduleRun:
    """Solve, independently validate, diagnose if needed, and gate explanation facts."""

    solution = solve_shift_schedule(
        schedule, time_limit_seconds=time_limit_seconds
    )
    validation = validate_shift_schedule(schedule, solution)
    diagnosis = None
    if solution.status.value == "infeasible":
        diagnosis = diagnose_shift_schedule_infeasibility(
            schedule, time_limit_seconds=min(time_limit_seconds or 2.0, 2.0)
        )
    explanation_facts = build_shift_schedule_explanation_payload(
        schedule, solution, validation
    )
    return ShiftScheduleRun(
        schedule=schedule,
        solution=solution,
        validation=validation,
        diagnosis=diagnosis,
        explanation_facts=explanation_facts,
    )


def parse_dayplan_request(
    request: str, *, client: Any = None, model: Optional[str] = None
) -> DayPlanExtraction:
    """Small adapter used by the UI and deterministic tests."""

    return parse_dayplan(request, client=client, model=model)


def clarify_dayplan_request(
    request: str,
    previous: DayPlanExtraction,
    clarification: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> DayPlanExtraction:
    """Run the single supported DayPlan clarification round."""

    return parse_dayplan(
        request,
        previous_extraction=previous,
        clarification=clarification,
        client=client,
        model=model,
    )


def parse_shift_schedule_request(
    request: str, *, client: Any = None, model: Optional[str] = None
) -> ShiftScheduleExtraction:
    """Small adapter used by the UI and deterministic tests."""

    return parse_shift_schedule(request, client=client, model=model)


def clarify_shift_schedule_request(
    request: str,
    previous: ShiftScheduleExtraction,
    clarification: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> ShiftScheduleExtraction:
    """Run the single supported ShiftSchedule clarification round."""

    return parse_shift_schedule(
        request,
        previous_extraction=previous,
        clarification=clarification,
        client=client,
        model=model,
    )


def format_schema_validation_error(error: ValidationError) -> str:
    """Format Pydantic errors without echoing potentially large input values."""

    parts = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "$"
        parts.append("%s: %s" % (location, item.get("msg", "validation error")))
    return "; ".join(parts) or "structured interpretation is invalid"


def _validate_json(raw_json: str, schema: Any) -> Any:
    try:
        payload = json.loads(raw_json)
    except JSONDecodeError as error:
        raise EditedInterpretationError(
            "Invalid JSON at line %d, column %d: %s"
            % (error.lineno, error.colno, error.msg)
        ) from error
    try:
        return schema.model_validate(payload)
    except ValidationError as error:
        raise EditedInterpretationError(format_schema_validation_error(error)) from error
