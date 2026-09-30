"""Structured Direct LLM scheduling baseline for both MVP domains."""

import json
from dataclasses import dataclass
from datetime import time as datetime_time
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer, model_validator

from .failures import ModelOutputFailure
from .config import openai_api_key, openai_model
from .dayplan import (
    DayPlan,
    DayPlanSolution,
    SolveStatus as DaySolveStatus,
    TaskAssignment,
    materialize_day_plan_solution,
)
from .dayplan.schema import minutes_to_time, time_to_minutes
from .shift_schedule import (
    ShiftAssignment,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus as ShiftSolveStatus,
    materialize_shift_schedule_solution,
)
from .telemetry import RunTelemetry, parse_openai_response


DAYPLAN_DIRECT_INSTRUCTIONS = """You schedule the supplied typed DayPlan.
Satisfy every hard constraint, including required tasks, time bounds, fixed
events, active-task non-overlap, and precedences. Optimize the stated soft
preferences. Return only the provided structured result. If no assignment can
satisfy all hard constraints, return infeasible with no assignments. Task names
must exactly match the supplied typed problem. Return task assignments only;
exclude fixed events from assignments (they are already supplied). Do not write solver code or
describe an optimization algorithm.
"""

SHIFT_DIRECT_INSTRUCTIONS = """You staff the supplied typed ShiftSchedule.
Satisfy every hard constraint, including exact coverage, eligibility,
availability, maximum hours, overlap, rest, consecutive-day, and required-day-
off rules. Optimize preferences and fairness using the supplied weights.
Return only the provided structured result. If no assignment can satisfy all
hard constraints, return infeasible with no assignments. Shift ids and employee
names must exactly match the supplied typed problem. Do not write solver code
or describe an optimization algorithm.
"""


class DirectSolverError(Exception):
    """Base class for direct-solver boundary failures."""


class DirectSolverAPIError(DirectSolverError):
    pass


class DirectSolverOutputError(DirectSolverError, ModelOutputFailure):
    pass


class DirectStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DirectDayAssignment(_StrictModel):
    task: str = Field(min_length=1)
    start: datetime_time
    end: datetime_time

    @field_serializer("start", "end")
    def serialize_time(self, value: datetime_time) -> str:
        return value.strftime("%H:%M")


class DirectDayPlanSolution(_StrictModel):
    status: DirectStatus
    assignments: List[DirectDayAssignment]
    unscheduled_tasks: List[str]
    explanation: str

    @model_validator(mode="after")
    def validate_infeasible_shape(self) -> "DirectDayPlanSolution":
        if self.status == DirectStatus.INFEASIBLE and self.assignments:
            raise ValueError("infeasible output cannot contain assignments")
        return self


class DirectShiftAssignment(_StrictModel):
    shift_id: str = Field(min_length=1)
    employee: str = Field(min_length=1)


class DirectShiftSolution(_StrictModel):
    status: DirectStatus
    assignments: List[DirectShiftAssignment]
    explanation: str

    @model_validator(mode="after")
    def validate_infeasible_shape(self) -> "DirectShiftSolution":
        if self.status == DirectStatus.INFEASIBLE and self.assignments:
            raise ValueError("infeasible output cannot contain assignments")
        return self


@dataclass
class DirectDayPlanResult:
    output: DirectDayPlanSolution
    solution: DayPlanSolution


@dataclass
class DirectShiftResult:
    output: DirectShiftSolution
    solution: ShiftScheduleSolution


def solve_day_plan_direct(
    plan: DayPlan,
    *,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
    request_timeout_seconds: Optional[float] = None,
) -> DirectDayPlanResult:
    """Ask the selected model to solve the final typed DayPlan directly."""

    output = _request_structured_solution(
        problem=plan,
        output_schema=DirectDayPlanSolution,
        instructions=DAYPLAN_DIRECT_INSTRUCTIONS,
        client=client,
        model=model,
        telemetry=telemetry,
        request_timeout_seconds=request_timeout_seconds,
    )
    if output.status == DirectStatus.INFEASIBLE:
        solution = materialize_day_plan_solution(
            plan,
            [],
            status=DaySolveStatus.INFEASIBLE,
            message=output.explanation or "The direct model reported infeasibility.",
        )
    else:
        assignments = [
            # Offsets are metadata, not UTC conversions, in this local-clock domain.
            TaskAssignment(
                name=item.task,
                start=minutes_to_time(time_to_minutes(item.start)),
                end=minutes_to_time(time_to_minutes(item.end)),
            )
            for item in output.assignments
        ]
        known_names = {task.name for task in plan.tasks}
        known_assignments = [
            assignment for assignment in assignments if assignment.name in known_names
        ]
        solution = materialize_day_plan_solution(plan, known_assignments)
        # Preserve unknown identifiers for the validator/evaluator instead of
        # crashing while deriving objective metadata from known entities.
        solution.assignments = assignments
    return DirectDayPlanResult(output=output, solution=solution)


def solve_shift_schedule_direct(
    schedule: ShiftSchedule,
    *,
    client: Any = None,
    model: Optional[str] = None,
    telemetry: Optional[RunTelemetry] = None,
    request_timeout_seconds: Optional[float] = None,
) -> DirectShiftResult:
    """Ask the selected model to solve the final typed workforce problem."""

    output = _request_structured_solution(
        problem=schedule,
        output_schema=DirectShiftSolution,
        instructions=SHIFT_DIRECT_INSTRUCTIONS,
        client=client,
        model=model,
        telemetry=telemetry,
        request_timeout_seconds=request_timeout_seconds,
    )
    if output.status == DirectStatus.INFEASIBLE:
        solution = materialize_shift_schedule_solution(
            schedule,
            [],
            status=ShiftSolveStatus.INFEASIBLE,
            message=output.explanation or "The direct model reported infeasibility.",
        )
    else:
        assignments = [
            ShiftAssignment(shift_id=item.shift_id, employee_name=item.employee)
            for item in output.assignments
        ]
        known_shifts = {shift.id for shift in schedule.shifts}
        known_employees = {employee.name for employee in schedule.employees}
        known_assignments = [
            assignment
            for assignment in assignments
            if assignment.shift_id in known_shifts
            and assignment.employee_name in known_employees
        ]
        solution = materialize_shift_schedule_solution(schedule, known_assignments)
        solution.assignments = assignments
    return DirectShiftResult(output=output, solution=solution)


def _request_structured_solution(
    *,
    problem: BaseModel,
    output_schema: Any,
    instructions: str,
    client: Any,
    model: Optional[str],
    telemetry: Optional[RunTelemetry],
    request_timeout_seconds: Optional[float],
) -> Any:
    if client is None:
        client = _create_openai_client()
    payload = json.dumps(problem.model_dump(mode="json"), indent=2, sort_keys=True)
    try:
        request_kwargs = {
            "model": model or openai_model(),
            "input": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": "Typed problem:\n" + payload},
            ],
            "text_format": output_schema,
        }
        if request_timeout_seconds is not None:
            request_kwargs["timeout"] = request_timeout_seconds
            if hasattr(client, "with_options"):
                client = client.with_options(max_retries=0)
        if telemetry is None:
            response = parse_openai_response(client, request_kwargs, telemetry)
        else:
            with telemetry.track("llm"):
                response = parse_openai_response(client, request_kwargs, telemetry)
    except ValidationError as exc:
        if exc.title != output_schema.__name__:
            raise DirectSolverAPIError("Non-output SDK validation failure: %s" % exc) from exc
        raise DirectSolverOutputError(
            "Direct solver returned invalid structured output: %s" % exc
        ) from exc
    except DirectSolverError:
        raise
    except Exception as exc:
        raise DirectSolverAPIError("Direct solver request failed: %s" % exc) from exc

    refusal = _response_refusal(response)
    if refusal:
        raise DirectSolverOutputError("The model refused direct solving: " + refusal)
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise DirectSolverOutputError("The model returned no parsed direct solution")
    try:
        return parsed if isinstance(parsed, output_schema) else output_schema.model_validate(parsed)
    except ValidationError as exc:
        raise DirectSolverOutputError(
            "Parsed direct solution failed validation: %s" % exc
        ) from exc


def _create_openai_client() -> Any:
    if not openai_api_key():
        raise DirectSolverAPIError("OPENAI_API_KEY is required for live direct solving")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DirectSolverAPIError("The openai package is not installed") from exc
    try:
        return OpenAI()
    except Exception as exc:
        raise DirectSolverAPIError("Could not create the OpenAI client: %s" % exc) from exc


def _response_refusal(response: Any) -> Optional[str]:
    direct = getattr(response, "refusal", None)
    if direct:
        return str(direct)
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None
