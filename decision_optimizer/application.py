"""UI-facing orchestration kept separate from Streamlit widgets."""

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Optional

from pydantic import ValidationError

from .dayplan import DayPlan, DayPlanSolution, solve_day_plan, validate_solution as validate_dayplan
from .direct_solver import (
    DirectDayPlanSolution,
    DirectShiftSolution,
    solve_day_plan_direct,
    solve_shift_schedule_direct,
)
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
from .experiment import ExperimentConfig
from .jev import JevDecision, apply_jev
from .shift_schedule import ShiftSchedule, ShiftScheduleSolution, solve_shift_schedule
from .shift_schedule import validate_solution as validate_shift_schedule
from .telemetry import RunTelemetry


class EditedInterpretationError(ValueError):
    """The user-edited JSON did not validate against the typed schema."""


@dataclass
class DayPlanRun:
    plan: DayPlan
    solution: DayPlanSolution
    validation: Any
    diagnosis: Optional[InfeasibilityDiagnostic] = None
    explanation_facts: Optional[DayPlanExplanationFacts] = None
    telemetry: Optional[RunTelemetry] = None
    direct_output: Optional[DirectDayPlanSolution] = None
    jev_decisions: list[JevDecision] = field(default_factory=list)


@dataclass
class ShiftScheduleRun:
    schedule: ShiftSchedule
    solution: ShiftScheduleSolution
    validation: Any
    diagnosis: Optional[InfeasibilityDiagnostic] = None
    explanation_facts: Optional[ShiftScheduleExplanationFacts] = None
    telemetry: Optional[RunTelemetry] = None
    direct_output: Optional[DirectShiftSolution] = None
    jev_decisions: list[JevDecision] = field(default_factory=list)


@dataclass
class DayPlanExperimentRun:
    extraction: DayPlanExtraction
    run: Optional[DayPlanRun]
    telemetry: RunTelemetry
    jev_decisions: list[JevDecision] = field(default_factory=list)


@dataclass
class ShiftScheduleExperimentRun:
    extraction: ShiftScheduleExtraction
    run: Optional[ShiftScheduleRun]
    telemetry: RunTelemetry
    jev_decisions: list[JevDecision] = field(default_factory=list)


def validate_edited_dayplan_json(raw_json: str) -> DayPlan:
    """Parse and validate edited DayPlan JSON before it can reach the solver."""

    return _validate_json(raw_json, DayPlan)


def validate_edited_shift_schedule_json(raw_json: str) -> ShiftSchedule:
    """Parse and validate edited ShiftSchedule JSON before it can reach the solver."""

    return _validate_json(raw_json, ShiftSchedule)


def run_dayplan_experiment(
    request: str,
    config: ExperimentConfig,
    *,
    client: Any = None,
    jev_client: Any = None,
    case_id: Optional[str] = None,
    time_limit_seconds: Optional[float] = 10.0,
    request_timeout_seconds: Optional[float] = None,
) -> DayPlanExperimentRun:
    """Interpret and solve one DayPlan arm with a single telemetry record."""

    telemetry = RunTelemetry.start(config, "dayplan", case_id)
    try:
        extraction = parse_dayplan_request(
            request,
            client=client,
            config=config,
            telemetry=telemetry,
            request_timeout_seconds=request_timeout_seconds,
        )
        if extraction.plan is None:
            telemetry.finish()
            return DayPlanExperimentRun(extraction, None, telemetry)
        jev_decisions: list[JevDecision] = []
        if config.use_jev:
            jev_result = apply_jev(
                request,
                extraction.plan,
                client=jev_client,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
            extraction = extraction.model_copy(update={"plan": jev_result.problem})
            jev_decisions = jev_result.decisions
        run = solve_confirmed_dayplan(
            extraction.plan,
            config=config,
            client=client,
            telemetry=telemetry,
            jev_decisions=jev_decisions,
            time_limit_seconds=time_limit_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        return DayPlanExperimentRun(extraction, run, telemetry, jev_decisions)
    except Exception as error:
        telemetry.finish(error)
        raise


def run_shift_schedule_experiment(
    request: str,
    config: ExperimentConfig,
    *,
    client: Any = None,
    jev_client: Any = None,
    case_id: Optional[str] = None,
    time_limit_seconds: Optional[float] = 10.0,
    request_timeout_seconds: Optional[float] = None,
) -> ShiftScheduleExperimentRun:
    """Interpret and solve one workforce arm with a single telemetry record."""

    telemetry = RunTelemetry.start(config, "shift_schedule", case_id)
    try:
        extraction = parse_shift_schedule_request(
            request,
            client=client,
            config=config,
            telemetry=telemetry,
            request_timeout_seconds=request_timeout_seconds,
        )
        if extraction.schedule is None:
            telemetry.finish()
            return ShiftScheduleExperimentRun(extraction, None, telemetry)
        jev_decisions: list[JevDecision] = []
        if config.use_jev:
            jev_result = apply_jev(
                request,
                extraction.schedule,
                client=jev_client,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
            extraction = extraction.model_copy(update={"schedule": jev_result.problem})
            jev_decisions = jev_result.decisions
        run = solve_confirmed_shift_schedule(
            extraction.schedule,
            config=config,
            client=client,
            telemetry=telemetry,
            jev_decisions=jev_decisions,
            time_limit_seconds=time_limit_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        return ShiftScheduleExperimentRun(extraction, run, telemetry, jev_decisions)
    except Exception as error:
        telemetry.finish(error)
        raise


def solve_confirmed_dayplan(
    plan: DayPlan,
    *,
    time_limit_seconds: Optional[float] = 10.0,
    config: Optional[ExperimentConfig] = None,
    client: Any = None,
    telemetry: Optional[RunTelemetry] = None,
    jev_decisions: Optional[list[JevDecision]] = None,
    request_timeout_seconds: Optional[float] = None,
) -> DayPlanRun:
    """Solve, independently validate, diagnose if needed, and gate explanation facts."""

    config = config or ExperimentConfig()
    telemetry = telemetry or RunTelemetry.start(config, "dayplan")
    direct_output = None
    try:
        if config.solution_engine == "direct_llm":
            direct_result = solve_day_plan_direct(
                plan,
                client=client,
                model=config.model,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
            solution = direct_result.solution
            direct_output = direct_result.output
        else:
            with telemetry.track("solver"):
                solution = solve_day_plan(plan, time_limit_seconds=time_limit_seconds)
    except Exception as error:
        telemetry.finish(error)
        raise
    validation = validate_dayplan(plan, solution)
    diagnosis = None
    if solution.status.value == "infeasible":
        diagnosis = diagnose_dayplan_infeasibility(
            plan, time_limit_seconds=min(time_limit_seconds or 2.0, 2.0)
        )
    explanation_facts = build_dayplan_explanation_payload(plan, solution, validation)
    telemetry.finish()
    return DayPlanRun(
        plan=plan,
        solution=solution,
        validation=validation,
        diagnosis=diagnosis,
        explanation_facts=explanation_facts,
        telemetry=telemetry,
        direct_output=direct_output,
        jev_decisions=list(jev_decisions or []),
    )


def solve_confirmed_shift_schedule(
    schedule: ShiftSchedule,
    *,
    time_limit_seconds: Optional[float] = 10.0,
    config: Optional[ExperimentConfig] = None,
    client: Any = None,
    telemetry: Optional[RunTelemetry] = None,
    jev_decisions: Optional[list[JevDecision]] = None,
    request_timeout_seconds: Optional[float] = None,
) -> ShiftScheduleRun:
    """Solve, independently validate, diagnose if needed, and gate explanation facts."""

    config = config or ExperimentConfig()
    telemetry = telemetry or RunTelemetry.start(config, "shift_schedule")
    direct_output = None
    try:
        if config.solution_engine == "direct_llm":
            direct_result = solve_shift_schedule_direct(
                schedule,
                client=client,
                model=config.model,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
            solution = direct_result.solution
            direct_output = direct_result.output
        else:
            with telemetry.track("solver"):
                solution = solve_shift_schedule(
                    schedule, time_limit_seconds=time_limit_seconds
                )
    except Exception as error:
        telemetry.finish(error)
        raise
    validation = validate_shift_schedule(schedule, solution)
    diagnosis = None
    if solution.status.value == "infeasible":
        diagnosis = diagnose_shift_schedule_infeasibility(
            schedule, time_limit_seconds=min(time_limit_seconds or 2.0, 2.0)
        )
    explanation_facts = build_shift_schedule_explanation_payload(
        schedule, solution, validation
    )
    telemetry.finish()
    return ShiftScheduleRun(
        schedule=schedule,
        solution=solution,
        validation=validation,
        diagnosis=diagnosis,
        explanation_facts=explanation_facts,
        telemetry=telemetry,
        direct_output=direct_output,
        jev_decisions=list(jev_decisions or []),
    )


def parse_dayplan_request(
    request: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    telemetry: Optional[RunTelemetry] = None,
    request_timeout_seconds: Optional[float] = None,
) -> DayPlanExtraction:
    """Small adapter used by the UI and deterministic tests."""

    return parse_dayplan(
        request,
        client=client,
        model=config.model if config is not None else model,
        telemetry=telemetry,
        request_timeout_seconds=request_timeout_seconds,
    )


def clarify_dayplan_request(
    request: str,
    previous: DayPlanExtraction,
    clarification: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    telemetry: Optional[RunTelemetry] = None,
) -> DayPlanExtraction:
    """Run the single supported DayPlan clarification round."""

    return parse_dayplan(
        request,
        previous_extraction=previous,
        clarification=clarification,
        client=client,
        model=config.model if config is not None else model,
        telemetry=telemetry,
    )


def parse_shift_schedule_request(
    request: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    telemetry: Optional[RunTelemetry] = None,
    request_timeout_seconds: Optional[float] = None,
) -> ShiftScheduleExtraction:
    """Small adapter used by the UI and deterministic tests."""

    return parse_shift_schedule(
        request,
        client=client,
        model=config.model if config is not None else model,
        telemetry=telemetry,
        request_timeout_seconds=request_timeout_seconds,
    )


def clarify_shift_schedule_request(
    request: str,
    previous: ShiftScheduleExtraction,
    clarification: str,
    *,
    client: Any = None,
    model: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    telemetry: Optional[RunTelemetry] = None,
) -> ShiftScheduleExtraction:
    """Run the single supported ShiftSchedule clarification round."""

    return parse_shift_schedule(
        request,
        previous_extraction=previous,
        clarification=clarification,
        client=client,
        model=config.model if config is not None else model,
        telemetry=telemetry,
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
