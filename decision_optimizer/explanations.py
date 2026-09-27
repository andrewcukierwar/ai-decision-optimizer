"""Grounded explanation facts and concise deterministic summaries.

These functions consume only typed schemas, materialized solver results, and
independent validator reports.  They do not inspect CP-SAT models or infer
unstated reasons for a schedule.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .dayplan import DayPlan, DayPlanSolution, SolveStatus as DayPlanSolveStatus
from .dayplan.validator import ValidationReport as DayPlanValidationReport
from .shift_schedule import (
    ObjectiveBreakdown,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus as ShiftScheduleSolveStatus,
)
from .shift_schedule.validator import ValidationReport as ShiftScheduleValidationReport


class DayPlanExplanationAssignment(BaseModel):
    name: str
    start: str
    end: str
    duration_min: int
    mode: str


class DayPlanExplanationPenalty(BaseModel):
    preference_index: int
    preference_type: str
    amount: int
    weighted_penalty: int


class DayPlanExplanationFacts(BaseModel):
    """Facts safe to pass to an explanation model or display directly."""

    problem_type: str = "dayplan"
    status: str
    optimal: bool
    objective_value: int
    assignments: List[DayPlanExplanationAssignment]
    preference_penalties: List[DayPlanExplanationPenalty]
    validation_valid: bool
    validation_errors: List[str] = Field(default_factory=list)


class ShiftScheduleExplanationAssignment(BaseModel):
    shift_id: str
    employee_name: str
    day: str
    start: str
    end: str
    location: str


class ShiftScheduleExplanationLoad(BaseModel):
    employee_name: str
    hours: float
    assigned_shift_ids: List[str]


class ShiftScheduleExplanationPenalty(BaseModel):
    employee_name: str
    shift_id: str
    amount: int
    weighted_penalty: int


class ShiftScheduleExplanationFacts(BaseModel):
    """Facts safe to pass to an explanation model or display directly."""

    problem_type: str = "shift_schedule"
    status: str
    optimal: bool
    objective_value: int
    assignments: List[ShiftScheduleExplanationAssignment]
    employee_hours: List[ShiftScheduleExplanationLoad]
    preference_penalties: List[ShiftScheduleExplanationPenalty]
    objective_breakdown: ObjectiveBreakdown
    validation_valid: bool
    validation_errors: List[str] = Field(default_factory=list)


def build_dayplan_explanation_payload(
    plan: DayPlan,
    solution: DayPlanSolution,
    validation: DayPlanValidationReport,
) -> Optional[DayPlanExplanationFacts]:
    """Build grounded DayPlan facts, refusing the normal path on bad validation."""

    if not validation.valid:
        return None
    if solution.status not in (DayPlanSolveStatus.OPTIMAL, DayPlanSolveStatus.FEASIBLE):
        return None

    task_modes = {task.name: task.mode.value for task in plan.tasks}
    task_durations = {task.name: task.duration_min for task in plan.tasks}
    return DayPlanExplanationFacts(
        status=solution.status.value,
        optimal=solution.optimal,
        objective_value=solution.objective_value,
        assignments=[
            DayPlanExplanationAssignment(
                name=assignment.name,
                start=assignment.start.strftime("%H:%M"),
                end=assignment.end.strftime("%H:%M"),
                duration_min=task_durations[assignment.name],
                mode=task_modes[assignment.name],
            )
            for assignment in sorted(solution.assignments, key=lambda item: item.start)
        ],
        preference_penalties=[
            DayPlanExplanationPenalty(
                preference_index=penalty.preference_index,
                preference_type=penalty.preference_type.value,
                amount=penalty.amount,
                weighted_penalty=penalty.weighted_penalty,
            )
            for penalty in solution.preference_penalties
        ],
        validation_valid=validation.valid,
        validation_errors=list(validation.errors),
    )


def build_shift_schedule_explanation_payload(
    schedule: ShiftSchedule,
    solution: ShiftScheduleSolution,
    validation: ShiftScheduleValidationReport,
) -> Optional[ShiftScheduleExplanationFacts]:
    """Build grounded ShiftSchedule facts, refusing the normal path on bad validation."""

    if not validation.valid:
        return None
    if solution.status not in (
        ShiftScheduleSolveStatus.OPTIMAL,
        ShiftScheduleSolveStatus.FEASIBLE,
    ):
        return None

    shifts_by_id = {shift.id: shift for shift in schedule.shifts}
    return ShiftScheduleExplanationFacts(
        status=solution.status.value,
        optimal=solution.optimal,
        objective_value=solution.objective_value,
        assignments=[
            ShiftScheduleExplanationAssignment(
                shift_id=assignment.shift_id,
                employee_name=assignment.employee_name,
                day=shifts_by_id[assignment.shift_id].day.isoformat(),
                start=shifts_by_id[assignment.shift_id].start.strftime("%H:%M"),
                end=shifts_by_id[assignment.shift_id].end.strftime("%H:%M"),
                location=shifts_by_id[assignment.shift_id].location,
            )
            for assignment in sorted(
                solution.assignments,
                key=lambda item: (
                    shifts_by_id[item.shift_id].day,
                    shifts_by_id[item.shift_id].start,
                    item.shift_id,
                    item.employee_name,
                ),
            )
        ],
        employee_hours=[
            ShiftScheduleExplanationLoad(
                employee_name=load.employee_name,
                hours=load.hours,
                assigned_shift_ids=list(load.assigned_shift_ids),
            )
            for load in solution.employee_hours
        ],
        preference_penalties=[
            ShiftScheduleExplanationPenalty(
                employee_name=penalty.employee_name,
                shift_id=penalty.shift_id,
                amount=penalty.amount,
                weighted_penalty=penalty.weighted_penalty,
            )
            for penalty in solution.preference_penalties
        ],
        objective_breakdown=solution.objective_breakdown,
        validation_valid=validation.valid,
        validation_errors=list(validation.errors),
    )


def render_dayplan_explanation(facts: DayPlanExplanationFacts) -> str:
    """Render a concise explanation without adding facts to the payload."""

    status = "optimal" if facts.optimal else facts.status
    lines = [
        "The deterministic solver returned an %s DayPlan with objective %d."
        % (status, facts.objective_value),
        "The independent plain-Python validator accepted every displayed assignment and penalty.",
    ]
    if facts.assignments:
        lines.append(
            "Activities are shown at the solver's materialized times; active/passive mode and duration come from the confirmed interpretation."
        )
    if facts.preference_penalties:
        penalty_text = ", ".join(
            "%s=%d (weighted %d)"
            % (item.preference_type, item.amount, item.weighted_penalty)
            for item in facts.preference_penalties
        )
        lines.append("Objective preference penalties: " + penalty_text + ".")
    else:
        lines.append("No preference penalties were recorded in the validated result.")
    return " ".join(lines)


def render_shift_schedule_explanation(facts: ShiftScheduleExplanationFacts) -> str:
    """Render a concise explanation without adding facts to the payload."""

    status = "optimal" if facts.optimal else facts.status
    breakdown = facts.objective_breakdown
    lines = [
        "The deterministic solver returned an %s Workforce Scheduler result with objective %d."
        % (status, facts.objective_value),
        "The independent plain-Python validator accepted the assignments, coverage, eligibility, availability, hours, and hard-rule checks.",
        "Fairness spread is %d minutes; weighted preference penalty is %d and weighted fairness is %d."
        % (
            breakdown.fairness_minutes_spread,
            breakdown.weighted_preference_penalty,
            breakdown.weighted_fairness,
        ),
    ]
    if facts.preference_penalties:
        lines.append(
            "%d preference assignment(s) incurred a penalty."
            % len(facts.preference_penalties)
        )
    else:
        lines.append("No preference assignment penalties were recorded.")
    return " ".join(lines)
