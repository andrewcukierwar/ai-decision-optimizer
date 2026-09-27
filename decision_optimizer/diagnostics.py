"""Bounded, deterministic infeasibility diagnosis for the two MVP problems.

The diagnosis layer deliberately reruns the existing deterministic solvers on a
small set of relaxed schema variants.  It does not inspect CP-SAT internals and
it never decides feasibility independently of CP-SAT.
"""

from datetime import date as datetime_date
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .dayplan import DayPlan, SolveStatus as DayPlanSolveStatus, TaskMode, solve_day_plan
from .shift_schedule import (
    Employee,
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    RequiredDaysOffRule,
    Shift,
    ShiftSchedule,
    SolveStatus as ShiftScheduleSolveStatus,
    Unavailability,
    solve_shift_schedule,
    time_to_minutes as shift_time_to_minutes,
)
from .dayplan.schema import time_to_minutes as dayplan_time_to_minutes
from .presentation import format_time


class DiagnosticFinding(BaseModel):
    """One concrete, solver-grounded conflict or relaxation suggestion."""

    family: str = Field(min_length=1)
    code: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)


class InfeasibilityDiagnostic(BaseModel):
    """A bounded set of deterministic findings for an infeasible input."""

    problem_type: str = Field(min_length=1)
    findings: List[DiagnosticFinding] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    tested_relaxations: List[str] = Field(default_factory=list)
    required_active_minutes: Optional[int] = None
    available_person_minutes: Optional[int] = None
    capacity_shortfall_minutes: Optional[int] = None


def diagnose_dayplan_infeasibility(
    plan: DayPlan, *, time_limit_seconds: Optional[float] = 2.0
) -> InfeasibilityDiagnostic:
    """Diagnose a DayPlan using direct checks and four bounded relaxations."""

    findings: List[DiagnosticFinding] = []
    tested: List[str] = []

    tasks_by_name = {task.name: task for task in plan.tasks}
    horizon_start = dayplan_time_to_minutes(plan.horizon.start)
    horizon_end = dayplan_time_to_minutes(plan.horizon.end)

    for task in plan.tasks:
        if not task.required:
            continue
        earliest = (
            dayplan_time_to_minutes(task.earliest_start)
            if task.earliest_start is not None
            else horizon_start
        )
        latest = (
            dayplan_time_to_minutes(task.latest_end)
            if task.latest_end is not None
            else horizon_end
        )
        available = latest - earliest
        if available < task.duration_min:
            findings.append(
                DiagnosticFinding(
                    family="task_timing",
                    code="task_window_too_small",
                    summary="Task timing window is too small",
                    evidence=(
                        "%s requires %d minutes, but only %d minutes are available "
                        "between %s and %s."
                        % (
                            task.name,
                            task.duration_min,
                            max(0, available),
                            _format_minutes(earliest),
                            _format_minutes(latest),
                        )
                    ),
                    suggestion=(
                        "Widen the earliest/latest bound for %s or reduce its duration."
                        % task.name
                    ),
                )
            )

    required_active_minutes = sum(
        task.duration_min
        for task in plan.tasks
        if task.required and task.mode == TaskMode.ACTIVE
    )
    fixed_blocked_minutes = _union_length(
        (
            dayplan_time_to_minutes(event.start),
            dayplan_time_to_minutes(event.end),
        )
        for event in plan.fixed_events
    )
    available_person_minutes = (horizon_end - horizon_start) - fixed_blocked_minutes
    available_person_minutes = max(0, available_person_minutes)
    capacity_shortfall_minutes = max(
        0, required_active_minutes - available_person_minutes
    )
    if required_active_minutes > available_person_minutes:
        findings.append(
            DiagnosticFinding(
                family="active_capacity",
                code="active_capacity_exceeded",
                summary="Active task time exceeds person capacity",
                evidence=(
                    "Required active work is %d minutes, but the horizon has at most "
                    "%d person minutes after fixed events."
                    % (required_active_minutes, max(0, available_person_minutes))
                ),
                suggestion=(
                    "Shorten or remove required active work."
                ),
            )
        )

    for precedence in plan.precedences:
        before = tasks_by_name[precedence.before]
        after = tasks_by_name[precedence.after]
        if not (before.required and after.required):
            continue
        earliest_before_end = _task_earliest_start(before, horizon_start) + before.duration_min
        latest_after_start = _task_latest_end(after, horizon_end) - after.duration_min
        if earliest_before_end + precedence.min_gap_min > latest_after_start:
            findings.append(
                DiagnosticFinding(
                    family="precedence",
                    code="precedence_min_gap_conflict",
                    summary="Precedence cannot fit inside task timing bounds",
                    evidence=(
                        "%s must finish no earlier than %s, but %s must start by %s "
                        "to meet its end bound."
                        % (
                            before.name,
                            _format_minutes(earliest_before_end),
                            after.name,
                            _format_minutes(latest_after_start),
                        )
                    ),
                    suggestion=(
                        "Widen the timing bound, reduce a duration, or relax the precedence "
                        "between %s and %s."
                        % (before.name, after.name)
                    ),
                )
            )
        if precedence.max_gap_min is not None:
            latest_before_end = _task_latest_end(before, horizon_end)
            earliest_after_start = _task_earliest_start(after, horizon_start)
            if earliest_after_start > latest_before_end + precedence.max_gap_min:
                findings.append(
                    DiagnosticFinding(
                        family="precedence",
                        code="precedence_max_gap_conflict",
                        summary="Precedence maximum gap cannot be met",
                        evidence=(
                            "%s cannot start until %s, which is more than %d minutes "
                            "after %s can finish."
                            % (
                                after.name,
                                _format_minutes(earliest_after_start),
                                precedence.max_gap_min,
                                before.name,
                            )
                        ),
                        suggestion=(
                            "Increase the maximum gap or widen the timing bounds for "
                            "%s and %s." % (before.name, after.name)
                        ),
                    )
                )

    def test_variant(
        label: str,
        relaxed_plan: DayPlan,
        finding: DiagnosticFinding,
    ) -> None:
        tested.append(label)
        result = solve_day_plan(relaxed_plan, time_limit_seconds=time_limit_seconds)
        if result.status in (DayPlanSolveStatus.OPTIMAL, DayPlanSolveStatus.FEASIBLE):
            findings.append(finding)

    if any(
        task.earliest_start is not None or task.latest_end is not None
        for task in plan.tasks
    ):
        relaxed_tasks = [
            task.model_copy(update={"earliest_start": None, "latest_end": None})
            for task in plan.tasks
        ]
        test_variant(
            "remove task timing bounds",
            plan.model_copy(deep=True, update={"tasks": relaxed_tasks}),
            DiagnosticFinding(
                family="task_timing",
                code="timing_bounds_binding",
                summary="Task timing bounds are binding",
                evidence="Removing all task earliest/latest bounds restored feasibility.",
                suggestion="Widen or remove the tightest task timing bound.",
            ),
        )

    if plan.fixed_events:
        test_variant(
            "remove fixed events",
            plan.model_copy(deep=True, update={"fixed_events": []}),
            DiagnosticFinding(
                family="fixed_events",
                code="fixed_event_conflict",
                summary="Fixed-event no-overlap is binding",
                evidence="Removing fixed events restored feasibility.",
                suggestion="Move or shorten a fixed event.",
            ),
        )

    if plan.precedences:
        test_variant(
            "remove precedences",
            plan.model_copy(deep=True, update={"precedences": []}),
            DiagnosticFinding(
                family="precedence",
                code="precedence_constraints_binding",
                summary="Precedence constraints are binding",
                evidence="Removing all precedence constraints restored feasibility.",
                suggestion="Relax the conflicting order, minimum gap, or maximum gap.",
            ),
        )

    active_tasks = [task for task in plan.tasks if task.mode == TaskMode.ACTIVE]
    if active_tasks:
        relaxed_tasks = [
            task.model_copy(update={"mode": TaskMode.PASSIVE})
            if task.mode == TaskMode.ACTIVE
            else task
            for task in plan.tasks
        ]
        test_variant(
            "relax active-person capacity",
            plan.model_copy(deep=True, update={"tasks": relaxed_tasks}),
            DiagnosticFinding(
                family="active_capacity",
                code="active_person_capacity_binding",
                summary="Active-task person capacity is binding",
                evidence="Treating active tasks as passive restored feasibility.",
                suggestion=(
                    "If an activity can legitimately run unattended, marking it passive "
                    "could add capacity."
                ),
            ),
        )

    findings = _deduplicate_findings(findings)
    if not findings:
        findings.append(
            DiagnosticFinding(
                family="hard_constraints",
                code="unidentified_hard_conflict",
                summary="The hard constraints conflict",
                evidence=(
                    "CP-SAT reported infeasible; the bounded checks did not isolate "
                    "one constraint family."
                ),
                suggestion="Relax one hard constraint family or add time/capacity.",
            )
        )
    findings = findings[:6]
    practical_suggestions: List[str] = []
    if capacity_shortfall_minutes > 0:
        practical_suggestions.extend(
            [
                "Extend the planning horizon.",
                "Shorten or remove required active work.",
            ]
        )
    if plan.fixed_events:
        practical_suggestions.append("Move or shorten a fixed event.")
    practical_suggestions.extend(item.suggestion for item in findings)
    return InfeasibilityDiagnostic(
        problem_type="dayplan",
        findings=findings,
        suggestions=_unique(practical_suggestions),
        tested_relaxations=tested,
        required_active_minutes=required_active_minutes,
        available_person_minutes=available_person_minutes,
        capacity_shortfall_minutes=capacity_shortfall_minutes,
    )


def diagnose_shift_schedule_infeasibility(
    schedule: ShiftSchedule, *, time_limit_seconds: Optional[float] = 2.0
) -> InfeasibilityDiagnostic:
    """Diagnose a ShiftSchedule using coverage checks and bounded relaxations."""

    findings: List[DiagnosticFinding] = []
    tested: List[str] = []
    locations = sorted({shift.location for shift in schedule.shifts})

    for shift in schedule.shifts:
        eligible = [
            employee
            for employee in schedule.employees
            if shift.location in employee.eligible_locations
        ]
        available = [
            employee
            for employee in eligible
            if not _is_unavailable(employee.unavailable, shift)
        ]
        if len(available) < shift.required_staff:
            if not eligible:
                evidence = (
                    "%s requires %d employee(s), but no employee is eligible for %s."
                    % (shift.id, shift.required_staff, shift.location)
                )
                suggestion = "Add an eligible employee for %s or change the shift location." % shift.location
            else:
                evidence = (
                    "%s requires %d employee(s), but only %d eligible employee(s) are "
                    "available."
                    % (shift.id, shift.required_staff, len(available))
                )
                suggestion = "Resolve an unavailable window or add another eligible employee."
            findings.append(
                DiagnosticFinding(
                    family="coverage",
                    code="coverage_eligibility_or_availability",
                    summary="Required coverage is not available",
                    evidence=evidence,
                    suggestion=suggestion,
                )
            )

    total_required_minutes = sum(
        shift.duration_minutes * shift.required_staff for shift in schedule.shifts
    )
    total_max_minutes = sum(employee.max_hours * 60 for employee in schedule.employees)
    if total_required_minutes > total_max_minutes:
        findings.append(
            DiagnosticFinding(
                family="max_hours",
                code="total_max_hours_too_small",
                summary="Employee maximum hours cannot cover demand",
                evidence=(
                    "Required coverage is %d minutes, but employee maximum hours provide "
                    "only %d minutes."
                    % (total_required_minutes, total_max_minutes)
                ),
                suggestion="Increase a max-hours limit or add another employee.",
            )
        )

    for location in locations:
        required_minutes = sum(
            shift.duration_minutes * shift.required_staff
            for shift in schedule.shifts
            if shift.location == location
        )
        eligible_capacity = sum(
            employee.max_hours * 60
            for employee in schedule.employees
            if location in employee.eligible_locations
        )
        if required_minutes > eligible_capacity:
            findings.append(
                DiagnosticFinding(
                    family="eligibility_capacity",
                    code="location_capacity_too_small",
                    summary="Location eligibility blocks enough staffing",
                    evidence=(
                        "%s requires %d eligible employee-minutes, but eligible "
                        "employees provide at most %d."
                        % (location, required_minutes, eligible_capacity)
                    ),
                    suggestion="Add eligibility for this location or add another eligible employee.",
                )
            )

    for rule in schedule.rules:
        if not isinstance(rule, RequiredDaysOffRule):
            continue
        employee = next(
            item for item in schedule.employees if item.name == rule.employee_name
        )
        for shift in schedule.shifts:
            if shift.day not in set(rule.days):
                continue
            if shift.location not in employee.eligible_locations:
                continue
            if _is_unavailable(employee.unavailable, shift):
                continue
            available_without_rule = [
                item
                for item in schedule.employees
                if shift.location in item.eligible_locations
                and not _is_unavailable(item.unavailable, shift)
            ]
            if len(available_without_rule) >= shift.required_staff:
                other_available = [item for item in available_without_rule if item.name != employee.name]
                if len(other_available) < shift.required_staff:
                    findings.append(
                        DiagnosticFinding(
                            family="required_days_off",
                            code="required_day_off_conflict",
                            summary="A required day off blocks coverage",
                            evidence=(
                                "%s is the only available eligible employee for %s, but "
                                "%s requires that day off."
                                % (employee.name, shift.id, employee.name)
                            ),
                            suggestion="Add coverage or relax the required day off.",
                        )
                    )

    def test_variant(
        label: str,
        relaxed_schedule: ShiftSchedule,
        finding: DiagnosticFinding,
    ) -> None:
        tested.append(label)
        result = solve_shift_schedule(
            relaxed_schedule, time_limit_seconds=time_limit_seconds
        )
        if result.status in (
            ShiftScheduleSolveStatus.OPTIMAL,
            ShiftScheduleSolveStatus.FEASIBLE,
        ):
            findings.append(finding)

    if any(employee.unavailable for employee in schedule.employees):
        employees = [
            employee.model_copy(update={"unavailable": []})
            for employee in schedule.employees
        ]
        test_variant(
            "remove availability restrictions",
            schedule.model_copy(deep=True, update={"employees": employees}),
            DiagnosticFinding(
                family="availability",
                code="availability_constraints_binding",
                summary="Availability restrictions are binding",
                evidence="Removing employee unavailability restored feasibility.",
                suggestion="Resolve an unavailable window or add another employee.",
            ),
        )

    employees = [
        employee.model_copy(update={"eligible_locations": locations})
        for employee in schedule.employees
    ]
    if any(
        set(employee.eligible_locations) != set(locations)
        for employee in schedule.employees
    ):
        test_variant(
            "relax location eligibility",
            schedule.model_copy(deep=True, update={"employees": employees}),
            DiagnosticFinding(
                family="eligibility",
                code="location_eligibility_constraints_binding",
                summary="Location eligibility restrictions are binding",
                evidence="Allowing every employee to work every listed location restored feasibility.",
                suggestion="Add an eligible employee for the blocked location.",
            ),
        )

    relaxed_max_hours = max(
        1,
        (total_required_minutes + 59) // 60,
    )
    if any(employee.max_hours * 60 < total_required_minutes for employee in schedule.employees):
        employees = [
            employee.model_copy(update={"max_hours": max(employee.max_hours, relaxed_max_hours)})
            for employee in schedule.employees
        ]
        test_variant(
            "relax employee max hours",
            schedule.model_copy(deep=True, update={"employees": employees}),
            DiagnosticFinding(
                family="max_hours",
                code="max_hours_constraints_binding",
                summary="Employee maximum hours are binding",
                evidence="Raising maximum hours enough to cover total demand restored feasibility.",
                suggestion="Raise a max-hours limit or add another employee.",
            ),
        )

    rule_variants = [
        (
            "minimum rest",
            MinimumRestRule,
            "minimum_rest_constraints_binding",
            "Minimum-rest rules are binding",
            "Removing minimum-rest rules restored feasibility.",
            "Reduce minimum rest or add another eligible employee.",
        ),
        (
            "maximum consecutive days",
            MaximumConsecutiveDaysRule,
            "maximum_consecutive_days_binding",
            "Maximum-consecutive-days rules are binding",
            "Removing maximum-consecutive-days rules restored feasibility.",
            "Relax the consecutive-day limit or add coverage.",
        ),
        (
            "required days off",
            RequiredDaysOffRule,
            "required_days_off_binding",
            "Required-days-off rules are binding",
            "Removing required-days-off rules restored feasibility.",
            "Relax a required day off or add coverage.",
        ),
    ]
    for label, rule_type, code, summary, evidence, suggestion in rule_variants:
        if not any(isinstance(rule, rule_type) for rule in schedule.rules):
            continue
        relaxed_rules = [rule for rule in schedule.rules if not isinstance(rule, rule_type)]
        test_variant(
            "remove " + label + " rules",
            schedule.model_copy(deep=True, update={"rules": relaxed_rules}),
            DiagnosticFinding(
                family="rules",
                code=code,
                summary=summary,
                evidence=evidence,
                suggestion=suggestion,
            ),
        )

    findings = _deduplicate_findings(findings)
    if not findings:
        findings.append(
            DiagnosticFinding(
                family="hard_constraints",
                code="unidentified_hard_conflict",
                summary="The hard constraints conflict",
                evidence=(
                    "CP-SAT reported infeasible; the bounded checks did not isolate "
                    "one constraint family."
                ),
                suggestion="Relax one hard constraint family or add staffing capacity.",
            )
        )
    findings = findings[:6]
    return InfeasibilityDiagnostic(
        problem_type="shift_schedule",
        findings=findings,
        suggestions=_unique(item.suggestion for item in findings),
        tested_relaxations=tested,
    )


def _task_earliest_start(task: object, horizon_start: int) -> int:
    value = getattr(task, "earliest_start")
    return dayplan_time_to_minutes(value) if value is not None else horizon_start


def _task_latest_end(task: object, horizon_end: int) -> int:
    value = getattr(task, "latest_end")
    return dayplan_time_to_minutes(value) if value is not None else horizon_end


def _union_length(intervals: Iterable[Tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _format_minutes(value: int) -> str:
    return format_time("%02d:%02d" % (value // 60, value % 60))


def _is_unavailable(unavailable: List[Unavailability], shift: Shift) -> bool:
    for period in unavailable:
        if period.shift_id == shift.id:
            return True
        if period.day != shift.day:
            continue
        if period.start is None:
            return True
        assert period.end is not None
        if _overlap_times(
            shift_time_to_minutes(shift.start),
            shift_time_to_minutes(shift.end),
            shift_time_to_minutes(period.start),
            shift_time_to_minutes(period.end),
        ):
            return True
    return False


def _overlap_times(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _deduplicate_findings(findings: List[DiagnosticFinding]) -> List[DiagnosticFinding]:
    seen: Set[Tuple[str, str]] = set()
    result: List[DiagnosticFinding] = []
    for finding in findings:
        key = (finding.code, finding.evidence)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
