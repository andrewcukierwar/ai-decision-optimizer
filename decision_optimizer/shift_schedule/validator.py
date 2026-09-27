"""Independent plain-Python validation of ShiftSchedule solver results.

This module intentionally does not import the compiler or inspect CP-SAT
variables.  It recomputes workforce semantics from the schema and materialized
assignments so a compiler bug or corrupted result can be detected separately.
"""

from dataclasses import dataclass, field
from datetime import date as datetime_date
from datetime import timedelta
from typing import Dict, List, Set, Tuple

from .schema import (
    Employee,
    EmployeeLoad,
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    ObjectiveBreakdown,
    PreferencePenalty,
    PREFERENCE_NORMALIZATION_MINUTES,
    RequiredDaysOffRule,
    Shift,
    ShiftAssignment,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus,
    Unavailability,
    time_to_minutes,
)


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_solution(
    schedule: ShiftSchedule, result: ShiftScheduleSolution
) -> ValidationReport:
    """Check a materialized solution against ShiftSchedule semantics."""

    errors: List[str] = []
    if result.optimal != (result.status == SolveStatus.OPTIMAL):
        errors.append("optimal flag does not match solve status")
    if result.status in (SolveStatus.INFEASIBLE, SolveStatus.UNKNOWN):
        if result.assignments:
            errors.append("non-feasible result must not contain assignments")
        if result.employee_hours:
            errors.append("non-feasible result must not contain employee hours")
        if result.preference_penalties:
            errors.append("non-feasible result must not contain preference penalties")
        if result.objective_value != 0 or result.objective_breakdown != ObjectiveBreakdown():
            errors.append("non-feasible result must have a zero objective")
        return ValidationReport(valid=not errors, errors=errors)

    shifts_by_id: Dict[str, Shift] = {shift.id: shift for shift in schedule.shifts}
    employees_by_name: Dict[str, Employee] = {
        employee.name: employee for employee in schedule.employees
    }
    assignments_by_shift: Dict[str, List[str]] = {shift.id: [] for shift in schedule.shifts}
    assignments_by_employee: Dict[str, List[str]] = {
        employee.name: [] for employee in schedule.employees
    }
    seen_pairs: Set[Tuple[str, str]] = set()

    for assignment in result.assignments:
        shift = shifts_by_id.get(assignment.shift_id)
        employee = employees_by_name.get(assignment.employee_name)
        if shift is None:
            errors.append("assignment references unknown shift: " + assignment.shift_id)
            continue
        if employee is None:
            errors.append(
                "assignment references unknown employee: " + assignment.employee_name
            )
            continue
        pair = (assignment.employee_name, assignment.shift_id)
        if pair in seen_pairs:
            errors.append(
                "duplicate employee assignment: %s -> %s"
                % (assignment.employee_name, assignment.shift_id)
            )
            continue
        seen_pairs.add(pair)
        assignments_by_shift[assignment.shift_id].append(assignment.employee_name)
        assignments_by_employee[assignment.employee_name].append(assignment.shift_id)

        if shift.location not in employee.eligible_locations:
            errors.append(
                "employee is not eligible for shift location: %s -> %s"
                % (assignment.employee_name, assignment.shift_id)
            )
        if _is_unavailable(employee.unavailable, shift):
            errors.append(
                "employee is unavailable for shift: %s -> %s"
                % (assignment.employee_name, assignment.shift_id)
            )

    for shift in schedule.shifts:
        actual_staff = len(assignments_by_shift[shift.id])
        if actual_staff != shift.required_staff:
            errors.append(
                "coverage mismatch for %s: expected %d, got %d"
                % (shift.id, shift.required_staff, actual_staff)
            )

    for employee in schedule.employees:
        assigned_shifts = [
            shifts_by_id[shift_id]
            for shift_id in assignments_by_employee[employee.name]
        ]
        total_minutes = sum(shift.duration_minutes for shift in assigned_shifts)
        if total_minutes > employee.max_hours * 60:
            errors.append("employee exceeds max hours: " + employee.name)

        ordered = sorted(
            assigned_shifts,
            key=lambda shift: (shift.day, time_to_minutes(shift.start), shift.id),
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if _overlap(left, right):
                    errors.append(
                        "employee shifts overlap: %s and %s for %s"
                        % (left.id, right.id, employee.name)
                    )

        for rule in schedule.rules:
            if not isinstance(rule, MinimumRestRule):
                continue
            rest_minutes = rule.min_rest_hours * 60
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    if _gap_minutes(left, right) < rest_minutes:
                        errors.append(
                            "minimum rest violated: %s and %s for %s"
                            % (left.id, right.id, employee.name)
                        )

        worked_days = {shift.day for shift in assigned_shifts}
        for rule in schedule.rules:
            if isinstance(rule, MaximumConsecutiveDaysRule):
                errors.extend(
                    _maximum_consecutive_day_errors(
                        employee.name, worked_days, rule.max_days
                    )
                )
            elif isinstance(rule, RequiredDaysOffRule) and rule.employee_name == employee.name:
                for day in rule.days:
                    if day in worked_days:
                        errors.append(
                            "required day off violated: %s on %s"
                            % (employee.name, day.isoformat())
                        )

    _validate_employee_hours(
        errors, schedule, assignments_by_employee, shifts_by_id, result.employee_hours
    )

    expected_penalties = _expected_preference_penalties(
        schedule, assignments_by_employee
    )
    if _penalty_keys(result.preference_penalties) != _penalty_keys(expected_penalties):
        errors.append("preference penalties do not match the assignments")

    expected_breakdown = _expected_objective(
        schedule, assignments_by_employee, shifts_by_id, expected_penalties
    )
    if result.objective_breakdown != expected_breakdown:
        errors.append("objective breakdown does not match the assignments")
    if result.objective_value != expected_breakdown.total:
        errors.append(
            "objective mismatch: expected %d, got %d"
            % (expected_breakdown.total, result.objective_value)
        )

    return ValidationReport(valid=not errors, errors=errors)


def _validate_employee_hours(
    errors: List[str],
    schedule: ShiftSchedule,
    assignments_by_employee: Dict[str, List[str]],
    shifts_by_id: Dict[str, Shift],
    actual_loads: List[EmployeeLoad],
) -> None:
    loads_by_name: Dict[str, EmployeeLoad] = {}
    for load in actual_loads:
        if load.employee_name not in assignments_by_employee:
            errors.append("employee hours reference unknown employee: " + load.employee_name)
        elif load.employee_name in loads_by_name:
            errors.append("duplicate employee hours: " + load.employee_name)
        else:
            loads_by_name[load.employee_name] = load

    for employee in schedule.employees:
        load = loads_by_name.get(employee.name)
        if load is None:
            errors.append("employee hours are missing: " + employee.name)
            continue
        expected_shift_ids = sorted(assignments_by_employee[employee.name])
        if sorted(load.assigned_shift_ids) != expected_shift_ids:
            errors.append("employee assigned shift list does not match: " + employee.name)
        expected_minutes = sum(
            shifts_by_id[shift_id].duration_minutes
            for shift_id in assignments_by_employee[employee.name]
        )
        if abs(load.hours - expected_minutes / 60.0) > 1e-9:
            errors.append("employee hours do not match assignments: " + employee.name)


def _expected_preference_penalties(
    schedule: ShiftSchedule, assignments_by_employee: Dict[str, List[str]]
) -> List[PreferencePenalty]:
    penalties: List[PreferencePenalty] = []
    for employee in schedule.employees:
        if not employee.preferred_shifts:
            continue
        preferred = set(employee.preferred_shifts)
        for shift_id in assignments_by_employee[employee.name]:
            if shift_id not in preferred:
                penalties.append(
                    PreferencePenalty(
                        employee_name=employee.name,
                        shift_id=shift_id,
                        amount=1,
                        weighted_penalty=schedule.objective_weights.preference_penalty,
                    )
                )
    return penalties


def _expected_objective(
    schedule: ShiftSchedule,
    assignments_by_employee: Dict[str, List[str]],
    shifts_by_id: Dict[str, Shift],
    penalties: List[PreferencePenalty],
) -> ObjectiveBreakdown:
    minutes_by_employee = {
        employee.name: sum(
            shifts_by_id[shift_id].duration_minutes
            for shift_id in assignments_by_employee[employee.name]
        )
        for employee in schedule.employees
    }
    preference_penalty = sum(item.amount for item in penalties)
    weighted_preference = sum(item.weighted_penalty for item in penalties)
    normalized_preference = weighted_preference * PREFERENCE_NORMALIZATION_MINUTES
    fairness_spread = max(minutes_by_employee.values()) - min(minutes_by_employee.values())
    weighted_fairness = fairness_spread * schedule.objective_weights.fairness
    return ObjectiveBreakdown(
        preference_penalty=preference_penalty,
        weighted_preference_penalty=weighted_preference,
        normalized_preference_penalty=normalized_preference,
        fairness_minutes_spread=fairness_spread,
        weighted_fairness=weighted_fairness,
        total=normalized_preference + weighted_fairness,
    )


def _penalty_keys(penalties: List[PreferencePenalty]) -> List[Tuple[str, str, int, int]]:
    return sorted(
        (
            penalty.employee_name,
            penalty.shift_id,
            penalty.amount,
            penalty.weighted_penalty,
        )
        for penalty in penalties
    )


def _maximum_consecutive_day_errors(
    employee_name: str, worked_days: Set[datetime_date], max_days: int
) -> List[str]:
    if len(worked_days) <= max_days:
        return []
    first_day = min(worked_days)
    last_day = max(worked_days)
    errors: List[str] = []
    window_size = max_days + 1
    day_count = (last_day - first_day).days + 1
    for offset in range(max(0, day_count - window_size + 1)):
        window_start = first_day + timedelta(days=offset)
        window = {
            window_start + timedelta(days=delta) for delta in range(window_size)
        }
        if window.issubset(worked_days):
            errors.append(
                "maximum consecutive days violated for %s starting %s"
                % (employee_name, window_start.isoformat())
            )
    return errors


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
            time_to_minutes(shift.start),
            time_to_minutes(shift.end),
            time_to_minutes(period.start),
            time_to_minutes(period.end),
        ):
            return True
    return False


def _overlap(left: Shift, right: Shift) -> bool:
    left_start = _absolute_minute(left)
    right_start = _absolute_minute(right)
    return _overlap_times(
        left_start,
        left_start + left.duration_minutes,
        right_start,
        right_start + right.duration_minutes,
    )


def _gap_minutes(left: Shift, right: Shift) -> int:
    return _absolute_minute(right) - (
        _absolute_minute(left) + left.duration_minutes
    )


def _absolute_minute(shift: Shift) -> int:
    return shift.day.toordinal() * 24 * 60 + time_to_minutes(shift.start)


def _overlap_times(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a
