"""Compile a ShiftSchedule schema into an OR-Tools CP-SAT model."""

from dataclasses import dataclass
from datetime import date as datetime_date
from datetime import timedelta
import re
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .schema import (
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    RequiredDaysOffRule,
    Shift,
    ShiftSchedule,
    Unavailability,
    PREFERENCE_NORMALIZATION_MINUTES,
    time_to_minutes,
)


@dataclass
class CompiledShiftSchedule:
    """The model and variable handles needed by the solver."""

    model: cp_model.CpModel
    assignment_vars: Dict[Tuple[str, str], Any]
    hours_vars: Dict[str, Any]
    preference_penalty_vars: Dict[Tuple[str, str], Any]
    fairness_spread_var: Any


def compile_shift_schedule(schedule: ShiftSchedule) -> CompiledShiftSchedule:
    """Build the deterministic CP-SAT model for ``schedule``.

    Each employee/shift pair has one Boolean assignment variable.  All
    constraints are hard except the two small objective components: assigning
    an employee outside their preferred shift list, and the spread between the
    highest and lowest assigned hours.  The primary score is
    ``preference_weight * violations * 60 + fairness_weight * spread_minutes``;
    raw violation count is a deterministic secondary tie-break.
    """

    model = cp_model.CpModel()
    ordered_shifts = sorted(
        schedule.shifts,
        key=lambda shift: (shift.day, time_to_minutes(shift.start), shift.id),
    )
    employee_names = [employee.name for employee in schedule.employees]
    origin_day = min(shift.day for shift in schedule.shifts)

    assignment_vars: Dict[Tuple[str, str], Any] = {}
    for employee in schedule.employees:
        for shift in schedule.shifts:
            assignment_vars[(employee.name, shift.id)] = model.NewBoolVar(
                "assign_%s_%s"
                % (_safe_name(employee.name), _safe_name(shift.id))
            )

    # Every shift is covered exactly at its required staffing level.
    for shift in schedule.shifts:
        model.Add(
            sum(assignment_vars[(employee.name, shift.id)] for employee in schedule.employees)
            == shift.required_staff
        )

    for employee in schedule.employees:
        for shift in schedule.shifts:
            variable = assignment_vars[(employee.name, shift.id)]
            if shift.location not in employee.eligible_locations:
                model.Add(variable == 0)
            if _is_unavailable(employee.unavailable, shift):
                model.Add(variable == 0)

        # The schema expresses max_hours as whole hours, while shifts retain
        # minute precision internally.
        model.Add(
            sum(
                shift.duration_minutes * assignment_vars[(employee.name, shift.id)]
                for shift in schedule.shifts
            )
            <= employee.max_hours * 60
        )

        for left_index, left in enumerate(ordered_shifts):
            for right in ordered_shifts[left_index + 1 :]:
                if _overlap(left, right, origin_day):
                    model.Add(
                        assignment_vars[(employee.name, left.id)]
                        + assignment_vars[(employee.name, right.id)]
                        <= 1
                    )

    minimum_rest_rules = [
        rule for rule in schedule.rules if isinstance(rule, MinimumRestRule)
    ]
    for rule in minimum_rest_rules:
        rest_minutes = rule.min_rest_hours * 60
        if rest_minutes == 0:
            continue
        for employee in schedule.employees:
            for left_index, left in enumerate(ordered_shifts):
                for right in ordered_shifts[left_index + 1 :]:
                    if _gap_minutes(left, right, origin_day) < rest_minutes:
                        model.Add(
                            assignment_vars[(employee.name, left.id)]
                            + assignment_vars[(employee.name, right.id)]
                            <= 1
                        )

    _add_maximum_consecutive_day_constraints(
        model, schedule, assignment_vars, ordered_shifts
    )
    _add_required_days_off_constraints(model, schedule, assignment_vars)

    hours_vars: Dict[str, Any] = {}
    total_shift_minutes = sum(shift.duration_minutes for shift in schedule.shifts)
    for employee in schedule.employees:
        hours = model.NewIntVar(
            0,
            total_shift_minutes,
            "hours_%s" % _safe_name(employee.name),
        )
        model.Add(
            hours
            == sum(
                shift.duration_minutes * assignment_vars[(employee.name, shift.id)]
                for shift in schedule.shifts
            )
        )
        hours_vars[employee.name] = hours

    max_hours_var = model.NewIntVar(0, total_shift_minutes, "max_hours_worked")
    min_hours_var = model.NewIntVar(0, total_shift_minutes, "min_hours_worked")
    model.AddMaxEquality(max_hours_var, list(hours_vars.values()))
    model.AddMinEquality(min_hours_var, list(hours_vars.values()))
    fairness_spread_var = model.NewIntVar(0, total_shift_minutes, "fairness_spread")
    model.Add(fairness_spread_var == max_hours_var - min_hours_var)

    preference_penalty_vars: Dict[Tuple[str, str], Any] = {}
    for employee in schedule.employees:
        if not employee.preferred_shifts:
            continue
        preferred = set(employee.preferred_shifts)
        for shift in schedule.shifts:
            penalty = model.NewBoolVar(
                "preference_%s_%s"
                % (_safe_name(employee.name), _safe_name(shift.id))
            )
            if shift.id in preferred:
                model.Add(penalty == 0)
            else:
                model.Add(
                    penalty == assignment_vars[(employee.name, shift.id)]
                )
            preference_penalty_vars[(employee.name, shift.id)] = penalty

    raw_preference_penalty = sum(preference_penalty_vars.values())
    primary_objective = (
        schedule.objective_weights.preference_penalty
        * PREFERENCE_NORMALIZATION_MINUTES
        * raw_preference_penalty
        + schedule.objective_weights.fairness * fairness_spread_var
    )
    # Preserve the normalized primary ordering exactly, then prefer fewer raw
    # preference violations for equal primary scores.  The scale is strictly
    # greater than the largest possible secondary value.
    tie_break_scale = len(preference_penalty_vars) + 1
    model.Minimize(primary_objective * tie_break_scale + raw_preference_penalty)

    return CompiledShiftSchedule(
        model=model,
        assignment_vars=assignment_vars,
        hours_vars=hours_vars,
        preference_penalty_vars=preference_penalty_vars,
        fairness_spread_var=fairness_spread_var,
    )


def _add_maximum_consecutive_day_constraints(
    model: cp_model.CpModel,
    schedule: ShiftSchedule,
    assignment_vars: Dict[Tuple[str, str], Any],
    ordered_shifts: List[Shift],
) -> None:
    rules = [
        rule
        for rule in schedule.rules
        if isinstance(rule, MaximumConsecutiveDaysRule)
    ]
    if not rules:
        return

    schedule_dates = sorted({shift.day for shift in ordered_shifts})
    first_day = min(schedule_dates)
    last_day = max(schedule_dates)
    day_count = (last_day - first_day).days + 1

    for employee in schedule.employees:
        worked_day_vars: Dict[datetime_date, Any] = {}
        for day in schedule_dates:
            day_assignments = [
                assignment_vars[(employee.name, shift.id)]
                for shift in ordered_shifts
                if shift.day == day
            ]
            worked = model.NewBoolVar(
                "worked_%s_%s" % (_safe_name(employee.name), day.isoformat())
            )
            for assignment in day_assignments:
                model.Add(worked >= assignment)
            model.Add(worked <= sum(day_assignments))
            worked_day_vars[day] = worked

        for rule in rules:
            window_size = rule.max_days + 1
            if day_count < window_size:
                continue
            for offset in range(day_count - window_size + 1):
                window_start = first_day + timedelta(days=offset)
                window_days = [
                    window_start + timedelta(days=delta)
                    for delta in range(window_size)
                ]
                model.Add(
                    sum(worked_day_vars.get(day, 0) for day in window_days)
                    <= rule.max_days
                )


def _add_required_days_off_constraints(
    model: cp_model.CpModel,
    schedule: ShiftSchedule,
    assignment_vars: Dict[Tuple[str, str], Any],
) -> None:
    for rule in schedule.rules:
        if not isinstance(rule, RequiredDaysOffRule):
            continue
        days_off = set(rule.days)
        for shift in schedule.shifts:
            if shift.day in days_off:
                model.Add(assignment_vars[(rule.employee_name, shift.id)] == 0)


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


def _overlap(left: Shift, right: Shift, origin_day: datetime_date) -> bool:
    return _overlap_times(
        _absolute_minute(left, origin_day),
        _absolute_minute(left, origin_day) + left.duration_minutes,
        _absolute_minute(right, origin_day),
        _absolute_minute(right, origin_day) + right.duration_minutes,
    )


def _gap_minutes(left: Shift, right: Shift, origin_day: datetime_date) -> int:
    left_end = _absolute_minute(left, origin_day) + left.duration_minutes
    right_start = _absolute_minute(right, origin_day)
    return right_start - left_end


def _absolute_minute(shift: Shift, origin_day: datetime_date) -> int:
    return (shift.day - origin_day).days * 24 * 60 + time_to_minutes(shift.start)


def _overlap_times(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "value"
