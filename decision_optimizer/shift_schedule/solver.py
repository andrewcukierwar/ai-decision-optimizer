"""Solve a compiled ShiftSchedule and materialize a typed result."""

from typing import Dict, Iterable, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from .compiler import compile_shift_schedule
from .schema import (
    EmployeeLoad,
    ObjectiveBreakdown,
    PreferencePenalty,
    PREFERENCE_NORMALIZATION_MINUTES,
    ShiftAssignment,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus,
)


def solve_shift_schedule(
    schedule: ShiftSchedule, time_limit_seconds: Optional[float] = 10.0
) -> ShiftScheduleSolution:
    """Compile and solve ``schedule`` with deterministic CP-SAT settings."""

    compiled = compile_shift_schedule(schedule)
    solver = cp_model.CpSolver()
    if time_limit_seconds is not None:
        solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(compiled.model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments: List[ShiftAssignment] = []
        for shift in sorted(schedule.shifts, key=lambda item: item.id):
            for employee in schedule.employees:
                if solver.Value(compiled.assignment_vars[(employee.name, shift.id)]):
                    assignments.append(
                        ShiftAssignment(
                            shift_id=shift.id,
                            employee_name=employee.name,
                        )
                    )

        return materialize_shift_schedule_solution(
            schedule,
            assignments,
            status=(
                SolveStatus.OPTIMAL
                if status == cp_model.OPTIMAL
                else SolveStatus.FEASIBLE
            ),
            optimal=status == cp_model.OPTIMAL,
        )

    if status == cp_model.INFEASIBLE:
        return ShiftScheduleSolution(
            status=SolveStatus.INFEASIBLE,
            message="No staffing schedule satisfies all hard ShiftSchedule constraints.",
        )

    return ShiftScheduleSolution(
        status=SolveStatus.UNKNOWN,
        message="CP-SAT did not determine feasibility within the configured limit.",
    )


def materialize_shift_schedule_solution(
    schedule: ShiftSchedule,
    assignments: Iterable[ShiftAssignment],
    *,
    status: SolveStatus = SolveStatus.FEASIBLE,
    optimal: bool = False,
    message: Optional[str] = None,
) -> ShiftScheduleSolution:
    """Recompute workloads and objective fields for external assignments."""

    materialized = list(assignments)
    if status in (SolveStatus.INFEASIBLE, SolveStatus.UNKNOWN):
        return ShiftScheduleSolution(status=status, optimal=False, message=message)

    employee_hours = _materialize_employee_hours(schedule, materialized)
    penalties, breakdown = _materialize_objective(schedule, materialized)
    return ShiftScheduleSolution(
        status=SolveStatus.OPTIMAL if optimal else status,
        assignments=materialized,
        employee_hours=employee_hours,
        preference_penalties=penalties,
        objective_breakdown=breakdown,
        objective_value=breakdown.total,
        optimal=optimal,
        message=message,
    )


def _materialize_employee_hours(
    schedule: ShiftSchedule, assignments: List[ShiftAssignment]
) -> List[EmployeeLoad]:
    assigned_by_employee: Dict[str, List[str]] = {
        employee.name: [] for employee in schedule.employees
    }
    shift_by_id = {shift.id: shift for shift in schedule.shifts}
    for assignment in assignments:
        assigned_by_employee[assignment.employee_name].append(assignment.shift_id)

    loads: List[EmployeeLoad] = []
    for employee in schedule.employees:
        shift_ids = assigned_by_employee[employee.name]
        minutes = sum(shift_by_id[shift_id].duration_minutes for shift_id in shift_ids)
        loads.append(
            EmployeeLoad(
                employee_name=employee.name,
                hours=minutes / 60.0,
                assigned_shift_ids=sorted(shift_ids),
            )
        )
    return loads


def _materialize_objective(
    schedule: ShiftSchedule, assignments: List[ShiftAssignment]
) -> Tuple[List[PreferencePenalty], ObjectiveBreakdown]:
    assigned_pairs: Set[Tuple[str, str]] = {
        (assignment.employee_name, assignment.shift_id) for assignment in assignments
    }
    penalties: List[PreferencePenalty] = []
    for employee in schedule.employees:
        if not employee.preferred_shifts:
            continue
        preferred = set(employee.preferred_shifts)
        for shift in schedule.shifts:
            if (employee.name, shift.id) in assigned_pairs and shift.id not in preferred:
                penalties.append(
                    PreferencePenalty(
                        employee_name=employee.name,
                        shift_id=shift.id,
                        amount=1,
                        weighted_penalty=schedule.objective_weights.preference_penalty,
                    )
                )

    minutes_by_employee = {employee.name: 0 for employee in schedule.employees}
    shift_by_id = {shift.id: shift for shift in schedule.shifts}
    for assignment in assignments:
        minutes_by_employee[assignment.employee_name] += shift_by_id[
            assignment.shift_id
        ].duration_minutes

    preference_penalty = sum(item.amount for item in penalties)
    weighted_preference = sum(item.weighted_penalty for item in penalties)
    normalized_preference = weighted_preference * PREFERENCE_NORMALIZATION_MINUTES
    fairness_spread = max(minutes_by_employee.values()) - min(minutes_by_employee.values())
    weighted_fairness = fairness_spread * schedule.objective_weights.fairness
    total = normalized_preference + weighted_fairness
    breakdown = ObjectiveBreakdown(
        preference_penalty=preference_penalty,
        weighted_preference_penalty=weighted_preference,
        normalized_preference_penalty=normalized_preference,
        fairness_minutes_spread=fairness_spread,
        weighted_fairness=weighted_fairness,
        total=total,
    )
    return penalties, breakdown
