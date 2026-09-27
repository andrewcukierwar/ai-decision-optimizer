"""Solve a compiled DayPlan and materialize a typed result."""

from typing import List, Optional

from ortools.sat.python import cp_model

from .compiler import compile_day_plan
from .schema import (
    DayPlan,
    DayPlanSolution,
    PreferencePenalty,
    SolveStatus,
    TaskAssignment,
    minutes_to_time,
    time_to_minutes,
)


def solve_day_plan(plan: DayPlan, time_limit_seconds: Optional[float] = 10.0) -> DayPlanSolution:
    """Compile and solve ``plan`` with a single deterministic CP-SAT worker."""

    compiled = compile_day_plan(plan)
    solver = cp_model.CpSolver()
    if time_limit_seconds is not None:
        solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(compiled.model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments: List[TaskAssignment] = []
        for task in plan.tasks:
            if solver.Value(compiled.presence_vars[task.name]):
                assignments.append(
                    TaskAssignment(
                        name=task.name,
                        start=minutes_to_time(solver.Value(compiled.start_vars[task.name])),
                        end=minutes_to_time(solver.Value(compiled.end_vars[task.name])),
                    )
                )

        penalties: List[PreferencePenalty] = []
        for index, preference in enumerate(plan.preferences):
            amount = _preference_amount(plan, assignments, index)
            penalties.append(
                PreferencePenalty(
                    preference_index=index,
                    preference_type=preference.type,
                    amount=amount,
                    weighted_penalty=amount * preference.weight,
                )
            )

        objective_value = int(round(solver.ObjectiveValue())) if plan.preferences else 0
        return DayPlanSolution(
            status=(SolveStatus.OPTIMAL if status == cp_model.OPTIMAL else SolveStatus.FEASIBLE),
            assignments=assignments,
            objective_value=objective_value,
            preference_penalties=penalties,
            optimal=status == cp_model.OPTIMAL,
        )

    if status == cp_model.INFEASIBLE:
        return DayPlanSolution(
            status=SolveStatus.INFEASIBLE,
            message="No schedule satisfies all hard DayPlan constraints.",
        )

    return DayPlanSolution(
        status=SolveStatus.UNKNOWN,
        message="CP-SAT did not determine feasibility within the configured limit.",
    )


def _preference_amount(plan: DayPlan, assignments: List[TaskAssignment], index: int) -> int:
    """Calculate the unweighted preference amount from the materialized result."""

    preference = plan.preferences[index]
    assignment_by_name = {assignment.name: assignment for assignment in assignments}
    if preference.type.value == "finish_before":
        assert preference.task is not None
        assert preference.time is not None
        assignment = assignment_by_name.get(preference.task)
        if assignment is None:
            return 0
        return max(0, time_to_minutes(assignment.end) - time_to_minutes(preference.time))

    if preference.type.value == "preferred_window":
        assert preference.task is not None
        assert preference.start is not None
        assert preference.end is not None
        assignment = assignment_by_name.get(preference.task)
        if assignment is None:
            return 0
        starts_early = max(
            0,
            time_to_minutes(preference.start) - time_to_minutes(assignment.start),
        )
        ends_late = max(
            0,
            time_to_minutes(assignment.end) - time_to_minutes(preference.end),
        )
        return starts_early + ends_late

    if plan.work_window is None:
        return 0
    work_start = time_to_minutes(plan.work_window.start)
    work_end = time_to_minutes(plan.work_window.end)
    task_by_name = {task.name: task for task in plan.tasks}
    return sum(
        1
        for assignment in assignments
        if task_by_name[assignment.name].mode.value == "active"
        and time_to_minutes(assignment.start) < work_end
        and time_to_minutes(assignment.end) > work_start
    )
