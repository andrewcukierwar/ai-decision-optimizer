"""Independent plain-Python validation of DayPlan solver results.

This module intentionally does not import the compiler or inspect CP-SAT
variables.  Its checks are written against the input schema and the
materialized result so a compiler bug can be caught independently.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .schema import DayPlan, DayPlanSolution, SolveStatus, Task, TaskAssignment, time_to_minutes


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str] = field(default_factory=list)


def validate_solution(plan: DayPlan, result: DayPlanSolution) -> ValidationReport:
    """Check a solution against DayPlan semantics using only plain Python."""

    errors: List[str] = []
    if result.status in (SolveStatus.INFEASIBLE, SolveStatus.UNKNOWN):
        if result.assignments:
            errors.append("non-feasible result must not contain assignments")
        if result.preference_penalties:
            errors.append("non-feasible result must not contain preference penalties")
        return ValidationReport(valid=not errors, errors=errors)

    tasks_by_name: Dict[str, Task] = {task.name: task for task in plan.tasks}
    assignments_by_name: Dict[str, TaskAssignment] = {}
    for assignment in result.assignments:
        if assignment.name not in tasks_by_name:
            errors.append("assignment references unknown task: " + assignment.name)
        elif assignment.name in assignments_by_name:
            errors.append("task has duplicate assignments: " + assignment.name)
        else:
            assignments_by_name[assignment.name] = assignment

    for task in plan.tasks:
        if task.required and task.name not in assignments_by_name:
            errors.append("required task is missing: " + task.name)

    horizon_start = time_to_minutes(plan.horizon.start)
    horizon_end = time_to_minutes(plan.horizon.end)
    for name, assignment in assignments_by_name.items():
        task = tasks_by_name[name]
        start = time_to_minutes(assignment.start)
        end = time_to_minutes(assignment.end)
        if end <= start:
            errors.append("task does not have positive duration: " + name)
            continue
        if end - start != task.duration_min:
            errors.append("task duration mismatch: " + name)
        if start < horizon_start or end > horizon_end:
            errors.append("task is outside the horizon: " + name)
        if task.earliest_start and start < time_to_minutes(task.earliest_start):
            errors.append("task starts before earliest_start: " + name)
        if task.latest_end and end > time_to_minutes(task.latest_end):
            errors.append("task ends after latest_end: " + name)

    active_intervals: List[Tuple[str, int, int]] = []
    for name, assignment in assignments_by_name.items():
        if tasks_by_name[name].mode.value == "active":
            active_intervals.append(
                (name, time_to_minutes(assignment.start), time_to_minutes(assignment.end))
            )

    for left_index, left in enumerate(active_intervals):
        for right in active_intervals[left_index + 1 :]:
            if _overlap(left[1], left[2], right[1], right[2]):
                errors.append("active tasks overlap: %s and %s" % (left[0], right[0]))

    for event in plan.fixed_events:
        event_start = time_to_minutes(event.start)
        event_end = time_to_minutes(event.end)
        for name, start, end in active_intervals:
            if _overlap(start, end, event_start, event_end):
                errors.append("active task overlaps fixed event: %s and %s" % (name, event.name))

    for precedence in plan.precedences:
        before = assignments_by_name.get(precedence.before)
        after = assignments_by_name.get(precedence.after)
        if before is None or after is None:
            continue
        before_end = time_to_minutes(before.end)
        after_start = time_to_minutes(after.start)
        gap = after_start - before_end
        if gap < precedence.min_gap_min:
            errors.append("precedence minimum gap violated: %s -> %s" % (precedence.before, precedence.after))
        if precedence.max_gap_min is not None and gap > precedence.max_gap_min:
            errors.append("precedence maximum gap violated: %s -> %s" % (precedence.before, precedence.after))

    expected_penalties = []
    for index, preference in enumerate(plan.preferences):
        if preference.type.value == "finish_before":
            assert preference.task is not None
            assert preference.time is not None
            assignment = assignments_by_name.get(preference.task)
            amount = (
                0
                if assignment is None
                else max(0, time_to_minutes(assignment.end) - time_to_minutes(preference.time))
            )
        else:
            amount = _work_interruptions(plan, assignments_by_name)
        expected_penalties.append((index, preference.type, amount, amount * preference.weight))

    if len(result.preference_penalties) != len(expected_penalties):
        errors.append("preference penalty count does not match the plan")
    else:
        for actual, expected in zip(result.preference_penalties, expected_penalties):
            if (
                actual.preference_index,
                actual.preference_type,
                actual.amount,
                actual.weighted_penalty,
            ) != expected:
                errors.append("preference penalty does not match the schedule")
                break

    expected_objective = sum(item[3] for item in expected_penalties)
    if result.objective_value != expected_objective:
        errors.append(
            "objective mismatch: expected %d, got %d" % (expected_objective, result.objective_value)
        )

    return ValidationReport(valid=not errors, errors=errors)


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _work_interruptions(
    plan: DayPlan, assignments_by_name: Dict[str, TaskAssignment]
) -> int:
    if plan.work_window is None:
        return 0
    work_start = time_to_minutes(plan.work_window.start)
    work_end = time_to_minutes(plan.work_window.end)
    return sum(
        1
        for task in plan.tasks
        if task.mode.value == "active"
        and task.name in assignments_by_name
        and _overlap(
            time_to_minutes(assignments_by_name[task.name].start),
            time_to_minutes(assignments_by_name[task.name].end),
            work_start,
            work_end,
        )
    )
