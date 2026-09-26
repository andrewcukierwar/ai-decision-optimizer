"""Compile a DayPlan schema into an OR-Tools CP-SAT model."""

from dataclasses import dataclass
from typing import Any, Dict, List

from ortools.sat.python import cp_model

from .schema import DayPlan, PreferenceType, TaskMode, time_to_minutes


@dataclass
class CompiledDayPlan:
    """The model and variable handles needed by the solver."""

    model: cp_model.CpModel
    # OR-Tools does not expose stable public Python classes for every variable
    # handle across releases, so the container uses Any while the model API
    # remains fully typed by its construction methods below.
    start_vars: Dict[str, Any]
    end_vars: Dict[str, Any]
    presence_vars: Dict[str, Any]
    penalty_vars: List[Any]


def compile_day_plan(plan: DayPlan) -> CompiledDayPlan:
    """Build the deterministic CP-SAT model for ``plan``.

    Active tasks and fixed events share a person resource. Passive tasks are
    represented as intervals for timing and precedence, but are deliberately
    omitted from that resource because the machine can run while the person
    works or performs another active task.
    """

    model = cp_model.CpModel()
    horizon_start = time_to_minutes(plan.horizon.start)
    horizon_end = time_to_minutes(plan.horizon.end)

    start_vars: Dict[str, Any] = {}
    end_vars: Dict[str, Any] = {}
    presence_vars: Dict[str, Any] = {}
    task_intervals: Dict[str, Any] = {}
    active_intervals: List[Any] = []

    for task in plan.tasks:
        presence = model.NewBoolVar("present_" + task.name)
        if task.required:
            model.Add(presence == 1)

        # The broad domain keeps even deliberately infeasible inputs
        # representable; feasibility is decided by CP-SAT constraints.
        start = model.NewIntVar(horizon_start, horizon_end, "start_" + task.name)
        end = model.NewIntVar(
            horizon_start, horizon_end + task.duration_min, "end_" + task.name
        )
        interval = model.NewOptionalIntervalVar(
            start, task.duration_min, end, presence, "task_" + task.name
        )

        if task.earliest_start is not None:
            model.Add(start >= time_to_minutes(task.earliest_start)).OnlyEnforceIf(presence)
        if task.latest_end is not None:
            model.Add(end <= time_to_minutes(task.latest_end)).OnlyEnforceIf(presence)
        model.Add(start >= horizon_start).OnlyEnforceIf(presence)
        model.Add(end <= horizon_end).OnlyEnforceIf(presence)

        start_vars[task.name] = start
        end_vars[task.name] = end
        presence_vars[task.name] = presence
        task_intervals[task.name] = interval
        if task.mode == TaskMode.ACTIVE:
            active_intervals.append(interval)

    for event in plan.fixed_events:
        event_start = time_to_minutes(event.start)
        event_duration = time_to_minutes(event.end) - event_start
        active_intervals.append(
            model.NewFixedSizeIntervalVar(event_start, event_duration, "fixed_" + event.name)
        )

    if active_intervals:
        model.AddNoOverlap(active_intervals)

    for precedence in plan.precedences:
        before_present = presence_vars[precedence.before]
        after_present = presence_vars[precedence.after]
        before_end = end_vars[precedence.before]
        after_start = start_vars[precedence.after]
        both_present = [before_present, after_present]
        model.Add(after_start >= before_end + precedence.min_gap_min).OnlyEnforceIf(both_present)
        if precedence.max_gap_min is not None:
            model.Add(after_start <= before_end + precedence.max_gap_min).OnlyEnforceIf(both_present)

    objective_terms = []
    penalty_vars: List[Any] = []

    for index, preference in enumerate(plan.preferences):
        if preference.type == PreferenceType.FINISH_BEFORE:
            assert preference.task is not None
            assert preference.time is not None
            target = time_to_minutes(preference.time)
            task = next(task for task in plan.tasks if task.name == preference.task)
            late_upper_bound = max(0, horizon_end - target)
            late = model.NewIntVar(0, late_upper_bound, "late_" + str(index))
            # If an optional task is absent, its unused end variable is
            # constrained below the target so the penalty remains zero.
            presence = presence_vars[preference.task]
            model.Add(end_vars[preference.task] <= target).OnlyEnforceIf(presence.Not())
            model.AddMaxEquality(late, [0, end_vars[preference.task] - target])
            penalty_vars.append(late)
            objective_terms.append(late * preference.weight)
        elif preference.type == PreferenceType.MINIMIZE_WORK_INTERRUPTIONS:
            if plan.work_window is None:
                continue
            work_start = time_to_minutes(plan.work_window.start)
            work_end = time_to_minutes(plan.work_window.end)
            overlap_vars: List[Any] = []
            for task in plan.tasks:
                if task.mode != TaskMode.ACTIVE:
                    continue
                presence = presence_vars[task.name]
                before_work = model.NewBoolVar("before_work_" + task.name)
                after_work = model.NewBoolVar("after_work_" + task.name)
                overlaps_work = model.NewBoolVar("interrupts_work_" + task.name)

                model.AddImplication(before_work, presence)
                model.AddImplication(after_work, presence)
                model.AddImplication(overlaps_work, presence)
                model.AddImplication(overlaps_work, before_work.Not())
                model.AddImplication(overlaps_work, after_work.Not())
                model.AddBoolOr([before_work, after_work, overlaps_work, presence.Not()])
                model.Add(end_vars[task.name] <= work_start).OnlyEnforceIf(before_work)
                model.Add(start_vars[task.name] >= work_end).OnlyEnforceIf(after_work)
                model.Add(end_vars[task.name] > work_start).OnlyEnforceIf(
                    [presence, before_work.Not()]
                )
                model.Add(start_vars[task.name] < work_end).OnlyEnforceIf(
                    [presence, after_work.Not()]
                )
                overlap_vars.append(overlaps_work)

            if overlap_vars:
                interruption_count = model.NewIntVar(
                    0, len(overlap_vars), "work_interruptions_" + str(index)
                )
                model.Add(interruption_count == sum(overlap_vars))
                penalty_vars.append(interruption_count)
                objective_terms.append(interruption_count * preference.weight)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    return CompiledDayPlan(
        model=model,
        start_vars=start_vars,
        end_vars=end_vars,
        presence_vars=presence_vars,
        penalty_vars=penalty_vars,
    )
