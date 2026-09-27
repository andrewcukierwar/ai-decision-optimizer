"""Compile a DayPlan schema into an OR-Tools CP-SAT model."""

from dataclasses import dataclass
from typing import Any, Dict, List

from ortools.sat.python import cp_model

from .schema import DayPlan, TaskMode, time_to_minutes


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
    tasks_by_name = {task.name: task for task in plan.tasks}

    start_vars: Dict[str, Any] = {}
    end_vars: Dict[str, Any] = {}
    presence_vars: Dict[str, Any] = {}
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
        if task.mode == TaskMode.ACTIVE:
            active_intervals.append(interval)

    for index, event in enumerate(plan.fixed_events):
        event_start = time_to_minutes(event.start)
        event_duration = time_to_minutes(event.end) - event_start
        active_intervals.append(
            model.NewFixedSizeIntervalVar(
                event_start,
                event_duration,
                "fixed_%d_%s" % (index, event.name),
            )
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
        if preference.type.value == "finish_before":
            assert preference.task is not None
            assert preference.time is not None
            target = time_to_minutes(preference.time)
            # The unused end variable of an absent optional task still has a
            # domain.  Compute the raw lateness independently, then reify the
            # actual penalty so absence is always zero even when the preferred
            # time lies outside the planning horizon.
            late_upper_bound = max(
                0,
                horizon_end
                + tasks_by_name[preference.task].duration_min
                - target,
            )
            raw_late = model.NewIntVar(
                0, late_upper_bound, "raw_late_" + str(index)
            )
            late = model.NewIntVar(0, late_upper_bound, "late_" + str(index))
            presence = presence_vars[preference.task]
            model.AddMaxEquality(
                raw_late, [0, end_vars[preference.task] - target]
            )
            model.Add(late == raw_late).OnlyEnforceIf(presence)
            model.Add(late == 0).OnlyEnforceIf(presence.Not())
            penalty_vars.append(late)
            objective_terms.append(late * preference.weight)
        elif preference.type.value == "preferred_window":
            assert preference.task is not None
            assert preference.start is not None
            assert preference.end is not None
            preferred_start = time_to_minutes(preference.start)
            preferred_end = time_to_minutes(preference.end)
            presence = presence_vars[preference.task]
            early_upper_bound = max(0, preferred_start - horizon_start)
            late_upper_bound = max(
                0,
                horizon_end
                + tasks_by_name[preference.task].duration_min
                - preferred_end,
            )
            raw_early = model.NewIntVar(
                0, early_upper_bound, "raw_early_window_penalty_" + str(index)
            )
            raw_late = model.NewIntVar(
                0, late_upper_bound, "raw_late_window_penalty_" + str(index)
            )
            raw_outside = model.NewIntVar(
                0,
                early_upper_bound + late_upper_bound,
                "raw_preferred_window_penalty_" + str(index),
            )
            outside = model.NewIntVar(
                0,
                early_upper_bound + late_upper_bound,
                "preferred_window_penalty_" + str(index),
            )
            model.AddMaxEquality(
                raw_early, [0, preferred_start - start_vars[preference.task]]
            )
            model.AddMaxEquality(
                raw_late, [0, end_vars[preference.task] - preferred_end]
            )
            model.Add(raw_outside == raw_early + raw_late)
            model.Add(outside == raw_outside).OnlyEnforceIf(presence)
            model.Add(outside == 0).OnlyEnforceIf(presence.Not())
            penalty_vars.append(outside)
            objective_terms.append(outside * preference.weight)
        elif preference.type.value == "minimize_work_interruptions":
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
