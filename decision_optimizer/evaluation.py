"""Canonical, deterministic evaluation for every experimental architecture.

Outputs are aligned and then scored against fixture ground truth.  The arm's
own parsed problem is used only for formulation comparison and identifier
context; it never defines whether the produced schedule is valid.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .dayplan import (
    DayPlan,
    DayPlanSolution,
    SolveStatus as DaySolveStatus,
    TaskAssignment,
    materialize_day_plan_solution,
    solve_day_plan,
    time_to_minutes as day_minutes,
    validate_solution as validate_dayplan,
)
from .shift_schedule import (
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    RequiredDaysOffRule,
    ShiftAssignment,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus as ShiftSolveStatus,
    materialize_shift_schedule_solution,
    solve_shift_schedule,
    time_to_minutes as shift_minutes,
    validate_solution as validate_shift_schedule,
)
from .shift_schedule.validator import _is_unavailable as shift_is_unavailable
from .telemetry import RunTelemetry


_TOKEN_ALIASES = {
    "appointments": "appointment",
    "calls": "call",
    "emails": "email",
    "groceries": "grocery",
    "lifting": "lift",
    "meetings": "meeting",
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "fri": "friday",
    "sat": "saturday",
    "sun": "sunday",
}
_IGNORED_TOKENS = {"a", "an", "the", "task", "shift"}


@dataclass(frozen=True)
class NameAlignment:
    """A one-to-one mapping from arm identifiers to canonical identifiers."""

    mapping: Dict[str, str] = field(default_factory=dict)
    unmatched_produced: List[str] = field(default_factory=list)
    missing_canonical: List[str] = field(default_factory=list)


class EvaluationRecord(BaseModel):
    """Metrics for one architecture/case pair."""

    model_config = ConfigDict(extra="forbid")

    valid_output: bool
    hard_constraints_satisfied: Optional[int]
    hard_constraints_total: Optional[int]
    required_completed: Optional[bool]
    feasible_correctly_reported: bool
    objective_value: Optional[int]
    optimal_objective: Optional[int]
    objective_gap: Optional[int]
    formulation_match: bool
    formulation_match_structural: bool
    canonical_feasible: bool
    constraint_violations: List[str] = Field(default_factory=list)
    canonical_validation_valid: bool
    alignment_errors: List[str] = Field(default_factory=list)
    latency_total_s: Optional[float] = None
    latency_llm_s: Optional[float] = None
    latency_jev_s: Optional[float] = None
    latency_solver_s: Optional[float] = None
    openai_input_tokens: Optional[int] = None
    openai_output_tokens: Optional[int] = None
    n_model_calls: Optional[int] = None
    estimated_cost: Optional[float] = None


def align_names(
    produced_names: Iterable[str], canonical_names: Iterable[str]
) -> NameAlignment:
    """Deterministically align minor spelling/wording variants one-to-one.

    Exact normalized names win, followed by token equivalence/subsets and a
    conservative string-similarity fallback. Ambiguous best matches stay
    unmatched rather than being guessed.
    """

    produced = list(dict.fromkeys(produced_names))
    canonical = list(dict.fromkeys(canonical_names))
    remaining_canonical = set(canonical)
    mapping: Dict[str, str] = {}

    for source in produced:
        exact = [
            target
            for target in remaining_canonical
            if _normal_name(source) == _normal_name(target) and _numeric_identity(source) == _numeric_identity(target)
        ]
        if len(exact) == 1:
            mapping[source] = exact[0]
            remaining_canonical.remove(exact[0])

    proposals: List[Tuple[Tuple[int, float], str, str]] = []
    for source in produced:
        if source in mapping:
            continue
        candidates = []
        for target in remaining_canonical:
            score = _name_score(source, target)
            if score is not None:
                candidates.append((score, target))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], _normal_name(item[1])))
        best_score = candidates[0][0]
        best = [item for item in candidates if item[0] == best_score]
        if len(best) == 1:
            proposals.append((best_score, source, best[0][1]))

    proposals.sort(key=lambda item: (item[0], _normal_name(item[1])))
    used_targets = set(mapping.values())
    for _, source, target in proposals:
        if target not in used_targets:
            mapping[source] = target
            used_targets.add(target)

    return NameAlignment(
        mapping=mapping,
        unmatched_produced=[name for name in produced if name not in mapping],
        missing_canonical=[name for name in canonical if name not in mapping.values()],
    )


def names_match(left: Optional[str], right: Optional[str]) -> bool:
    if left is None or right is None:
        return left == right
    alignment = align_names([left], [right])
    return alignment.mapping.get(left) == right


def evaluate_dayplan(
    canonical: DayPlan,
    solution: Optional[DayPlanSolution],
    *,
    arm_spec: Optional[DayPlan] = None,
    telemetry: Optional[RunTelemetry] = None,
    formulation_match: Optional[bool] = None,
) -> EvaluationRecord:
    """Score a DayPlan output exclusively against ``canonical``."""

    canonical_result = solve_day_plan(canonical)
    if canonical_result.status == DaySolveStatus.UNKNOWN:
        raise RuntimeError("Canonical feasibility was not established; evaluation cannot proceed")
    canonical_feasible = canonical_result.status in {
        DaySolveStatus.OPTIMAL,
        DaySolveStatus.FEASIBLE,
    }
    optimal_objective = (
        canonical_result.objective_value
        if canonical_result.status == DaySolveStatus.OPTIMAL
        else None
    )
    if solution is None:
        return _empty_record(
            score=_dayplan_hard_score(canonical, [], []),
            optimal_objective=optimal_objective,
            formulation_match=(
                formulation_match
                if formulation_match is not None
                else arm_spec is not None and dayplan_formulation_matches(canonical, arm_spec)
            ),
            canonical_feasible=canonical_feasible,
            formulation_match_structural=arm_spec is not None and dayplan_formulation_matches(canonical, arm_spec, structural=True),
            telemetry=telemetry,
        )

    source_names = (
        [task.name for task in arm_spec.tasks]
        if arm_spec is not None
        else [assignment.name for assignment in solution.assignments]
    )
    alignment = align_names(source_names, [task.name for task in canonical.tasks])
    mapped_assignments: List[TaskAssignment] = []
    alignment_errors: List[str] = []
    for assignment in solution.assignments:
        canonical_name = alignment.mapping.get(assignment.name)
        if canonical_name is None:
            fallback = align_names(
                [assignment.name],
                [
                    task.name
                    for task in canonical.tasks
                    if task.name not in {item.name for item in mapped_assignments}
                ],
            )
            canonical_name = fallback.mapping.get(assignment.name)
        if canonical_name is None:
            alignment_errors.append("unmatched task assignment: " + assignment.name)
            continue
        mapped_assignments.append(
            TaskAssignment(
                name=canonical_name, start=assignment.start, end=assignment.end
            )
        )

    reported_feasible = solution.status in {
        DaySolveStatus.OPTIMAL,
        DaySolveStatus.FEASIBLE,
    }
    canonical_solution = materialize_day_plan_solution(
        canonical,
        mapped_assignments if reported_feasible else [],
        status=DaySolveStatus.FEASIBLE if reported_feasible else solution.status,
    )
    validation = validate_dayplan(canonical, canonical_solution)
    score = _dayplan_hard_score(canonical, mapped_assignments, alignment_errors)
    required_names = {task.name for task in canonical.tasks if task.required}
    completed_names = {item.name for item in mapped_assignments}
    objective_value = canonical_solution.objective_value if reported_feasible else None
    return _record(
        valid_output=True,
        score=score,
        required_completed=required_names.issubset(completed_names),
        feasible_correctly_reported=(reported_feasible if canonical_feasible else solution.status == DaySolveStatus.INFEASIBLE),
        objective_value=objective_value,
        optimal_objective=optimal_objective,
        formulation_match=(
            formulation_match
            if formulation_match is not None
            else arm_spec is not None and dayplan_formulation_matches(canonical, arm_spec)
        ),
        canonical_validation_valid=(
            (validation.valid and not alignment_errors and reported_feasible)
            if canonical_feasible else solution.status == DaySolveStatus.INFEASIBLE and not solution.assignments
        ),
        alignment_errors=alignment_errors,
        canonical_feasible=canonical_feasible,
        formulation_match_structural=arm_spec is not None and dayplan_formulation_matches(canonical, arm_spec, structural=True),
        constraint_violations=[str(item) for item in validation.errors] + alignment_errors,
        telemetry=telemetry,
    )


def evaluate_shift_schedule(
    canonical: ShiftSchedule,
    solution: Optional[ShiftScheduleSolution],
    *,
    arm_spec: Optional[ShiftSchedule] = None,
    telemetry: Optional[RunTelemetry] = None,
    formulation_match: Optional[bool] = None,
) -> EvaluationRecord:
    """Score a workforce output exclusively against ``canonical``."""

    canonical_result = solve_shift_schedule(canonical)
    if canonical_result.status == ShiftSolveStatus.UNKNOWN:
        raise RuntimeError("Canonical feasibility was not established; evaluation cannot proceed")
    canonical_feasible = canonical_result.status in {
        ShiftSolveStatus.OPTIMAL,
        ShiftSolveStatus.FEASIBLE,
    }
    optimal_objective = (
        canonical_result.objective_value
        if canonical_result.status == ShiftSolveStatus.OPTIMAL
        else None
    )
    if solution is None:
        return _empty_record(
            score=_shift_hard_score(canonical, [], []),
            optimal_objective=optimal_objective,
            formulation_match=(
                formulation_match
                if formulation_match is not None
                else arm_spec is not None
                and shift_formulation_matches(canonical, arm_spec)
            ),
            canonical_feasible=canonical_feasible,
            formulation_match_structural=arm_spec is not None and shift_formulation_matches(canonical, arm_spec, structural=True),
            telemetry=telemetry,
        )

    source_employees = (
        [item.name for item in arm_spec.employees]
        if arm_spec is not None
        else [item.employee_name for item in solution.assignments]
    )
    source_shifts = (
        [item.id for item in arm_spec.shifts]
        if arm_spec is not None
        else [item.shift_id for item in solution.assignments]
    )
    employee_alignment = align_names(
        source_employees, [item.name for item in canonical.employees]
    )
    shift_alignment = align_names(source_shifts, [item.id for item in canonical.shifts])
    mapped_assignments: List[ShiftAssignment] = []
    alignment_errors: List[str] = []
    for assignment in solution.assignments:
        employee_name = employee_alignment.mapping.get(assignment.employee_name)
        shift_id = shift_alignment.mapping.get(assignment.shift_id)
        if employee_name is None:
            employee_name = align_names(
                [assignment.employee_name],
                [item.name for item in canonical.employees],
            ).mapping.get(assignment.employee_name)
        if shift_id is None:
            shift_id = align_names(
                [assignment.shift_id], [item.id for item in canonical.shifts]
            ).mapping.get(assignment.shift_id)
        if employee_name is None:
            alignment_errors.append("unmatched employee: " + assignment.employee_name)
        if shift_id is None:
            alignment_errors.append("unmatched shift: " + assignment.shift_id)
        if employee_name is not None and shift_id is not None:
            mapped_assignments.append(
                ShiftAssignment(shift_id=shift_id, employee_name=employee_name)
            )

    reported_feasible = solution.status in {
        ShiftSolveStatus.OPTIMAL,
        ShiftSolveStatus.FEASIBLE,
    }
    canonical_solution = materialize_shift_schedule_solution(
        canonical,
        mapped_assignments if reported_feasible else [],
        status=ShiftSolveStatus.FEASIBLE if reported_feasible else solution.status,
    )
    validation = validate_shift_schedule(canonical, canonical_solution)
    score = _shift_hard_score(canonical, mapped_assignments, alignment_errors)
    assigned_by_shift: Dict[str, int] = {item.id: 0 for item in canonical.shifts}
    for assignment in mapped_assignments:
        assigned_by_shift[assignment.shift_id] += 1
    required_completed = all(
        assigned_by_shift[shift.id] == shift.required_staff
        for shift in canonical.shifts
    )
    objective_value = canonical_solution.objective_value if reported_feasible else None
    return _record(
        valid_output=True,
        score=score,
        required_completed=required_completed if canonical_feasible else None,
        feasible_correctly_reported=(reported_feasible if canonical_feasible else solution.status == ShiftSolveStatus.INFEASIBLE),
        objective_value=objective_value,
        optimal_objective=optimal_objective,
        formulation_match=(
            formulation_match
            if formulation_match is not None
            else arm_spec is not None and shift_formulation_matches(canonical, arm_spec)
        ),
        canonical_validation_valid=(
            (validation.valid and not alignment_errors and reported_feasible)
            if canonical_feasible else solution.status == ShiftSolveStatus.INFEASIBLE and not solution.assignments
        ),
        alignment_errors=alignment_errors,
        canonical_feasible=canonical_feasible,
        formulation_match_structural=arm_spec is not None and shift_formulation_matches(canonical, arm_spec, structural=True),
        constraint_violations=[str(item) for item in validation.errors] + alignment_errors,
        telemetry=telemetry,
    )


def dayplan_formulation_matches(canonical: DayPlan, arm: DayPlan, *, structural: bool = False) -> bool:
    # DayPlan uses local clock minutes, and its time serializers discard timezone
    # metadata. Compare the same representation that is persisted and solved,
    # so a provider time suffix cannot change scoring after a Jev/cache roundtrip.
    canonical = DayPlan.model_validate(canonical.model_dump(mode="json"))
    arm = DayPlan.model_validate(arm.model_dump(mode="json"))
    alignment = align_names(
        [task.name for task in arm.tasks], [task.name for task in canonical.tasks]
    )
    if alignment.unmatched_produced or alignment.missing_canonical:
        return False
    if canonical.horizon != arm.horizon or canonical.work_window != arm.work_window:
        return False
    canonical_tasks = {item.name: item for item in canonical.tasks}
    for task in arm.tasks:
        expected = canonical_tasks[alignment.mapping[task.name]]
        if task.model_dump(exclude={"name"}) != expected.model_dump(exclude={"name"}):
            return False
    if not _fixed_events_match(canonical.fixed_events, arm.fixed_events):
        return False
    actual_precedences = sorted(
        (
            alignment.mapping.get(item.before),
            alignment.mapping.get(item.after),
            item.min_gap_min,
            item.max_gap_min,
        )
        for item in arm.precedences
    )
    expected_precedences = sorted(
        (item.before, item.after, item.min_gap_min, item.max_gap_min)
        for item in canonical.precedences
    )
    if actual_precedences != expected_precedences:
        return False
    actual_preferences = sorted(
        _day_preference_key(item, alignment.mapping)[:-1] if structural else _day_preference_key(item, alignment.mapping) for item in arm.preferences
    )
    expected_preferences = sorted(
        _day_preference_key(item, {})[:-1] if structural else _day_preference_key(item, {}) for item in canonical.preferences
    )
    return actual_preferences == expected_preferences


def shift_formulation_matches(canonical: ShiftSchedule, arm: ShiftSchedule, *, structural: bool = False) -> bool:
    # Scheduling uses local clock time; normalize exactly as persistence does.
    canonical = ShiftSchedule.model_validate(canonical.model_dump(mode="json"))
    arm = ShiftSchedule.model_validate(arm.model_dump(mode="json"))
    employee_alignment = align_names(
        [item.name for item in arm.employees],
        [item.name for item in canonical.employees],
    )
    shift_alignment = align_names(
        [item.id for item in arm.shifts], [item.id for item in canonical.shifts]
    )
    if any(
        (
            employee_alignment.unmatched_produced,
            employee_alignment.missing_canonical,
            shift_alignment.unmatched_produced,
            shift_alignment.missing_canonical,
        )
    ):
        return False
    canonical_shifts = {item.id: item for item in canonical.shifts}
    for shift in arm.shifts:
        expected = canonical_shifts[shift_alignment.mapping[shift.id]]
        if shift.model_dump(exclude={"id"}) != expected.model_dump(exclude={"id"}):
            return False
    canonical_employees = {item.name: item for item in canonical.employees}
    for employee in arm.employees:
        expected = canonical_employees[employee_alignment.mapping[employee.name]]
        if employee.max_hours != expected.max_hours:
            return False
        if sorted(employee.eligible_locations) != sorted(expected.eligible_locations):
            return False
        actual_unavailable = set(
            _unavailability_key(item, shift_alignment.mapping)
            for item in employee.unavailable
        )
        expected_unavailable = set(
            _unavailability_key(item, {}) for item in expected.unavailable
        )
        if actual_unavailable != expected_unavailable:
            return False
        if sorted(shift_alignment.mapping.get(item) for item in employee.preferred_shifts) != sorted(
            expected.preferred_shifts
        ):
            return False
    if not structural and arm.objective_weights != canonical.objective_weights:
        return False
    actual_rules = sorted(
        _rule_key(item, employee_alignment.mapping) for item in arm.rules
    )
    expected_rules = sorted(_rule_key(item, {}) for item in canonical.rules)
    return actual_rules == expected_rules


def _dayplan_hard_score(
    plan: DayPlan,
    assignments: Sequence[TaskAssignment],
    alignment_errors: Sequence[str],
) -> Tuple[int, int]:
    # Every entry below corresponds to a canonical constraint slot.  Output
    # mistakes change pass/fail values, never the number of checks.
    checks: List[bool] = [not alignment_errors]
    tasks = {task.name: task for task in plan.tasks}
    by_name: Dict[str, List[TaskAssignment]] = {name: [] for name in tasks}
    for assignment in assignments:
        by_name.setdefault(assignment.name, []).append(assignment)
    for task in plan.tasks:
        task_assignments = by_name[task.name]
        checks.append(
            len(task_assignments) == 1
            if task.required
            else len(task_assignments) <= 1
        )
        if len(task_assignments) == 1:
            assignment = task_assignments[0]
            start = day_minutes(assignment.start)
            end = day_minutes(assignment.end)
            checks.append(
                end - start == task.duration_min
                and start >= day_minutes(plan.horizon.start)
                and end <= day_minutes(plan.horizon.end)
                and (task.earliest_start is None or start >= day_minutes(task.earliest_start))
                and (task.latest_end is None or end <= day_minutes(task.latest_end))
            )
        else:
            # Optional absence is valid; a missing required task or duplicate
            # assignments cannot silently remove its timing check.
            checks.append(not task.required and not task_assignments)

    active_tasks = [task for task in plan.tasks if task.mode.value == "active"]
    for index, left_task in enumerate(active_tasks):
        for right_task in active_tasks[index + 1 :]:
            left = by_name[left_task.name]
            right = by_name[right_task.name]
            if len(left) > 1 or len(right) > 1:
                checks.append(False)
            elif not left or not right:
                checks.append(
                    (bool(left) or not left_task.required)
                    and (bool(right) or not right_task.required)
                )
            else:
                checks.append(
                    not _overlap(
                        day_minutes(left[0].start),
                        day_minutes(left[0].end),
                        day_minutes(right[0].start),
                        day_minutes(right[0].end),
                    )
                )
    for task in active_tasks:
        task_assignments = by_name[task.name]
        for event in plan.fixed_events:
            if len(task_assignments) > 1:
                checks.append(False)
            elif not task_assignments:
                checks.append(not task.required)
            else:
                checks.append(
                    not _overlap(
                        day_minutes(task_assignments[0].start),
                        day_minutes(task_assignments[0].end),
                        day_minutes(event.start),
                        day_minutes(event.end),
                    )
                )
    for index, left in enumerate(plan.fixed_events):
        for right in plan.fixed_events[index + 1 :]:
            checks.append(
                not _overlap(
                    day_minutes(left.start),
                    day_minutes(left.end),
                    day_minutes(right.start),
                    day_minutes(right.end),
                )
            )
    for precedence in plan.precedences:
        before_values = by_name[precedence.before]
        after_values = by_name[precedence.after]
        if len(before_values) > 1 or len(after_values) > 1:
            checks.append(False)
            continue
        if not before_values or not after_values:
            checks.append(
                (bool(before_values) or not tasks[precedence.before].required)
                and (bool(after_values) or not tasks[precedence.after].required)
            )
            continue
        before = before_values[0]
        after = after_values[0]
        gap = day_minutes(after.start) - day_minutes(before.end)
        checks.append(
            gap >= precedence.min_gap_min
            and (precedence.max_gap_min is None or gap <= precedence.max_gap_min)
        )
    return sum(checks), len(checks)


def _shift_hard_score(
    schedule: ShiftSchedule,
    assignments: Sequence[ShiftAssignment],
    alignment_errors: Sequence[str],
) -> Tuple[int, int]:
    # Reference validity and assignment uniqueness are fixed inventory items;
    # subsequent checks enumerate canonical variables and rules only.
    checks: List[bool] = [not alignment_errors]
    shifts = {item.id: item for item in schedule.shifts}
    employees = {item.name: item for item in schedule.employees}
    pairs = [(item.employee_name, item.shift_id) for item in assignments]
    pair_set = set(pairs)
    checks.append(len(pairs) == len(pair_set))
    for shift in schedule.shifts:
        checks.append(sum(shift_id == shift.id for _, shift_id in pairs) == shift.required_staff)
    ordered_shifts = sorted(
        schedule.shifts, key=lambda item: (item.day, shift_minutes(item.start), item.id)
    )
    for employee in schedule.employees:
        for shift in schedule.shifts:
            assigned = (employee.name, shift.id) in pair_set
            checks.append(
                not assigned
                or (
                    shift.location in employee.eligible_locations
                    and not shift_is_unavailable(employee.unavailable, shift)
                )
            )
        assigned = [shifts[shift_id] for name, shift_id in pairs if name == employee.name]
        checks.append(sum(item.duration_minutes for item in assigned) <= employee.max_hours * 60)
        for index, left in enumerate(ordered_shifts):
            for right in ordered_shifts[index + 1 :]:
                both = (employee.name, left.id) in pair_set and (employee.name, right.id) in pair_set
                if _shift_overlap(left, right):
                    checks.append(not both)
        for rule in schedule.rules:
            if isinstance(rule, MinimumRestRule):
                for index, left in enumerate(ordered_shifts):
                    for right in ordered_shifts[index + 1 :]:
                        if _shift_gap(left, right) < rule.min_rest_hours * 60:
                            both = (employee.name, left.id) in pair_set and (employee.name, right.id) in pair_set
                            checks.append(not both)
            elif isinstance(rule, MaximumConsecutiveDaysRule):
                worked_days = {item.day for item in assigned}
                first = min(item.day for item in schedule.shifts)
                last = max(item.day for item in schedule.shifts)
                for offset in range(max(0, (last - first).days - rule.max_days + 1)):
                    start = first + timedelta(days=offset)
                    window = {start + timedelta(days=day) for day in range(rule.max_days + 1)}
                    checks.append(not window.issubset(worked_days))
            elif isinstance(rule, RequiredDaysOffRule) and rule.employee_name == employee.name:
                worked_days = {item.day for item in assigned}
                checks.extend(day not in worked_days for day in rule.days)
    return sum(checks), len(checks)


def _record(
    *,
    valid_output: bool,
    score: Tuple[int, int],
    required_completed: bool,
    feasible_correctly_reported: bool,
    objective_value: Optional[int],
    optimal_objective: Optional[int],
    formulation_match: bool,
    canonical_validation_valid: bool,
    canonical_feasible: bool,
    formulation_match_structural: bool,
    constraint_violations: List[str],
    alignment_errors: List[str],
    telemetry: Optional[RunTelemetry],
) -> EvaluationRecord:
    metric = _telemetry_metrics(telemetry)
    return EvaluationRecord(
        valid_output=valid_output,
        hard_constraints_satisfied=score[0] if canonical_feasible else None,
        hard_constraints_total=score[1] if canonical_feasible else None,
        required_completed=required_completed if canonical_feasible else None,
        feasible_correctly_reported=feasible_correctly_reported,
        objective_value=objective_value,
        optimal_objective=optimal_objective,
        objective_gap=(
            objective_value - optimal_objective
            if canonical_validation_valid
            and objective_value is not None
            and optimal_objective is not None
            else None
        ),
        formulation_match=formulation_match,
        formulation_match_structural=formulation_match_structural,
        canonical_feasible=canonical_feasible,
        constraint_violations=constraint_violations if canonical_feasible else [],
        canonical_validation_valid=canonical_validation_valid,
        alignment_errors=alignment_errors,
        **metric,
    )


def _empty_record(
    *,
    score: Tuple[int, int],
    optimal_objective: Optional[int],
    formulation_match: bool,
    canonical_feasible: bool,
    formulation_match_structural: bool,
    telemetry: Optional[RunTelemetry],
) -> EvaluationRecord:
    return _record(
        valid_output=False,
        score=score,
        required_completed=False,
        feasible_correctly_reported=False,
        objective_value=None,
        optimal_objective=optimal_objective,
        formulation_match=formulation_match,
        formulation_match_structural=formulation_match_structural,
        canonical_feasible=canonical_feasible,
        canonical_validation_valid=False,
        alignment_errors=[],
        constraint_violations=["No usable model output"],
        telemetry=telemetry,
    )


def _telemetry_metrics(telemetry: Optional[RunTelemetry]) -> Dict[str, Any]:
    if telemetry is None:
        return {}
    return {
        "latency_total_s": telemetry.latency_total_s,
        "latency_llm_s": telemetry.latency_llm_s,
        "latency_jev_s": telemetry.latency_jev_s,
        "latency_solver_s": telemetry.latency_solver_s,
        "openai_input_tokens": telemetry.openai_input_tokens,
        "openai_output_tokens": telemetry.openai_output_tokens,
        "n_model_calls": telemetry.n_model_calls,
        "estimated_cost": telemetry.estimated_cost,
    }


def _numeric_identity(value: str) -> List[str]:
    value = re.sub(r"(\d{4})[- /](\d{2})[- /](\d{2})", r"\1\2\3", value)
    return re.findall(r"\d+", value)


def _name_score(left: str, right: str) -> Optional[Tuple[int, float]]:
    if _numeric_identity(left) != _numeric_identity(right):
        return None
    left_tokens = _name_tokens(left)
    right_tokens = _name_tokens(right)
    if left_tokens and left_tokens == right_tokens:
        return (1, 0.0)
    if left_tokens and right_tokens and (
        left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)
    ):
        return (2, float(abs(len(left_tokens) - len(right_tokens))))
    ratio = SequenceMatcher(None, _normal_name(left), _normal_name(right)).ratio()
    if ratio >= 0.82:
        return (3, 1.0 - ratio)
    return None


def _name_tokens(value: str) -> set:
    return {
        _TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", _ascii(value).lower())
        if token not in _IGNORED_TOKENS
    }


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii(value).lower())


def _ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()


def _fixed_events_match(canonical: Sequence[Any], arm: Sequence[Any]) -> bool:
    remaining = list(canonical)
    for event in arm:
        match = next(
            (
                candidate
                for candidate in remaining
                if names_match(event.name, candidate.name)
                and event.start == candidate.start
                and event.end == candidate.end
            ),
            None,
        )
        if match is None:
            return False
        remaining.remove(match)
    return not remaining


def _day_preference_key(item: Any, mapping: Dict[str, str]) -> Tuple[Any, ...]:
    return (
        item.type.value,
        mapping.get(item.task, item.task),
        item.time,
        item.start,
        item.end,
        item.weight,
    )


def _unavailability_key(item: Any, mapping: Dict[str, str]) -> Tuple[Any, ...]:
    return (
        mapping.get(item.shift_id, item.shift_id),
        item.day,
        item.start,
        item.end,
    )


def _rule_key(item: Any, employee_mapping: Dict[str, str]) -> Tuple[Any, ...]:
    if isinstance(item, MinimumRestRule):
        return (item.type, item.min_rest_hours)
    if isinstance(item, MaximumConsecutiveDaysRule):
        return (item.type, item.max_days)
    return (
        item.type,
        employee_mapping.get(item.employee_name, item.employee_name),
        tuple(sorted(item.days)),
    )


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _shift_absolute(shift: Any) -> int:
    return shift.day.toordinal() * 24 * 60 + shift_minutes(shift.start)


def _shift_overlap(left: Any, right: Any) -> bool:
    left_start = _shift_absolute(left)
    right_start = _shift_absolute(right)
    return _overlap(
        left_start,
        left_start + left.duration_minutes,
        right_start,
        right_start + right.duration_minutes,
    )


def _shift_gap(left: Any, right: Any) -> int:
    return _shift_absolute(right) - (_shift_absolute(left) + left.duration_minutes)
