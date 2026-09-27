import json
from pathlib import Path

import pytest

from decision_optimizer.dayplan import (
    DayPlan,
    DayPlanSolution,
    SolveStatus,
    compile_day_plan,
    solve_day_plan,
    validate_solution,
)
from decision_optimizer.dayplan.schema import time_to_minutes


CASES = Path(__file__).parent / "cases"


def load_case(name):
    with (CASES / name).open() as handle:
        return DayPlan.model_validate(json.load(handle))


@pytest.mark.parametrize(
    "case_name",
    [
        "wfh_laundry_lift.json",
        "simple_active.json",
        "passive_overlap.json",
        "nonzero_objective.json",
        "optional_task.json",
    ],
)
def test_hand_authored_feasible_cases_are_optimal_and_valid(case_name):
    plan = load_case(case_name)
    result = solve_day_plan(plan)

    assert result.status == SolveStatus.OPTIMAL
    assert result.optimal is True
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def overlaps(start_a, end_a, start_b, end_b):
    return start_a < end_b and start_b < end_a


def repeated_meetings_plan(tasks=None):
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "18:00"},
            "fixed_events": [
                {"name": "Meeting", "start": "10:00", "end": "10:30"},
                {"name": "Meeting", "start": "13:00", "end": "13:30"},
                {"name": "Meeting", "start": "16:00", "end": "16:30"},
            ],
            "tasks": tasks or [],
            "precedences": [],
            "preferences": [],
        }
    )


def test_duplicate_fixed_event_names_validate_and_have_unique_compiler_intervals():
    plan = repeated_meetings_plan()

    result = solve_day_plan(plan)
    compiled = compile_day_plan(plan)
    fixed_interval_names = [
        constraint.name
        for constraint in compiled.model.Proto().constraints
        if constraint.name.startswith("fixed_")
    ]

    assert result.status == SolveStatus.OPTIMAL
    assert validate_solution(plan, result).valid
    assert fixed_interval_names == [
        "fixed_0_Meeting",
        "fixed_1_Meeting",
        "fixed_2_Meeting",
    ]


@pytest.mark.parametrize("conflict_start", ["10:00", "13:00", "16:00"])
def test_each_duplicate_fixed_event_is_enforced_against_active_tasks(conflict_start):
    conflict_hour, conflict_minute = (int(part) for part in conflict_start.split(":"))
    conflict_end = conflict_hour * 60 + conflict_minute + 30
    end = "%02d:%02d" % (conflict_end // 60, conflict_end % 60)
    plan = repeated_meetings_plan(
        tasks=[
            {
                "name": "active task",
                "duration_min": 30,
                "mode": "active",
                "earliest_start": conflict_start,
                "latest_end": end,
                "required": True,
            }
        ]
    )

    result = solve_day_plan(plan)

    assert result.status == SolveStatus.INFEASIBLE
    assert validate_solution(plan, result).valid


def test_active_task_does_not_overlap_any_duplicate_fixed_event():
    plan = repeated_meetings_plan(
        tasks=[
            {
                "name": "active task",
                "duration_min": 30,
                "mode": "active",
                "required": True,
            }
        ]
    )

    result = solve_day_plan(plan)
    assignment = result.assignments[0]

    assert result.status == SolveStatus.OPTIMAL
    assert all(
        not overlaps(
            time_to_minutes(assignment.start),
            time_to_minutes(assignment.end),
            time_to_minutes(event.start),
            time_to_minutes(event.end),
        )
        for event in plan.fixed_events
    )
    assert validate_solution(plan, result).valid


def test_task_and_fixed_event_display_names_may_match():
    plan = repeated_meetings_plan(
        tasks=[
            {
                "name": "Meeting",
                "duration_min": 30,
                "mode": "active",
                "required": True,
            }
        ]
    )

    result = solve_day_plan(plan)

    assert result.status == SolveStatus.OPTIMAL
    assert validate_solution(plan, result).valid


def test_task_names_must_still_be_unique():
    with pytest.raises(ValueError, match="task names must be unique"):
        repeated_meetings_plan(
            tasks=[
                {"name": "duplicate", "duration_min": 30, "mode": "active"},
                {"name": "duplicate", "duration_min": 30, "mode": "active"},
            ]
        )


def test_wfh_laundry_lift_has_expected_semantics_and_optimal_objective():
    plan = load_case("wfh_laundry_lift.json")
    result = solve_day_plan(plan)
    assignments = {item.name: item for item in result.assignments}

    assert time_to_minutes(assignments["transfer"].start) >= time_to_minutes(
        assignments["wash"].end
    )
    assert time_to_minutes(assignments["transfer"].start) - time_to_minutes(
        assignments["wash"].end
    ) <= 15
    assert assignments["dry"].start == assignments["transfer"].end
    interruption_penalty = result.preference_penalties[1]
    assert interruption_penalty.amount == 1
    assert result.preference_penalties[0].amount == 0
    assert result.objective_value == 3
    assert time_to_minutes(assignments["lift"].end) <= time_to_minutes(
        plan.preferences[0].time
    )
    assert all(task.earliest_start is None for task in plan.tasks if task.name in {"lift", "groceries"})
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def test_passive_task_must_overlap_fixed_event_and_validator_accepts_it():
    plan = load_case("passive_overlap.json")
    result = solve_day_plan(plan)
    assignments = {item.name: item for item in result.assignments}
    meeting = plan.fixed_events[0]
    machine = assignments["machine"]

    assert overlaps(
        time_to_minutes(machine.start),
        time_to_minutes(machine.end),
        time_to_minutes(meeting.start),
        time_to_minutes(meeting.end),
    )
    assert not overlaps(
        time_to_minutes(assignments["call"].start),
        time_to_minutes(assignments["call"].end),
        time_to_minutes(meeting.start),
        time_to_minutes(meeting.end),
    )
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def test_nonzero_optimal_objective_is_materialized_and_validated():
    plan = load_case("nonzero_objective.json")
    result = solve_day_plan(plan)

    assert result.status == SolveStatus.OPTIMAL
    assert result.optimal is True
    assert result.preference_penalties[0].amount == 20
    assert result.preference_penalties[0].weighted_penalty == 100
    assert result.objective_value == 100
    report = validate_solution(plan, result)
    assert report.valid, report.errors


@pytest.mark.parametrize("case_name", ["infeasible_fixed_event.json", "infeasible_precedence.json"])
def test_infeasible_cases_are_first_class_and_validate(case_name):
    plan = load_case(case_name)
    result = solve_day_plan(plan)

    assert result.status == SolveStatus.INFEASIBLE
    assert result.assignments == []
    report = validate_solution(plan, result)
    assert report.valid, report.errors


def test_validator_catches_a_corrupted_solver_result():
    plan = load_case("simple_active.json")
    result = solve_day_plan(plan)
    result.assignments[0].end = result.assignments[0].start

    report = validate_solution(plan, result)

    assert report.valid is False
    assert any("positive duration" in error for error in report.errors)


def test_validator_rejects_feasible_status_when_fixed_events_overlap():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "fixed_events": [
                {"name": "meeting", "start": "09:00", "end": "10:00"},
                {"name": "appointment", "start": "09:30", "end": "10:30"},
            ],
        }
    )
    corrupted = DayPlanSolution(status=SolveStatus.OPTIMAL, optimal=True)

    report = validate_solution(plan, corrupted)

    assert report.valid is False
    assert any("fixed events overlap" in error for error in report.errors)


def test_validator_rejects_corrupted_nonfeasible_metadata():
    plan = load_case("infeasible_fixed_event.json")
    corrupted = DayPlanSolution(
        status=SolveStatus.INFEASIBLE,
        objective_value=1,
        optimal=True,
    )

    report = validate_solution(plan, corrupted)

    assert report.valid is False
    assert "optimal flag does not match solve status" in report.errors
    assert "non-feasible result must have a zero objective" in report.errors


def test_compiler_exposes_integer_minute_model_handles():
    plan = load_case("simple_active.json")
    compiled = compile_day_plan(plan)

    assert set(compiled.start_vars) == {"email", "focus"}
    assert set(compiled.end_vars) == {"email", "focus"}
