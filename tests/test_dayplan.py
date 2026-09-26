import json
from pathlib import Path

import pytest

from dayplan import (
    DayPlan,
    SolveStatus,
    compile_day_plan,
    solve_day_plan,
    validate_solution,
)
from dayplan.schema import time_to_minutes


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


def test_wfh_laundry_lift_has_expected_dependencies_and_zero_interruptions():
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
    assert interruption_penalty.amount == 0
    assert result.objective_value == 0


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


def test_compiler_exposes_integer_minute_model_handles():
    plan = load_case("simple_active.json")
    compiled = compile_day_plan(plan)

    assert set(compiled.start_vars) == {"email", "focus"}
    assert set(compiled.end_vars) == {"email", "focus"}
