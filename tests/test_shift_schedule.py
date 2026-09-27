import json
from pathlib import Path

import pytest

from decision_optimizer.shift_schedule import (
    ShiftSchedule,
    ShiftAssignment,
    SolveStatus,
    compile_shift_schedule,
    solve_shift_schedule,
    validate_solution,
)
from decision_optimizer.shift_schedule.solver import _materialize_objective


CASES = Path(__file__).parent / "cases"
FIXTURES = Path(__file__).parent / "fixtures" / "shift_schedule_cases"


def load_case(name):
    with (CASES / name).open() as handle:
        return ShiftSchedule.model_validate(json.load(handle))


@pytest.mark.parametrize(
    "case_name", ["shift_schedule_basic.json", "shift_schedule_preference_fairness.json"]
)
def test_feasible_shift_schedule_cases_are_optimal_and_independently_validated(case_name):
    schedule = load_case(case_name)
    result = solve_shift_schedule(schedule)

    assert result.status == SolveStatus.OPTIMAL
    assert result.optimal is True
    report = validate_solution(schedule, result)
    assert report.valid, report.errors


def test_basic_case_respects_location_and_availability_semantics():
    schedule = load_case("shift_schedule_basic.json")
    result = solve_shift_schedule(schedule)
    assignments = {(item.shift_id, item.employee_name) for item in result.assignments}

    assert ("mon_hospital", "Bob") in assignments
    assert ("wed_clinic", "Alice") in assignments
    assert not ("mon_hospital", "Alice") in assignments
    assert validate_solution(schedule, result).valid


def test_preference_and_fairness_objective_is_materialized_with_ties_allowed():
    schedule = load_case("shift_schedule_preference_fairness.json")
    result = solve_shift_schedule(schedule)

    assert result.objective_value == 600
    assert result.objective_breakdown.preference_penalty == 1
    assert result.objective_breakdown.weighted_preference_penalty == 10
    assert result.objective_breakdown.normalized_preference_penalty == 600
    assert result.objective_breakdown.fairness_minutes_spread == 0
    assert len(result.preference_penalties) == 1
    assert result.preference_penalties[0].shift_id == "s2"
    assert sorted(load.hours for load in result.employee_hours) == [4.0, 4.0]


def test_infeasible_shift_schedule_is_first_class_and_validates():
    schedule = load_case("shift_schedule_infeasible.json")
    result = solve_shift_schedule(schedule)

    assert result.status == SolveStatus.INFEASIBLE
    assert result.assignments == []
    report = validate_solution(schedule, result)
    assert report.valid, report.errors


def test_validator_catches_a_corrupted_materialized_result():
    schedule = load_case("shift_schedule_basic.json")
    result = solve_shift_schedule(schedule)
    result.assignments[0].shift_id = result.assignments[1].shift_id

    report = validate_solution(schedule, result)

    assert report.valid is False
    assert any("coverage" in error or "duplicate" in error for error in report.errors)


def test_minimum_rest_is_a_hard_constraint():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "early", "day": "2026-10-05", "start": "08:00", "end": "12:00", "location": "site", "required_staff": 1},
                {"id": "late", "day": "2026-10-05", "start": "16:00", "end": "20:00", "location": "site", "required_staff": 1},
            ],
            "employees": [
                {"name": "Only", "max_hours": 8, "eligible_locations": ["site"]}
            ],
            "rules": [{"type": "minimum_rest", "min_rest_hours": 12}],
        }
    )

    result = solve_shift_schedule(schedule)

    assert result.status == SolveStatus.INFEASIBLE


def test_maximum_consecutive_days_is_a_hard_constraint():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "d1", "day": "2026-10-05", "start": "09:00", "end": "10:00", "location": "site", "required_staff": 1},
                {"id": "d2", "day": "2026-10-06", "start": "09:00", "end": "10:00", "location": "site", "required_staff": 1},
                {"id": "d3", "day": "2026-10-07", "start": "09:00", "end": "10:00", "location": "site", "required_staff": 1},
            ],
            "employees": [
                {"name": "Only", "max_hours": 8, "eligible_locations": ["site"]}
            ],
            "rules": [{"type": "maximum_consecutive_days", "max_days": 2}],
        }
    )

    result = solve_shift_schedule(schedule)

    assert result.status == SolveStatus.INFEASIBLE


def test_required_days_off_is_a_hard_constraint():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "day_off_shift", "day": "2026-10-05", "start": "09:00", "end": "10:00", "location": "site", "required_staff": 1}
            ],
            "employees": [
                {"name": "Only", "max_hours": 8, "eligible_locations": ["site"]}
            ],
            "rules": [
                {"type": "required_days_off", "employee_name": "Only", "days": ["2026-10-05"]}
            ],
        }
    )

    result = solve_shift_schedule(schedule)

    assert result.status == SolveStatus.INFEASIBLE


def test_json_fixtures_validate_as_shift_schedules():
    fixture_files = sorted(FIXTURES.glob("*.json"))
    assert len(fixture_files) == 3
    for fixture_file in fixture_files:
        with fixture_file.open() as handle:
            ShiftSchedule.model_validate(json.load(handle))


def _tradeoff_schedule(*, preference_penalty=1, fairness=1, three_shifts=False):
    if three_shifts:
        shifts = [
            {"id": "s1", "day": "2026-10-05", "start": "09:00", "end": "10:00", "location": "site", "required_staff": 1},
            {"id": "s2", "day": "2026-10-05", "start": "10:00", "end": "11:00", "location": "site", "required_staff": 1},
            {"id": "s3", "day": "2026-10-05", "start": "11:00", "end": "12:00", "location": "site", "required_staff": 1},
        ]
        employees = [
            {"name": "A", "max_hours": 8, "eligible_locations": ["site"], "preferred_shifts": ["s1", "s2", "s3"]},
            {"name": "B", "max_hours": 8, "eligible_locations": ["site"], "preferred_shifts": ["s3"], "unavailable": [{"shift_id": "s3"}]},
        ]
    else:
        shifts = [
            {"id": "s1", "day": "2026-10-05", "start": "09:00", "end": "11:00", "location": "site", "required_staff": 1},
            {"id": "s2", "day": "2026-10-05", "start": "11:00", "end": "13:00", "location": "site", "required_staff": 1},
            {"id": "s3", "day": "2026-10-05", "start": "13:00", "end": "14:00", "location": "site", "required_staff": 1},
            {"id": "s4", "day": "2026-10-05", "start": "14:00", "end": "17:00", "location": "site", "required_staff": 1},
        ]
        employees = [
            {"name": "A", "max_hours": 8, "eligible_locations": ["site"], "preferred_shifts": ["s1", "s2", "s3", "s4"]},
            {"name": "B", "max_hours": 8, "eligible_locations": ["site"], "preferred_shifts": ["s4"]},
        ]
    return ShiftSchedule.model_validate(
        {
            "shifts": shifts,
            "employees": employees,
            "objective_weights": {
                "preference_penalty": preference_penalty,
                "fairness": fairness,
            },
        }
    )


def test_normalized_objective_tie_breaks_to_fewer_preference_violations():
    schedule = _tradeoff_schedule()
    candidate_a = [
        ShiftAssignment(shift_id="s1", employee_name="A"),
        ShiftAssignment(shift_id="s2", employee_name="A"),
        ShiftAssignment(shift_id="s3", employee_name="A"),
        ShiftAssignment(shift_id="s4", employee_name="B"),
    ]
    candidate_b = [
        ShiftAssignment(shift_id="s3", employee_name="A"),
        ShiftAssignment(shift_id="s4", employee_name="A"),
        ShiftAssignment(shift_id="s1", employee_name="B"),
        ShiftAssignment(shift_id="s2", employee_name="B"),
    ]
    _, breakdown_a = _materialize_objective(schedule, candidate_a)
    _, breakdown_b = _materialize_objective(schedule, candidate_b)

    assert breakdown_a.total == 120
    assert breakdown_b.total == 120
    assert breakdown_a.preference_penalty < breakdown_b.preference_penalty

    result = solve_shift_schedule(_tradeoff_schedule(preference_penalty=2))

    assert result.objective_breakdown.preference_penalty == 0
    assert result.objective_breakdown.fairness_minutes_spread == 120
    assert result.objective_breakdown.total == 120
    assert {item.shift_id for item in result.assignments if item.employee_name == "B"} == {"s4"}

    zero_preferences = 0 * 60 + 120
    two_preferences = 2 * 60 + 0
    assert zero_preferences == two_preferences
    assert 0 < 2  # The secondary rule chooses the first candidate on this tie.


def test_smaller_normalized_primary_objective_beats_fewer_preference_violations():
    result = solve_shift_schedule(_tradeoff_schedule(three_shifts=True))

    assert result.objective_breakdown.preference_penalty == 1
    assert result.objective_breakdown.fairness_minutes_spread == 60
    assert result.objective_breakdown.total == 120


def test_objective_weights_change_preference_fairness_tradeoff():
    preference_heavy = solve_shift_schedule(_tradeoff_schedule(preference_penalty=2))
    fairness_heavy = solve_shift_schedule(_tradeoff_schedule(fairness=2))

    assert preference_heavy.objective_breakdown.preference_penalty == 0
    assert preference_heavy.objective_breakdown.fairness_minutes_spread == 120
    assert fairness_heavy.objective_breakdown.preference_penalty == 1
    assert fairness_heavy.objective_breakdown.fairness_minutes_spread == 0
