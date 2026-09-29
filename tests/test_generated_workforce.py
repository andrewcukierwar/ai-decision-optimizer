import json

from decision_optimizer.shift_schedule import (
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    RequiredDaysOffRule,
    ShiftSchedule,
    solve_shift_schedule,
    validate_solution,
)
from scripts.generate_workforce_cases import DEFAULT_OUTPUT, generate_cases


def test_generated_workforce_fixture_matches_generator_and_is_canonical_first():
    generated = generate_cases()
    checked_in = json.loads(DEFAULT_OUTPUT.read_text())

    assert checked_in == generated
    assert len(generated) == 3
    for case in generated:
        schedule = ShiftSchedule.model_validate(case["canonical"])
        assert 8 <= len(schedule.employees) <= 10
        assert 14 <= len(schedule.shifts) <= 28
        assert all(employee.unavailable for employee in schedule.employees)
        assert any(
            isinstance(rule, MinimumRestRule) for rule in schedule.rules
        )
        assert any(
            isinstance(rule, MaximumConsecutiveDaysRule) for rule in schedule.rules
        )
        assert any(
            isinstance(rule, RequiredDaysOffRule) for rule in schedule.rules
        )
        assert schedule.shifts[0].id in case["prompt"]
        assert schedule.employees[0].name in case["prompt"]


def test_generated_workforce_cases_are_feasible_and_validate():
    for case in generate_cases():
        schedule = ShiftSchedule.model_validate(case["canonical"])
        solution = solve_shift_schedule(schedule)
        report = validate_solution(schedule, solution)

        assert solution.status.value == "optimal", case["name"]
        assert report.valid, (case["name"], report.errors)
