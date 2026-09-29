from decision_optimizer.dayplan import (
    DayPlan,
    SolveStatus,
    TaskAssignment,
    materialize_day_plan_solution,
)
from decision_optimizer.evaluation import (
    align_names,
    dayplan_formulation_matches,
    evaluate_dayplan,
    evaluate_shift_schedule,
    shift_formulation_matches,
)
from decision_optimizer.shift_schedule import (
    ShiftAssignment,
    ShiftSchedule,
    materialize_shift_schedule_solution,
)


def test_name_alignment_handles_minor_variants_and_leaves_ambiguous_names_unmatched():
    alignment = align_names(
        ["Morning groceries", "FOCUS-block", "unknown chore"],
        ["grocery", "Focus block", "email"],
    )

    assert alignment.mapping == {
        "Morning groceries": "grocery",
        "FOCUS-block": "Focus block",
    }
    assert alignment.unmatched_produced == ["unknown chore"]
    assert alignment.missing_canonical == ["email"]

    ambiguous = align_names(["meeting"], ["team meeting", "client meeting"])
    assert ambiguous.mapping == {}
    assert ambiguous.unmatched_produced == ["meeting"]


def test_dayplan_is_scored_against_canonical_not_arms_relaxed_parse():
    canonical = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {
                    "name": "grocery",
                    "duration_min": 60,
                    "mode": "active",
                    "latest_end": "10:00",
                    "required": True,
                }
            ],
        }
    )
    arm_spec = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {
                    "name": "Morning groceries",
                    "duration_min": 60,
                    "mode": "active",
                    "required": True,
                }
            ],
        }
    )
    arm_solution = materialize_day_plan_solution(
        arm_spec,
        [TaskAssignment(name="Morning groceries", start="10:00", end="11:00")],
    )

    result = evaluate_dayplan(canonical, arm_solution, arm_spec=arm_spec)

    assert result.valid_output is True
    assert result.required_completed is True
    assert result.canonical_validation_valid is False
    assert result.hard_constraints_satisfied < result.hard_constraints_total
    assert result.formulation_match is False
    assert result.optimal_objective == 0


def test_dayplan_formulation_match_allows_only_name_wording_difference():
    canonical = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {"name": "grocery", "duration_min": 60, "mode": "active"}
            ],
        }
    )
    renamed = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {
                    "name": "Morning groceries",
                    "duration_min": 60,
                    "mode": "active",
                }
            ],
        }
    )

    assert dayplan_formulation_matches(canonical, renamed) is True


def test_objective_and_gap_are_recomputed_under_canonical_preferences():
    canonical = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {"name": "focus", "duration_min": 60, "mode": "active"}
            ],
            "preferences": [
                {
                    "type": "finish_before",
                    "task": "focus",
                    "time": "10:00",
                    "weight": 2,
                }
            ],
        }
    )
    arm_spec = canonical.model_copy(update={"preferences": []})
    arm_solution = materialize_day_plan_solution(
        arm_spec,
        [TaskAssignment(name="focus", start="10:00", end="11:00")],
    )
    assert arm_solution.objective_value == 0

    result = evaluate_dayplan(canonical, arm_solution, arm_spec=arm_spec)

    assert result.canonical_validation_valid is True
    assert result.objective_value == 120
    assert result.optimal_objective == 0
    assert result.objective_gap == 120


def test_shift_evaluation_aligns_names_and_catches_canonical_unavailability():
    canonical = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "monday_front",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                }
            ],
            "employees": [
                {
                    "name": "Ava Smith",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                    "unavailable": [{"shift_id": "monday_front"}],
                },
                {
                    "name": "Ben",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                },
            ],
        }
    )
    arm_spec = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "Monday front shift",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                }
            ],
            "employees": [
                {
                    "name": "ava-smith",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                },
                {
                    "name": "Ben",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                },
            ],
        }
    )
    arm_solution = materialize_shift_schedule_solution(
        arm_spec,
        [ShiftAssignment(shift_id="Monday front shift", employee_name="ava-smith")],
    )

    result = evaluate_shift_schedule(canonical, arm_solution, arm_spec=arm_spec)

    assert result.valid_output is True
    assert result.required_completed is True
    assert result.canonical_validation_valid is False
    assert result.hard_constraints_satisfied < result.hard_constraints_total
    assert result.formulation_match is False


def test_shift_formulation_match_accepts_aligned_identifiers_and_references():
    canonical = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "monday_front",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                }
            ],
            "employees": [
                {
                    "name": "Ava Smith",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                    "preferred_shifts": ["monday_front"],
                }
            ],
            "rules": [
                {
                    "type": "required_days_off",
                    "employee_name": "Ava Smith",
                    "days": ["2026-10-06"],
                }
            ],
        }
    )
    renamed = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "Monday front shift",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                }
            ],
            "employees": [
                {
                    "name": "ava-smith",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                    "preferred_shifts": ["Monday front shift"],
                }
            ],
            "rules": [
                {
                    "type": "required_days_off",
                    "employee_name": "ava-smith",
                    "days": ["2026-10-06"],
                }
            ],
        }
    )

    assert shift_formulation_matches(canonical, renamed) is True


def test_unmatched_entities_count_missing_instead_of_crashing():
    canonical = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [{"name": "focus", "duration_min": 60, "mode": "active"}],
        }
    )
    foreign = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [{"name": "lunch", "duration_min": 60, "mode": "active"}],
        }
    )
    output = materialize_day_plan_solution(
        foreign, [TaskAssignment(name="lunch", start="09:00", end="10:00")]
    )

    result = evaluate_dayplan(canonical, output, arm_spec=foreign)

    assert result.valid_output is True
    assert result.required_completed is False
    assert result.alignment_errors == ["unmatched task assignment: lunch"]
    assert result.canonical_validation_valid is False


def test_feasible_infeasible_correctness_comes_from_canonical_cp_sat_solve():
    impossible = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "10:00"},
            "tasks": [
                {"name": "one", "duration_min": 60, "mode": "active"},
                {"name": "two", "duration_min": 60, "mode": "active"},
            ],
        }
    )
    reported = materialize_day_plan_solution(
        impossible, [], status=SolveStatus.INFEASIBLE
    )

    result = evaluate_dayplan(impossible, reported, arm_spec=impossible)

    assert result.feasible_correctly_reported is True
    assert result.canonical_validation_valid is True
    assert result.optimal_objective is None
    assert result.objective_gap is None
