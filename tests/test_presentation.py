import json
from datetime import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision_optimizer.dayplan import DayPlan, solve_day_plan, validate_solution
from decision_optimizer.diagnostics import _format_minutes
from decision_optimizer.explanations import build_dayplan_explanation_payload
from decision_optimizer.presentation import (
    build_dayplan_schedule_rows,
    dayplan_explanation_lines,
    dayplan_preference_statements,
    dayplan_preference_summary,
    format_dayplan_text_names,
    format_entity_name,
    format_shift_label,
    shift_infeasibility_summary,
    shift_preference_statements,
    shift_preference_violation_count,
    shift_workload_explanation,
    format_time,
    format_time_range,
)
from decision_optimizer.shift_schedule import ShiftSchedule


CASES = Path(__file__).parent / "cases"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("09:00", "9:00 AM"),
        (time(12, 0), "12:00 PM"),
        ("13:30", "1:30 PM"),
        ("00:00", "12:00 AM"),
    ],
)
def test_format_time_uses_us_12_hour_display(value, expected):
    assert format_time(value) == expected


def test_diagnostic_time_display_uses_the_same_12_hour_helper():
    assert _format_minutes(0) == "12:00 AM"
    assert _format_minutes(810) == "1:30 PM"
    assert _format_minutes(-30) == "30 minutes before the planning day"
    assert _format_minutes(1470) == "12:30 AM (+1 day)"


def test_diagnostic_name_formatting_treats_backslashes_as_literal_text():
    assert format_dayplan_text_names(
        r"relax \1 task", [r"\1 task"]
    ) == r"relax \1 task"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("breakfast", "Breakfast"),
        ("prepare for interview", "Prepare for interview"),
        ("pharmacy errand", "Pharmacy errand"),
        ("work out", "Work out"),
        ("take baked item out and clean up", "Take baked item out and clean up"),
        ("", ""),
    ],
)
def test_format_entity_name_uses_sentence_style_only(value, expected):
    assert format_entity_name(value) == expected


def test_shift_labels_use_structured_fields_instead_of_machine_ids():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "front_desk_morning", "day": "2026-10-05", "start": "09:00", "end": "13:00", "location": "front_desk", "required_staff": 1}
            ],
            "employees": [{"name": "Grace", "max_hours": 8, "eligible_locations": ["front_desk"], "preferred_shifts": ["front_desk_morning"]}],
        }
    )

    assert format_shift_label(schedule.shifts[0]) == "Front desk · 9:00 AM–1:00 PM"
    assert "front_desk_morning" not in format_shift_label(schedule.shifts[0])


def test_workforce_preference_language_describes_outside_assignments_only():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {"id": "preferred", "day": "2026-10-05", "start": "09:00", "end": "10:00", "location": "office", "required_staff": 1},
                {"id": "other", "day": "2026-10-05", "start": "10:00", "end": "11:00", "location": "office", "required_staff": 1},
            ],
            "employees": [{"name": "Grace", "max_hours": 8, "eligible_locations": ["office"], "preferred_shifts": ["preferred"]}],
        }
    )
    no_penalties = SimpleNamespace(preference_penalties=[])
    violated = SimpleNamespace(
        preference_penalties=[
            SimpleNamespace(employee_name="Grace", shift_id="other", amount=1)
        ]
    )

    assert shift_preference_statements(schedule, no_penalties) == [
        "✓ No employees were assigned outside their listed preferred shifts"
    ]
    assert shift_preference_statements(schedule, violated) == [
        "△ Grace was assigned Office · Oct 5, 2026 · 10:00–11:00 AM outside their listed preferred shifts"
    ]


@pytest.mark.parametrize(
    ("loads", "expected"),
    [
        ([SimpleNamespace(hours=15), SimpleNamespace(hours=15)], "Workloads are evenly distributed at 15 hours per employee."),
        ([SimpleNamespace(hours=4), SimpleNamespace(hours=8)], "Assigned hours range from 4 hours to 8 hours, a 4-hour spread."),
    ],
)
def test_workforce_fairness_explanation_uses_min_max_assigned_hours(loads, expected):
    assert shift_workload_explanation(loads) == expected


def test_workforce_preference_count_and_infeasible_heading_are_deterministic():
    assert shift_preference_violation_count(
        [SimpleNamespace(amount=1), SimpleNamespace(amount=3)]
    ) == 2
    assert shift_preference_violation_count([]) == 0
    assert shift_infeasibility_summary(None) == [
        "No feasible staffing schedule",
        "Deterministic diagnosis completed",
    ]


@pytest.mark.parametrize(
    ("penalties", "expected"),
    [
        ([SimpleNamespace(amount=0), SimpleNamespace(amount=0)], "2 / 2 satisfied"),
        (
            [SimpleNamespace(amount=0), SimpleNamespace(amount=2), SimpleNamespace(amount=0)],
            "2 / 3 satisfied",
        ),
        ([], "None specified"),
        ([SimpleNamespace(amount=3)], "0 / 1 satisfied"),
    ],
)
def test_dayplan_preference_summary_counts_preferences_not_penalty_minutes(
    penalties, expected
):
    assert dayplan_preference_summary(penalties) == expected


def test_schedule_rows_combine_tasks_and_fixed_events_in_time_order():
    plan = DayPlan.model_validate(
        json.loads((CASES / "passive_overlap.json").read_text())
    )
    solution = solve_day_plan(plan)

    rows = build_dayplan_schedule_rows(plan, solution.assignments)

    assert [row["Activity"] for row in rows] == ["Meeting", "Machine", "Call"]
    assert [row["Type"] for row in rows] == [
        "Fixed event",
        "Passive task",
        "Active task",
    ]
    assert rows[0]["Time"] == "9:00–10:00 AM"
    assert rows[2]["Time"] == "10:30–11:00 AM"


def test_schedule_rows_show_duplicate_fixed_event_display_names_individually():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "18:00"},
            "fixed_events": [
                {"name": "Meeting", "start": "10:00", "end": "10:30"},
                {"name": "Meeting", "start": "13:00", "end": "13:30"},
                {"name": "Meeting", "start": "16:00", "end": "16:30"},
            ],
            "tasks": [],
        }
    )

    rows = build_dayplan_schedule_rows(plan, [])

    assert [row["Activity"] for row in rows] == ["Meeting", "Meeting", "Meeting"]
    assert [row["Time"] for row in rows] == [
        "10:00–10:30 AM",
        "1:00–1:30 PM",
        "4:00–4:30 PM",
    ]


def test_preference_result_text_uses_grounded_penalty_facts_only():
    plan = DayPlan.model_validate(
        json.loads((CASES / "wfh_laundry_lift.json").read_text())
    )
    solution = solve_day_plan(plan)
    facts = build_dayplan_explanation_payload(plan, solution, validate_solution(plan, solution))

    assert facts is not None
    statements = dayplan_preference_statements(facts)

    assert "✓ Lift finished before 4:00 PM" in statements
    assert "△ 1 work-window interruption recorded" in statements
    assert all("12:00" not in statement for statement in statements)
    assert format_time_range("12:00", "13:00") == "12:00–1:00 PM"


def test_preferred_window_penalty_text_does_not_misstate_which_endpoint_missed():
    facts = SimpleNamespace(
        preference_penalties=[
            SimpleNamespace(
                preference_type="preferred_window",
                task="focus",
                preferred_start="10:00",
                preferred_end="11:00",
                amount=30,
                weighted_penalty=30,
            )
        ]
    )

    assert dayplan_preference_statements(facts) == [
        "△ Focus was scheduled 30 minutes outside the preferred 10:00–11:00 AM window"
    ]


def test_explanation_identifies_active_tasks_interrupting_work_window():
    plan = DayPlan.model_validate(
        {
            "horizon": {"start": "08:00", "end": "18:00"},
            "work_window": {"start": "09:00", "end": "17:00"},
            "tasks": [
                {
                    "name": "Grocery trip",
                    "duration_min": 30,
                    "mode": "active",
                    "earliest_start": "09:00",
                    "latest_end": "09:30",
                    "required": True,
                },
                {
                    "name": "Lunch",
                    "duration_min": 30,
                    "mode": "active",
                    "earliest_start": "10:00",
                    "latest_end": "10:30",
                    "required": True,
                },
                {
                    "name": "Workout",
                    "duration_min": 30,
                    "mode": "active",
                    "earliest_start": "11:00",
                    "latest_end": "11:30",
                    "required": True,
                },
                {
                    "name": "Machine cycle",
                    "duration_min": 60,
                    "mode": "passive",
                    "earliest_start": "09:00",
                    "latest_end": "10:00",
                    "required": True,
                },
            ],
            "preferences": [
                {"type": "minimize_work_interruptions", "weight": 1}
            ],
        }
    )
    solution = solve_day_plan(plan)
    facts = build_dayplan_explanation_payload(
        plan, solution, validate_solution(plan, solution)
    )

    assert facts is not None
    assert facts.work_window_interrupting_tasks == [
        "Grocery trip",
        "Lunch",
        "Workout",
    ]
    assert facts.preference_penalties[0].amount == 3
    assert "Work was interrupted by Grocery trip, Lunch, and Workout." in dayplan_explanation_lines(facts)
