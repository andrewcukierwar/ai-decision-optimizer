import json
from datetime import time
from pathlib import Path

import pytest

from decision_optimizer.dayplan import DayPlan, solve_day_plan, validate_solution
from decision_optimizer.diagnostics import _format_minutes
from decision_optimizer.explanations import build_dayplan_explanation_payload
from decision_optimizer.presentation import (
    build_dayplan_schedule_rows,
    dayplan_preference_statements,
    dayplan_explanation_lines,
    format_time,
    format_time_range,
)


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


def test_schedule_rows_combine_tasks_and_fixed_events_in_time_order():
    plan = DayPlan.model_validate(
        json.loads((CASES / "passive_overlap.json").read_text())
    )
    solution = solve_day_plan(plan)

    rows = build_dayplan_schedule_rows(plan, solution.assignments)

    assert [row["Activity"] for row in rows] == ["meeting", "machine", "call"]
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

    assert "✓ lift finished before 4:00 PM" in statements
    assert "△ 1 work-window interruption recorded" in statements
    assert all("12:00" not in statement for statement in statements)
    assert format_time_range("12:00", "13:00") == "12:00–1:00 PM"


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
