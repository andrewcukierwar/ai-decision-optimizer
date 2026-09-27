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
