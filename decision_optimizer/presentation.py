"""Presentation-only helpers for the Streamlit application.

The scheduling schemas intentionally keep their compact, machine-friendly
representations.  This module converts already-confirmed facts into readable
display values without changing solver inputs or serialized schema values.
"""

from datetime import date as datetime_date
from datetime import datetime, time as datetime_time
from typing import Any, Dict, Iterable, List, Union


TimeValue = Union[str, datetime_time]


def format_time(value: TimeValue) -> str:
    """Format a minute-precision ``HH:MM`` value as a US 12-hour time."""

    parsed = _coerce_time(value)
    hour = parsed.hour % 12 or 12
    period = "AM" if parsed.hour < 12 else "PM"
    return f"{hour}:{parsed.minute:02d} {period}"


def format_time_window(start: TimeValue, end: TimeValue) -> str:
    """Format a window with a full period marker on both endpoints."""

    return f"{format_time(start)} – {format_time(end)}"


def format_time_range(start: TimeValue, end: TimeValue) -> str:
    """Format a compact range, repeating AM/PM only when it changes."""

    start_time = _coerce_time(start)
    end_time = _coerce_time(end)
    start_display = format_time(start_time)
    end_display = format_time(end_time)
    start_period = "AM" if start_time.hour < 12 else "PM"
    end_period = "AM" if end_time.hour < 12 else "PM"
    if start_period == end_period:
        return f"{_clock(start_time)}–{_clock(end_time)} {end_period}"
    return f"{start_display}–{end_display}"


def format_date(value: datetime_date) -> str:
    """Format a date without a leading zero in the day number."""

    return f"{value.strftime('%b')} {value.day}, {value.year}"


def format_duration(minutes: int) -> str:
    """Format a duration in the compact form used by result tables."""

    return f"{minutes} min"


def format_minutes_duration(minutes: int) -> str:
    """Format a minute count as a readable workload spread or duration."""

    if minutes == 0:
        return "0 min"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hr" if hours == 1 else f"{hours} hrs"
    if minutes > 60:
        return f"{minutes / 60:.1f} hrs"
    return f"{minutes} min"


def build_dayplan_schedule_rows(plan: Any, assignments: Iterable[Any]) -> List[Dict[str, str]]:
    """Combine task assignments and fixed events into one chronological table."""

    task_by_name = {task.name: task for task in plan.tasks}
    entries = []
    for assignment in assignments:
        task = task_by_name[assignment.name]
        entries.append(
            (
                assignment.start,
                assignment.end,
                1,
                assignment.name,
                {
                    "Time": format_time_range(assignment.start, assignment.end),
                    "Activity": assignment.name,
                    "Type": f"{task.mode.value.capitalize()} task",
                    "Duration": format_duration(task.duration_min),
                },
            )
        )
    for event in plan.fixed_events:
        duration = _minutes_between(event.start, event.end)
        entries.append(
            (
                event.start,
                event.end,
                0,
                event.name,
                {
                    "Time": format_time_range(event.start, event.end),
                    "Activity": event.name,
                    "Type": "Fixed event",
                    "Duration": format_duration(duration),
                },
            )
        )

    entries.sort(key=lambda item: (item[0], item[1], item[2], item[3].lower()))
    return [item[4] for item in entries]


def dayplan_preference_statements(facts: Any) -> List[str]:
    """Render DayPlan preference outcomes using only grounded penalty facts."""

    statements: List[str] = []
    for item in facts.preference_penalties:
        if item.preference_type == "finish_before" and item.task and item.target_time:
            target = format_time(item.target_time)
            if item.amount == 0:
                statements.append(f"✓ {item.task} finished before {target}")
            else:
                statements.append(
                    f"△ {item.task} finished {item.amount} minutes after the preferred {target} target"
                )
        elif (
            item.preference_type == "preferred_window"
            and item.task
            and item.preferred_start
            and item.preferred_end
        ):
            window = format_time_range(item.preferred_start, item.preferred_end)
            if item.amount == 0:
                statements.append(
                    f"✓ {item.task} scheduled within the preferred {window} window"
                )
            else:
                statements.append(
                    f"△ {item.task} ended {item.amount} minutes outside the preferred {window} window"
                )
        elif item.preference_type == "minimize_work_interruptions":
            if item.amount == 0:
                statements.append("✓ No active tasks interrupted the work window")
            else:
                noun = "interruption" if item.amount == 1 else "interruptions"
                statements.append(f"△ {item.amount} work-window {noun} recorded")
        else:
            statements.append(
                f"△ {item.preference_type} incurred a weighted penalty of {item.weighted_penalty}"
            )
    return statements


def dayplan_explanation_lines(facts: Any) -> List[str]:
    """Build concise, user-facing explanation lines from grounded DayPlan facts."""

    statements = dayplan_preference_statements(facts)
    satisfied = [_without_marker(item) for item in statements if item.startswith("✓")]
    violated = [_without_marker(item) for item in statements if item.startswith("△")]
    lines: List[str] = []

    if violated:
        lines.append("Tradeoff: " + _join_items(violated) + ".")
    elif satisfied:
        lines.append("All encoded soft preferences were satisfied.")
    else:
        lines.append("No soft preferences were encoded for this plan.")

    passive = [item.name for item in facts.assignments if item.mode == "passive"]
    if passive:
        lines.append("Passive activities included: " + _join_items(passive) + ".")
    if satisfied:
        lines.append("Satisfied preferences: " + _join_items(satisfied) + ".")
    if not lines:
        lines.append("The confirmed activities were placed in the recommended schedule.")
    return lines[:4]


def shift_preference_statements(schedule: Any, facts: Any) -> List[str]:
    """Render employee preference outcomes from the confirmed schedule and facts."""

    if facts.preference_penalties:
        shifts_by_id = {shift.id: shift for shift in schedule.shifts}
        statements = []
        for penalty in facts.preference_penalties:
            shift = shifts_by_id.get(penalty.shift_id)
            shift_detail = penalty.shift_id
            if shift is not None:
                shift_detail = (
                    f"{penalty.shift_id} ({format_date(shift.day)}, "
                    f"{format_time_range(shift.start, shift.end)})"
                )
            statements.append(
                f"△ {penalty.employee_name} was assigned {shift_detail} outside their preferred shifts"
            )
        return statements

    if any(employee.preferred_shifts for employee in schedule.employees):
        return ["✓ All encoded employee shift preferences were satisfied"]
    return ["No employee shift preferences were encoded"]


def shift_explanation_lines(facts: Any) -> List[str]:
    """Build concise, user-facing explanation lines from grounded shift facts."""

    lines: List[str] = []
    if facts.preference_penalties:
        details = [
            f"{item.employee_name} was assigned {item.shift_id} outside their preferred shifts"
            for item in facts.preference_penalties
        ]
        lines.append("Tradeoff: " + _join_items(details) + ".")
    else:
        lines.append("No employee preference penalties were recorded.")

    spread = facts.objective_breakdown.fairness_minutes_spread
    if spread == 0:
        lines.append("Assigned workload is evenly distributed across employees.")
    else:
        lines.append(
            "The resulting workload spread is "
            + format_minutes_duration(spread)
            + "."
        )
    return lines[:3]


def _coerce_time(value: TimeValue) -> datetime_time:
    if isinstance(value, datetime_time):
        return value
    if isinstance(value, str):
        try:
            return datetime_time.fromisoformat(value)
        except ValueError:
            return datetime.strptime(value, "%H:%M").time()
    raise TypeError("time value must be datetime.time or HH:MM text")


def _clock(value: datetime_time) -> str:
    return f"{value.hour % 12 or 12}:{value.minute:02d}"


def _minutes_between(start: datetime_time, end: datetime_time) -> int:
    return (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)


def _without_marker(value: str) -> str:
    return value[2:] if len(value) > 2 else value


def _join_items(items: List[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]
