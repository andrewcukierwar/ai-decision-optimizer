"""Presentation-only helpers for the Streamlit application.

The scheduling schemas intentionally keep their compact, machine-friendly
representations.  This module converts already-confirmed facts into readable
display values without changing solver inputs or serialized schema values.
"""

from datetime import date as datetime_date
from datetime import datetime, time as datetime_time
import re
from typing import Any, Dict, Iterable, List, Union

from .dayplan.schema import time_to_minutes

TimeValue = Union[str, datetime_time]


def format_entity_name(value: str) -> str:
    """Capitalize only the first character of a user-facing entity name."""

    return value[:1].upper() + value[1:]


def format_shift_location(value: str) -> str:
    """Render a Workforce location without changing its schema value."""

    return format_entity_name(value.replace("_", " "))


def format_shift_label(shift: Any, *, include_date: bool = False) -> str:
    """Build a readable shift label from structured fields, never from its id."""

    parts = [format_shift_location(shift.location)]
    if include_date:
        parts.append(format_date(shift.day))
    parts.append(format_time_range(shift.start, shift.end))
    return " · ".join(parts)


def format_shift_references(schedule: Any, shift_ids: Iterable[str]) -> str:
    """Render assigned/preferred shift references while preserving unknown ids."""

    shifts_by_id = {shift.id: shift for shift in schedule.shifts}
    return ", ".join(
        format_shift_label(shifts_by_id[shift_id], include_date=True)
        if shift_id in shifts_by_id
        else shift_id
        for shift_id in shift_ids
    )


def shift_preference_violation_count(preference_penalties: Iterable[Any]) -> int:
    """Count outside-preference assignments, not the magnitude of each amount."""

    return sum(1 for item in preference_penalties if item.amount > 0)


def shift_infeasibility_summary(diagnostic: Any) -> List[str]:
    """Return the separate Workforce infeasible-result presentation heading."""

    return ["No feasible staffing schedule", "Deterministic diagnosis completed"]


def shift_workload_explanation(employee_hours: Iterable[Any]) -> str:
    """Describe the measured minimum, maximum, and spread of assigned hours."""

    hours = [float(load.hours) for load in employee_hours]
    if not hours:
        return "No employee workloads were materialized."
    minimum = min(hours)
    maximum = max(hours)
    spread = maximum - minimum
    if spread == 0:
        return "Workloads are evenly distributed at %s per employee." % _hours_label(
            minimum
        )
    return (
        "Assigned hours range from %s to %s, a %s spread."
        % (
            _hours_label(minimum),
            _hours_label(maximum),
            _hours_label(spread, hyphenate=True),
        )
    )


def format_dayplan_text_names(text: str, names: Iterable[str]) -> str:
    """Format known Day Planner names inside already-rendered diagnostic text."""

    formatted = text
    for name in sorted({item for item in names if len(item) > 1}, key=len, reverse=True):
        formatted = re.sub(
            r"(?<!\w)" + re.escape(name) + r"(?!\w)",
            lambda _: format_entity_name(name),
            formatted,
        )
    return formatted


def dayplan_preference_summary(preference_penalties: Iterable[Any]) -> str:
    """Summarize encoded preferences by count, not by penalty magnitude."""

    penalties = list(preference_penalties)
    if not penalties:
        return "None specified"
    satisfied = sum(item.amount == 0 for item in penalties)
    return f"{satisfied} / {len(penalties)} satisfied"


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
                    "Activity": format_entity_name(assignment.name),
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
                    "Activity": format_entity_name(event.name),
                    "Type": "Fixed event",
                    "Duration": format_duration(duration),
                },
            )
        )

    # DayPlan uses local clock minutes, regardless of incidental tzinfo.
    entries.sort(key=lambda item: (
        time_to_minutes(item[0]), time_to_minutes(item[1]), item[2], item[3].lower()
    ))
    return [item[4] for item in entries]


def dayplan_preference_statements(facts: Any) -> List[str]:
    """Render DayPlan preference outcomes using only grounded penalty facts."""

    statements: List[str] = []
    for item in facts.preference_penalties:
        task_name = format_entity_name(item.task) if item.task else item.task
        if item.preference_type == "finish_before" and item.task and item.target_time:
            target = format_time(item.target_time)
            if item.amount == 0:
                statements.append(f"✓ {task_name} finished before {target}")
            else:
                statements.append(
                    f"△ {task_name} finished {item.amount} minutes after the preferred {target} target"
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
                    f"✓ {task_name} scheduled within the preferred {window} window"
                )
            else:
                statements.append(
                    f"△ {task_name} was scheduled {item.amount} minutes outside the preferred {window} window"
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

    if getattr(facts, "work_window_interrupting_tasks", None):
        lines.append(
            "Work was interrupted by "
            + _join_items(
                [format_entity_name(item) for item in facts.work_window_interrupting_tasks]
            )
            + "."
        )

    passive = [
        format_entity_name(item.name)
        for item in facts.assignments
        if item.mode == "passive"
    ]
    if passive:
        lines.append("Passive activities included: " + _join_items(passive) + ".")
    if satisfied:
        lines.append("Satisfied preferences: " + _join_items(satisfied) + ".")
    if not lines:
        lines.append("The confirmed activities were placed in the recommended schedule.")
    return lines[:4]


def dayplan_infeasibility_summary(diagnostic: Any) -> List[str]:
    """Build deterministic, user-facing summary lines for an infeasible DayPlan."""

    lines = ["No feasible schedule"]
    if diagnostic is None:
        return lines + ["No deterministic diagnosis was available."]

    lines.append("Deterministic diagnosis completed")
    if (
        diagnostic.required_active_minutes is not None
        and diagnostic.available_person_minutes is not None
        and diagnostic.capacity_shortfall_minutes
        and diagnostic.capacity_shortfall_minutes > 0
    ):
        lines.extend(
            [
                "Required active work: %d minutes" % diagnostic.required_active_minutes,
                "Available person-time after fixed events: %d minutes"
                % diagnostic.available_person_minutes,
                "Capacity shortfall: %d minutes" % diagnostic.capacity_shortfall_minutes,
            ]
        )
    return lines


def shift_preference_statements(schedule: Any, facts: Any) -> List[str]:
    """Render employee preference outcomes from the confirmed schedule and facts."""

    if facts.preference_penalties:
        shifts_by_id = {shift.id: shift for shift in schedule.shifts}
        statements = []
        for penalty in facts.preference_penalties:
            shift = shifts_by_id.get(penalty.shift_id)
            shift_detail = penalty.shift_id
            if shift is not None:
                shift_detail = format_shift_label(shift, include_date=True)
            statements.append(
                f"△ {penalty.employee_name} was assigned {shift_detail} outside their listed preferred shifts"
            )
        return statements

    if any(employee.preferred_shifts for employee in schedule.employees):
        return ["✓ No employees were assigned outside their listed preferred shifts"]
    return ["No employee shift preferences were encoded"]


def shift_explanation_lines(facts: Any) -> List[str]:
    """Build concise, user-facing explanation lines from grounded shift facts."""

    lines: List[str] = []
    if facts.preference_penalties:
        details = [
            f"{item.employee_name} was assigned {getattr(item, 'shift_label', None) or item.shift_id} outside their listed preferred shifts"
            for item in facts.preference_penalties
        ]
        lines.append("Tradeoff: " + _join_items(details) + ".")
    else:
        if getattr(facts, "has_preferred_shifts", False):
            lines.append(
                "No employees were assigned outside their listed preferred shifts."
            )
        else:
            lines.append("No employee shift preferences were encoded.")

    lines.append(shift_workload_explanation(facts.employee_hours))
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


def _hours_label(value: float, *, hyphenate: bool = False) -> str:
    number = f"{value:g}"
    unit = "hour" if value == 1 else "hours"
    if hyphenate:
        return f"{number}-hour"
    return f"{number} {unit}"


def _without_marker(value: str) -> str:
    return value[2:] if len(value) > 2 else value


def _join_items(items: List[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]
