"""Generate larger workforce fixtures from canonical typed specifications."""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any, Dict, List

from decision_optimizer import config as _config  # noqa: F401 - loads local .env
from decision_optimizer.shift_schedule import ShiftSchedule


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "generated_workforce_cases.json"
CASE_SIZES = (("regional_week", 8, 7), ("ten_day_rotation", 9, 10), ("two_week_rotation", 10, 14))
EMPLOYEE_NAMES = (
    "Avery",
    "Blake",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
)


def generate_cases() -> List[Dict[str, Any]]:
    """Build canonical specs first, then render requests from those specs."""

    cases = []
    for case_index, (name, employee_count, day_count) in enumerate(CASE_SIZES):
        schedule = _build_schedule(case_index, employee_count, day_count)
        cases.append(
            {
                "name": name,
                "prompt": render_request(schedule),
                "canonical": schedule.model_dump(mode="json"),
            }
        )
    return cases


def render_request(schedule: ShiftSchedule) -> str:
    """Render a complete natural-language request without inventing facts."""

    lines = [
        "Create a fair workforce schedule for these shifts. Cover every shift exactly at its stated staffing level.",
        "Shifts:",
    ]
    for shift in schedule.shifts:
        lines.append(
            "- %s on %s from %s to %s at %s needs %d staff."
            % (
                shift.id,
                shift.day.isoformat(),
                shift.start.strftime("%H:%M"),
                shift.end.strftime("%H:%M"),
                shift.location,
                shift.required_staff,
            )
        )
    lines.append("Employees:")
    for employee in schedule.employees:
        details = [
            "%s can work at most %d hours" % (employee.name, employee.max_hours),
            "is eligible at " + " and ".join(employee.eligible_locations),
        ]
        unavailable = [item.shift_id for item in employee.unavailable if item.shift_id]
        if unavailable:
            details.append("is unavailable for " + ", ".join(unavailable))
        if employee.preferred_shifts:
            details.append("prefers " + ", ".join(employee.preferred_shifts))
        lines.append("- " + "; ".join(details) + ".")
    for rule in schedule.rules:
        if rule.type == "minimum_rest":
            lines.append(
                "Require at least %d hours of rest between shifts."
                % rule.min_rest_hours
            )
        elif rule.type == "maximum_consecutive_days":
            lines.append(
                "No employee may work more than %d consecutive days."
                % rule.max_days
            )
        else:
            lines.append(
                "%s must have %s off."
                % (
                    rule.employee_name,
                    ", ".join(day.isoformat() for day in rule.days),
                )
            )
    lines.append(
        "Use preference penalty weight %d and fairness weight %d."
        % (
            schedule.objective_weights.preference_penalty,
            schedule.objective_weights.fairness,
        )
    )
    return "\n".join(lines)


def write_cases(path: Path = DEFAULT_OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(generate_cases(), indent=2) + "\n")
    return path


def _build_schedule(
    case_index: int, employee_count: int, day_count: int
) -> ShiftSchedule:
    start_day = date(2026, 10, 5) + timedelta(days=case_index * 21)
    shifts = []
    for day_offset in range(day_count):
        day = start_day + timedelta(days=day_offset)
        location = "north" if day_offset % 2 == 0 else "south"
        for period, start, end in (
            ("morning", "08:00", "12:00"),
            ("evening", "16:00", "20:00"),
        ):
            shifts.append(
                {
                    "id": "%s_%s" % (day.strftime("%Y%m%d"), period),
                    "day": day.isoformat(),
                    "start": start,
                    "end": end,
                    "location": location,
                    "required_staff": 2,
                }
            )

    shift_ids = [item["id"] for item in shifts]
    employees = []
    for employee_index, name in enumerate(EMPLOYEE_NAMES[:employee_count]):
        if employee_index % 5 == 0:
            locations = ["north"]
        elif employee_index % 5 == 1:
            locations = ["south"]
        else:
            locations = ["north", "south"]
        unavailable = [
            {"shift_id": shift_ids[(employee_index * 3 + case_index) % len(shift_ids)]},
            {"shift_id": shift_ids[(employee_index * 5 + 7) % len(shift_ids)]},
        ]
        unavailable = list({item["shift_id"]: item for item in unavailable}.values())
        unavailable_ids = {item["shift_id"] for item in unavailable}
        preferred = [
            shift_id
            for shift_number, shift_id in enumerate(shift_ids)
            if shift_number % employee_count in {
                employee_index,
                (employee_index + 2) % employee_count,
                (employee_index + 4) % employee_count,
            }
            and shift_id not in unavailable_ids
        ]
        employees.append(
            {
                "name": name,
                "max_hours": 32 if day_count == 7 else (40 if day_count == 10 else 48),
                "unavailable": unavailable,
                "eligible_locations": locations,
                "preferred_shifts": preferred,
            }
        )

    rules = [
        {"type": "minimum_rest", "min_rest_hours": 12},
        {"type": "maximum_consecutive_days", "max_days": 4},
        {
            "type": "required_days_off",
            "employee_name": EMPLOYEE_NAMES[2],
            "days": [(start_day + timedelta(days=2)).isoformat()],
        },
        {
            "type": "required_days_off",
            "employee_name": EMPLOYEE_NAMES[-1 if employee_count == 10 else employee_count - 1],
            "days": [(start_day + timedelta(days=day_count - 2)).isoformat()],
        },
    ]
    return ShiftSchedule.model_validate(
        {
            "shifts": shifts,
            "employees": employees,
            "rules": rules,
            "objective_weights": {"preference_penalty": 2, "fairness": 1},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(write_cases(args.output))


if __name__ == "__main__":
    main()
