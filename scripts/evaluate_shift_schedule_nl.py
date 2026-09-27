"""Opt-in live ShiftSchedule natural-language evaluation.

Run this script with an OPENAI_API_KEY.  It is intentionally not collected by
pytest because every prompt makes a paid API request.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from decision_optimizer.parsing.shift_schedule import (
    ShiftScheduleError,
    _create_openai_client,
    parse_shift_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "shift_schedule_nl_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for this run")
    parser.add_argument(
        "--show-actual",
        action="store_true",
        help="Print the parsed extraction JSON for formulation failures",
    )
    args = parser.parse_args()

    cases = json.loads(args.fixture.read_text())
    try:
        client = _create_openai_client()
    except ShiftScheduleError as exc:
        parser.error(str(exc))

    matches = 0
    for case in cases:
        try:
            extraction = parse_shift_schedule(case["prompt"], client=client, model=args.model)
        except ShiftScheduleError as exc:
            print("FAIL %-32s %s" % (case["name"], exc))
            continue

        differences = formulation_differences(extraction, case["expected"])
        if not differences:
            matches += 1
            print("PASS " + case["name"])
        else:
            print("MISS " + case["name"])
            for difference in differences:
                print("  - " + difference)
            if args.show_actual:
                print(json.dumps(extraction.model_dump(mode="json"), indent=2, sort_keys=True))

    print("%d/%d ShiftSchedule formulations matched expected key fields" % (matches, len(cases)))


def formulation_differences(extraction: Any, expected: Dict[str, Any]) -> List[str]:
    differences: List[str] = []
    if expected.get("missing_info") is True:
        if extraction.schedule is not None:
            differences.append("expected schedule=null for missing-info case")
        if not extraction.missing_info:
            differences.append("expected non-empty missing_info")
        return differences

    if extraction.schedule is None:
        differences.append("expected complete schedule, actual schedule=null")
        return differences
    if extraction.missing_info:
        differences.append("expected empty missing_info")

    schedule = extraction.schedule
    shifts = {shift.id: shift for shift in schedule.shifts}
    employees = {employee.name: employee for employee in schedule.employees}
    for shift_id, expected_shift in expected.get("shifts", {}).items():
        actual = shifts.get(shift_id)
        if actual is None:
            differences.append("missing shift: " + shift_id)
            continue
        for field, expected_value in expected_shift.items():
            actual_value = getattr(actual, field)
            if hasattr(actual_value, "strftime"):
                actual_value = actual_value.strftime("%H:%M")
            elif hasattr(actual_value, "isoformat"):
                actual_value = actual_value.isoformat()
            if actual_value != expected_value:
                differences.append(
                    "shift %s %s expected %s, actual %s"
                    % (shift_id, field, expected_value, actual_value)
                )
    for name, expected_employee in expected.get("employees", {}).items():
        actual = employees.get(name)
        if actual is None:
            differences.append("missing employee: " + name)
            continue
        for field, expected_value in expected_employee.items():
            if field == "unavailable":
                if not any(
                    item.day and item.day.isoformat() == expected_value["day"]
                    and item.start and item.start.strftime("%H:%M") == expected_value["start"]
                    and item.end and item.end.strftime("%H:%M") == expected_value["end"]
                    for item in actual.unavailable
                ):
                    differences.append("employee %s unavailable window mismatch" % name)
            elif field == "unavailable_shift":
                if not any(item.shift_id == expected_value for item in actual.unavailable):
                    differences.append("employee %s unavailable shift mismatch" % name)
            elif getattr(actual, field) != expected_value:
                differences.append(
                    "employee %s %s expected %s, actual %s"
                    % (name, field, expected_value, getattr(actual, field))
                )

    actual_rules = []
    for rule in schedule.rules:
        if rule.type == "minimum_rest":
            actual_rules.append(["minimum_rest", rule.min_rest_hours])
        elif rule.type == "maximum_consecutive_days":
            actual_rules.append(["maximum_consecutive_days", rule.max_days])
    if sorted(actual_rules) != sorted(expected.get("rules", [])):
        differences.append("rule set mismatch")
    return differences


if __name__ == "__main__":
    main()
