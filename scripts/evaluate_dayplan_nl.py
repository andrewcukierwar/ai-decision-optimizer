"""Opt-in live DayPlan natural-language evaluation.

Run with an OPENAI_API_KEY; this script is intentionally not collected by
pytest because every prompt makes a paid API request.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from decision_optimizer.dayplan import DayPlan, solve_day_plan, validate_solution
from decision_optimizer.parsing.dayplan import (
    DayPlanError,
    _create_openai_client,
    parse_dayplan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "dayplan_nl_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for this run")
    args = parser.parse_args()

    cases = json.loads(args.fixture.read_text())
    try:
        client = _create_openai_client()
    except DayPlanError as exc:
        parser.error(str(exc))
    formulation_matches = 0
    ready_cases = 0
    solved_successfully = 0
    validation_passes = 0

    for case in cases:
        try:
            plan = parse_dayplan(case["prompt"], client=client, model=args.model)
        except DayPlanError as exc:
            print("FAIL %-32s %s" % (case["name"], exc))
            continue

        matches = matches_expected(plan, case["expected"])
        if matches:
            formulation_matches += 1
        else:
            print("MISS %-32s key fields differ" % case["name"])

        if case["expected"].get("missing_info") is True:
            continue
        ready_cases += 1
        if plan.missing_info or plan.horizon is None:
            continue

        solution = solve_day_plan(plan)
        report = validate_solution(plan, solution)
        if solution.status.value in {"optimal", "feasible"}:
            solved_successfully += 1
        if report.valid:
            validation_passes += 1

    print(
        "%d/%d formulations matched expected key fields" % (formulation_matches, len(cases))
    )
    print(
        "%d/%d formulation-valid cases solved successfully"
        % (solved_successfully, ready_cases)
    )
    print(
        "%d/%d solver results passed independent validation"
        % (validation_passes, ready_cases)
    )


def matches_expected(plan: DayPlan, expected: Dict[str, Any]) -> bool:
    if expected.get("missing_info") is True:
        return bool(plan.missing_info)
    if plan.missing_info:
        return False

    if "horizon" in expected and not _window_matches(plan.horizon, expected["horizon"]):
        return False
    if "work_window" in expected and not _window_matches(
        plan.work_window, expected["work_window"]
    ):
        return False
    if not _events_match(plan.fixed_events, expected.get("fixed_events", {})):
        return False
    if not _tasks_match(plan.tasks, expected.get("tasks", {})):
        return False
    if not _precedences_match(plan, expected.get("precedences", [])):
        return False
    return _preferences_match(plan, expected.get("preferences", []))


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _window_matches(actual: Any, expected: Optional[Dict[str, str]]) -> bool:
    if expected is None:
        return actual is None
    return (
        actual is not None
        and actual.start.strftime("%H:%M") == expected["start"]
        and actual.end.strftime("%H:%M") == expected["end"]
    )


def _events_match(actual: Iterable[Any], expected: Dict[str, Dict[str, str]]) -> bool:
    by_name = {_normal(item.name): item for item in actual}
    return set(by_name) == {_normal(name) for name in expected} and all(
        _normal(name) in by_name
        and _window_matches(by_name[_normal(name)], window)
        for name, window in expected.items()
    )


def _tasks_match(actual: Iterable[Any], expected: Dict[str, Dict[str, Any]]) -> bool:
    by_name = {_normal(item.name): item for item in actual}
    if set(by_name) != {_normal(name) for name in expected}:
        return False
    for name, fields in expected.items():
        task = by_name.get(_normal(name))
        if task is None:
            return False
        for field, value in fields.items():
            actual_value = getattr(task, field)
            if hasattr(actual_value, "value"):
                actual_value = actual_value.value
            elif hasattr(actual_value, "strftime"):
                actual_value = actual_value.strftime("%H:%M")
            if actual_value != value:
                return False
    return True


def _precedences_match(plan: DayPlan, expected: Iterable[Any]) -> bool:
    actual = {
        (
            item.before,
            item.after,
            item.min_gap_min,
            item.max_gap_min,
        )
        for item in plan.precedences
    }
    return actual == {tuple(item) for item in expected}


def _preferences_match(plan: DayPlan, expected: Iterable[Any]) -> bool:
    actual = {
        (
            item.type.value,
            item.task,
            item.time.strftime("%H:%M") if item.time else None,
        )
        for item in plan.preferences
    }
    return actual == {
        (item[0], item[1] if len(item) > 1 else None, item[2] if len(item) > 2 else None)
        for item in expected
    }


if __name__ == "__main__":
    main()
