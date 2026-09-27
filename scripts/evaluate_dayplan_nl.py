"""Opt-in live DayPlan natural-language evaluation.

Run with an OPENAI_API_KEY; this script is intentionally not collected by
pytest because every prompt makes a paid API request.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from decision_optimizer.dayplan import DayPlan, solve_day_plan, validate_solution
from decision_optimizer.parsing.dayplan import (
    DayPlanError,
    DayPlanExtraction,
    _create_openai_client,
    parse_dayplan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "dayplan_nl_eval.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for this run")
    parser.add_argument(
        "--show-actual",
        action="store_true",
        help="Print the parsed DayPlanExtraction JSON for formulation failures",
    )
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
            extraction = parse_dayplan(case["prompt"], client=client, model=args.model)
            if case.get("clarification") and extraction.plan is None:
                extraction = parse_dayplan(
                    case["prompt"],
                    previous_extraction=extraction,
                    clarification=case["clarification"],
                    client=client,
                    model=args.model,
                )
        except DayPlanError as exc:
            print("FAIL %-32s %s" % (case["name"], exc))
            continue

        differences = formulation_differences(extraction, case["expected"])
        if not differences:
            formulation_matches += 1
        else:
            print("MISS %s" % case["name"])
            for difference in differences:
                print("  - %s" % difference)
            if args.show_actual:
                print(json.dumps(extraction.model_dump(mode="json"), indent=2, sort_keys=True))

        if case["expected"].get("missing_info") is True:
            continue
        ready_cases += 1
        if extraction.plan is None:
            continue

        solution = solve_day_plan(extraction.plan)
        report = validate_solution(extraction.plan, solution)
        if solution.status.value in {"optimal", "feasible"}:
            solved_successfully += 1
        if report.valid:
            validation_passes += 1

    print(
        "%d/%d formulations matched expected key fields"
        % (formulation_matches, len(cases))
    )
    print(
        "%d/%d formulation-valid cases solved successfully"
        % (solved_successfully, ready_cases)
    )
    print(
        "%d/%d solver results passed independent validation"
        % (validation_passes, ready_cases)
    )


def matches_expected(extraction: DayPlanExtraction, expected: Dict[str, Any]) -> bool:
    """Return whether consequential expected fields match the extraction."""

    return not formulation_differences(extraction, expected)


def formulation_differences(
    extraction: DayPlanExtraction, expected: Dict[str, Any]
) -> List[str]:
    """Explain semantic differences without judging prose or style."""

    differences: List[str] = []
    expected_missing = expected.get("missing_info") is True

    if expected_missing:
        if extraction.plan is not None:
            differences.append("expected plan=null for missing-info case, actual complete plan")
        if not extraction.missing_info:
            differences.append("expected non-empty missing_info, actual empty")
        return differences

    if extraction.plan is None:
        differences.append("expected complete plan, actual plan=null")
        if extraction.missing_info:
            differences.append("actual missing_info: " + "; ".join(extraction.missing_info))
        return differences
    if extraction.missing_info:
        differences.append(
            "expected empty missing_info, actual: " + "; ".join(extraction.missing_info)
        )

    plan = extraction.plan
    if "horizon" in expected:
        _compare_window(differences, "horizon", plan.horizon, expected["horizon"])
    if "work_window" in expected:
        _compare_window(differences, "work window", plan.work_window, expected["work_window"])

    _compare_events(differences, plan.fixed_events, expected.get("fixed_events", {}))
    _compare_tasks(differences, plan.tasks, expected.get("tasks", {}))
    actual_task_names = [task.name for task in plan.tasks]
    for required_task in expected.get("required_tasks", []):
        if not any(_names_match(required_task, actual_name) for actual_name in actual_task_names):
            differences.append("missing required task: " + required_task)
    _compare_precedences(differences, plan, expected.get("precedences", []))
    _compare_preferences(differences, plan, expected.get("preferences", []))
    return differences


def _compare_window(
    differences: List[str], label: str, actual: Any, expected: Optional[Dict[str, str]]
) -> None:
    expected_value = "null" if expected is None else "%s-%s" % (
        expected["start"],
        expected["end"],
    )
    actual_value = _format_window(actual)
    if actual_value != expected_value:
        differences.append(
            "expected %s %s, actual %s" % (label, expected_value, actual_value)
        )


def _compare_events(
    differences: List[str], actual: Iterable[Any], expected: Dict[str, Dict[str, str]]
) -> None:
    matched, missing, extra = _match_named(expected.keys(), actual)
    for name in missing:
        differences.append("missing fixed event: " + name)
    for event in extra:
        differences.append("unexpected fixed event: " + event.name)
    for name, event in matched.items():
        expected_value = "%s-%s" % (expected[name]["start"], expected[name]["end"])
        actual_value = _format_window(event)
        if actual_value != expected_value:
            differences.append(
                "fixed event %s time expected %s, actual %s"
                % (name, expected_value, actual_value)
            )


def _compare_tasks(
    differences: List[str], actual: Iterable[Any], expected: Dict[str, Dict[str, Any]]
) -> None:
    matched, missing, extra = _match_named(expected.keys(), actual)
    for name in missing:
        differences.append("missing task: " + name)
    for task in extra:
        differences.append("extra task: " + task.name)
    for name, task in matched.items():
        for field, expected_value in expected[name].items():
            actual_value = _field_value(getattr(task, field))
            if actual_value != expected_value:
                label = {"duration_min": "duration", "latest_end": "deadline"}.get(
                    field, field
                )
                differences.append(
                    "task %s %s expected %s, actual %s"
                    % (name, label, _display(expected_value), _display(actual_value))
                )


def _compare_precedences(
    differences: List[str], plan: DayPlan, expected: Iterable[Any]
) -> None:
    actual = list(plan.precedences)
    used: set = set()
    for item in expected:
        before, after, min_gap, max_gap = item
        candidate_index = next(
            (
                index
                for index, actual_item in enumerate(actual)
                if index not in used
                and _names_match(before, actual_item.before)
                and _names_match(after, actual_item.after)
            ),
            None,
        )
        if candidate_index is None:
            differences.append("missing precedence %s -> %s" % (before, after))
            continue
        used.add(candidate_index)
        actual_item = actual[candidate_index]
        if actual_item.min_gap_min != min_gap:
            differences.append(
                "precedence %s -> %s min_gap expected %s, actual %s"
                % (before, after, min_gap, actual_item.min_gap_min)
            )
        if actual_item.max_gap_min != max_gap:
            differences.append(
                "precedence %s -> %s max_gap expected %s, actual %s"
                % (before, after, max_gap, actual_item.max_gap_min)
            )
    for index, item in enumerate(actual):
        if index not in used:
            differences.append("unexpected precedence %s -> %s" % (item.before, item.after))


def _compare_preferences(
    differences: List[str], plan: DayPlan, expected: Iterable[Any]
) -> None:
    actual = list(plan.preferences)
    used: set = set()
    for item in expected:
        preference_type = item[0]
        if preference_type == "preferred_window":
            task = item[1]
            start = item[2]
            end = item[3]
            candidate_index = next(
                (
                    index
                    for index, actual_item in enumerate(actual)
                    if index not in used
                    and actual_item.type.value == preference_type
                    and _optional_names_match(task, actual_item.task)
                    and _field_value(actual_item.start) == start
                    and _field_value(actual_item.end) == end
                ),
                None,
            )
        else:
            task = item[1] if len(item) > 1 else None
            time = item[2] if len(item) > 2 else None
            candidate_index = next(
                (
                    index
                    for index, actual_item in enumerate(actual)
                    if index not in used
                    and actual_item.type.value == preference_type
                    and _optional_names_match(task, actual_item.task)
                    and _field_value(actual_item.time) == time
                ),
                None,
            )
        if candidate_index is None:
            end_text = item[3] if preference_type == "preferred_window" else None
            differences.append(
                "missing preference: "
                + _preference_text(preference_type, task, item[2], end_text)
            )
            continue
        used.add(candidate_index)
        if preference_type == "preferred_window":
            expected_weight = item[4] if len(item) > 4 else None
        else:
            expected_weight = item[3] if len(item) > 3 else None
        if expected_weight is not None and actual[candidate_index].weight != expected_weight:
            differences.append(
                "preference %s weight expected %s, actual %s"
                % (preference_type, expected_weight, actual[candidate_index].weight)
            )
    for index, item in enumerate(actual):
        if index not in used:
            differences.append(
                "unexpected preference: "
                + _preference_text(
                    item.type.value,
                    item.task,
                    _field_value(item.time)
                    if item.type.value != "preferred_window"
                    else _field_value(item.start),
                    _field_value(item.end)
                    if item.type.value == "preferred_window"
                    else None,
                )
            )


def _match_named(
    expected_names: Iterable[str], actual_items: Iterable[Any]
) -> Tuple[Dict[str, Any], List[str], List[Any]]:
    remaining = list(actual_items)
    matched: Dict[str, Any] = {}
    missing: List[str] = []
    for expected_name in expected_names:
        exact = [item for item in remaining if _normal(item.name) == _normal(expected_name)]
        candidates = exact or [
            item for item in remaining if _names_match(expected_name, item.name)
        ]
        if len(candidates) == 1:
            matched[expected_name] = candidates[0]
            remaining.remove(candidates[0])
        else:
            missing.append(expected_name)
    return matched, missing, remaining


def _names_match(expected: Optional[str], actual: Optional[str]) -> bool:
    if expected is None or actual is None:
        return expected == actual
    expected_tokens = _name_tokens(expected)
    actual_tokens = _name_tokens(actual)
    return bool(expected_tokens and actual_tokens) and (
        expected_tokens == actual_tokens
        or expected_tokens.issubset(actual_tokens)
        or actual_tokens.issubset(expected_tokens)
    )


def _optional_names_match(expected: Optional[str], actual: Optional[str]) -> bool:
    return _names_match(expected, actual)


def _name_tokens(value: str) -> set:
    replacements = {
        "groceries": "grocery",
        "lifting": "lift",
        "meetings": "meeting",
        "appointments": "appointment",
        "emails": "email",
        "calls": "call",
    }
    return {
        replacements.get(token, token)
        for token in re.findall(r"[a-z0-9]+", value.lower())
    }


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _format_window(value: Any) -> str:
    if value is None:
        return "null"
    return "%s-%s" % (_field_value(value.start), _field_value(value.end))


def _field_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    return value


def _display(value: Any) -> str:
    value = _field_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _preference_text(
    preference_type: str,
    task: Optional[str],
    time: Optional[str],
    end: Optional[str] = None,
) -> str:
    parts = [preference_type]
    if task is not None:
        parts.append("task=" + task)
    if time is not None:
        parts.append("time=" + time)
    if end is not None:
        parts.append("end=" + end)
    return ", ".join(parts)


if __name__ == "__main__":
    main()
