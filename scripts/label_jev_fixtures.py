"""Create a reviewable *draft* Jev answer key for selected benchmark cases.

This helper never declares its proposed labels to be ground truth.  Its output
is marked ``draft_needs_human_review`` at both the file and decision level.
Use ``--baseline gpt`` to perform one OpenAI extraction per selected fixture;
the default ``fixture`` mode is deterministic and uses checked-in canonical or
expected formulations so the review workflow can be exercised without paid
calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from decision_optimizer import config as _config  # noqa: F401 - loads .env
from decision_optimizer.dayplan import DayPlan
from decision_optimizer.jev import build_jev_questions
from decision_optimizer.parsing.dayplan import parse_dayplan
from decision_optimizer.parsing.shift_schedule import parse_shift_schedule
from decision_optimizer.shift_schedule import ShiftSchedule


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
DEFAULT_SELECTION = FIXTURES / "jev_benchmark_selection.json"
DEFAULT_OUTPUT = FIXTURES / "jev_expected_draft.json"


def load_selected_cases(path: Path = DEFAULT_SELECTION) -> List[Dict[str, Any]]:
    """Resolve the selection manifest against its source fixture files."""

    manifest = json.loads(path.read_text())
    loaded_sources: Dict[str, List[Dict[str, Any]]] = {}
    selected: List[Dict[str, Any]] = []
    for reference in manifest["cases"]:
        source_name = reference["source"]
        if source_name not in loaded_sources:
            loaded_sources[source_name] = json.loads((path.parent / source_name).read_text())
        source_case = next(
            (item for item in loaded_sources[source_name] if item["name"] == reference["name"]),
            None,
        )
        if source_case is None:
            raise ValueError("Missing selected case %s in %s" % (reference["name"], source_name))
        selected.append({**source_case, **reference})
    return selected


def build_draft_answer_key(
    cases: Iterable[Dict[str, Any]],
    *,
    baseline: str = "fixture",
    model: Optional[str] = None,
    dayplan_extractor: Optional[Callable[[Dict[str, Any]], DayPlan]] = None,
    shift_extractor: Optional[Callable[[Dict[str, Any]], ShiftSchedule]] = None,
) -> Dict[str, Any]:
    """Derive applicable questions and prefill explicitly unreviewed labels."""

    if baseline not in {"fixture", "gpt"}:
        raise ValueError("baseline must be fixture or gpt")
    output_cases = []
    for case in cases:
        domain = case["domain"]
        if domain == "dayplan":
            problem = (
                dayplan_extractor(case)
                if dayplan_extractor is not None
                else _dayplan_baseline(case, baseline=baseline, model=model)
            )
        else:
            problem = (
                shift_extractor(case)
                if shift_extractor is not None
                else _shift_baseline(case, baseline=baseline, model=model)
            )
        questions = build_jev_questions(case["prompt"], problem)
        proposed = []
        for question in questions:
            gpt_value = question.question_type.gpt_value(problem, question.target)
            proposed.append(
                {
                    "question_type": question.question_type.key,
                    "item": question.target.item,
                    "primitive": question.question_type.primitive,
                    "gpt_value": gpt_value,
                    "proposed_expected_jev": gpt_value,
                    "review_status": "needs_human_review",
                }
            )
        output_cases.append(
            {
                "name": case["name"],
                "domain": domain,
                "source": case["source"],
                "prompt": case["prompt"],
                "baseline_source": "gpt_extraction" if baseline == "gpt" else "fixture_formulation",
                "expected_jev_review_status": "needs_human_review",
                "expected_jev": proposed,
            }
        )
    return {
        "schema_version": 1,
        "label_status": "draft_needs_human_review",
        "warning": (
            "These are proposed labels, not human ground truth. A human must review every "
            "expected_jev entry before benchmark calibration or accuracy analysis."
        ),
        "baseline": baseline,
        "cases": output_cases,
    }


def _dayplan_baseline(case: Dict[str, Any], *, baseline: str, model: Optional[str]) -> DayPlan:
    if baseline == "gpt":
        extraction = parse_dayplan(case["prompt"], model=model)
        if case.get("clarification") and extraction.plan is None:
            extraction = parse_dayplan(
                case["prompt"],
                previous_extraction=extraction,
                clarification=case["clarification"],
                model=model,
            )
        if extraction.plan is None:
            raise ValueError("GPT baseline remained incomplete for " + case["name"])
        return extraction.plan
    if "canonical" in case:
        return DayPlan.model_validate(case["canonical"])
    return DayPlan.model_validate(_dayplan_expected_to_schema(case["expected"]))


def _shift_baseline(case: Dict[str, Any], *, baseline: str, model: Optional[str]) -> ShiftSchedule:
    if baseline == "gpt":
        extraction = parse_shift_schedule(case["prompt"], model=model)
        if extraction.schedule is None:
            raise ValueError("GPT baseline remained incomplete for " + case["name"])
        return extraction.schedule
    if "canonical" in case:
        return ShiftSchedule.model_validate(case["canonical"])
    return ShiftSchedule.model_validate(_shift_expected_to_schema(case["expected"]))


def _dayplan_expected_to_schema(expected: Dict[str, Any]) -> Dict[str, Any]:
    events = expected.get("fixed_events", {})
    if isinstance(events, dict):
        events = [{"name": name, **window} for name, window in events.items()]
    tasks = [
        {"name": name, **fields}
        for name, fields in expected.get("tasks", {}).items()
    ]
    precedences = [
        {"before": item[0], "after": item[1], "min_gap_min": item[2], "max_gap_min": item[3]}
        for item in expected.get("precedences", [])
    ]
    preferences = []
    for item in expected.get("preferences", []):
        if item[0] == "finish_before":
            preferences.append(
                {"type": item[0], "task": item[1], "time": item[2], "weight": item[3] if len(item) > 3 else 1}
            )
        elif item[0] == "preferred_window":
            preferences.append(
                {
                    "type": item[0], "task": item[1], "start": item[2], "end": item[3],
                    "weight": item[4] if len(item) > 4 else 1,
                }
            )
        else:
            preferences.append({"type": item[0], "weight": item[3] if len(item) > 3 else 1})
    return {
        "horizon": expected["horizon"],
        "work_window": expected.get("work_window"),
        "fixed_events": events,
        "tasks": tasks,
        "precedences": precedences,
        "preferences": preferences,
    }


def _shift_expected_to_schema(expected: Dict[str, Any]) -> Dict[str, Any]:
    shifts = [{"id": shift_id, **fields} for shift_id, fields in expected["shifts"].items()]
    employees = []
    for name, fields in expected["employees"].items():
        employee = {"name": name, **fields}
        unavailable = []
        if "unavailable" in employee:
            unavailable.append(employee.pop("unavailable"))
        if "unavailable_shift" in employee:
            unavailable.append({"shift_id": employee.pop("unavailable_shift")})
        employee["unavailable"] = unavailable
        employees.append(employee)
    rules = []
    for item in expected.get("rules", []):
        if item[0] == "minimum_rest":
            rules.append({"type": item[0], "min_rest_hours": item[1]})
        elif item[0] == "maximum_consecutive_days":
            rules.append({"type": item[0], "max_days": item[1]})
    return {
        "shifts": shifts,
        "employees": employees,
        "rules": rules,
        "objective_weights": expected.get("objective_weights", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", choices=("fixture", "gpt"), default="fixture")
    parser.add_argument("--model", default=None, help="OpenAI model override for --baseline gpt")
    args = parser.parse_args()

    draft = build_draft_answer_key(
        load_selected_cases(args.selection), baseline=args.baseline, model=args.model
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(draft, indent=2) + "\n")
    question_count = sum(len(case["expected_jev"]) for case in draft["cases"])
    print(
        "Wrote %d cases and %d proposed labels to %s; every label still needs human review."
        % (len(draft["cases"]), question_count, args.output)
    )


if __name__ == "__main__":
    main()
