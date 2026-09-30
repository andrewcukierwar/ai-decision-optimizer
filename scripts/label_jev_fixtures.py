"""Build a reviewable draft Jev answer key without conflating baselines.

The canonical proposed answer is derived from the fixture's intended typed
meaning. GPT and Jev values are optional observations from actual calls and
are never copied into the canonical column. Generated workforce labels are
marked generator-derived, not human-reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from decision_optimizer import config as _config  # noqa: F401 - loads .env
from decision_optimizer.config import openai_model
from decision_optimizer.dayplan import DayPlan
from decision_optimizer.evaluation import align_names
from decision_optimizer.jev_alignment import indexed_questions as _indexed_questions
from decision_optimizer.jev import JevQuestion, apply_jev, build_jev_questions
from decision_optimizer.parsing.dayplan import parse_dayplan
from decision_optimizer.parsing.shift_schedule import parse_shift_schedule
from decision_optimizer.shift_schedule import ShiftSchedule


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
DEFAULT_SELECTION = FIXTURES / "jev_benchmark_selection.json"
DEFAULT_OUTPUT = FIXTURES / "jev_expected_draft.json"
DEFAULT_REVIEW_OUTPUT = FIXTURES / "jev_expected_review.tsv"


def load_selected_cases(path: Path = DEFAULT_SELECTION) -> List[Dict[str, Any]]:
    """Resolve the selection manifest against its source fixture files."""

    manifest = json.loads(path.read_text())
    loaded_sources: Dict[str, List[Dict[str, Any]]] = {}
    selected: List[Dict[str, Any]] = []
    for reference in manifest["cases"]:
        source_name = reference["source"]
        if source_name not in loaded_sources:
            loaded_sources[source_name] = json.loads(
                (path.parent / source_name).read_text()
            )
        source_case = next(
            (
                item
                for item in loaded_sources[source_name]
                if item["name"] == reference["name"]
            ),
            None,
        )
        if source_case is None:
            raise ValueError(
                "Missing selected case %s in %s"
                % (reference["name"], source_name)
            )
        selected.append({**source_case, **reference, "prompt": reference.get("benchmark_prompt_override", source_case["prompt"])})
    return selected


def canonical_problem(case: Mapping[str, Any]) -> DayPlan | ShiftSchedule:
    """Materialize the independent fixture-intent formulation."""

    if case["domain"] == "dayplan":
        if "canonical" in case:
            return DayPlan.model_validate(case["canonical"])
        return DayPlan.model_validate(_dayplan_expected_to_schema(case["expected"]))
    if "canonical" in case:
        return ShiftSchedule.model_validate(case["canonical"])
    return ShiftSchedule.model_validate(_shift_expected_to_schema(case["expected"]))


def build_draft_answer_key(
    cases: Iterable[Dict[str, Any]],
    *,
    baseline: str = "fixture",
    model: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    include_jev: bool = False,
    dayplan_extractor: Optional[Callable[[Dict[str, Any]], DayPlan]] = None,
    shift_extractor: Optional[Callable[[Dict[str, Any]], ShiftSchedule]] = None,
    jev_client: Any = None,
) -> Dict[str, Any]:
    """Create canonical proposals and optionally attach observed model answers.

    ``baseline='fixture'`` performs no paid calls. ``baseline='gpt'`` is an independent development sample, not benchmark evidence, and observes
    each requested model's extraction; ``include_jev`` additionally records
    Jev's answers. Missing and unaligned questions remain explicit in coverage.
    """

    if baseline not in {"fixture", "gpt"}:
        raise ValueError("baseline must be fixture or gpt")
    observed_models = list(models or ([model] if model else []))
    if baseline == "gpt" and not observed_models:
        observed_models = [openai_model()]
    if baseline == "fixture":
        observed_models = []

    output_cases = []
    for case in cases:
        canonical = canonical_problem(case)
        canonical_questions = build_jev_questions(case["prompt"], canonical)
        indexed_canonical = _indexed_questions(
            case["domain"], canonical_questions, canonical, canonical
        )
        labels = []
        for question_id, question in indexed_canonical:
            generated = bool(case.get("generated"))
            labels.append(
                {
                    "question_id": question_id,
                    "question_type": question.question_type.key,
                    "canonical_item": question.target.item,
                    "primitive": question.question_type.primitive,
                    "canonical_expected_answer": question.question_type.gpt_value(
                        canonical, question.target
                    ),
                    "supporting_request_context": _supporting_context(
                        case["prompt"], question
                    ),
                    "label_provenance": (
                        "generator_specification"
                        if generated
                        else "fixture_intended_meaning"
                    ),
                    "review_status": (
                        "generator_derived_not_human_reviewed"
                        if generated
                        else "needs_human_review"
                    ),
                    "review_reason": _review_reason(case, question),
                    "primary_accuracy_calibration_eligible": not _excluded_subjective_label(case, question),
                    "primary_exclusion_reason": "subjective_weight_without_agreed_rubric" if _excluded_subjective_label(case, question) else None,
                    "observations": {},
                }
            )

        label_by_id = {label["question_id"]: label for label in labels}
        coverage: Dict[str, Any] = {}
        unaligned: Dict[str, List[Dict[str, Any]]] = {}
        for observed_model in observed_models:
            extracted = _extract_problem(
                case,
                observed_model,
                dayplan_extractor=dayplan_extractor,
                shift_extractor=shift_extractor,
            )
            if extracted is None:
                for label in labels:
                    label["observations"][observed_model] = {
                        "alignment_status": "not_generated",
                        "gpt_baseline_answer": None,
                        "jev_answer": None,
                    }
                coverage[observed_model] = {
                    "canonical_questions": len(labels),
                    "aligned_questions": 0,
                    "not_generated": len(labels),
                    "unaligned_generated": 0,
                }
                unaligned[observed_model] = []
                continue

            generated_questions = build_jev_questions(case["prompt"], extracted)
            indexed_generated = _indexed_questions(
                case["domain"], generated_questions, extracted, canonical
            )
            jev_by_item: Dict[tuple[str, str], Any] = {}
            if include_jev:
                jev_result = apply_jev(
                    case["prompt"], extracted, client=jev_client
                )
                jev_by_item = {
                    (decision.question_type, decision.item): decision
                    for decision in jev_result.decisions
                }

            matched = set()
            model_unaligned = []
            for question_id, question in indexed_generated:
                gpt_answer = question.question_type.gpt_value(
                    extracted, question.target
                )
                decision = jev_by_item.get(
                    (question.question_type.key, question.target.item)
                )
                observation = {
                    "alignment_status": "aligned",
                    "generated_item": question.target.item,
                    "gpt_baseline_answer": gpt_answer,
                    "jev_answer": None if decision is None else decision.jev_value,
                    "jev_probability": None
                    if decision is None
                    else decision.confidence,
                }
                if question_id in label_by_id and question_id not in matched:
                    label_by_id[question_id]["observations"][observed_model] = observation
                    matched.add(question_id)
                else:
                    observation["alignment_status"] = "unaligned_generated"
                    model_unaligned.append(
                        {
                            "semantic_candidate_id": question_id,
                            "question_type": question.question_type.key,
                            "generated_item": question.target.item,
                            "gpt_baseline_answer": gpt_answer,
                            "jev_answer": observation["jev_answer"],
                        }
                    )
            for label in labels:
                if observed_model not in label["observations"]:
                    label["observations"][observed_model] = {
                        "alignment_status": "not_generated",
                        "gpt_baseline_answer": None,
                        "jev_answer": None,
                    }
            coverage[observed_model] = {
                "canonical_questions": len(labels),
                "aligned_questions": len(matched),
                "not_generated": len(labels) - len(matched),
                "unaligned_generated": len(model_unaligned),
            }
            unaligned[observed_model] = model_unaligned

        output_cases.append(
            {
                "name": case["name"],
                "domain": case["domain"],
                "source": case["source"],
                "generated": bool(case.get("generated")),
                "prompt": case["prompt"],
                "canonical_source": (
                    "generator_specification"
                    if case.get("generated")
                    else "fixture_intended_meaning"
                ),
                "expected_jev_review_status": (
                    "generator_derived_not_human_reviewed"
                    if case.get("generated")
                    else "needs_human_review"
                ),
                "expected_jev": labels,
                "observation_coverage": coverage,
                "unaligned_observed_questions": unaligned,
            }
        )
    return {
        "schema_version": 2,
        "label_status": "draft_not_human_approved",
        "warning": (
            "Canonical answers are proposed from fixture intent. Generator-derived "
            "labels are deterministic but not human-reviewed; all other labels require "
            "human review before calibration or accuracy claims. GPT and Jev fields are "
            "observations only and never define canonical truth."
        ),
        "methodology": {
            "canonical_expected_answer": "Independent fixture intent or generator specification.",
            "gpt_baseline_answer": "Observed from the named model's actual extraction.",
            "jev_answer": "Observed from TypeSafe for that extracted formulation.",
            "alignment": (
                "Semantic identities align task/employee/shift names to canonical names "
                "and normalize equivalent hard/soft timing representations. Missing and "
                "unaligned questions are retained in coverage rather than dropped."
            ),
        },
        "observation_source": "independent_development_sample" if observed_models else "none",
        "observed_models": observed_models,
        "jev_observed": include_jev,
        "cases": output_cases,
    }


def write_review_tsv(draft: Mapping[str, Any], path: Path) -> Path:
    """Write the compact human-review queue in case/question-type order."""

    fieldnames = [
        "case",
        "domain",
        "question_type",
        "question_id",
        "canonical_item",
        "canonical_expected_answer",
        "supporting_request_context",
        "label_provenance",
        "review_status",
        "review_reason",
        "primary_accuracy_calibration_eligible",
        "primary_exclusion_reason",
        "model_observations",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        for case in draft["cases"]:
            for label in case["expected_jev"]:
                writer.writerow(
                    {
                        "case": case["name"],
                        "domain": case["domain"],
                        "question_type": label["question_type"],
                        "question_id": label["question_id"],
                        "canonical_item": label["canonical_item"],
                        "canonical_expected_answer": json.dumps(
                            label["canonical_expected_answer"]
                        ),
                        "supporting_request_context": label[
                            "supporting_request_context"
                        ],
                        "label_provenance": label["label_provenance"],
                        "review_status": label["review_status"],
                        "review_reason": label["review_reason"],
                        "primary_accuracy_calibration_eligible": label["primary_accuracy_calibration_eligible"],
                        "primary_exclusion_reason": label["primary_exclusion_reason"],
                        "model_observations": json.dumps(
                            label["observations"], sort_keys=True
                        ),
                    }
                )
    return path


def _extract_problem(
    case: Dict[str, Any],
    model: str,
    *,
    dayplan_extractor: Optional[Callable[[Dict[str, Any]], DayPlan]],
    shift_extractor: Optional[Callable[[Dict[str, Any]], ShiftSchedule]],
) -> Optional[DayPlan | ShiftSchedule]:
    if case["domain"] == "dayplan":
        if dayplan_extractor is not None:
            return dayplan_extractor(case)
        extraction = parse_dayplan(case["prompt"], model=model)
        if case.get("clarification") and extraction.plan is None:
            extraction = parse_dayplan(
                case["prompt"],
                previous_extraction=extraction,
                clarification=case["clarification"],
                model=model,
            )
        return extraction.plan
    if shift_extractor is not None:
        return shift_extractor(case)
    return parse_shift_schedule(case["prompt"], model=model).schedule


def _supporting_context(prompt: str, question: JevQuestion) -> str:
    names = [
        str(value)
        for key, value in question.target.locator.items()
        if key in {"task", "employee", "shift"} and value
    ]
    if question.target.locator.get("signature", {}).get("task"):
        names.append(str(question.target.locator["signature"]["task"]))
    fragments = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", prompt)
        if part.strip()
    ]
    if question.question_type.key == "availability_applies":
        employee, shift = str(question.target.locator["employee"]), str(question.target.locator["shift"])
        employee_facts = [f for f in fragments if re.search(r"\b" + re.escape(employee) + r"\b", f, re.I)]
        shift_facts = [f for f in fragments if f.lstrip("- ").startswith(shift + " on ")]
        return " ".join((employee_facts[:1] + shift_facts[:1])) or prompt[:240]
    matches = [
        fragment
        for fragment in fragments
        if any(name.lower() in fragment.lower() for name in names)
    ]
    if not matches and question.question_type.key == "pref_weight":
        matches = [
            fragment
            for fragment in fragments
            if re.search(r"\b(prefer|priority|weight|fair)\w*\b", fragment, re.I)
        ]
    return " ".join(matches[:2]) or prompt[:240]


def _excluded_subjective_label(case: Mapping[str, Any], question: JevQuestion) -> bool:
    if question.question_type.key != "pref_weight":
        return False
    if case["name"] in {"workout_before_four_soft", "soft_finish_preference"}:
        return True
    return case["name"] == "preferences_and_availability" and question.target.locator.get("field") == "preference_penalty"


def _review_reason(case: Mapping[str, Any], question: JevQuestion) -> str:
    if case.get("generated"):
        return "unambiguous_generator_template"
    if question.question_type.key == "pref_weight" and not re.search(
        r"\b(?:priority|weight)\s*(?:is\s*)?[1-5]\b", case["prompt"], re.I
    ):
        return "subjective_preference_strength"
    if case["source"] == "dayplan_jev_ambiguous.json":
        return "intentionally_ambiguous_language"
    return "fixture_proposal_requires_human_confirmation"


def _dayplan_expected_to_schema(expected: Dict[str, Any]) -> Dict[str, Any]:
    events = expected.get("fixed_events", {})
    if isinstance(events, dict):
        events = [{"name": name, **window} for name, window in events.items()]
    tasks = [
        {"name": name, **fields} for name, fields in expected.get("tasks", {}).items()
    ]
    precedences = [
        {
            "before": item[0],
            "after": item[1],
            "min_gap_min": item[2],
            "max_gap_min": item[3],
        }
        for item in expected.get("precedences", [])
    ]
    preferences = []
    for item in expected.get("preferences", []):
        if item[0] == "finish_before":
            preferences.append(
                {
                    "type": item[0],
                    "task": item[1],
                    "time": item[2],
                    "weight": item[3] if len(item) > 3 else 1,
                }
            )
        elif item[0] == "preferred_window":
            preferences.append(
                {
                    "type": item[0],
                    "task": item[1],
                    "start": item[2],
                    "end": item[3],
                    "weight": item[4] if len(item) > 4 else 1,
                }
            )
        else:
            preferences.append(
                {"type": item[0], "weight": item[3] if len(item) > 3 else 1}
            )
    return {
        "horizon": expected["horizon"],
        "work_window": expected.get("work_window"),
        "fixed_events": events,
        "tasks": tasks,
        "precedences": precedences,
        "preferences": preferences,
    }


def _shift_expected_to_schema(expected: Dict[str, Any]) -> Dict[str, Any]:
    shifts = [
        {"id": shift_id, **fields} for shift_id, fields in expected["shifts"].items()
    ]
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


def attach_benchmark_observations(
    draft: Dict[str, Any], cases: Sequence[Dict[str, Any]], results_dir: Path,
    models: Optional[Sequence[str]] = None,
) -> None:
    """Attach the original experimental observations from verified stage caches.

    No GPT or Jev calls, and no second extraction sample. Missing caches and
    failed stages are reported separately from model omissions.
    """
    import hashlib
    from decision_optimizer.benchmark import (
        BenchmarkCase, _extraction_fingerprint, _jev_fingerprint,
        _problem_from_json, _problem_hash, _read_cache, question_observations,
    )
    from decision_optimizer.jev import JevDecision
    selected_models = list(models or ("gpt-6-luna", "gpt-6.1-sol"))
    by_name = {case["name"]: case for case in cases}
    any_jev = False
    for output_case in draft["cases"]:
        raw = by_name[output_case["name"]]
        case = BenchmarkCase(raw["name"], raw["domain"], raw["prompt"], canonical_problem(raw), raw["source"], raw.get("clarification"))
        slug = hashlib.sha256(case.case_id.encode()).hexdigest()[:12]
        label_by_id = {label["question_id"]: label for label in output_case["expected_jev"]}
        for model in selected_models:
            path = results_dir / "cache" / ("%s_%s_extraction.json" % (slug, model))
            if not path.exists():
                output_case["observation_coverage"][model] = {"observation_source": "benchmark_cache", "cache_status": "missing_extraction_cache"}
                for label in label_by_id.values():
                    label["observations"][model] = {"alignment_status": "not_observed", "gpt_baseline_answer": None, "jev_answer": None}
                continue
            raw_cache = json.loads(path.read_text())
            fingerprint = _extraction_fingerprint(case, model, raw_cache.get("runtime_configuration"))
            cache = _read_cache(path, fingerprint)
            if cache is None:
                raise ValueError("Incompatible extraction cache: " + str(path))
            extracted = _problem_from_json(case.domain, cache.get("problem"))
            if "terminal_error" not in cache and cache.get("problem_hash") != _problem_hash(extracted):
                raise ValueError("Extraction cache problem hash mismatch: " + str(path))
            decisions = []
            jev_status = "not_observed"
            jev_fingerprint = None
            jev_path = path.with_name("%s_%s_jev.json" % (slug, model))
            if extracted is not None and jev_path.exists():
                raw_jev = json.loads(jev_path.read_text())
                jev_fingerprint = _jev_fingerprint(fingerprint, raw_jev["jev_batch_size"], raw_jev["jev_apply_threshold"], raw_jev["requested_jev_model"], problem_hash=_problem_hash(extracted), runtime=raw_jev["runtime_configuration"])
                jev_cache = _read_cache(jev_path, jev_fingerprint)
                if jev_cache is None:
                    raise ValueError("Incompatible Jev cache: " + str(jev_path))
                jev_status = "terminal_failure" if "terminal_error" in jev_cache else "observed"
                decisions = [JevDecision.model_validate(item) for item in jev_cache.get("decisions", [])]
                any_jev = any_jev or bool(decisions)
            observations = question_observations(case, extracted, decisions)
            for label in label_by_id.values():
                label["observations"][model] = {"alignment_status": "not_generated", "gpt_baseline_answer": None, "jev_answer": None}
            for observation in observations["observed_questions"]:
                if observation["canonical_question_id"] in label_by_id:
                    label_by_id[observation["canonical_question_id"]]["observations"][model] = {**observation, "observation_source": "benchmark_cache", "extraction_fingerprint": fingerprint, "jev_fingerprint": jev_fingerprint}
            output_case["observation_coverage"][model] = {
                **{key: value for key, value in observations.items() if key not in {"observed_questions", "unaligned_jev_decisions"}},
                "observation_source": "benchmark_cache", "cache_status": "terminal_failure" if "terminal_error" in cache else "observed",
                "jev_cache_status": jev_status, "extraction_fingerprint": fingerprint,
                "extraction_problem_hash": _problem_hash(extracted), "jev_fingerprint": jev_fingerprint,
            }
            output_case["unaligned_observed_questions"][model] = [item for item in observations["observed_questions"] if item["alignment_status"] != "aligned"] + observations["unaligned_jev_decisions"]
    draft["observed_models"] = selected_models
    draft["jev_observed"] = any_jev
    draft["observation_source"] = "benchmark_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--baseline", choices=("fixture", "benchmark"), default="fixture")
    parser.add_argument("--benchmark-results-dir", type=Path, help="Attach the actual benchmark extraction/Jev cache; makes no API calls.")
    parser.add_argument(
        "--model",
        action="append",
        choices=("gpt-6-luna", "gpt-6.1-sol"),
        help="Model cached observations to attach; repeat for both.",
    )
    args = parser.parse_args()
    if args.baseline == "benchmark" and args.benchmark_results_dir is None:
        parser.error("--baseline benchmark requires --benchmark-results-dir")
    cases = load_selected_cases(args.selection)
    draft = build_draft_answer_key(cases, baseline="fixture")
    if args.baseline == "benchmark":
        attach_benchmark_observations(draft, cases, args.benchmark_results_dir, args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(draft, indent=2) + "\n")
    write_review_tsv(draft, args.review_output)
    question_count = sum(len(case["expected_jev"]) for case in draft["cases"])
    print(
        "Wrote %d cases and %d canonical proposals to %s and %s; no human approval was inferred."
        % (len(draft["cases"]), question_count, args.output, args.review_output)
    )


if __name__ == "__main__":
    main()
