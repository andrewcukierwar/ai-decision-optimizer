"""Deterministic, offline Phase 10 analysis of the frozen reconciled benchmark.

Run from the repository root: .venv/bin/python -m scripts.analyze_benchmark
Only the output directory and calibration image are written. No benchmark,
SDK, parser, solver, .env loader, or provider module is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results/final_sol61_v2"
ANSWER_KEY = ROOT / "tests/fixtures/jev_expected_draft.json"
MODELS = ("gpt-6-luna", "gpt-6.1-sol")
ENGINES = ("direct_llm", "cp_sat")
QUESTION_TYPES = ("constraint_hardness", "task_requirement", "task_mode",
                  "availability_applies", "pref_weight")
CHOICES = {"constraint_hardness": ("hard", "soft"),
           "task_requirement": ("required", "optional"),
           "task_mode": ("active", "passive")}
VERSION = "phase10-offline-analysis-v1"
LIMITATIONS = [
    "14 curated cases (nine Day Planner, five Workforce), with one extraction sample per case/model; no repeated-run variance is measured.",
    "Four architecture variants per case/model share one extraction; both Jev-on engines share one Jev stage. The 112 cells are not independent observations.",
    "Two Workforce cases are generated from templates with explicit canonical specifications and numeric weights; their many availability labels dominate pooled accuracy.",
    "Extraction prompts were developed with awareness of benchmark-style examples (benchmark-prompt tuning).",
    "Direct LLM solves the same typed JSON used by CP-SAT, rather than the raw natural-language request.",
    "Jev can review only questions generated from GPT-extracted content; omitted content, extraction failures, and clarification abstentions constrain coverage.",
    "Primary accuracy and calibration exclude three subjective preference-strength labels. Score observations are few; do not interpret them as a general calibration claim.",
    "Custom user prompts have no canonical ground truth and are not evaluated here.",
    "This is an exploratory Resume MVP benchmark: no significance tests, independence-based confidence intervals, composite scores, or claims of general model superiority.",
    "Standalone architecture telemetry is an isolation estimate. Recorded experimental costs exclude unknown usage/billing for two OpenAI timeout requests and are not invoices. No TypeSafe dollar cost is assigned.",
    "Reconciliation normalized eight Workforce formulation rows and recovered upstream Jev evidence for one failed Direct solve. Its exact timed-out request body was not persisted; no solution was inferred.",
]


def require(condition, message):
    if not condition:
        raise ValueError("Analysis stopped: " + message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def primary_eligible(label):
    """Mirror the frozen answer-key gate without importing provider-bearing code."""
    return (label.get("primary_accuracy_calibration_eligible") is True
            and not label.get("primary_exclusion_reason")
            and (label.get("review_status"), label.get("label_provenance")) in {
                ("human_reviewed_approved", "fixture_intended_meaning"),
                ("generator_derived", "generator_derived")})


def outcome_correct(row):
    evaluation = row["canonical_evaluation"]
    return evaluation["canonical_validation_valid" if evaluation["canonical_feasible"]
                      else "feasible_correctly_reported"] is True


def optimal(row):
    e = row["canonical_evaluation"]
    return (e["canonical_feasible"] is True and e["canonical_validation_valid"] is True
            and e["objective_gap"] == 0)


def cell_key(row):
    c = row["configuration"]
    return row["case_id"], row["model"], c["use_jev"], c["solution_engine"]


def build_pairs(rows, dimension):
    """All 56 frozen pairs, including unavailable stages and failed outputs."""
    index = {cell_key(r): r for r in rows}
    require(len(index) == len(rows), "duplicate architecture cell")
    position, values = {"jev": (2, (False, True)), "engine": (3, ENGINES),
                        "model": (1, MODELS)}[dimension]
    pairs = []
    for key in sorted(index):
        other = list(key)
        other[position] = values[1] if key[position] == values[0] else values[0]
        require(tuple(other) in index, "missing paired cell")
        if key[position] == values[0]:
            pairs.append((index[key], index[tuple(other)]))
    return pairs


def transition(before, after):
    if before is None or after is None:
        return "not_comparable"
    return "improved" if after > before else "worsened" if after < before else "same"


def mean(values):
    return statistics.mean(values) if values else None


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def architecture_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        c = row["configuration"]
        groups[(MODELS.index(row["model"]), c["use_jev"], ENGINES.index(c["solution_engine"]))].append(row)
    summaries = []
    for _, group in sorted(groups.items()):
        e = [r["canonical_evaluation"] for r in group]
        t = [r["standalone_architecture_telemetry_estimate"] for r in group]
        valid = [r for r in group if r["canonical_evaluation"]["canonical_feasible"]
                 and r["canonical_evaluation"]["canonical_validation_valid"]]
        feasible = sum(x["canonical_feasible"] for x in e)
        summaries.append({
            "architecture": group[0]["architecture"], "cases": len(group),
            "outcome_correct": sum(outcome_correct(r) for r in group),
            "outcome_denominator": len(group),
            "usable_output": sum(x["valid_output"] is True for x in e),
            "usable_output_denominator": len(group),
            "terminal_failure": sum(r["status"] == "terminal_failure" for r in group),
            "terminal_failure_denominator": len(group),
            "abstention_no_usable_output": sum(r["status"] == "success" and not r["canonical_evaluation"]["valid_output"] for r in group),
            "abstention_denominator": len(group),
            "feasible_canonical_valid": len(valid), "feasible_denominator": feasible,
            "infeasible_correct": sum(not x["canonical_feasible"] and x["feasible_correctly_reported"] is True for x in e),
            "infeasible_denominator": len(group) - feasible,
            "structural_formulation_match": sum(x["formulation_match_structural"] is True for x in e),
            "full_formulation_match": sum(x["formulation_match"] is True for x in e),
            "formulation_denominator": len(group),
            "optimal": sum(optimal(r) for r in valid), "optimal_denominator": len(valid),
            "optimal_percent_among_valid_feasible": 100 * rate(sum(optimal(r) for r in valid), len(valid)) if valid else None,
            "latency_median_s": statistics.median(x["latency_total_s"] for x in t),
            "latency_mean_s": mean([x["latency_total_s"] for x in t]),
            "latency_denominator": len(t),
            "terminal_timeout": sum(r.get("failure_category") == "timeout" for r in group),
            "timeout_denominator": len(group),
            "openai_cost_median_usd": statistics.median(x["estimated_cost"] for x in t),
            "openai_cost_mean_usd": mean([x["estimated_cost"] for x in t]),
            "cost_denominator": len(t),
            "cost_unknown_usage_cells": sum(x["n_model_call_attempts"] > x["n_model_calls"] for x in t),
        })
    return summaries


def decision_instances(rows, key):
    """Read Jev-on CP-SAT alignment once per extraction, never per engine.

    Final accuracy uses the persisted represented value. Threshold fallbacks
    retain GPT; removed targets remain wrong, never credited with a proposal.
    Unaligned generated questions receive no canonical answer.
    """
    records = []
    cases = {c["name"]: c for c in key["cases"]}
    for row in sorted(rows, key=cell_key):
        c = row["configuration"]
        if not c["use_jev"] or c["solution_engine"] != "cp_sat":
            continue
        case = cases[row["case_id"]]
        labels = {label["question_id"]: label for label in case["expected_jev"]}
        aligned = {}
        base = {"case_id": row["case_id"], "model": row["model"], "domain": row["domain"]}
        for observation in row["question_alignment"]["observed_questions"]:
            qid = observation["canonical_question_id"]
            if qid is None:
                records.append({**base, "question_type": observation["question_type"],
                                "status": "unaligned_generated", "primary_eligible": False,
                                "canonical_question_id": None, "expected": None,
                                "provenance": "unaligned", "observation": observation})
            else:
                require(qid in labels and qid not in aligned, "invalid/duplicate canonical join")
                aligned[qid] = observation
        require(set(labels) - set(aligned) == set(row["question_alignment"]["missing_canonical_question_ids"]), "missing canonical IDs disagree")
        for qid, label in sorted(labels.items()):
            observation = aligned.get(qid)
            eligible = primary_eligible(label)
            expected = label["canonical_expected_answer"]
            final = None if observation is None else observation["final_applied_answer"]
            represented = observation is not None and observation["final_value_status"] in {"represented", "gpt_baseline"}
            records.append({**base, "question_type": label["question_type"],
                            "primitive": label["primitive"], "canonical_question_id": qid,
                            "status": "aligned" if observation else "not_generated",
                            "primary_eligible": eligible, "expected": expected,
                            "provenance": "subjective_excluded" if not eligible else "human_reviewed" if label["review_status"] == "human_reviewed_approved" else "generator_derived",
                            "gpt_correct": observation is not None and observation["gpt_baseline_answer"] == expected,
                            "jev_correct": observation is not None and observation["jev_observation_status"] == "observed" and observation["jev_answer"] == expected,
                            "final_correct": represented and final == expected,
                            "final_represented": represented, "observation": observation})
    return records


def grouped_records(records):
    """Prominent model/type results plus provenance and domain breakdowns."""
    groups = defaultdict(list)
    for r in records:
        for key in [("overall", "all", "all"), ("model", r["model"], "all"),
                    ("question_type", "all", r["question_type"]),
                    ("model_question_type", r["model"], r["question_type"]),
                    ("provenance", "all", r["provenance"]),
                    ("model_provenance", r["model"], r["provenance"]),
                    ("domain", "all", r["domain"])]:
            groups[key].append(r)
    return sorted(groups.items())


def accuracy_and_coverage(records):
    accuracy, coverage = [], []
    for (scope, model, category), group in grouped_records(records):
        base = {"scope": scope, "model": model, "category": category}
        eligible = [r for r in group if r["primary_eligible"]]
        aligned = [r for r in eligible if r["status"] == "aligned"]
        observed = [r for r in aligned if r["observation"]["jev_observation_status"] == "observed"]
        coverage.append({**base, "canonical_primary_instances": len(eligible),
                         "aligned_primary": len(aligned), "missing_not_generated_primary": len(eligible) - len(aligned),
                         "jev_observed_primary": len(observed), "aligned_missing_jev_answer": len(aligned) - len(observed),
                         "unaligned_generated": sum(r["status"] == "unaligned_generated" for r in group),
                         "excluded_subjective_instances": sum(r["provenance"] == "subjective_excluded" for r in group),
                         "excluded_subjective_aligned": sum(r["provenance"] == "subjective_excluded" and r["status"] == "aligned" for r in group),
                         "final_not_represented_aligned": sum(not r["final_represented"] for r in aligned)})
        if not eligible:
            continue
        result = {**base, "canonical_eligible": len(eligible), "aligned_observed": len(aligned),
                  "jev_observed": len(observed)}
        for who in ("gpt", "jev", "final"):
            correct = sum(r[who + "_correct"] for r in aligned)
            denominator = len(observed) if who == "jev" else len(aligned)
            result.update({who + "_aligned_correct": correct,
                           who + "_aligned_denominator": denominator,
                           who + "_aligned_accuracy": rate(correct, denominator),
                           who + "_end_to_end_correct": correct,
                           who + "_end_to_end_denominator": len(eligible),
                           who + "_end_to_end_accuracy": rate(correct, len(eligible))})
        accuracy.append(result)
    return accuracy, coverage


def validate_distribution(probabilities, classes):
    require(set(probabilities) == set(classes), "probability classes differ")
    require(all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                for p in probabilities.values()), "invalid probability")
    require(math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5), "probabilities do not sum to one")


def multiclass_brier(probabilities, expected):
    require(expected in probabilities, "canonical answer outside probability classes")
    return sum((p - int(k == expected)) ** 2 for k, p in probabilities.items())


def binary_brier(probabilities, expected):
    """Half the two-class sum = (p_positive - y_positive)^2, range [0,1]."""
    require(len(probabilities) == 2, "binary Brier needs two classes")
    validate_distribution(probabilities, tuple(probabilities))
    return multiclass_brier(probabilities, expected) / 2


def score_probabilities(sdk_probabilities):
    validate_distribution(sdk_probabilities, tuple(str(k) for k in range(5)))
    return {int(k) + 1: p for k, p in sdk_probabilities.items()}


def ranked_probability_score(probabilities, expected):
    """Normalized ordinal RPS: 1/(K-1) sum_{k=1}^{K-1}(F_k-Y_k)^2."""
    require(set(probabilities) == set(range(1, 6)) and expected in probabilities, "RPS needs weights 1–5")
    cumulative = 0
    errors = []
    for k in range(1, 5):
        cumulative += probabilities[k]
        errors.append((cumulative - int(expected <= k)) ** 2)
    return sum(errors) / 4


def reliability_bins(samples):
    bins = []
    for index in range(5):
        selected = [s for s in samples if min(int(s["confidence"] * 5), 4) == index]
        bins.append({"lower": index / 5, "upper": (index + 1) / 5,
                     "upper_inclusive": index == 4, "n": len(selected),
                     "mean_selected_confidence": mean([s["confidence"] for s in selected]),
                     "empirical_correctness": mean([int(s["correct"]) for s in selected])})
    return bins


def calibration_metrics(records):
    groups = defaultdict(list)
    for r in records:
        if not r["primary_eligible"] or r["status"] != "aligned":
            continue
        o = r["observation"]
        if o["jev_observation_status"] != "observed":
            continue
        qt, probabilities, expected = r["question_type"], o["probabilities"], r["expected"]
        sample = {"confidence": o["jev_probability"], "correct": r["jev_correct"]}
        if qt in CHOICES:
            validate_distribution(probabilities, CHOICES[qt])
            sample["binary_brier"] = binary_brier(probabilities, expected)
            sample["two_class_sum_brier"] = multiclass_brier(probabilities, expected)
        elif qt == "availability_applies":
            validate_distribution(probabilities, ("yes", "no"))
            sample["binary_brier"] = (probabilities["yes"] - int(expected)) ** 2
        else:
            require(qt == "pref_weight", "unrecognized calibration primitive")
            mapped = score_probabilities(probabilities)
            sample.update(multiclass_brier=multiclass_brier(mapped, expected),
                          mae_levels=abs(o["jev_answer"] - expected),
                          normalized_rps=ranked_probability_score(mapped, expected))
        require(math.isclose(probabilities[str(o["jev_answer"] - 1)] if qt == "pref_weight"
                             else probabilities[("yes" if o["jev_answer"] else "no") if qt == "availability_applies" else o["jev_answer"]],
                             sample["confidence"], abs_tol=1e-5), "selected confidence disagrees with distribution")
        groups[qt].append(sample)
        groups[r["model"] + "/" + qt].append(sample)
        if qt in CHOICES:
            groups["pooled_choice"].append(sample)
    result = {}
    for name, samples in sorted(groups.items()):
        fields = sorted(set().union(*(s.keys() for s in samples)) - {"confidence", "correct"})
        result[name] = {"n": len(samples), **{k: mean([s[k] for s in samples]) for k in fields},
                        "selected_answer_correct": sum(s["correct"] for s in samples),
                        "reliability_bins": reliability_bins(samples)}
    return {"definitions": {
        "choice_binary_brier": "0.5 * sum over both classes (p_k-y_k)^2; equivalent to positive-class squared error; range 0–1. Also report the unscaled two-class sum (range 0–2).",
        "noul_binary_brier": "(p_yes - y_yes)^2; range 0–1; separate from Choice.",
        "score_multiclass_brier": "sum over five classes (p_k-y_k)^2; SDK 0–4 maps to canonical weights 1–5; range 0–2; never pooled with binary Brier.",
        "score_mae": "Absolute error of the modal selected weight in optimizer levels 1–5.",
        "score_normalized_rps": "1/4 * sum at thresholds 1–4 (cumulative predicted probability - cumulative one-hot truth)^2; range 0–1.",
        "reliability": "Confidence of the selected Jev answer versus exact correctness, five equal-width bins [0,.2), …, [.8,1]. Missing/unaligned/subjective labels excluded; empty bins are null.",
    }, "groups": result,
            "score_interpretation": "Small eligible Score sample; descriptive metrics only, insufficient for a broad calibration claim."}


def pair_record(before, after, dimension):
    a, b = before["canonical_evaluation"], after["canonical_evaluation"]
    valid_a = a["canonical_feasible"] and a["canonical_validation_valid"]
    valid_b = b["canonical_feasible"] and b["canonical_validation_valid"]
    comparable = valid_a and valid_b and a["optimal_objective"] == b["optimal_objective"]
    ta, tb = [r["standalone_architecture_telemetry_estimate"] for r in (before, after)]
    result = {"case_id": before["case_id"], "domain": before["domain"], "model": before["model"],
              "engine": before["configuration"]["solution_engine"], "use_jev": before["configuration"]["use_jev"],
              "before_architecture": before["architecture"], "after_architecture": after["architecture"],
              "before_status": before["status"], "after_status": after["status"],
              "before_outcome_correct": outcome_correct(before), "after_outcome_correct": outcome_correct(after),
              "outcome_effect": transition(outcome_correct(before), outcome_correct(after)),
              "structural_effect": transition(a["formulation_match_structural"], b["formulation_match_structural"]),
              "before_structural_match": a["formulation_match_structural"], "after_structural_match": b["formulation_match_structural"],
              "before_usable_output": a["valid_output"], "after_usable_output": b["valid_output"],
              "before_feasible_valid": bool(valid_a), "after_feasible_valid": bool(valid_b),
              "canonical_validity_effect": transition(a["canonical_validation_valid"], b["canonical_validation_valid"]) if a["canonical_feasible"] else "not_applicable_infeasible",
              "before_optimal": optimal(before), "after_optimal": optimal(after),
              "optimum_effect": transition(optimal(before), optimal(after)) if a["canonical_feasible"] else "not_applicable_infeasible",
              "objective_effect": transition(-a["objective_gap"], -b["objective_gap"]) if comparable else "not_comparable",
              "before_objective_gap": a["objective_gap"], "after_objective_gap": b["objective_gap"],
              "before_latency_s": ta["latency_total_s"], "after_latency_s": tb["latency_total_s"],
              "latency_delta_s": tb["latency_total_s"] - ta["latency_total_s"],
              "before_cost_usd": ta["estimated_cost"], "after_cost_usd": tb["estimated_cost"],
              "cost_delta_usd": tb["estimated_cost"] - ta["estimated_cost"],
              "cost_delta_has_unknown_usage": any(t["n_model_call_attempts"] > t["n_model_calls"] for t in (ta, tb))}
    if dimension == "jev":
        available = before.get("final_problem") is not None and after.get("final_problem") is not None
        changed = before.get("final_problem") != after.get("final_problem") if available else None
        decisions = after["jev_decisions"]
        edits = sum(d["changed"] for d in decisions)
        category = ("unavailable_formulation" if not available else "typed_formulation_changed" if changed
                    else "applied_edit_no_net_typed_change" if edits else "asked_no_applied_formulation_change" if decisions
                    else "no_questions")
        result.update(formulation_available=available, typed_formulation_changed=changed,
                      asked_questions=len(decisions), accepted_answers=sum(d["applied"] for d in decisions),
                      applied_typed_edits=edits, formulation_change_category=category)
    return result


def pair_summaries(pairs, dimension):
    groups = defaultdict(list)
    for p in pairs:
        keys = [("overall", "all"), ("domain", p["domain"])]
        if dimension != "model":
            keys += [("model", p["model"])]
        if dimension != "engine":
            keys += [("engine", p["engine"])]
        if dimension != "jev":
            keys += [("jev", str(p["use_jev"]).lower())]
        # Compare models separately within every identical engine/Jev setting.
        if dimension == "model":
            keys += [("engine_jev", p["engine"] + "/jev=" + str(p["use_jev"]).lower())]
        if dimension == "engine":
            keys += [("model_jev", p["model"] + "/jev=" + str(p["use_jev"]).lower())]
        for key in keys:
            groups[key].append(p)
    summaries = []
    for (scope, category), group in sorted(groups.items()):
        s = {"scope": scope, "category": category, "pairs": len(group),
             "before_outcome_correct": sum(p["before_outcome_correct"] for p in group),
             "after_outcome_correct": sum(p["after_outcome_correct"] for p in group),
             "before_usable_output": sum(p["before_usable_output"] for p in group),
             "after_usable_output": sum(p["after_usable_output"] for p in group),
             "before_structural_match": sum(p["before_structural_match"] for p in group),
             "after_structural_match": sum(p["after_structural_match"] for p in group),
             "before_optimal": sum(p["before_optimal"] for p in group),
             "after_optimal": sum(p["after_optimal"] for p in group),
             "before_feasible_valid": sum(p["before_feasible_valid"] for p in group),
             "after_feasible_valid": sum(p["after_feasible_valid"] for p in group),
             "feasible_pair_denominator": sum(p["canonical_validity_effect"] != "not_applicable_infeasible" for p in group),
             "before_terminal_failure": sum(p["before_status"] == "terminal_failure" for p in group),
             "after_terminal_failure": sum(p["after_status"] == "terminal_failure" for p in group),
             "before_abstention": sum(p["before_status"] == "success" and not p["before_usable_output"] for p in group),
             "after_abstention": sum(p["after_status"] == "success" and not p["after_usable_output"] for p in group),
             "latency_delta_median_s": statistics.median(p["latency_delta_s"] for p in group),
             "latency_delta_mean_s": mean([p["latency_delta_s"] for p in group]),
             "cost_delta_median_usd": statistics.median(p["cost_delta_usd"] for p in group),
             "cost_delta_mean_usd": mean([p["cost_delta_usd"] for p in group]),
             "cost_delta_unknown_usage_pairs": sum(p["cost_delta_has_unknown_usage"] for p in group)}
        for metric in ("outcome", "structural", "canonical_validity", "optimum", "objective"):
            s[metric + "_effects"] = dict(sorted(Counter(p[metric + "_effect"] for p in group).items()))
        if dimension == "jev":
            category_counts = Counter(p["formulation_change_category"] for p in group)
            s["formulation_change_categories"] = {
                name: category_counts[name] for name in (
                    "asked_no_applied_formulation_change", "applied_edit_no_net_typed_change",
                    "typed_formulation_changed", "unavailable_formulation", "no_questions")}
        summaries.append(s)
    return summaries


def immutable_snapshot(results_dir):
    """Protect every frozen input, including raw caches, answer keys and reports."""
    paths = [p for p in Path(results_dir).rglob("*") if p.is_file()
             and "analysis" not in p.relative_to(results_dir).parts and "__pycache__" not in p.parts]
    paths += list((ROOT / "tests/fixtures").rglob("*"))
    paths += [ROOT / "project_plan.md", *(ROOT / "docs").glob("*benchmark*.md"),
              ROOT / "docs/final_benchmark_integrity_audit_details.json"]
    return {p.resolve(): sha256(p) for p in paths if p.is_file()}


def load_inputs(results_dir=RESULTS, answer_key=ANSWER_KEY):
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "reconciliation_manifest.json").read_text())
    require(manifest["authoritative_phase_10_input"] == "reconciled.jsonl", "wrong authoritative input")
    require(sha256(results_dir / "reconciled.jsonl") == manifest["derived_sha256"]["reconciled.jsonl"], "reconciled hash differs")
    require(sha256(answer_key) == manifest["auxiliary_input_sha256"]["tests/fixtures/jev_expected_draft.json"], "answer-key hash differs")
    raw_hashes = {name: digest for name, digest in manifest["input_sha256"].items()
                  if name not in {"audit_details", "audit_report"}}
    require(len(raw_hashes) == 55 and all(sha256(results_dir / name) == digest
                                        for name, digest in raw_hashes.items()), "audited raw input hashes differ")
    rows = read_jsonl(results_dir / "reconciled.jsonl")
    key = json.loads(Path(answer_key).read_text())
    cases = {c["name"] for c in key["cases"]}
    require(len(cases) == 14 and len(key["cases"]) == 14, "expected 14 unique cases")
    expected = {(case, model, jev, engine) for case in cases for model in MODELS
                for jev in (False, True) for engine in ENGINES}
    require(len(rows) == 112 and {cell_key(r) for r in rows} == expected, "expected exactly 112 unique reconciled cells")
    require(all(r.get("reconciliation", {}).get("paid_observations_generated_or_replaced") is False for r in rows), "not reconciled observations")
    labels = [label for c in key["cases"] for label in c["expected_jev"]]
    require(Counter(l["review_status"] for l in labels) == {"human_reviewed_approved": 54, "generator_derived": 71, "subjective_excluded_from_primary_scoring": 3}
            and sum(primary_eligible(l) for l in labels) == 125, "frozen primary label policy differs")
    for case in cases:
        require(len({r["canonical_evaluation"]["canonical_feasible"] for r in rows if r["case_id"] == case}) == 1, "canonical feasibility differs by arm")
    for r in rows:
        e = r["canonical_evaluation"]
        require(isinstance(e["canonical_feasible"], bool), "canonical feasibility unscorable")
        if e["canonical_validation_valid"] and e["canonical_feasible"]:
            require(e["objective_gap"] is not None and e["objective_value"] is not None
                    and e["optimal_objective"] is not None
                    and e["objective_gap"] == e["objective_value"] - e["optimal_objective"], "objective gap inconsistent")
    # All four variants share the same extraction; both Jev engines share observations.
    for a, b in build_pairs(rows, "engine"):
        require(a["question_alignment"] == b["question_alignment"], "shared decision observations differ")
    for case in cases:
        for model in MODELS:
            require(len({r.get("extraction_problem_hash") for r in rows if (r["case_id"], r["model"]) == (case, model)}) == 1, "shared extraction hashes differ")
    totals = manifest["experiment_totals"]
    expected_totals = {"openai_call_attempts": 76, "openai_responses_with_metadata": 74,
                       "openai_input_tokens": 86583, "openai_output_tokens": 46461,
                       "openai_estimated_cost_usd": .2874103, "openai_usage_unknown_calls": 2,
                       "jev_completed_calls": 24, "jev_input_tokens": 50946, "jev_questions": 230}
    for field, expected_value in expected_totals.items():
        require(math.isclose(totals[field], expected_value, abs_tol=1e-10), "experiment accounting differs")
        require(math.isclose(sum(r["actual_paid_call_attempts_created_for_this_row"][field] for r in rows), expected_value, abs_tol=1e-10), "row allocations disagree with experiment ledger")
    return rows, key, manifest


def analyze(rows, key, manifest):
    instances = decision_instances(rows, key)
    accuracy, coverage = accuracy_and_coverage(instances)
    overall = next(r for r in coverage if r["scope"] == "overall")
    require((overall["canonical_primary_instances"], overall["aligned_primary"],
             overall["missing_not_generated_primary"], overall["unaligned_generated"],
             overall["excluded_subjective_instances"]) == (250, 223, 27, 2, 6), "frozen coverage differs")
    pairs = {dimension: [pair_record(a, b, dimension) for a, b in build_pairs(rows, dimension)]
             for dimension in ("jev", "engine", "model")}
    require(all(len(p) == 56 for p in pairs.values()), "expected 56 pairs per comparison")
    architecture = architecture_summary(rows)
    require(len(architecture) == 8, "expected eight architectures")
    failures = [{"case_id": r["case_id"], "architecture": r["architecture"], "status": r["status"],
                 "failure_stage": r.get("failure_stage"), "failure_category": r.get("failure_category"),
                 "missing_info": (r.get("extraction") or {}).get("missing_info", []),
                 "constraint_violations": r["canonical_evaluation"]["constraint_violations"]}
                for r in sorted(rows, key=cell_key) if r["status"] != "success" or not r["canonical_evaluation"]["valid_output"]]
    extraction_coverage = []
    for model in MODELS:
        stages = [r for r in rows if r["model"] == model and not r["configuration"]["use_jev"] and r["configuration"]["solution_engine"] == "cp_sat"]
        extraction_coverage.append({"model": model, "case_model_stages": len(stages),
                                    "typed_extractions": sum(r.get("final_problem") is not None for r in stages),
                                    "terminal_extraction_failure": sum(r["status"] == "terminal_failure" and r.get("failure_stage") == "extraction" for r in stages),
                                    "clarification_null_formulation": sum(r["status"] == "success" and r.get("final_problem") is None for r in stages)})
    summary = {"analysis_version": VERSION, "observations_consumed": len(rows), "cases": 14,
               "architecture_summary": architecture, "label_policy": manifest["label_counts"],
               "decision_coverage": overall, "experiment_accounting": manifest["experiment_totals"],
               "extraction_coverage": extraction_coverage, "failures_and_abstentions": failures,
               "paired_comparisons": {d: pair_summaries(p, d) for d, p in pairs.items()},
               "jev_outcome_fixed": [p for p in pairs["jev"] if p["outcome_effect"] == "improved"],
               "jev_outcome_broken": [p for p in pairs["jev"] if p["outcome_effect"] == "worsened"],
               "limitations": LIMITATIONS}
    jev_stages = [entry["telemetry"]["latency_jev_s"] for entry in manifest["unique_stage_ledger"] if entry["stage"] == "jev"]
    summary["jev_stage_latency"] = {"completed_calls": len(jev_stages),
                                    "median_s": statistics.median(jev_stages), "mean_s": mean(jev_stages)}
    return summary, accuracy, coverage, instances, pairs, calibration_metrics(instances)


def write_csv(path, rows):
    require(bool(rows), "empty CSV")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in rows)])


def fraction(n, d):
    return f"{n} / {d}"


def accuracy_table(rows):
    return markdown_table(["Model/slice", "Type", "GPT aligned", "Jev aligned", "Final aligned", "GPT end-to-end", "Jev end-to-end", "Final end-to-end"],
        [[r["model"] if r["model"] != "all" else r["scope"], r["category"],
          *(fraction(r[w + "_aligned_correct"], r[w + "_aligned_denominator"]) for w in ("gpt", "jev", "final")),
          *(fraction(r[w + "_end_to_end_correct"], r["canonical_eligible"]) for w in ("gpt", "jev", "final"))] for r in rows])


def paired_table(summaries):
    return markdown_table(["Slice", "Pairs", "Outcome before → after", "Outcome improved/worsened/same", "Valid feasible before → after", "Optimal before → after (valid denominators)", "Objective improved/worsened/same/comparable", "Median latency Δ s", "Median cost Δ $"],
        [[s["category"], s["pairs"], f"{s['before_outcome_correct']} → {s['after_outcome_correct']} / {s['pairs']}",
          "/".join(str(s["outcome_effects"].get(k, 0)) for k in ("improved", "worsened", "same")),
          f"{s['before_feasible_valid']} → {s['after_feasible_valid']} / {s['feasible_pair_denominator']}",
          f"{s['before_optimal']}/{s['before_feasible_valid']} → {s['after_optimal']}/{s['after_feasible_valid']}",
          "/".join(str(s["objective_effects"].get(k, 0)) for k in ("improved", "worsened", "same")) + "/" + str(sum(s["objective_effects"].get(k, 0) for k in ("improved", "worsened", "same"))),
          f"{s['latency_delta_median_s']:.3f}", f"{s['cost_delta_median_usd']:.7f}"] for s in summaries])


def report(summary, accuracy, coverage, calibration, pairs, objectives):
    a = summary["architecture_summary"]
    sections = ["# Phase 10 — Final benchmark analysis", "## 1. Dataset and execution",
        "Offline analysis of `../reconciled.jsonl`, verified against the reconciliation manifest and finalized answer-key hash. 14 cases × two models × two Jev settings × two engines = 112 observations. Nine Day Planner and five Workforce cases; 12 canonically feasible and two infeasible. 103 execution successes include eight clarification/null-formulation outputs; nine terminal failures remain in all relevant denominators.",
        "Outcome correctness: for feasible cases, `canonical_validation_valid == true`; for infeasible cases, `feasible_correctly_reported == true`. Usable output is `valid_output`; an invalid canonical schedule can still be usable structured output. Abstention means execution success without usable output and excludes terminal failures. Formulation accuracy counts every cell, including absent formulations as incorrect. Hard-constraint k/n is not a headline metric.",
        "## 2. Eight architectures",
        markdown_table(["Architecture", "Outcome", "Usable", "Terminal", "Abstention", "Valid feasible", "Infeasible correct", "Structural", "Full", "Optimal / valid feasible", "% optimal"],
            [[r["architecture"], fraction(r["outcome_correct"],14), fraction(r["usable_output"],14), fraction(r["terminal_failure"],14), fraction(r["abstention_no_usable_output"],14), fraction(r["feasible_canonical_valid"],12), fraction(r["infeasible_correct"],2), fraction(r["structural_formulation_match"],14), fraction(r["full_formulation_match"],14), fraction(r["optimal"],r["optimal_denominator"]), f"{r['optimal_percent_among_valid_feasible']:.1f}%"] for r in a]),
        "Objectives are evaluated under each case's canonical objective. `objective_gaps.csv` lists every cell, its domain, canonical optimum, objective and gap. Raw units are never averaged across problems. No relative normalization is used: zero optima are frequent, and cross-case objective scales are not comparable.",
        "## 3. Luna 6 versus Sol 6.1",
        "Before = Luna; after = Sol 6.1. These are paired descriptive observations, not independent model trials. Deltas are Sol minus Luna and include failed/abstained cells.",
        paired_table(summary["paired_comparisons"]["model"]),
        markdown_table(["Model", "Typed extraction / cases", "Terminal extraction / cases", "Clarification / cases"],
                       [[r["model"], fraction(r["typed_extractions"],14), fraction(r["terminal_extraction_failure"],14), fraction(r["clarification_null_formulation"],14)] for r in summary["extraction_coverage"]]),
        markdown_table(["Slice", "Structural before → after", "Usable before → after", "Terminal before → after", "Abstention before → after"],
                       [[s["category"], f"{s['before_structural_match']} → {s['after_structural_match']} / {s['pairs']}", f"{s['before_usable_output']} → {s['after_usable_output']} / {s['pairs']}", f"{s['before_terminal_failure']} → {s['after_terminal_failure']} / {s['pairs']}", f"{s['before_abstention']} → {s['after_abstention']} / {s['pairs']}"] for s in summary["paired_comparisons"]["model"]]),
        "Model outcome discordances (all cases and settings):",
        markdown_table(["Case", "Engine", "Jev", "Outcome Luna → Sol", "Objective gap Luna → Sol"],
                       [[p["case_id"],p["engine"],p["use_jev"],f"{p['before_outcome_correct']} → {p['after_outcome_correct']}",f"{p['before_objective_gap']} → {p['after_objective_gap']}"] for p in pairs["model"] if p["outcome_effect"] != "same" or p["objective_effect"] in {"improved", "worsened"}]),
        "## 4. Direct LLM versus CP-SAT",
        "Before = Direct LLM; after = CP-SAT, within the same case/model/Jev setting. Deltas are CP-SAT minus Direct. Optimum counts use each side's valid-feasible denominator; objective comparisons require both outputs valid for the same case.",
        paired_table(summary["paired_comparisons"]["engine"]),
        "Noteworthy discordant pairs:",
        markdown_table(["Case", "Model", "Jev", "Outcome Direct → CP", "Optimum Direct → CP", "Gap Direct → CP"],
            [[p["case_id"], p["model"], p["use_jev"], f"{p['before_outcome_correct']} → {p['after_outcome_correct']}", f"{p['before_optimal']} → {p['after_optimal']}", f"{p['before_objective_gap']} → {p['after_objective_gap']}"] for p in pairs["engine"] if p["outcome_effect"] != "same" or p["objective_effect"] in {"improved", "worsened"}]),
        "## 5. GPT versus Jev decision accuracy",
        "Frozen policy: 54 human-reviewed + 71 generator-derived = 125 unique primary labels; three subjective weight labels excluded. Across two models: 250 eligible instances, 223 aligned and observed, 27 missing, plus two explicitly unaligned generated questions. The exclusions yield six model-label instances (five observed, one missing). Decisions are counted once per extraction, not twice per engine.",
        "All tables show exact counts; the CSV also reports aligned and end-to-end rates. Missing canonical questions count wrong end-to-end. Jev proposed answers are scored before thresholding. Final accuracy scores the persisted represented/applied value: accepted Jev where represented, otherwise GPT threshold fallback; a removed/unrepresented target counts wrong. No unaligned question receives an inferred label.",
        accuracy_table([r for r in accuracy if r["scope"] in {"overall", "model", "question_type", "model_question_type"}]),
        "Label provenance:", accuracy_table([r for r in accuracy if r["scope"] in {"provenance", "model_provenance"}]),
        "Coverage (primary denominator is canonical instances; unaligned and subjective counts are separate):",
        markdown_table(["Slice", "Type", "Primary", "Aligned", "Missing", "Unaligned generated", "Excluded subjective"],
            [[r["model"] if r["model"] != "all" else r["scope"],r["category"],r["canonical_primary_instances"],r["aligned_primary"],r["missing_not_generated_primary"],r["unaligned_generated"],r["excluded_subjective_instances"]] for r in coverage if r["scope"] in {"overall", "model", "question_type", "model_question_type", "provenance"}]),
        "## 6. Calibration",
        *(f"**{name}:** {definition}" for name,definition in calibration["definitions"].items()),
        markdown_table(["Group", "n", "Binary Brier", "Two-class sum Brier", "Five-class sum Brier", "MAE levels", "Normalized RPS"],
                       [[name, g["n"], *(f"{g[k]:.6f}" if k in g else "—" for k in ("binary_brier", "two_class_sum_brier", "multiclass_brier", "mae_levels", "normalized_rps"))] for name,g in calibration["groups"].items()]),
        calibration["score_interpretation"],
        "![Five-bin selected-answer reliability by question type](../../../docs/images/calibration.png)",
        "Separate panels identify the primitive and question type. Marker labels show bin sample sizes; empty bins are omitted. This is selected-answer confidence reliability, not a positive-class probability curve. Numerical bins and model-specific groups are in `calibration_metrics.json`.",
        "## 7. Paired Jev downstream effects",
        "Before = Jev off; after = Jev on. All 56 pairs remain, including unavailable formulations. A typed change means exact equality of the two persisted typed-JSON objects changed, not that a canonical semantic error was necessarily corrected. Accepted answers that retain the same value do not count as edits. Structural/full correctness uses reconciled canonical semantic comparisons.",
        paired_table(summary["paired_comparisons"]["jev"]),
        markdown_table(["Slice", "Pairs", "Change categories", "Structural effects", "Canonical validity effects"],
            [[s["category"],s["pairs"],json.dumps(s["formulation_change_categories"],sort_keys=True),json.dumps(s["structural_effects"],sort_keys=True),json.dumps(s["canonical_validity_effects"],sort_keys=True)] for s in summary["paired_comparisons"]["jev"]]),
        "Changed formulations or discordant outcomes (every fixed/broken case is included):",
        markdown_table(["Case", "Model", "Engine", "Typed change", "Structural effect", "Outcome effect", "Objective effect"],
            [[p["case_id"],p["model"],p["engine"],p["typed_formulation_changed"],p["structural_effect"],p["outcome_effect"],p["objective_effect"]] for p in pairs["jev"] if p["typed_formulation_changed"] or p["outcome_effect"] != "same"]),
        "These paired associations do not establish causality. Direct solves are separate calls; output differences without a formulation change and a timeout cannot be attributed to Jev's semantic edits. `jev_effects.csv` preserves every pair, including questions asked, accepted answers, applied edit counts, change categories, and metric transitions.",
        "The two observed outcome fixes are Luna Direct `hard_timing` and `ten_day_rotation`; the observed break is Luna Jev Direct `workout_before_four_hard` (solve timeout). None of these three pairs changed its formulation. Genuine formulation changes occur only on `infeasible_staffing_is_extracted_not_repaired` and `regional_week`, under both models: four shared case/model stages, appearing in eight engine pairs. All eight reduce structural formulation correctness; none changes case-level outcome correctness. No structural formulation fixes occur. No pair records applied edits with zero net typed change.",
        "## 8. Latency and cost",
        "Standalone estimates include the stages each architecture would require in isolation. All 14 cells contribute to each median/mean, including terminal waits and abstentions. Timeout costs contain only recorded usage and have unknown additional usage; cost deltas involving them are flagged. No sum of standalone costs is presented as actual expenditure.",
        markdown_table(["Architecture", "Latency median s", "Latency mean s", "Timeout / cases", "OpenAI cost median $", "OpenAI cost mean $", "Unknown-usage cells / cases"],
            [[r["architecture"],f"{r['latency_median_s']:.3f}",f"{r['latency_mean_s']:.3f}",fraction(r["terminal_timeout"],14),f"{r['openai_cost_median_usd']:.7f}",f"{r['openai_cost_mean_usd']:.7f}",fraction(r["cost_unknown_usage_cells"],14)] for r in a]),
        "Latency and cost statistics above each have n=14. Actual shared-stage experimental accounting from reconciliation: 76 OpenAI attempts, 74 responses with metadata, 86,583 input tokens, 46,461 output tokens, $0.2874103 recorded estimated OpenAI cost; two OpenAI timeout requests have unknown usage/billing. Jev: 24 completed calls, 50,946 input tokens, 230 questions. TypeSafe dollar cost is unknown/unassigned.",
        f"The 24 unique completed Jev stages have median latency {summary['jev_stage_latency']['median_s']:.3f} s and mean {summary['jev_stage_latency']['mean_s']:.3f} s. These stage timings are distinct from end-to-end architecture deltas.",
        "## 9. Failures and abstentions",
        markdown_table(["Case", "Architecture", "Status", "Stage/category", "Clarification"],
            [[r["case_id"],r["architecture"],r["status"],f"{r['failure_stage'] or '—'}/{r['failure_category'] or '—'}", "; ".join(r["missing_info"])] for r in summary["failures_and_abstentions"]]),
        "Nine terminal cells arise from three failed requests/stages: Luna soft-workout extraction timeout affects four arms, Luna impossible-workout invalid extraction affects four, and Luna hard-workout Jev Direct solve timeout affects one. Eight abstention cells arise from two clarification extractions: Luna WFH and Sol optional-walk each affect four arms. Sol's claimed inability to represent optional tasks and Luna's claimed inability to represent exact starts were extraction abstentions, not canonical ambiguity. All benchmark cases have fixed canonical ground truth.",
        "## 10. Limitations", "\n".join("- " + s for s in LIMITATIONS),
        "## 11. Defensible findings for later drafting", "\n".join("- " + s for s in conclusions(summary, accuracy, calibration)),
        "## Reproduction and artifacts",
        "Prepare the existing test environment with the additional plotting requirement in `requirements-analysis.txt` (for example, `uv pip install --offline -r requirements-analysis.txt` from a populated local cache), then run `.venv/bin/python -m scripts.analyze_benchmark`. Analysis itself requires no network access. Output flags: `--results-dir`, `--answer-key`, `--output-dir`, `--plot-path`; the reconciled input and key must match the frozen manifest. Paths overlapping frozen inputs are rejected. Outputs contain no generation timestamp; identical inputs, source and plotting dependencies produce identical bytes. `analysis_summary.json` records input/source/output hashes and plot versions. Run `.venv/bin/python -m pytest` for the complete deterministic suite.",
        "Artifacts: architecture, decision accuracy, coverage, Jev effects, engine pairs, model pairs, per-cell objective gaps, calibration metrics, decision instances, JSON summary, this report, and the calibration image. No README or resume changes are included.",
    ]
    return "\n\n".join(sections) + "\n"


def conclusions(summary, accuracy, calibration):
    a = summary["architecture_summary"]
    overall = next(r for r in accuracy if r["scope"] == "overall")
    engine = next(s for s in summary["paired_comparisons"]["engine"] if s["scope"] == "overall")
    model = next(s for s in summary["paired_comparisons"]["model"] if s["scope"] == "overall")
    jev = next(s for s in summary["paired_comparisons"]["jev"] if s["scope"] == "overall")
    return [
        "Architecture outcome correctness ranges from " + str(min(r["outcome_correct"] for r in a)) + " to " + str(max(r["outcome_correct"] for r in a)) + " of 14; these are canonical case outcomes, not constraint fractions.",
        f"In 56 engine pairs, Direct and CP-SAT have {engine['before_outcome_correct']}/56 and {engine['after_outcome_correct']}/56 correct outcomes; CP-SAT improves {engine['outcome_effects'].get('improved',0)} and worsens {engine['outcome_effects'].get('worsened',0)} paired outcomes.",
        f"Among valid feasible outputs, Direct attains {engine['before_optimal']}/{engine['before_feasible_valid']} canonical optima and CP-SAT {engine['after_optimal']}/{engine['after_feasible_valid']}; objective comparisons remain within cases.",
        f"Luna and Sol record {model['before_outcome_correct']}/56 and {model['after_outcome_correct']}/56 correct cells, respectively, with shared extractions and different extraction failures/abstentions; this does not establish general superiority.",
        f"GPT, Jev proposed, and final represented decisions score {overall['gpt_aligned_correct']}/223, {overall['jev_aligned_correct']}/223, and {overall['final_aligned_correct']}/223 aligned primary answers; the end-to-end denominator is 250.",
        "Per-question-type accuracy is necessary: generated Workforce availability labels dominate pooled counts and cannot substitute for Day Planner results.",
        f"Across 56 Jev pairs, outcomes improve in {jev['outcome_effects'].get('improved',0)}, worsen in {jev['outcome_effects'].get('worsened',0)}, and remain the same in {jev['outcome_effects'].get('same',0)}; formulation edits and independent Direct solve variation must be distinguished.",
        f"Calibration is descriptive and primitive-specific: pooled Choice binary Brier {calibration['groups']['pooled_choice']['binary_brier']:.6f}, Noul Brier {calibration['groups']['availability_applies']['binary_brier']:.6f}, and five-class Score Brier {calibration['groups']['pref_weight']['multiclass_brier']:.6f} from only {calibration['groups']['pref_weight']['n']} eligible Score observations.",
        "Nine terminal and eight abstention cells remain scored; they arise from three failed stages and two clarification extractions rather than 17 independent extraction attempts.",
        "Actual recorded OpenAI experiment cost is $0.2874103 plus unknown timeout billing; standalone medians/means describe isolation requirements and Jev dollar cost is unassigned.",
    ]


def plot_calibration(calibration, path):
    # Keep matplotlib's font/config cache outside frozen artifacts and user home.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phase10-matplotlib"))
    import matplotlib
    import numpy
    import PIL
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = [*CHOICES, "availability_applies", "pref_weight"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for ax, name in zip(axes.flat, names):
            g = calibration["groups"][name]
            bins = [b for b in g["reliability_bins"] if b["n"]]
            primitive = "Choice (binary)" if name in CHOICES else "Noul (yes/no)" if name == "availability_applies" else "Score (five levels)"
            ax.plot([0, 1], [0, 1], linestyle="--", color="#8492a6", linewidth=1, label="Perfect reliability")
            ax.plot([b["mean_selected_confidence"] for b in bins], [b["empirical_correctness"] for b in bins],
                    "o-", color="#176f9e", markersize=7)
            for b in bins:
                ax.annotate(f"n={b['n']}", (b["mean_selected_confidence"], b["empirical_correctness"]),
                            xytext=(-8, -18), textcoords="offset points", ha="right", fontsize=9)
            ax.set(xlim=(-.03,1.05), ylim=(-.08,1.07), xticks=[0,.2,.4,.6,.8,1], yticks=[0,.2,.4,.6,.8,1],
                   title=f"{name}\n{primitive}; n={g['n']}", xlabel="Mean selected-answer confidence", ylabel="Empirical exact correctness")
            ax.grid(alpha=.18)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.05,.9,"Five equal-width confidence bins\nPrimary aligned observations only\nEmpty bins omitted; counts on markers\n\nModels pooled within question type\nScore sample is small\nNo uncertainty or generalization claims", transform=axes.flat[-1].transAxes, va="top", linespacing=1.8)
        fig.suptitle("Jev selected-answer reliability — curated benchmark", fontsize=16)
        fig.savefig(path, dpi=160, metadata={"Software": "Phase 10 deterministic analysis"})
        plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "numpy": numpy.__version__,
            "pillow": PIL.__version__}


def run_analysis(results_dir=RESULTS, answer_key=ANSWER_KEY, output_dir=None, plot_path=None):
    results_dir, answer_key = Path(results_dir).resolve(), Path(answer_key).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else results_dir / "analysis"
    plot_path = Path(plot_path).resolve() if plot_path else ROOT / "docs/images/calibration.png"
    snapshot = immutable_snapshot(results_dir)
    snapshot[answer_key] = sha256(answer_key)
    filenames = ["architecture_summary.csv", "jev_accuracy.csv", "jev_coverage.csv", "jev_effects.csv",
                 "engine_pairs.csv", "model_pairs.csv", "objective_gaps.csv", "calibration_metrics.json",
                 "decision_instances.jsonl", "analysis_summary.json", "analysis_report.md"]
    destinations = [(output_dir / name).resolve() for name in filenames] + [plot_path.resolve()]
    require(len(set(destinations)) == len(destinations), "output destinations overlap")
    require(all(p not in snapshot and not any(p in s.parents for s in snapshot) for p in destinations), "output overlaps frozen input")
    require(all(not p.is_relative_to(results_dir) or p.is_relative_to(results_dir / "analysis")
                for p in destinations), "benchmark outputs must stay inside analysis directory")
    require(all(p.suffix == ".png" for p in [plot_path]), "plot must be a PNG")
    rows, key, manifest = load_inputs(results_dir, answer_key)
    summary, accuracy, coverage, instances, pairs, calibration = analyze(rows, key, manifest)
    objectives = [{"case_id": r["case_id"], "domain": r["domain"], "architecture": r["architecture"],
                   **{k: r["canonical_evaluation"][k] for k in ("canonical_feasible", "canonical_validation_valid", "objective_value", "optimal_objective", "objective_gap")},
                   "optimal": optimal(r)} for r in sorted(rows,key=cell_key)]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    for name, data in [("architecture_summary.csv", summary["architecture_summary"]),
                       ("jev_accuracy.csv", accuracy), ("jev_coverage.csv", coverage),
                       ("jev_effects.csv", pairs["jev"]), ("engine_pairs.csv", pairs["engine"]),
                       ("model_pairs.csv", pairs["model"]), ("objective_gaps.csv", objectives)]:
        write_csv(output_dir / name, data)
    write_json(output_dir / "calibration_metrics.json", calibration)
    (output_dir / "decision_instances.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in instances))
    (output_dir / "analysis_report.md").write_text(report(summary, accuracy, coverage, calibration, pairs, objectives))
    summary["plot_dependencies"] = plot_calibration(calibration, plot_path)
    summary["defensible_findings"] = conclusions(summary, accuracy, calibration)
    summary["provenance"] = {"authoritative_input": "reconciled.jsonl",
                             "reconciled_sha256": sha256(results_dir / "reconciled.jsonl"),
                             "answer_key_sha256": sha256(answer_key),
                             "manifest_sha256": sha256(results_dir / "reconciliation_manifest.json"),
                             "analysis_script_sha256": sha256(Path(__file__)),
                             "immutable_inputs_verified": len(snapshot)}
    summary["output_sha256"] = {p.name: sha256(p) for p in destinations if p.name != "analysis_summary.json"}
    require(all(sha256(p) == digest for p, digest in snapshot.items()), "frozen inputs modified")
    write_json(output_dir / "analysis_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--answer-key", type=Path, default=ANSWER_KEY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-path", type=Path)
    args = parser.parse_args()
    summary = run_analysis(**vars(args))
    print(f"Analyzed {summary['observations_consumed']} reconciled observations; eight architectures; frozen inputs unchanged.")


if __name__ == "__main__":
    main()
