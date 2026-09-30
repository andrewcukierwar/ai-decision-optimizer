"""Offline research views and runtime display helpers; no provider execution.

Only the frozen final directory is read. Metrics are displayed from Phase 10,
never recomputed, ranked, or inferred from custom prompts.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "benchmark_results" / "final_sol61_v2"
CALIBRATION_IMAGE = ROOT / "docs" / "images" / "calibration.png"
CUSTOM_VALIDATION_NOTE = (
    "Runtime validation checks the interpreted problem, including any manual JSON edits. "
    "Custom prompts have no canonical ground truth: benchmark correctness, formulation "
    "match, and canonical optimum scores are available only in Benchmark / Compare."
)
QUESTION_LABELS = {
    "constraint_hardness": "Constraint hardness",
    "task_requirement": "Task requirement",
    "task_mode": "Task mode",
    "pref_weight": "Preference weight",
    "availability_applies": "Availability applicability",
}


@dataclass(frozen=True)
class BenchmarkView:
    summary: dict
    cells: list[dict]
    gaps: dict
    calibration: dict

    @property
    def case_ids(self) -> list[str]:
        return list(dict.fromkeys(cell["case_id"] for cell in self.cells))


def load_final_benchmark() -> BenchmarkView:
    """Read only committed final artifacts, verifying the published provenance."""
    summary = json.loads((FINAL / "analysis/analysis_summary.json").read_text())
    reconciled = (FINAL / "reconciled.jsonl").read_bytes()
    manifest = (FINAL / "reconciliation_manifest.json").read_bytes()
    for data, key in ((reconciled, "reconciled_sha256"), (manifest, "manifest_sha256")):
        if hashlib.sha256(data).hexdigest() != summary["provenance"][key]:
            raise ValueError("Final benchmark provenance mismatch; results were not loaded.")
    artifacts = {}
    for name in ("objective_gaps.csv", "calibration_metrics.json"):
        data = (FINAL / "analysis" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != summary["output_sha256"][name]:
            raise ValueError("Phase 10 artifact mismatch; results were not loaded.")
        artifacts[name] = data.decode()
    cells = [json.loads(line) for line in reconciled.splitlines() if line.strip()]
    gaps = {
        (row["case_id"], row["architecture"]): row
        for row in csv.DictReader(artifacts["objective_gaps.csv"].splitlines())
    }
    return BenchmarkView(summary, cells, gaps, json.loads(artifacts["calibration_metrics.json"]))


def _ratio(row: dict, numerator: str, denominator: str) -> str:
    return f"{row[numerator]}/{row[denominator]}"


def architecture_summary_rows(view: BenchmarkView) -> list[dict]:
    """Preserve Phase 10 order and denominators, including failures/abstentions."""
    return [
        {
            "Architecture": row["architecture"],
            "Correct outcome": _ratio(row, "outcome_correct", "outcome_denominator"),
            "Usable output": _ratio(row, "usable_output", "usable_output_denominator"),
            "Canonical valid (feasible)": _ratio(row, "feasible_canonical_valid", "feasible_denominator"),
            "Correct infeasible": _ratio(row, "infeasible_correct", "infeasible_denominator"),
            "Structural match": _ratio(row, "structural_formulation_match", "formulation_denominator"),
            "Full match": _ratio(row, "full_formulation_match", "formulation_denominator"),
            "Optimum / valid feasible": _ratio(row, "optimal", "optimal_denominator"),
            "Terminal failures": row["terminal_failure"],
            "Abstentions": row["abstention_no_usable_output"],
            "Median latency (s)": round(row["latency_median_s"], 3),
            "Median OpenAI cost ($)": f'{row["openai_cost_median_usd"]:.7f}',
            "Unknown usage cells": row["cost_unknown_usage_cells"],
        }
        for row in view.summary["architecture_summary"]
    ]


def _truth(value: Any) -> str:
    return "N/A" if value is None else "Yes" if value else "No"


def case_outcome_rows(view: BenchmarkView, case_id: str) -> list[dict]:
    rows = []
    order = {row["architecture"]: i for i, row in enumerate(view.summary["architecture_summary"])}
    for cell in sorted((c for c in view.cells if c["case_id"] == case_id), key=lambda c: order[c["architecture"]]):
        evaluation = cell["canonical_evaluation"]
        feasible = evaluation["canonical_feasible"]
        terminal = cell["status"] == "terminal_failure"
        usable = evaluation["valid_output"]
        status = (
            "🔴 Terminal failure" if terminal else "🟠 Abstention" if not usable
            else "🟢 Correct outcome" if (evaluation["canonical_validation_valid"] if feasible else evaluation["feasible_correctly_reported"])
            else "⚠️ Incorrect outcome"
        )
        gap = view.gaps[(case_id, cell["architecture"])]
        telemetry = cell["standalone_architecture_telemetry_estimate"]
        rows.append({
            "Architecture": cell["architecture"],
            "Status": status,
            "Correct outcome": _truth(evaluation["canonical_validation_valid"] if feasible else evaluation["feasible_correctly_reported"]),
            "Usable output": _truth(usable),
            "Canonical validity": _truth(evaluation["canonical_validation_valid"]) if feasible else "N/A (infeasible case)",
            "Structural match": _truth(evaluation["formulation_match_structural"]),
            "Full match": _truth(evaluation["formulation_match"]),
            "Canonical optimum attained": _truth(gap["optimal"] == "True") if feasible else "N/A",
            "Canonical objective": gap["objective_value"] or "N/A",
            "Canonical optimum": gap["optimal_objective"] if feasible else "N/A",
            "Objective gap": gap["objective_gap"] or "N/A",
            "Latency (s)": round(telemetry["latency_total_s"], 3),
            "Estimated OpenAI cost ($)": f'{telemetry["estimated_cost"]:.7f}',
            "Failure stage / category": " / ".join(filter(None, (cell.get("failure_stage"), cell.get("failure_category")))) or "—",
        })
    return rows


def jev_decision_rows(decisions: list) -> list[dict]:
    rows = []
    for decision in decisions:
        d = decision.model_dump() if hasattr(decision, "model_dump") else decision
        # Persisted locators/context identify targets without inventing semantics.
        locator = d.get("locator") or {}
        target = locator.get("signature", locator)
        target = {k: v for k, v in target.items() if k not in {"weight", "kind", "domain"} and v is not None}
        rows.append({
            "Question type": QUESTION_LABELS.get(d["question_type"], d["question_type"]),
            "Item / semantic target": ", ".join(f"{k}: {v}" for k, v in target.items()) or d["item"],
            "GPT baseline": str(d["gpt_value"]),
            "Jev proposal": str(d["jev_value"]),
            "Selected probability": d["confidence"],
            "Applied?": _truth(d["applied"]),
            "Final represented value": str(d["final_value"]) if d.get("final_value") is not None else "Not represented / not recorded",
            "Changed formulation?": _truth(d["changed"]),
        })
    return rows


def runtime_summary(run: Any) -> dict:
    """Do not equate validator acceptance of an empty verdict with feasibility proof."""
    solution, telemetry = run.solution, run.telemetry
    feasible = solution.status.value in {"optimal", "feasible"}
    valid = bool(run.validation.valid)
    return {
        "Architecture": telemetry.architecture if telemetry else "Selected architecture",
        "Execution": "Completed" if telemetry and telemetry.success else "Not recorded",
        "Solve status": solution.status.value.capitalize(),
        "Usable result": "Yes" if feasible and valid else "No feasible schedule" if solution.status.value == "infeasible" else "No",
        "Runtime hard constraints": ("Passed" if valid else "Failed") if feasible else "N/A (no feasible schedule)",
        "Objective": solution.objective_value if feasible and valid else "N/A",
        "Optimal for interpreted problem": "Yes (CP-SAT proven)" if feasible and valid and solution.optimal and telemetry and telemetry.solution_engine == "cp_sat" else "Not established",
        "Total pipeline latency (s)": round(telemetry.latency_total_s, 3) if telemetry else "N/A",
        "Estimated OpenAI cost ($)": f"{telemetry.estimated_cost:.7f}" if telemetry else "N/A",
        "Model calls": telemetry.n_model_calls if telemetry else "N/A",
        **({"Jev questions / calls": f"{telemetry.n_jev_questions} / {telemetry.n_jev_calls}"} if telemetry and telemetry.use_jev else {}),
    }


def research_findings(view: BenchmarkView) -> dict[str, str]:
    """Concise presentation of finalized findings; no new calculations or claims."""
    model = next(r for r in view.summary["paired_comparisons"]["model"] if r["scope"] == "overall")
    engine = next(r for r in view.summary["paired_comparisons"]["engine"] if r["scope"] == "overall")
    coverage = {r["model"]: r for r in view.summary["extraction_coverage"]}
    return {
        "Base model": (
            f'Sol 6.1: {model["after_outcome_correct"]}/56 correct cells; Luna: {model["before_outcome_correct"]}/56. '
            f'Typed extraction coverage: {coverage["gpt-6.1-sol"]["typed_extractions"]}/14 versus '
            f'{coverage["gpt-6-luna"]["typed_extractions"]}/14. '
            "Descriptive results from one extraction sample per case/model; extraction coverage was an end-to-end bottleneck."
        ),
        "Solution engine": (
            f'Across {engine["pairs"]} matched pairs, Direct LLM: {engine["before_outcome_correct"]} correct outcomes; '
            f'CP-SAT: {engine["after_outcome_correct"]}. CP-SAT improved five and worsened zero; all five improvements involved Luna. '
            "Every valid feasible CP-SAT result attained the canonical optimum (42/42). "
            "With Sol 6.1, Direct and CP-SAT matched on outcome correctness and optimum attainment."
        ),
        "Jev: negative / null result": (
            "125 primary canonical labels; 223 aligned observed decisions of 250 model-label opportunities; "
            "27 missing due to extraction/coverage. GPT baseline: 223/223 correct; raw Jev: 208/223; "
            "thresholded/final represented: 216/223. Raw errors occurred only in availability questions. "
            "Eight genuine formulation changes reduced structural correctness; none fixed a case-level outcome. "
            "Threshold fallback retained eight correct GPT answers that raw Jev would have replaced incorrectly. "
            "Two improved and one worsened paired outcomes occurred without formulation changes, reflecting independent Direct solve variation. "
            "Generated Workforce availability labels dominate pooled decision counts."
        ),
    }
