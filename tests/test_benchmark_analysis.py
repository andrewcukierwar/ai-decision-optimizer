"""Hand-checkable statistics and offline regressions for the frozen Phase 10 inputs."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from scripts import analyze_benchmark as analysis


@pytest.fixture(scope="module")
def inputs():
    return analysis.load_inputs()


@pytest.fixture(scope="module")
def computed(inputs):
    return analysis.analyze(*inputs)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    output = tmp_path_factory.mktemp("phase10")
    frozen = analysis.immutable_snapshot(analysis.RESULTS)
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("provider/benchmark execution and network are forbidden")
        # An independent guard even if an accidental future import is added.
        import decision_optimizer.benchmark as benchmark
        import openai
        import typesafe_sdk
        patch.setattr(socket.socket, "connect", forbidden)
        patch.setattr(benchmark.BenchmarkRunner, "run", forbidden)
        for name in ("extract", "decide", "solve"):
            patch.setattr(benchmark.LiveBenchmarkBackend, name, forbidden)
        patch.setattr(openai, "OpenAI", forbidden)
        patch.setattr(typesafe_sdk, "TypeSafeClient", forbidden)
        result = analysis.run_analysis(output_dir=output / "analysis", plot_path=output / "calibration.png")
        first = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        assert analysis.run_analysis(output_dir=output / "analysis", plot_path=output / "calibration.png") == result
        assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == first
    assert all(analysis.sha256(path) == digest for path, digest in frozen.items())
    return output, result


def test_eight_architectures_consume_all_112_observations(inputs, computed):
    rows, _, _ = inputs
    summary = computed[0]
    assert len(rows) == len({analysis.cell_key(r) for r in rows}) == 112
    assert summary["observations_consumed"] == 112
    assert len(summary["architecture_summary"]) == 8
    assert [r["outcome_correct"] for r in summary["architecture_summary"]] == [8, 11, 9, 11, 13, 13, 13, 13]
    for row in summary["architecture_summary"]:
        assert row["cases"] == row["outcome_denominator"] == row["usable_output_denominator"] == 14
        assert row["feasible_denominator"] == 12 and row["infeasible_denominator"] == 2
        assert row["cost_denominator"] == row["latency_denominator"] == row["formulation_denominator"] == 14


def test_failures_and_abstentions_stay_in_all_relevant_denominators(computed):
    summary = computed[0]
    arch = summary["architecture_summary"]
    assert sum(r["terminal_failure"] for r in arch) == 9
    assert sum(r["abstention_no_usable_output"] for r in arch) == 8
    assert sum(r["usable_output"] for r in arch) == 95
    assert sum(r["outcome_denominator"] for r in arch) == 112
    assert len(summary["failures_and_abstentions"]) == 17
    for row in arch:
        assert row["usable_output"] + row["terminal_failure"] + row["abstention_no_usable_output"] == 14
    assert summary["extraction_coverage"] == [
        {"model": "gpt-6-luna", "case_model_stages": 14, "typed_extractions": 11,
         "terminal_extraction_failure": 2, "clarification_null_formulation": 1},
        {"model": "gpt-6.1-sol", "case_model_stages": 14, "typed_extractions": 13,
         "terminal_extraction_failure": 0, "clarification_null_formulation": 1}]


def test_outcome_correctness_uses_canonical_feasibility_not_constraint_fraction():
    feasible = {"canonical_evaluation": {"canonical_feasible": True,
                "canonical_validation_valid": False, "feasible_correctly_reported": True,
                "hard_constraints_satisfied": 99, "hard_constraints_total": 100}}
    assert not analysis.outcome_correct(feasible)
    feasible["canonical_evaluation"]["canonical_feasible"] = False
    assert analysis.outcome_correct(feasible)
    feasible["canonical_evaluation"]["feasible_correctly_reported"] = False
    assert not analysis.outcome_correct(feasible)


def test_frozen_125_label_policy_250_instances_223_aligned_27_missing(inputs, computed):
    key = inputs[1]
    labels = [l for c in key["cases"] for l in c["expected_jev"]]
    assert len(labels) == 128 and sum(analysis.primary_eligible(l) for l in labels) == 125
    assert Counter(l["review_status"] for l in labels) == {
        "human_reviewed_approved": 54, "generator_derived": 71,
        "subjective_excluded_from_primary_scoring": 3}
    instances = computed[3]
    primary = [r for r in instances if r["primary_eligible"]]
    assert len(primary) == 250
    assert Counter(r["status"] for r in primary) == {"aligned": 223, "not_generated": 27}
    assert Counter(r["model"] for r in primary if r["status"] == "not_generated") == {
        "gpt-6-luna": 23, "gpt-6.1-sol": 4}
    accuracy = next(r for r in computed[1] if r["scope"] == "overall")
    assert [accuracy[w + "_aligned_correct"] for w in ("gpt", "jev", "final")] == [223, 208, 216]
    assert all(accuracy[w + "_end_to_end_denominator"] == 250 for w in ("gpt", "jev", "final"))
    assert accuracy["gpt_end_to_end_accuracy"] == pytest.approx(223 / 250)


def test_subjective_exclusions_never_enter_accuracy_or_calibration(inputs, computed):
    rows, key, manifest = inputs
    changed = deepcopy(key)
    excluded = [l for c in changed["cases"] for l in c["expected_jev"]
                if l["review_status"] == "subjective_excluded_from_primary_scoring"]
    assert len(excluded) == 3
    for label in excluded:
        label["canonical_expected_answer"] = 999
        label["primary_accuracy_calibration_eligible"] = True
        label["review_status"] = "human_reviewed_approved"
        assert label["primary_exclusion_reason"]
        assert not analysis.primary_eligible(label)
    altered = analysis.analyze(rows, changed, manifest)
    assert altered[1] == computed[1]  # All primary accuracy rows identical.
    assert altered[5] == computed[5]  # All calibration identical.
    assert computed[5]["groups"]["pref_weight"]["n"] == 10


def test_unaligned_questions_never_receive_labels(computed):
    unaligned = [r for r in computed[3] if r["status"] == "unaligned_generated"]
    assert len(unaligned) == 2
    assert all(r["expected"] is None and r["canonical_question_id"] is None
               and not r["primary_eligible"] and "gpt_correct" not in r for r in unaligned)
    assert {r["case_id"] for r in unaligned} == {"active_and_passive"}
    # 63 Choice + 150 Noul + 10 Score = 223, without the two extra hardness questions.
    groups = computed[5]["groups"]
    assert groups["constraint_hardness"]["n"] == 11
    assert groups["pooled_choice"]["n"] + groups["availability_applies"]["n"] + groups["pref_weight"]["n"] == 223


@pytest.mark.parametrize("expected, binary, full", [("yes", .09, .18), ("no", .49, .98)])
def test_binary_brier_hand_examples(expected, binary, full):
    distribution = {"yes": .7, "no": .3}
    assert analysis.binary_brier(distribution, expected) == pytest.approx(binary)
    assert analysis.multiclass_brier(distribution, expected) == pytest.approx(full)
    assert (distribution["yes"] - int(expected == "yes")) ** 2 == pytest.approx(binary)


def test_five_class_brier_and_score_mapping_hand_examples():
    sdk = {"0": .6, "1": 0, "2": 0, "3": 0, "4": .4}
    mapped = analysis.score_probabilities(sdk)
    assert mapped == {1: .6, 2: 0, 3: 0, 4: 0, 5: .4}
    assert analysis.multiclass_brier(mapped, 1) == pytest.approx(.32)
    assert analysis.multiclass_brier(mapped, 5) == pytest.approx(.72)
    assert analysis.multiclass_brier({i: .2 for i in range(1, 6)}, 3) == pytest.approx(.8)
    onehot = {i: int(i == 1) for i in range(1, 6)}
    assert analysis.multiclass_brier(onehot, 1) == 0
    assert analysis.multiclass_brier(onehot, 5) == 2


def test_ranked_probability_score_hand_examples():
    onehot = {i: int(i == 1) for i in range(1, 6)}
    assert analysis.ranked_probability_score(onehot, 1) == 0
    assert analysis.ranked_probability_score(onehot, 5) == 1
    assert analysis.ranked_probability_score({1: .6, 2: 0, 3: 0, 4: 0, 5: .4}, 5) == pytest.approx(.36)
    assert analysis.ranked_probability_score({i: .2 for i in range(1, 6)}, 3) == pytest.approx(.1)


@pytest.mark.parametrize("probabilities", [
    {"0": 1}, {"0": .5, "1": .5, "2": 0, "3": 0, "5": 0},
    {str(i): .3 for i in range(5)}, {"0": float("nan"), **{str(i): 0 for i in range(1, 5)}}])
def test_bad_score_distributions_fail_closed(probabilities):
    with pytest.raises(ValueError):
        analysis.score_probabilities(probabilities)


def test_final_decision_scores_threshold_fallback_and_missing_representation(inputs):
    rows, key, _ = deepcopy(inputs)
    row = next(r for r in rows if r["case_id"] == "straightforward_staffing"
               and r["model"] == "gpt-6-luna" and r["configuration"]["use_jev"]
               and r["configuration"]["solution_engine"] == "cp_sat")
    o = row["question_alignment"]["observed_questions"][0]
    assert o["gpt_baseline_answer"] is True
    o.update(jev_answer=False, jev_probability=.6, applied=False, changed=False,
             probabilities={"yes": .4, "no": .6}, final_applied_answer=True,
             final_value_status="represented")
    instance = next(r for r in analysis.decision_instances(rows, key)
                    if r["case_id"] == row["case_id"] and r["model"] == row["model"]
                    and r["canonical_question_id"] == o["canonical_question_id"])
    assert instance["gpt_correct"] and not instance["jev_correct"] and instance["final_correct"]
    o.update(jev_answer=True, applied=True, final_applied_answer=None, final_value_status="not_represented")
    instance = next(r for r in analysis.decision_instances(rows, key)
                    if r["case_id"] == row["case_id"] and r["model"] == row["model"]
                    and r["canonical_question_id"] == o["canonical_question_id"])
    assert instance["jev_correct"] and not instance["final_correct"]


def test_reliability_five_bins_boundaries_and_empty_bins():
    samples = [{"confidence": confidence, "correct": confidence == 1}
               for confidence in (0, .2, .4, .6, .8, 1)]
    bins = analysis.reliability_bins(samples)
    assert [b["n"] for b in bins] == [1, 1, 1, 1, 2]
    assert bins[-1]["mean_selected_confidence"] == pytest.approx(.9)
    assert bins[-1]["empirical_correctness"] == .5
    assert all(b["mean_selected_confidence"] is None for b in analysis.reliability_bins([]))


@pytest.mark.parametrize("dimension", ["jev", "engine", "model"])
def test_pair_construction_is_exact_and_retains_failed_stages(inputs, dimension):
    rows = inputs[0]
    pairs = analysis.build_pairs(rows, dimension)
    assert len(pairs) == 56
    position = {"jev": 2, "engine": 3, "model": 1}[dimension]
    for a, b in pairs:
        keys = analysis.cell_key(a), analysis.cell_key(b)
        assert keys[0][position] != keys[1][position]
        assert all(keys[0][i] == keys[1][i] for i in range(4) if i != position)
    assert any(a["status"] == "terminal_failure" or b["status"] == "terminal_failure" for a,b in pairs)
    assert len({analysis.cell_key(r) for pair in pairs for r in pair}) == 112
    with pytest.raises(ValueError, match="missing paired cell"):
        analysis.build_pairs(rows[1:], dimension)


def test_jev_effects_distinguish_formulation_edits_from_direct_variation(computed):
    pairs = computed[4]["jev"]
    assert Counter(p["formulation_change_category"] for p in pairs) == {
        "typed_formulation_changed": 8, "asked_no_applied_formulation_change": 40,
        "unavailable_formulation": 8}
    assert Counter(p["structural_effect"] for p in pairs) == {"same": 48, "worsened": 8}
    fixes = [p for p in pairs if p["outcome_effect"] == "improved"]
    breaks = [p for p in pairs if p["outcome_effect"] == "worsened"]
    assert {p["case_id"] for p in fixes} == {"hard_timing", "ten_day_rotation"}
    assert [p["case_id"] for p in breaks] == ["workout_before_four_hard"]
    assert all(p["model"] == "gpt-6-luna" and p["engine"] == "direct_llm"
               and p["typed_formulation_changed"] is False for p in fixes + breaks)
    assert all(p["outcome_effect"] == "same" for p in pairs if p["typed_formulation_changed"])
    assert sum(p["objective_effect"] == "improved" for p in pairs) == 1


def test_jev_applied_edits_with_zero_net_change_are_a_distinct_category(inputs):
    a, b = deepcopy(analysis.build_pairs(inputs[0], "jev")[0])
    assert a["final_problem"] == b["final_problem"] and b["jev_decisions"]
    b["jev_decisions"][0]["changed"] = True
    assert analysis.pair_record(a,b,"jev")["formulation_change_category"] == "applied_edit_no_net_typed_change"


def test_engine_comparison_and_objective_denominators(computed):
    s = next(s for s in computed[0]["paired_comparisons"]["engine"] if s["scope"] == "overall")
    assert s["pairs"] == 56 and s["feasible_pair_denominator"] == 48
    assert (s["before_outcome_correct"],s["after_outcome_correct"]) == (43,48)
    assert s["outcome_effects"] == {"improved": 5, "same": 51}
    assert (s["before_feasible_valid"], s["after_feasible_valid"]) == (37,42)
    assert (s["before_optimal"],s["after_optimal"]) == (34,42)
    assert s["objective_effects"] == {"improved": 3, "not_comparable": 19, "same": 34}


def test_standalone_telemetry_and_actual_expenditure_are_separate(inputs, computed):
    rows, _, manifest = inputs
    summary = computed[0]
    for r in summary["architecture_summary"]:
        group = [x["standalone_architecture_telemetry_estimate"] for x in rows if x["architecture"] == r["architecture"]]
        assert r["openai_cost_mean_usd"] == pytest.approx(sum(t["estimated_cost"] for t in group)/14)
        assert r["latency_mean_s"] == pytest.approx(sum(t["latency_total_s"] for t in group)/14)
    assert summary["experiment_accounting"] == manifest["experiment_totals"]
    assert summary["experiment_accounting"]["openai_estimated_cost_usd"] == .2874103
    assert summary["experiment_accounting"]["openai_usage_unknown_calls"] == 2
    assert summary["experiment_accounting"]["jev_cost_usd"] is None
    assert summary["jev_stage_latency"]["completed_calls"] == 24


def test_all_artifacts_reproduce_byte_for_byte_without_network_or_raw_modifications(artifacts):
    output, summary = artifacts
    assert len(list((output/"analysis").iterdir())) == 11
    assert (output/"calibration.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert summary["provenance"]["reconciled_sha256"] == analysis.sha256(analysis.RESULTS/"reconciled.jsonl")
    assert summary["provenance"]["answer_key_sha256"] == analysis.sha256(analysis.ANSWER_KEY)
    for name, digest in summary["output_sha256"].items():
        path = output/name if name == "calibration.png" else output/"analysis"/name
        assert analysis.sha256(path) == digest


def test_output_overlapping_frozen_artifacts_is_rejected_before_writing(tmp_path):
    before = analysis.sha256(analysis.RESULTS/"reconciled.jsonl")
    with pytest.raises(ValueError, match="overlaps frozen input"):
        analysis.run_analysis(output_dir=tmp_path, plot_path=analysis.RESULTS/"reconciled.jsonl")
    assert analysis.sha256(analysis.RESULTS/"reconciled.jsonl") == before
    with pytest.raises(ValueError, match="inside analysis directory"):
        analysis.run_analysis(output_dir=analysis.RESULTS/"cache", plot_path=tmp_path/"calibration.png")


def test_symlink_output_cannot_overwrite_frozen_input(tmp_path):
    (tmp_path/"analysis_summary.json").symlink_to(analysis.ANSWER_KEY)
    with pytest.raises(ValueError, match="overlaps frozen input"):
        analysis.run_analysis(output_dir=tmp_path, plot_path=tmp_path/"calibration.png")


def test_changed_reconciled_input_stops_before_any_output(tmp_path, monkeypatch):
    original = analysis.sha256
    monkeypatch.setattr(analysis, "sha256", lambda p: "changed" if Path(p).name == "reconciled.jsonl" else original(p))
    with pytest.raises(ValueError, match="reconciled hash differs"):
        analysis.run_analysis(output_dir=tmp_path/"must_not_exist", plot_path=tmp_path/"plot.png")
    assert not (tmp_path/"must_not_exist").exists() and not (tmp_path/"plot.png").exists()


def test_analysis_module_imports_no_provider_execution_code():
    import ast
    source = Path(analysis.__file__).read_text()
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = [node.module for node in imports if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.startswith(("openai", "typesafe", "decision_optimizer", "scripts.run_benchmark", "scripts.label_jev")) for name in names)
