import csv
import json

import pytest

from decision_optimizer.dayplan import DayPlan, solve_day_plan
from scripts.label_jev_fixtures import (
    DEFAULT_SELECTION,
    DEFAULT_OUTPUT,
    DEFAULT_REVIEW_OUTPUT,
    build_draft_answer_key,
    is_primary_scoring_eligible,
    load_selected_cases,
    write_review_tsv,
)


def test_benchmark_selection_has_planned_composition_and_infeasible_cases():
    cases = load_selected_cases()
    dayplans = [case for case in cases if case["domain"] == "dayplan"]
    workforce = [case for case in cases if case["domain"] == "shift_schedule"]

    assert len(cases) == 14
    assert len(dayplans) == 9
    assert len(workforce) == 5
    assert sum(case.get("generated", False) for case in workforce) == 2
    assert any(case.get("infeasible") for case in dayplans)
    assert any(case.get("infeasible") for case in workforce)
    assert {
        "workout_before_four_soft",
        "workout_before_four_hard",
        "optional_walk_if_time",
    } <= {case["name"] for case in dayplans}


def test_new_dayplan_cases_are_typed_and_declared_infeasible_case_is_infeasible():
    cases = load_selected_cases()
    new_cases = [case for case in cases if case["source"] == "dayplan_jev_ambiguous.json"]
    for case in new_cases:
        DayPlan.model_validate(case["canonical"])

    impossible = next(
        case for case in new_cases if case["name"] == "impossible_hard_workout_window"
    )
    assert solve_day_plan(DayPlan.model_validate(impossible["canonical"])).status.value == "infeasible"


def test_labeling_helper_separates_canonical_proposals_from_observations():
    draft = build_draft_answer_key(load_selected_cases(), baseline="fixture")

    assert draft["label_status"] == "draft_not_human_approved"
    assert "not human-reviewed" in draft["warning"]
    assert len(draft["cases"]) == 14
    labels = [label for case in draft["cases"] for label in case["expected_jev"]]
    assert labels
    assert all("canonical_expected_answer" in label for label in labels)
    assert all("gpt_value" not in label for label in labels)
    assert all(label["observations"] == {} for label in labels)
    generated = [
        label
        for case in draft["cases"]
        if case["generated"]
        for label in case["expected_jev"]
    ]
    reviewed_queue = [
        label
        for case in draft["cases"]
        if not case["generated"]
        for label in case["expected_jev"]
    ]
    assert generated
    assert all(
        label["review_status"] == "generator_derived_not_human_reviewed"
        and label["label_provenance"] == "generator_specification"
        for label in generated
    )
    assert all(
        label["review_status"] == "needs_human_review"
        for label in reviewed_queue
    )
    assert all(
        case["expected_jev_review_status"]
        in {"needs_human_review", "generator_derived_not_human_reviewed"}
        for case in draft["cases"]
    )


def test_semantic_alignment_keeps_gpt_disagreement_and_coverage_explicit():
    case = next(
        item
        for item in load_selected_cases()
        if item["name"] == "workout_before_four_hard"
    )
    extracted = DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "18:00"},
            "tasks": [
                {
                    "name": "workout session",
                    "duration_min": 60,
                    "mode": "active",
                    "required": True,
                }
            ],
            "preferences": [
                {
                    "type": "finish_before",
                    "task": "workout session",
                    "time": "16:00",
                    "weight": 2,
                }
            ],
        }
    )

    draft = build_draft_answer_key(
        [case],
        baseline="gpt",
        model="gpt-6-luna",
        dayplan_extractor=lambda _: extracted,
    )
    output_case = draft["cases"][0]
    hardness = next(
        label
        for label in output_case["expected_jev"]
        if label["question_type"] == "constraint_hardness"
    )

    assert hardness["canonical_expected_answer"] == "hard"
    assert hardness["observations"]["gpt-6-luna"]["gpt_baseline_answer"] == "soft"
    assert hardness["observations"]["gpt-6-luna"]["alignment_status"] == "aligned"
    assert output_case["observation_coverage"]["gpt-6-luna"] == {
        "canonical_questions": 3,
        "aligned_questions": 3,
        "not_generated": 0,
        "unaligned_generated": 1,
    }
    assert output_case["unaligned_observed_questions"]["gpt-6-luna"][0][
        "question_type"
    ] == "pref_weight"


def test_review_tsv_contains_provenance_context_and_status(tmp_path):
    draft = build_draft_answer_key(load_selected_cases(), baseline="fixture")
    output = write_review_tsv(draft, tmp_path / "review.tsv")
    rows = output.read_text().splitlines()

    assert rows[0].startswith("case\tdomain\tquestion_type\tquestion_id")
    assert "supporting_request_context" in rows[0]
    assert "label_provenance" in rows[0]
    assert "review_status" in rows[0]
    assert len(rows) == 1 + sum(
        len(case["expected_jev"]) for case in draft["cases"]
    )


def test_selection_manifest_points_to_completed_review_without_changing_cases():
    manifest = json.loads(DEFAULT_SELECTION.read_text())
    key = json.loads(DEFAULT_OUTPUT.read_text())

    assert manifest["label_status"] == "review_complete_with_subjective_exclusions"
    assert manifest["answer_key_file"] == DEFAULT_OUTPUT.name
    assert manifest["review_summary"] == key["review_summary"]
    assert [c["name"] for c in manifest["cases"]] == [c["name"] for c in key["cases"]]


def test_completed_answer_key_has_exact_review_counts_and_unchanged_answers():
    from collections import Counter

    key = json.loads(DEFAULT_OUTPUT.read_text())
    labels = [label for case in key["cases"] for label in case["expected_jev"]]
    assert len(labels) == 128
    assert Counter(label["review_status"] for label in labels) == {
        "human_reviewed_approved": 54,
        "generator_derived": 71,
        "subjective_excluded_from_primary_scoring": 3,
    }
    assert sum(is_primary_scoring_eligible(label) for label in labels) == 125
    assert key["review_summary"] == {
        "human_reviewed_approved": 54,
        "generator_derived_scored": 71,
        "subjective_excluded": 3,
        "primary_scored_total": 125,
        "labels_total": 128,
    }
    approved_by_case = {
        "wfh_laundry_lift_groceries": 15, "active_and_passive": 4,
        "laundry_precedence": 6, "hard_timing": 5,
        "soft_finish_preference": 3, "workout_before_four_soft": 3,
        "workout_before_four_hard": 3, "optional_walk_if_time": 4,
        "impossible_hard_workout_window": 3, "straightforward_staffing": 2,
        "preferences_and_availability": 2,
        "infeasible_staffing_is_extracted_not_repaired": 4,
    }
    for case in key["cases"]:
        if case["generated"]:
            assert all(label["label_provenance"] == "generator_derived"
                       and label["review_status"] == "generator_derived"
                       and is_primary_scoring_eligible(label)
                       for label in case["expected_jev"])
        else:
            approved = [label for label in case["expected_jev"]
                        if label["review_status"] == "human_reviewed_approved"]
            assert len(approved) == approved_by_case[case["name"]]
            assert all(label["label_provenance"] == "fixture_intended_meaning"
                       and is_primary_scoring_eligible(label) for label in approved)
    proposals = build_draft_answer_key(load_selected_cases(), baseline="fixture")
    def answers(artifact):
        return {(case["name"], label["question_id"]): label["canonical_expected_answer"]
                for case in artifact["cases"] for label in case["expected_jev"]}
    assert answers(key) == answers(proposals)
    # A newly generated draft never inherits approval merely from the manifest.
    assert not any(is_primary_scoring_eligible(label)
                   for case in proposals["cases"] for label in case["expected_jev"])


@pytest.mark.parametrize("case_name,weight", [
    ("soft_finish_preference", 1),
    ("workout_before_four_soft", 3),
    ("preferences_and_availability", 1),
])
def test_subjective_labels_cannot_enter_primary_scoring(case_name, weight):
    key = json.loads(DEFAULT_OUTPUT.read_text())
    case = next(case for case in key["cases"] if case["name"] == case_name)
    label = next(label for label in case["expected_jev"] if label["question_type"] == "pref_weight")
    assert label["canonical_expected_answer"] == weight
    assert label["review_status"] == "subjective_excluded_from_primary_scoring"
    assert label["primary_accuracy_calibration_eligible"] is False
    assert label["primary_exclusion_reason"] == "subjective_weight_without_agreed_rubric"
    assert not is_primary_scoring_eligible(label)
    # Either an accidental eligibility toggle or status edit remains excluded.
    assert not is_primary_scoring_eligible({**label, "primary_accuracy_calibration_eligible": True})
    assert not is_primary_scoring_eligible({**label, "review_status": "human_reviewed_approved"})
    assert not is_primary_scoring_eligible({**label, "primary_accuracy_calibration_eligible": True,
                                          "review_status": "human_reviewed_approved"})


def test_reviewed_tsv_matches_json_status_answers_and_eligibility():
    key = json.loads(DEFAULT_OUTPUT.read_text())
    labels = {(case["name"], label["question_id"]): label
              for case in key["cases"] for label in case["expected_jev"]}
    with DEFAULT_REVIEW_OUTPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    assert len(rows) == len(labels) == 128
    assert len({(row["case"], row["question_id"]) for row in rows}) == 128
    for row in rows:
        label = labels[(row["case"], row["question_id"])]
        assert json.loads(row["canonical_expected_answer"]) == label["canonical_expected_answer"]
        assert row["review_status"] == label["review_status"]
        assert row["label_provenance"] == label["label_provenance"]
        assert row["primary_accuracy_calibration_eligible"] == str(label["primary_accuracy_calibration_eligible"])
        assert row["primary_exclusion_reason"] == (label["primary_exclusion_reason"] or "")
    assert sum(row["primary_accuracy_calibration_eligible"] == "True" for row in rows) == 125
