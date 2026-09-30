import json

from decision_optimizer.dayplan import DayPlan, solve_day_plan
from scripts.label_jev_fixtures import (
    DEFAULT_SELECTION,
    build_draft_answer_key,
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


def test_selection_manifest_itself_does_not_claim_reviewed_labels():
    manifest = json.loads(DEFAULT_SELECTION.read_text())

    assert manifest["label_status"] == "draft_needs_human_review"
    assert "human-reviewed" in manifest["notes"]
