import json

from decision_optimizer.dayplan import DayPlan, solve_day_plan
from scripts.label_jev_fixtures import (
    DEFAULT_SELECTION,
    build_draft_answer_key,
    load_selected_cases,
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


def test_labeling_helper_emits_only_explicitly_unreviewed_draft_labels():
    draft = build_draft_answer_key(load_selected_cases(), baseline="fixture")

    assert draft["label_status"] == "draft_needs_human_review"
    assert "not human ground truth" in draft["warning"]
    assert len(draft["cases"]) == 14
    labels = [label for case in draft["cases"] for label in case["expected_jev"]]
    assert labels
    assert all(label["review_status"] == "needs_human_review" for label in labels)
    assert all(
        label["proposed_expected_jev"] == label["gpt_value"] for label in labels
    )
    assert all(
        case["expected_jev_review_status"] == "needs_human_review"
        for case in draft["cases"]
    )


def test_selection_manifest_itself_does_not_claim_reviewed_labels():
    manifest = json.loads(DEFAULT_SELECTION.read_text())

    assert manifest["label_status"] == "draft_needs_human_review"
    assert "human-reviewed" in manifest["notes"]

