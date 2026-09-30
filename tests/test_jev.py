from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, Noul, Score

from decision_optimizer.application import (
    run_dayplan_experiment,
    run_shift_schedule_experiment,
)
from decision_optimizer.dayplan import DayPlan
from decision_optimizer.direct_solver import (
    DirectDayAssignment,
    DirectDayPlanSolution,
)
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.jev import (
    JEV_QUESTION_TYPES,
    apply_jev,
    build_jev_questions,
    score_to_weight,
)
from decision_optimizer.parsing.dayplan import DayPlanExtraction
from decision_optimizer.parsing.shift_schedule import ShiftScheduleExtraction
from decision_optimizer.shift_schedule import ShiftSchedule
from decision_optimizer.telemetry import RunTelemetry


def sample_plan() -> DayPlan:
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "18:00"},
            "tasks": [
                {
                    "name": "workout",
                    "duration_min": 60,
                    "mode": "active",
                    "latest_end": "16:00",
                    "required": True,
                },
                {
                    "name": "report",
                    "duration_min": 45,
                    "mode": "active",
                    "required": True,
                },
                {
                    "name": "dishwasher",
                    "duration_min": 30,
                    "mode": "passive",
                    "required": False,
                },
            ],
            "preferences": [
                {
                    "type": "finish_before",
                    "task": "report",
                    "time": "15:00",
                    "weight": 1,
                },
                {"type": "minimize_work_interruptions", "weight": 2},
            ],
        }
    )


def sample_schedule() -> ShiftSchedule:
    return ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "s1",
                    "day": "2026-10-05",
                    "start": "09:00",
                    "end": "13:00",
                    "location": "store",
                    "required_staff": 1,
                },
                {
                    "id": "s2",
                    "day": "2026-10-05",
                    "start": "14:00",
                    "end": "18:00",
                    "location": "store",
                    "required_staff": 1,
                },
            ],
            "employees": [
                {
                    "name": "Alice",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                    "unavailable": [{"shift_id": "s1"}],
                    "preferred_shifts": ["s2"],
                },
                {
                    "name": "Bob",
                    "max_hours": 8,
                    "eligible_locations": ["store"],
                },
            ],
        }
    )


class FakeJevClient:
    def __init__(self, answer_builder=None, input_tokens=123):
        self.answer_builder = answer_builder or _baseline_answers
        self.input_tokens = input_tokens
        self.calls = []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            answers=self.answer_builder(kwargs["state"], kwargs["questions"]),
            usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=10),
        )


def _baseline_answers(state, questions):
    answers = {}
    for context, (name, question) in zip(
        state["extracted_context"], questions.items()
    ):
        gpt_value = context["gpt_value"]
        if isinstance(question, Choice):
            choices = list(question.criteria)
            answers[name] = SimpleNamespace(
                choice=gpt_value,
                confidence=0.91,
                probabilities={
                    choice: (0.91 if choice == gpt_value else 0.09)
                    for choice in choices
                },
            )
        elif isinstance(question, Score):
            expected = int(gpt_value) - 1
            answers[name] = SimpleNamespace(
                score=float(expected),
                confidence=0.91,
                probabilities={
                    level: (1.0 if level == expected else 0.0)
                    for level in range(5)
                },
            )
        else:
            answers[name] = SimpleNamespace(noul=0.91 if gpt_value else 0.09)
    return answers


def _custom_answers(changes):
    def build(state, questions):
        answers = _baseline_answers(state, questions)
        for context, (name, question) in zip(
            state["extracted_context"], questions.items()
        ):
            replacement = changes.get((context["question"], context["item"]))
            if replacement is None:
                continue
            if isinstance(question, Choice):
                selected, probability = replacement
                answers[name] = SimpleNamespace(
                    choice=selected,
                    confidence=0.99,
                    probabilities={
                        choice: (
                            probability
                            if choice == selected
                            else (1 - probability) / (len(question.criteria) - 1)
                        )
                        for choice in question.criteria
                    },
                )
            elif isinstance(question, Score):
                expected, confidence, probabilities = replacement
                answers[name] = SimpleNamespace(
                    score=expected,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            else:
                answers[name] = SimpleNamespace(noul=replacement)
        return answers

    return build


def test_registry_has_the_five_core_question_types_in_priority_order():
    assert [item.key for item in JEV_QUESTION_TYPES] == [
        "constraint_hardness",
        "task_requirement",
        "task_mode",
        "pref_weight",
        "availability_applies",
    ]


def test_question_applicability_and_one_call_batching(monkeypatch):
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-latest")
    plan = sample_plan()
    questions = build_jev_questions("Plan this day", plan)
    counts = {}
    for item in questions:
        counts[item.question_type.key] = counts.get(item.question_type.key, 0) + 1

    assert counts == {
        "constraint_hardness": 2,
        "task_requirement": 3,
        "task_mode": 3,
        "pref_weight": 2,
    }
    client = FakeJevClient()
    result = apply_jev("Plan this day", plan, client=client)

    assert len(client.calls) == 1
    assert len(client.calls[0]["questions"]) == 10
    assert client.calls[0]["model"] == "jev-latest"
    assert client.calls[0]["state"]["original_request"] == "Plan this day"
    assert result.problem == plan


def test_dayplan_rewrites_all_four_core_dayplan_judgments_and_revalidates():
    changes = {
        ("constraint_hardness", "preference:0:report"): ("hard", 0.9),
        ("constraint_hardness", "task:workout:hard_latest"): ("soft", 0.9),
        ("task_requirement", "task:dishwasher"): ("required", 0.9),
        ("task_mode", "task:dishwasher"): ("active", 0.9),
        (
            "pref_weight",
            "preference:1:minimize_work_interruptions",
        ): (3.6, 0.92, {0: 0.0, 1: 0.0, 2: 0.1, 3: 0.2, 4: 0.7}),
    }
    result = apply_jev(
        "I need the report by three and care a lot about interruptions.",
        sample_plan(),
        client=FakeJevClient(_custom_answers(changes)),
    )
    plan = DayPlan.model_validate(result.problem.model_dump(mode="python"))
    tasks = {task.name: task for task in plan.tasks}

    assert tasks["report"].latest_end.strftime("%H:%M") == "15:00"
    assert tasks["workout"].latest_end is None
    assert tasks["dishwasher"].required is True
    assert tasks["dishwasher"].mode.value == "active"
    interruption = next(
        preference
        for preference in plan.preferences
        if preference.type.value == "minimize_work_interruptions"
    )
    assert interruption.weight == 5
    assert any(
        preference.task == "workout"
        and preference.type.value == "finish_before"
        for preference in plan.preferences
    )
    score_decision = next(
        item
        for item in result.decisions
        if item.item == "preference:1:minimize_work_interruptions"
    )
    assert score_decision.expected_score == 3.6
    assert score_decision.probabilities == {
        "0": 0.0,
        "1": 0.0,
        "2": 0.1,
        "3": 0.2,
        "4": 0.7,
    }
    assert score_decision.jev_value == 5
    assert score_decision.applied and score_decision.changed


def test_choice_uses_selected_probability_for_threshold_and_preserves_gpt_value():
    client = FakeJevClient(
        _custom_answers(
            {("task_requirement", "task:dishwasher"): ("required", 0.69)}
        )
    )
    result = apply_jev("Maybe run it", sample_plan(), client=client, threshold=0.7)
    dishwasher = next(
        task for task in result.problem.tasks if task.name == "dishwasher"
    )
    decision = next(
        item
        for item in result.decisions
        if item.question_type == "task_requirement"
        and item.item == "task:dishwasher"
    )

    assert dishwasher.required is False
    assert decision.confidence == pytest.approx(0.69)
    assert decision.model_confidence == pytest.approx(0.99)
    assert decision.applied is False
    assert decision.probabilities["required"] == pytest.approx(0.69)
    assert decision.probabilities["optional"] == pytest.approx(0.31)


@pytest.mark.parametrize(
    ("expected", "weight"),
    [(0.0, 1), (0.49, 1), (0.5, 2), (2.5, 4), (4.0, 5)],
)
def test_score_to_integer_mapping_is_half_up_and_clamped(expected, weight):
    assert score_to_weight(expected) == weight


def test_score_below_confidence_threshold_keeps_existing_weight():
    changes = {
        ("pref_weight", "preference:1:minimize_work_interruptions"): (
            4.0,
            0.69,
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
        )
    }
    result = apply_jev(
        "Keep interruptions down",
        sample_plan(),
        client=FakeJevClient(_custom_answers(changes)),
        threshold=0.7,
    )
    preference = result.problem.preferences[1]
    decision = next(
        item for item in result.decisions if item.item.startswith("preference:1:")
    )

    assert preference.weight == 2
    assert decision.jev_value == 5
    assert decision.expected_score == 4.0
    assert decision.applied is False


def test_noul_confidence_rewrites_availability_and_logs_yes_probability():
    changes = {
        ("availability_applies", "employee:Alice:shift:s1"): 0.2,
        ("availability_applies", "employee:Alice:shift:s2"): 0.8,
    }
    result = apply_jev(
        "Alice has an availability statement.",
        sample_schedule(),
        client=FakeJevClient(_custom_answers(changes)),
    )
    schedule = ShiftSchedule.model_validate(result.problem.model_dump(mode="python"))
    alice = next(employee for employee in schedule.employees if employee.name == "Alice")
    unavailable = [item.shift_id for item in alice.unavailable]

    assert unavailable == ["s2"]
    s1 = next(item for item in result.decisions if item.item.endswith("shift:s1"))
    assert s1.jev_value is False
    assert s1.yes_probability == pytest.approx(0.2)
    assert s1.confidence == pytest.approx(0.8)
    assert s1.probabilities == {"no": 0.8, "yes": 0.2}
    assert s1.applied and s1.changed


def test_noul_uncertainty_does_not_turn_into_a_rewrite():
    changes = {("availability_applies", "employee:Alice:shift:s1"): 0.4}
    result = apply_jev(
        "Alice is unavailable.",
        sample_schedule(),
        client=FakeJevClient(_custom_answers(changes)),
    )
    alice = next(
        employee for employee in result.problem.employees if employee.name == "Alice"
    )
    decision = next(item for item in result.decisions if item.item.endswith("shift:s1"))

    assert [item.shift_id for item in alice.unavailable] == ["s1"]
    assert decision.confidence == pytest.approx(0.6)
    assert decision.applied is False


def test_workforce_scaling_guard_only_fans_out_flagged_employees():
    questions = build_jev_questions("Staff these shifts", sample_schedule())
    availability = [
        item for item in questions if item.question_type.key == "availability_applies"
    ]

    assert len(availability) == 2
    assert all("Alice" in item.target.item for item in availability)
    assert not any("Bob" in item.target.item for item in availability)


def test_preferred_shift_alone_is_not_availability_evidence():
    schedule = sample_schedule().model_copy(deep=True)
    alice = schedule.employees[0].model_copy(update={"unavailable": []})
    schedule.employees[0] = alice

    availability = [
        item
        for item in build_jev_questions("Alice prefers s2.", schedule)
        if item.question_type.key == "availability_applies"
    ]

    assert availability == []


def test_availability_targets_all_covered_shifts_and_two_deterministic_controls():
    schedule = ShiftSchedule.model_validate(
        {
            "shifts": [
                {
                    "id": "morning",
                    "day": "2026-10-05",
                    "start": "08:00",
                    "end": "12:00",
                    "location": "store",
                    "required_staff": 1,
                },
                {
                    "id": "afternoon",
                    "day": "2026-10-05",
                    "start": "12:00",
                    "end": "16:00",
                    "location": "store",
                    "required_staff": 1,
                },
                {
                    "id": "evening",
                    "day": "2026-10-05",
                    "start": "16:00",
                    "end": "20:00",
                    "location": "store",
                    "required_staff": 1,
                },
                {
                    "id": "next_day",
                    "day": "2026-10-06",
                    "start": "08:00",
                    "end": "12:00",
                    "location": "store",
                    "required_staff": 1,
                },
            ],
            "employees": [
                {
                    "name": "Alice",
                    "max_hours": 16,
                    "eligible_locations": ["store"],
                    "unavailable": [
                        {
                            "day": "2026-10-05",
                            "start": "07:00",
                            "end": "15:00",
                        }
                    ],
                }
            ],
        }
    )

    availability = [
        item
        for item in build_jev_questions("Alice cannot work before 15:00.", schedule)
        if item.question_type.key == "availability_applies"
    ]

    assert [item.target.locator["shift"] for item in availability] == [
        "morning",
        "afternoon",
        "evening",
        "next_day",
    ]
    assert [item.target.context["selection_role"] for item in availability] == [
        "covered_by_extracted_unavailability",
        "covered_by_extracted_unavailability",
        "negative_control",
        "negative_control",
    ]


def test_jev_telemetry_counts_logical_questions_and_input_tokens():
    telemetry = RunTelemetry.start(
        ExperimentConfig(use_jev=True), "shift_schedule"
    )
    client = FakeJevClient(input_tokens=77)

    apply_jev("Staff these shifts", sample_schedule(), client=client, telemetry=telemetry)

    assert len(client.calls) == 1
    assert telemetry.n_jev_questions == len(client.calls[0]["questions"])
    assert telemetry.n_jev_questions == 3  # two shifts plus preference weight
    assert telemetry.jev_input_tokens == 77
    assert telemetry.latency_jev_s >= 0


def test_deterministic_chunking_accumulates_calls_tokens_and_questions():
    telemetry = RunTelemetry.start(ExperimentConfig(use_jev=True), "dayplan")
    client = FakeJevClient(input_tokens=11)

    result = apply_jev(
        "Plan this day",
        sample_plan(),
        client=client,
        telemetry=telemetry,
        max_questions_per_request=3,
    )

    assert len(result.decisions) == 10
    assert [len(call["questions"]) for call in client.calls] == [3, 3, 3, 1]
    assert telemetry.n_jev_calls == 4
    assert telemetry.n_jev_questions == 10
    assert telemetry.jev_input_tokens == 44


def test_changed_is_false_when_prior_hardness_rewrite_removes_weight_target():
    changes = {
        ("constraint_hardness", "preference:0:report"): ("hard", 0.95),
        ("pref_weight", "preference:0:finish_before"): (
            4.0,
            0.95,
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
        ),
    }

    result = apply_jev(
        "The report deadline is firm.",
        sample_plan(),
        client=FakeJevClient(_custom_answers(changes)),
    )

    hardness = next(
        item
        for item in result.decisions
        if item.question_type == "constraint_hardness"
        and item.item == "preference:0:report"
    )
    weight = next(
        item
        for item in result.decisions
        if item.question_type == "pref_weight"
        and item.item == "preference:0:finish_before"
    )
    assert hardness.changed is True
    assert weight.applied is True
    assert weight.jev_value == 5
    assert weight.gpt_value == 1
    assert weight.changed is False


class FakeOpenAIClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.responses = self
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.outputs.pop(0),
            refusal=None,
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )


class MustNotCallJev:
    def system_one(self, **kwargs):
        raise AssertionError("Jev must remain off")


def test_use_jev_false_preserves_existing_behavior_without_a_jev_call():
    plan = sample_plan()
    result = run_dayplan_experiment(
        "Plan it",
        ExperimentConfig(use_jev=False, solution_engine="cp_sat"),
        client=FakeOpenAIClient([DayPlanExtraction(plan=plan, missing_info=[])]),
        jev_client=MustNotCallJev(),
    )

    assert result.extraction.plan == plan
    assert result.jev_decisions == []
    assert result.telemetry.n_jev_questions == 0


@pytest.mark.parametrize("engine", ["cp_sat", "direct_llm"])
def test_jev_adjustment_precedes_both_solution_engines(engine):
    plan = sample_plan()
    changes = {("task_requirement", "task:dishwasher"): ("required", 0.9)}
    outputs = [DayPlanExtraction(plan=plan, missing_info=[])]
    if engine == "direct_llm":
        outputs.append(
            DirectDayPlanSolution(
                status="feasible",
                assignments=[
                    DirectDayAssignment(task="workout", start="09:00", end="10:00"),
                    DirectDayAssignment(task="report", start="10:00", end="10:45"),
                    DirectDayAssignment(
                        task="dishwasher", start="10:45", end="11:15"
                    ),
                ],
                unscheduled_tasks=[],
                explanation="All tasks scheduled.",
            )
        )
    openai_client = FakeOpenAIClient(outputs)
    result = run_dayplan_experiment(
        "The dishwasher is required.",
        ExperimentConfig(use_jev=True, solution_engine=engine),
        client=openai_client,
        jev_client=FakeJevClient(_custom_answers(changes)),
    )

    assert result.run is not None
    dishwasher = next(task for task in result.run.plan.tasks if task.name == "dishwasher")
    assert dishwasher.required is True
    assert result.run.jev_decisions == result.jev_decisions
    if engine == "direct_llm":
        typed_problem = result.run.direct_output
        assert typed_problem is not None
        assert '"name": "dishwasher"' in openai_client.calls[1]["input"][1]["content"]
        assert '"required": true' in openai_client.calls[1]["input"][1]["content"]


def test_shift_experiment_uses_adjusted_schedule_before_cp_sat():
    schedule = sample_schedule()
    changes = {
        ("availability_applies", "employee:Alice:shift:s1"): 0.1,
        ("availability_applies", "employee:Alice:shift:s2"): 0.9,
    }
    result = run_shift_schedule_experiment(
        "Alice cannot work s2.",
        ExperimentConfig(use_jev=True),
        client=FakeOpenAIClient(
            [ShiftScheduleExtraction(schedule=schedule, missing_info=[])]
        ),
        jev_client=FakeJevClient(_custom_answers(changes)),
    )

    assert result.run is not None
    alice = next(
        employee for employee in result.run.schedule.employees if employee.name == "Alice"
    )
    assert [item.shift_id for item in alice.unavailable] == ["s2"]
