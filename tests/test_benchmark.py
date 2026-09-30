import json

import pytest

from decision_optimizer.benchmark import (
    ALL_EXPERIMENT_CONFIGS,
    BenchmarkCase,
    BenchmarkRunner,
    ExtractionStage,
    SolveStage,
)
from decision_optimizer.dayplan import DayPlan, solve_day_plan
from decision_optimizer.jev import JevDecision, JevResult


def small_plan() -> DayPlan:
    return DayPlan.model_validate(
        {
            "horizon": {"start": "09:00", "end": "12:00"},
            "tasks": [
                {
                    "name": "report",
                    "duration_min": 30,
                    "mode": "active",
                    "required": True,
                }
            ],
        }
    )


class FakeBenchmarkBackend:
    def __init__(self, *, interrupt_on_solve=None, fail_config=None):
        self.plan = small_plan()
        self.interrupt_on_solve = interrupt_on_solve
        self.fail_config = fail_config
        self.extract_calls = []
        self.decide_calls = []
        self.solve_calls = []

    def extract(
        self,
        *,
        domain,
        prompt,
        clarification,
        model,
        telemetry,
        request_timeout_seconds,
    ):
        self.extract_calls.append((model, prompt, request_timeout_seconds))
        telemetry.n_model_calls = 1
        telemetry.openai_input_tokens = 10
        telemetry.openai_output_tokens = 2
        telemetry.latency_llm_s = 0.1
        telemetry.latency_total_s = 0.1
        telemetry.success = True
        return ExtractionStage(self.plan.model_copy(deep=True), [], telemetry)

    def decide(
        self,
        *,
        prompt,
        problem,
        telemetry,
        request_timeout_seconds,
        jev_batch_size,
        api_retries,
    ):
        self.decide_calls.append(
            (telemetry.model, problem.model_dump(mode="json"), jev_batch_size)
        )
        telemetry.n_jev_calls = 1
        telemetry.n_jev_questions = 1
        telemetry.jev_input_tokens = 5
        telemetry.latency_jev_s = 0.01
        telemetry.latency_total_s = 0.01
        telemetry.success = True
        decision = JevDecision(
            question_type="task_requirement",
            item="task:report",
            primitive="choice",
            gpt_value="required",
            jev_value="required",
            confidence=0.95,
            probabilities={"required": 0.95, "optional": 0.05},
            model_confidence=0.95,
            applied=True,
            changed=False,
        )
        return JevResult(
            problem=problem.model_copy(deep=True),
            decisions=[decision],
            resolved_model="jev-test-1.0",
        )

    def solve(
        self,
        *,
        domain,
        problem,
        config,
        telemetry,
        request_timeout_seconds,
        solver_timeout_seconds,
    ):
        self.solve_calls.append(
            (config, problem.model_dump(mode="json"), request_timeout_seconds)
        )
        call_number = len(self.solve_calls)
        if self.interrupt_on_solve == call_number:
            raise KeyboardInterrupt("simulated interruption")
        if self.fail_config == (
            config.model,
            config.use_jev,
            config.solution_engine,
        ):
            raise RuntimeError("simulated paid failure")
        if config.solution_engine == "direct_llm":
            telemetry.n_model_calls = 1
            telemetry.openai_input_tokens = 7
            telemetry.openai_output_tokens = 3
            telemetry.latency_llm_s = 0.2
        else:
            telemetry.latency_solver_s = 0.02
        telemetry.latency_total_s = telemetry.latency_llm_s + telemetry.latency_solver_s
        telemetry.success = True
        return SolveStage(solve_day_plan(problem), None, telemetry)


def benchmark_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="tiny",
        domain="dayplan",
        prompt="Plan 09:00-12:00 with a required active 30-minute report.",
        canonical=small_plan(),
        source="unit-test",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_complete_eight_architecture_flow_is_paired_scored_and_persisted(tmp_path):
    backend = FakeBenchmarkBackend()
    runner = BenchmarkRunner(tmp_path, backend, jev_batch_size=3)

    counts = runner.run([benchmark_case()], ALL_EXPERIMENT_CONFIGS)

    assert counts == {"succeeded": 8, "failed": 0, "skipped": 0}
    assert len(backend.extract_calls) == 2  # once per base model
    assert len(backend.decide_calls) == 2  # once per case/model, shared by engines
    assert len(backend.solve_calls) == 8

    for model in ("gpt-6-luna", "gpt-6.1-sol"):
        model_solves = [item for item in backend.solve_calls if item[0].model == model]
        off_specs = [item[1] for item in model_solves if not item[0].use_jev]
        on_specs = [item[1] for item in model_solves if item[0].use_jev]
        assert off_specs[0] == off_specs[1]
        assert on_specs[0] == on_specs[1]

    records = read_jsonl(tmp_path / "latest.jsonl")
    assert len(records) == 8
    assert all(record["status"] == "success" for record in records)
    assert all(
        record["canonical_evaluation"]["canonical_validation_valid"]
        for record in records
    )
    assert all(record["canonical_evaluation"]["formulation_match"] for record in records)
    assert {
        record["resolved_jev_model"]
        for record in records
        if record["configuration"]["use_jev"]
    } == {"jev-test-1.0"}
    assert len(read_jsonl(tmp_path / "jev_decisions.jsonl")) == 2
    assert (tmp_path / "latest_summary.csv").exists()

    actual_openai = sum(
        record["actual_paid_call_attempts_created_for_this_row"][
            "openai_call_attempts"
        ]
        for record in records
    )
    actual_jev = sum(
        record["actual_paid_call_attempts_created_for_this_row"]["jev_call_attempts"]
        for record in records
    )
    assert actual_openai == 6  # two shared extractions + four direct solves
    assert actual_jev == 2
    assert all(
        record["standalone_architecture_telemetry_estimate"]["n_model_calls"]
        >= 1
        for record in records
    )


def test_interruption_keeps_completed_rows_and_resume_skips_them(tmp_path):
    interrupted = FakeBenchmarkBackend(interrupt_on_solve=3)
    with pytest.raises(KeyboardInterrupt):
        BenchmarkRunner(tmp_path, interrupted).run(
            [benchmark_case()], ALL_EXPERIMENT_CONFIGS
        )

    assert len(read_jsonl(tmp_path / "latest.jsonl")) == 2
    resumed = FakeBenchmarkBackend()
    counts = BenchmarkRunner(tmp_path, resumed).run(
        [benchmark_case()], ALL_EXPERIMENT_CONFIGS
    )

    assert counts == {"succeeded": 6, "failed": 0, "skipped": 2}
    # Luna's shared stages survived; Sol had not started when interrupted.
    assert len(resumed.extract_calls) == 1
    assert resumed.extract_calls[0][0] == "gpt-6.1-sol"
    assert len(resumed.decide_calls) == 1
    assert len(resumed.solve_calls) == 6
    assert len(read_jsonl(tmp_path / "latest.jsonl")) == 8
    assert len(read_jsonl(tmp_path / "jev_decisions.jsonl")) == 2


def test_failure_is_recorded_not_completed_and_only_failed_cell_retries(tmp_path):
    failed_key = ("gpt-6.1-sol", True, "direct_llm")
    backend = FakeBenchmarkBackend(fail_config=failed_key)
    first = BenchmarkRunner(tmp_path, backend).run(
        [benchmark_case()], ALL_EXPERIMENT_CONFIGS
    )

    assert first == {"succeeded": 7, "failed": 1, "skipped": 0}
    records = read_jsonl(tmp_path / "latest.jsonl")
    failure = next(record for record in records if record["status"] == "failure")
    assert failure["failure_stage"] == "solve_or_evaluate"
    assert "simulated paid failure" in failure["error"]

    healthy = FakeBenchmarkBackend()
    second = BenchmarkRunner(tmp_path, healthy).run(
        [benchmark_case()], ALL_EXPERIMENT_CONFIGS
    )
    assert second == {"succeeded": 1, "failed": 0, "skipped": 7}
    assert len(healthy.solve_calls) == 1
    assert healthy.extract_calls == []
    assert healthy.decide_calls == []
    final_records = read_jsonl(tmp_path / "latest.jsonl")
    assert sum(record["status"] == "success" for record in final_records) == 8


def test_preview_full_fixture_shape_requires_no_backend_calls(tmp_path):
    backend = FakeBenchmarkBackend()
    runner = BenchmarkRunner(tmp_path, backend)
    preview = runner.preview([benchmark_case()], ALL_EXPERIMENT_CONFIGS)

    assert preview["total_cells"] == 8
    assert preview["pending_cells"] == 8
    assert preview["expected_paid_calls"] == {
        "openai_min": 6,
        "openai_max": 6,
        "typesafe_jev": 2,
        "total_min": 8,
        "total_max": 8,
    }
    assert backend.extract_calls == []
    assert backend.decide_calls == []
    assert backend.solve_calls == []
