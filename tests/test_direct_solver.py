import json
from datetime import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision_optimizer.application import (
    parse_dayplan_request,
    parse_shift_schedule_request,
    run_dayplan_experiment,
    solve_confirmed_dayplan,
    solve_confirmed_shift_schedule,
)
from decision_optimizer.dayplan import DayPlan
from decision_optimizer.direct_solver import (
    DAYPLAN_DIRECT_INSTRUCTIONS,
    SHIFT_DIRECT_INSTRUCTIONS,
    DirectDayAssignment,
    DirectDayPlanSolution,
    DirectShiftAssignment,
    DirectShiftSolution,
    DirectStatus,
)
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.parsing.dayplan import DayPlanExtraction
from decision_optimizer.parsing.shift_schedule import ShiftScheduleExtraction
from decision_optimizer.shift_schedule import ShiftSchedule


CASES = Path(__file__).parent / "cases"


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.output,
            refusal=None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=25),
        )


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


def load_dayplan():
    return DayPlan.model_validate(
        json.loads((CASES / "simple_active.json").read_text())
    )


def load_schedule():
    return ShiftSchedule.model_validate(
        json.loads((CASES / "shift_schedule_basic.json").read_text())
    )


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-sol"])
def test_selected_model_routes_through_existing_parsers(model):
    dayplan = load_dayplan()
    day_client = FakeClient(DayPlanExtraction(plan=dayplan, missing_info=[]))
    shift = load_schedule()
    shift_client = FakeClient(
        ShiftScheduleExtraction(schedule=shift, missing_info=[])
    )
    config = ExperimentConfig(model=model)

    parse_dayplan_request("Plan it", client=day_client, config=config)
    parse_shift_schedule_request("Staff it", client=shift_client, config=config)

    assert day_client.responses.calls[0]["model"] == model
    assert shift_client.responses.calls[0]["model"] == model


def test_cp_sat_experiment_records_parse_and_solver_in_one_telemetry_record():
    plan = load_dayplan()
    client = FakeClient(DayPlanExtraction(plan=plan, missing_info=[]))

    result = run_dayplan_experiment(
        "Plan it",
        ExperimentConfig(model="gpt-6.1-sol", solution_engine="cp_sat"),
        client=client,
        case_id="simple-active",
    )

    assert result.run is not None
    assert result.run.validation.valid
    assert result.telemetry.case_id == "simple-active"
    assert result.telemetry.n_model_calls == 1
    assert result.telemetry.latency_llm_s >= 0
    assert result.telemetry.latency_solver_s > 0
    assert result.telemetry.success is True


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-sol"])
def test_direct_dayplan_is_structured_materialized_and_validated(model):
    plan = load_dayplan()
    direct = DirectDayPlanSolution(
        status=DirectStatus.FEASIBLE,
        assignments=[
            DirectDayAssignment(task="email", start="09:00", end="09:30"),
            DirectDayAssignment(task="focus", start="09:30", end="10:30"),
        ],
        unscheduled_tasks=[],
        explanation="All hard constraints are satisfied.",
    )
    client = FakeClient(direct)
    run = solve_confirmed_dayplan(
        plan,
        config=ExperimentConfig(model=model, solution_engine="direct_llm"),
        client=client,
    )

    assert run.direct_output == direct
    assert run.solution.status.value == "feasible"
    assert run.validation.valid, run.validation.errors
    assert client.responses.calls[0]["model"] == model
    assert client.responses.calls[0]["text_format"] is DirectDayPlanSolution
    assert "Typed problem:" in client.responses.calls[0]["input"][1]["content"]
    assert "CP-SAT" not in DAYPLAN_DIRECT_INSTRUCTIONS
    assert run.telemetry.n_model_calls == 1
    assert run.telemetry.openai_input_tokens == 100


@pytest.mark.parametrize("offset", ["-05:00", "+09:00"])
def test_direct_dayplan_materializes_offset_times_without_shifting_local_clock(offset):
    plan = DayPlan.model_validate({
        "horizon": {"start": "09:00", "end": "18:00"},
        "tasks": [{"name": "focus", "duration_min": 60, "mode": "active"}],
    })
    direct = DirectDayPlanSolution.model_validate({
        "status": "feasible",
        "assignments": [{"task": "focus", "start": "10:00" + offset, "end": "11:00" + offset}],
        "unscheduled_tasks": [],
        "explanation": "A feasible local-clock schedule.",
    })

    run = solve_confirmed_dayplan(
        plan, config=ExperimentConfig(solution_engine="direct_llm"), client=FakeClient(direct),
    )

    assignment = run.solution.assignments[0]
    assert assignment.start == time(10)
    assert assignment.end == time(11)
    assert assignment.start.tzinfo is assignment.end.tzinfo is None
    assert assignment.model_dump(mode="json") == {"name": "focus", "start": "10:00", "end": "11:00"}
    assert run.validation.valid, run.validation.errors
    assert run.direct_output.assignments[0].start.tzinfo is not None


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-sol"])
def test_direct_shift_schedule_is_structured_materialized_and_validated(model):
    schedule = load_schedule()
    deterministic = __import__(
        "decision_optimizer.shift_schedule", fromlist=["solve_shift_schedule"]
    ).solve_shift_schedule(schedule)
    direct = DirectShiftSolution(
        status=DirectStatus.FEASIBLE,
        assignments=[
            DirectShiftAssignment(
                shift_id=item.shift_id, employee=item.employee_name
            )
            for item in deterministic.assignments
        ],
        explanation="Coverage and hard constraints are satisfied.",
    )
    client = FakeClient(direct)
    run = solve_confirmed_shift_schedule(
        schedule,
        config=ExperimentConfig(model=model, solution_engine="direct_llm"),
        client=client,
    )

    assert run.direct_output == direct
    assert run.validation.valid, run.validation.errors
    assert client.responses.calls[0]["model"] == model
    assert client.responses.calls[0]["text_format"] is DirectShiftSolution
    assert "CP-SAT" not in SHIFT_DIRECT_INSTRUCTIONS


def test_direct_infeasible_verdict_converts_to_existing_solution_type():
    plan = load_dayplan()
    direct = DirectDayPlanSolution(
        status="infeasible",
        assignments=[],
        unscheduled_tasks=[task.name for task in plan.tasks],
        explanation="No feasible schedule.",
    )

    run = solve_confirmed_dayplan(
        plan,
        config=ExperimentConfig(solution_engine="direct_llm"),
        client=FakeClient(direct),
    )

    assert run.solution.status.value == "infeasible"
    assert run.solution.assignments == []


def test_direct_unknown_identifier_is_materialized_for_validation_not_crashed():
    schedule = load_schedule()
    direct = DirectShiftSolution(
        status="feasible",
        assignments=[
            DirectShiftAssignment(
                shift_id=schedule.shifts[0].id, employee="invented employee"
            )
        ],
        explanation="Proposed assignment.",
    )

    run = solve_confirmed_shift_schedule(
        schedule,
        config=ExperimentConfig(solution_engine="direct_llm"),
        client=FakeClient(direct),
    )

    assert run.validation.valid is False
    assert any("unknown employee" in error for error in run.validation.errors)
