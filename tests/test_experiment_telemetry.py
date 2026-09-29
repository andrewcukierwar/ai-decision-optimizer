import importlib
from types import SimpleNamespace

import dotenv
import pytest
from pydantic import ValidationError

import decision_optimizer.config as config_module
from decision_optimizer.experiment import ExperimentConfig
from decision_optimizer.telemetry import RunTelemetry, estimate_openai_cost


def test_config_module_is_the_dotenv_bootstrap(monkeypatch):
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: calls.append(True))

    importlib.reload(config_module)

    assert calls == [True]


@pytest.mark.parametrize(
    ("model", "use_jev", "engine", "label"),
    [
        ("gpt-6-luna", False, "direct_llm", "Luna 6 · Direct LLM"),
        ("gpt-6-luna", True, "direct_llm", "Luna 6 · Jev · Direct LLM"),
        ("gpt-6-luna", False, "cp_sat", "Luna 6 · CP-SAT"),
        ("gpt-6-luna", True, "cp_sat", "Luna 6 · Jev · CP-SAT"),
        ("gpt-6-sol", False, "direct_llm", "Sol 6 · Direct LLM"),
        ("gpt-6-sol", True, "direct_llm", "Sol 6 · Jev · Direct LLM"),
        ("gpt-6-sol", False, "cp_sat", "Sol 6 · CP-SAT"),
        ("gpt-6-sol", True, "cp_sat", "Sol 6 · Jev · CP-SAT"),
    ],
)
def test_canonical_architecture_labels(model, use_jev, engine, label):
    assert ExperimentConfig(
        model=model, use_jev=use_jev, solution_engine=engine
    ).label() == label


def test_experiment_config_rejects_unplanned_models_and_engines():
    with pytest.raises(ValidationError):
        ExperimentConfig(model="gpt-5.6-sol")
    with pytest.raises(ValidationError):
        ExperimentConfig(solution_engine="provider_plugin")


def test_telemetry_accumulates_usage_and_estimates_cost():
    experiment = ExperimentConfig(model="gpt-6-sol")
    telemetry = RunTelemetry.start(experiment, "dayplan", "case-1")
    telemetry.record_openai_response(
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000)
        )
    )
    telemetry.record_openai_response(
        SimpleNamespace(usage={"input_tokens": 50, "output_tokens": 20})
    )
    telemetry.finish()

    assert telemetry.architecture == "Sol 6 · CP-SAT"
    assert telemetry.n_model_calls == 2
    assert telemetry.openai_input_tokens == 1_000_050
    assert telemetry.openai_output_tokens == 100_020
    assert telemetry.estimated_cost == pytest.approx(
        estimate_openai_cost("gpt-6-sol", 1_000_050, 100_020)
    )
    assert telemetry.success is True
    assert "_started_at" not in telemetry.to_dict()


def test_telemetry_records_failure_information():
    telemetry = RunTelemetry.start(ExperimentConfig(), "shift_schedule")

    telemetry.finish(ValueError("bad structured output"))

    assert telemetry.success is False
    assert telemetry.error == "ValueError: bad structured output"


def test_jev_threshold_is_validated(monkeypatch):
    monkeypatch.setenv("JEV_APPLY_THRESHOLD", "1.2")
    with pytest.raises(ValueError, match="between 0 and 1"):
        config_module.jev_apply_threshold()
