"""Lightweight in-memory telemetry for research runs."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterator, Optional

from .experiment import ExperimentConfig


# Standard processing, short-context text-token prices per 1M tokens.
# Source checked 2026-09-30: https://developers.openai.com/api/docs/models/gpt-6.1-sol
# Luna: https://developers.openai.com/api/docs/models/gpt-6-luna
OPENAI_PRICING_USD_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00},
}
OPENAI_PRICING_AS_OF = "2026-09-30"


def estimate_openai_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float:
    """Estimate standard-processing text cost using the dated table above."""

    price = OPENAI_PRICING_USD_PER_MILLION.get(model)
    if price is None:
        raise ValueError("No OpenAI pricing configured for model: " + model)
    return (
        input_tokens * price["input"] + output_tokens * price["output"]
    ) / 1_000_000


@dataclass
class RunTelemetry:
    """One serializable record for an interpretation-and-solve run."""

    architecture: str
    problem_type: str
    case_id: Optional[str]
    model: str
    use_jev: bool
    solution_engine: str
    latency_total_s: float = 0.0
    latency_llm_s: float = 0.0
    latency_jev_s: float = 0.0
    latency_solver_s: float = 0.0
    openai_input_tokens: int = 0
    openai_output_tokens: int = 0
    n_model_calls: int = 0
    n_model_call_attempts: int = 0
    resolved_openai_models: list[str] = field(default_factory=list)
    jev_input_tokens: int = 0
    n_jev_questions: int = 0
    n_jev_calls: int = 0
    n_jev_call_attempts: int = 0
    success: bool = False
    error: Optional[str] = None
    estimated_cost: float = 0.0
    _started_at: float = field(default_factory=perf_counter, repr=False, compare=False)

    @classmethod
    def start(
        cls,
        config: ExperimentConfig,
        problem_type: str,
        case_id: Optional[str] = None,
    ) -> "RunTelemetry":
        return cls(
            architecture=config.label(),
            problem_type=problem_type,
            case_id=case_id,
            model=config.model,
            use_jev=config.use_jev,
            solution_engine=config.solution_engine,
        )

    @contextmanager
    def track(self, component: str) -> Iterator[None]:
        """Accumulate wall time for ``llm``, ``jev``, or ``solver`` work."""

        if component not in {"llm", "jev", "solver"}:
            raise ValueError("telemetry component must be llm, jev, or solver")
        started = perf_counter()
        try:
            yield
        finally:
            field_name = "latency_%s_s" % component
            setattr(self, field_name, getattr(self, field_name) + perf_counter() - started)

    def record_openai_response(self, response: Any) -> None:
        """Record one Responses API call without depending on an SDK response type."""

        resolved = getattr(response, "model", None)
        if resolved and str(resolved) not in self.resolved_openai_models:
            self.resolved_openai_models.append(str(resolved))
        usage = getattr(response, "usage", None)
        self.openai_input_tokens += _usage_value(usage, "input_tokens")
        self.openai_output_tokens += _usage_value(usage, "output_tokens")
        self.n_model_calls += 1
        self.estimated_cost = estimate_openai_cost(
            self.model, self.openai_input_tokens, self.openai_output_tokens
        )

    def record_jev_response(self, response: Any, question_count: int) -> None:
        """Record one batched System One response and its logical decisions."""

        usage = getattr(response, "usage", None)
        self.jev_input_tokens += _usage_value(usage, "input_tokens")
        self.n_jev_questions += question_count
        self.n_jev_calls += 1

    def finish(self, error: Optional[BaseException] = None) -> "RunTelemetry":
        self.latency_total_s = perf_counter() - self._started_at
        self.success = error is None
        self.error = None if error is None else "%s: %s" % (type(error).__name__, error)
        self.estimated_cost = estimate_openai_cost(
            self.model, self.openai_input_tokens, self.openai_output_tokens
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("_started_at", None)
        return payload


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    return int(value or 0)


def parse_openai_response(client: Any, request_kwargs: Dict[str, Any], telemetry: Optional[RunTelemetry]) -> Any:
    """Retain response metadata even if SDK output-schema parsing fails.

    Real SDK raw responses defer structured-output parsing. Test clients and
    older compatible clients can use the ordinary parse boundary.
    """
    from types import SimpleNamespace
    if telemetry is not None:
        telemetry.n_model_call_attempts += 1
    if telemetry is not None and hasattr(client.responses, "with_raw_response"):
        raw = client.responses.with_raw_response.parse(**request_kwargs)
        payload = raw.http_response.json()
        telemetry.record_openai_response(SimpleNamespace(
            model=payload.get("model"), usage=payload.get("usage"),
        ))
        return raw.parse()
    response = client.responses.parse(**request_kwargs)
    if telemetry is not None:
        telemetry.record_openai_response(response)
    return response
