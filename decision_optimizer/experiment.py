"""Canonical configuration for one experimental architecture."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .config import openai_model


ModelId = Literal["gpt-6-luna", "gpt-6-sol"]
SolutionEngine = Literal["direct_llm", "cp_sat"]


class ExperimentConfig(BaseModel):
    """The three experimental dimensions in the Research MVP."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelId = "gpt-6-sol"
    use_jev: bool = False
    solution_engine: SolutionEngine = "cp_sat"

    @classmethod
    def from_env(
        cls, *, use_jev: bool = False, solution_engine: SolutionEngine = "cp_sat"
    ) -> "ExperimentConfig":
        return cls(
            model=openai_model(),
            use_jev=use_jev,
            solution_engine=solution_engine,
        )

    def label(self) -> str:
        """Return the single canonical architecture label used everywhere."""

        model_label = "Luna 6" if self.model == "gpt-6-luna" else "Sol 6"
        engine_label = (
            "Direct LLM" if self.solution_engine == "direct_llm" else "CP-SAT"
        )
        parts = [model_label]
        if self.use_jev:
            parts.append("Jev")
        parts.append(engine_label)
        return " · ".join(parts)
