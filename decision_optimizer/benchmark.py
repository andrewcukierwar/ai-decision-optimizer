"""Lightweight, paired, resumable benchmark execution.

For each (case, model), one extraction is shared by all requested architecture
variants. One Jev-adjusted formulation is likewise shared by both Jev engines.
Canonical specifications are passed only to deterministic evaluation after
extraction, decision, and solving calls have completed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence

from .config import jev_apply_threshold, typesafe_model
from .dayplan import DayPlan, DayPlanSolution, solve_day_plan
from .direct_solver import solve_day_plan_direct, solve_shift_schedule_direct
from .evaluation import evaluate_dayplan, evaluate_shift_schedule
from .experiment import ExperimentConfig
from .jev import DEFAULT_JEV_BATCH_SIZE, JevDecision, JevResult, apply_jev, build_jev_questions
from .parsing.dayplan import parse_dayplan
from .parsing.shift_schedule import parse_shift_schedule
from .shift_schedule import ShiftSchedule, ShiftScheduleSolution, solve_shift_schedule
from .telemetry import RunTelemetry


Domain = Literal["dayplan", "shift_schedule"]
Problem = DayPlan | ShiftSchedule
Solution = DayPlanSolution | ShiftScheduleSolution
CACHE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


ALL_EXPERIMENT_CONFIGS: tuple[ExperimentConfig, ...] = tuple(
    ExperimentConfig(model=model, use_jev=use_jev, solution_engine=engine)
    for model in ("gpt-6-luna", "gpt-6-sol")
    for use_jev in (False, True)
    for engine in ("direct_llm", "cp_sat")
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    domain: Domain
    prompt: str
    canonical: Problem
    source: str = ""
    clarification: Optional[str] = None


@dataclass
class ExtractionStage:
    problem: Optional[Problem]
    missing_info: List[str]
    telemetry: RunTelemetry


@dataclass
class SolveStage:
    solution: Solution
    direct_output: Optional[Dict[str, Any]]
    telemetry: RunTelemetry


class BenchmarkBackend(Protocol):
    def extract(
        self,
        *,
        domain: Domain,
        prompt: str,
        clarification: Optional[str],
        model: str,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
    ) -> ExtractionStage: ...

    def decide(
        self,
        *,
        prompt: str,
        problem: Problem,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
        jev_batch_size: int,
        api_retries: int,
    ) -> JevResult: ...

    def solve(
        self,
        *,
        domain: Domain,
        problem: Problem,
        config: ExperimentConfig,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
        solver_timeout_seconds: float,
    ) -> SolveStage: ...


class LiveBenchmarkBackend:
    """Existing project boundaries adapted to the paired benchmark protocol."""

    def __init__(self, *, openai_client: Any = None, jev_client: Any = None):
        self.openai_client = openai_client
        self.jev_client = jev_client

    def extract(
        self,
        *,
        domain: Domain,
        prompt: str,
        clarification: Optional[str],
        model: str,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
    ) -> ExtractionStage:
        if domain == "dayplan":
            extraction = parse_dayplan(
                prompt,
                client=self.openai_client,
                model=model,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
            if clarification and extraction.plan is None:
                extraction = parse_dayplan(
                    prompt,
                    previous_extraction=extraction,
                    clarification=clarification,
                    client=self.openai_client,
                    model=model,
                    telemetry=telemetry,
                    request_timeout_seconds=request_timeout_seconds,
                )
            telemetry.finish()
            return ExtractionStage(extraction.plan, extraction.missing_info, telemetry)

        extraction = parse_shift_schedule(
            prompt,
            client=self.openai_client,
            model=model,
            telemetry=telemetry,
            request_timeout_seconds=request_timeout_seconds,
        )
        if clarification and extraction.schedule is None:
            extraction = parse_shift_schedule(
                prompt,
                previous_extraction=extraction,
                clarification=clarification,
                client=self.openai_client,
                model=model,
                telemetry=telemetry,
                request_timeout_seconds=request_timeout_seconds,
            )
        telemetry.finish()
        return ExtractionStage(extraction.schedule, extraction.missing_info, telemetry)

    def decide(
        self,
        *,
        prompt: str,
        problem: Problem,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
        jev_batch_size: int,
        api_retries: int,
    ) -> JevResult:
        result = apply_jev(
            prompt,
            problem,
            client=self.jev_client,
            telemetry=telemetry,
            max_questions_per_request=jev_batch_size,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=api_retries,
        )
        telemetry.finish()
        return result

    def solve(
        self,
        *,
        domain: Domain,
        problem: Problem,
        config: ExperimentConfig,
        telemetry: RunTelemetry,
        request_timeout_seconds: float,
        solver_timeout_seconds: float,
    ) -> SolveStage:
        direct_output = None
        if domain == "dayplan":
            assert isinstance(problem, DayPlan)
            if config.solution_engine == "direct_llm":
                result = solve_day_plan_direct(
                    problem,
                    client=self.openai_client,
                    model=config.model,
                    telemetry=telemetry,
                    request_timeout_seconds=request_timeout_seconds,
                )
                solution = result.solution
                direct_output = result.output.model_dump(mode="json")
            else:
                with telemetry.track("solver"):
                    solution = solve_day_plan(
                        problem, time_limit_seconds=solver_timeout_seconds
                    )
        else:
            assert isinstance(problem, ShiftSchedule)
            if config.solution_engine == "direct_llm":
                result = solve_shift_schedule_direct(
                    problem,
                    client=self.openai_client,
                    model=config.model,
                    telemetry=telemetry,
                    request_timeout_seconds=request_timeout_seconds,
                )
                solution = result.solution
                direct_output = result.output.model_dump(mode="json")
            else:
                with telemetry.track("solver"):
                    solution = solve_shift_schedule(
                        problem, time_limit_seconds=solver_timeout_seconds
                    )
        telemetry.finish()
        return SolveStage(solution, direct_output, telemetry)


@dataclass
class _LoadedStage:
    payload: Any
    cache_hit: bool
    fingerprint: str


class BenchmarkRunner:
    """Execute selected cells, appending each result before moving on."""

    def __init__(
        self,
        results_dir: Path,
        backend: BenchmarkBackend,
        *,
        request_timeout_seconds: float = 180.0,
        solver_timeout_seconds: float = 10.0,
        jev_batch_size: int = DEFAULT_JEV_BATCH_SIZE,
        api_retries: int = 0,
    ):
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if solver_timeout_seconds <= 0:
            raise ValueError("solver_timeout_seconds must be positive")
        if jev_batch_size < 1:
            raise ValueError("jev_batch_size must be at least 1")
        if api_retries < 0:
            raise ValueError("api_retries must be nonnegative")
        self.results_dir = results_dir
        self.backend = backend
        self.request_timeout_seconds = request_timeout_seconds
        self.solver_timeout_seconds = solver_timeout_seconds
        self.jev_batch_size = jev_batch_size
        self.api_retries = api_retries
        self.results_path = results_dir / "latest.jsonl"
        self.decisions_path = results_dir / "jev_decisions.jsonl"
        self.summary_path = results_dir / "latest_summary.csv"
        self.cache_dir = results_dir / "cache"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.successful_keys = self._successful_result_keys()
        self.decision_ids = self._existing_decision_ids()

    def run(
        self,
        cases: Sequence[BenchmarkCase],
        configs: Sequence[ExperimentConfig],
    ) -> Dict[str, int]:
        """Run every pending cell once; failures are append-only and retryable."""

        counts = {"succeeded": 0, "failed": 0, "skipped": 0}
        for case in cases:
            for model in dict.fromkeys(config.model for config in configs):
                pair_configs = [config for config in configs if config.model == model]
                pending = []
                for config in pair_configs:
                    key = result_key(case.case_id, config)
                    if key in self.successful_keys:
                        counts["skipped"] += 1
                    else:
                        pending.append(config)
                if not pending:
                    continue

                try:
                    extraction = self._load_or_extract(case, model)
                except Exception as error:
                    for index, config in enumerate(pending):
                        self._append_failure(
                            case,
                            config,
                            "extraction",
                            error,
                            actual_paid={
                                "openai_call_attempts": 1 if index == 0 else 0,
                                "jev_call_attempts": 0,
                            },
                        )
                        counts["failed"] += 1
                    continue

                jev_stage: Optional[_LoadedStage] = None
                jev_error: Optional[Exception] = None
                if extraction.payload.problem is not None and any(
                    config.use_jev for config in pending
                ):
                    try:
                        jev_stage = self._load_or_decide(
                            case, model, extraction
                        )
                    except Exception as error:
                        jev_error = error

                extraction_charge_available = not extraction.cache_hit
                jev_charge_available = bool(jev_stage and not jev_stage.cache_hit)
                jev_failure_charge_available = jev_error is not None
                for config in pending:
                    if config.use_jev and jev_error is not None:
                        self._append_failure(
                            case,
                            config,
                            "jev",
                            jev_error,
                            actual_paid={
                                "openai_call_attempts": (
                                    extraction.payload.telemetry.n_model_calls
                                    if extraction_charge_available
                                    else 0
                                ),
                                "jev_call_attempts": 1
                                if jev_failure_charge_available
                                else 0,
                            },
                        )
                        extraction_charge_available = False
                        jev_failure_charge_available = False
                        counts["failed"] += 1
                        continue
                    try:
                        final_problem = extraction.payload.problem
                        decisions: List[JevDecision] = []
                        resolved_jev_model = None
                        jev_telemetry = None
                        if config.use_jev and jev_stage is not None:
                            final_problem = jev_stage.payload["problem"]
                            decisions = jev_stage.payload["decisions"]
                            resolved_jev_model = jev_stage.payload["resolved_model"]
                            jev_telemetry = jev_stage.payload["telemetry"]

                        solve_stage = None
                        if final_problem is not None:
                            solve_telemetry = RunTelemetry.start(
                                config, case.domain, case.case_id
                            )
                            solve_stage = self.backend.solve(
                                domain=case.domain,
                                problem=final_problem.model_copy(deep=True),
                                config=config,
                                telemetry=solve_telemetry,
                                request_timeout_seconds=self.request_timeout_seconds,
                                solver_timeout_seconds=self.solver_timeout_seconds,
                            )

                        combined = _combine_telemetry(
                            config,
                            case,
                            extraction.payload.telemetry,
                            jev_telemetry,
                            None if solve_stage is None else solve_stage.telemetry,
                        )
                        evaluation = _evaluate(
                            case,
                            None if solve_stage is None else solve_stage.solution,
                            final_problem,
                            combined,
                        )
                        actual_paid = {
                            "openai_call_attempts": (
                                (extraction.payload.telemetry.n_model_calls if extraction_charge_available else 0)
                                + (1 if config.solution_engine == "direct_llm" and solve_stage is not None else 0)
                            ),
                            "jev_call_attempts": (
                                jev_telemetry.n_jev_calls
                                if config.use_jev and jev_charge_available and jev_telemetry
                                else 0
                            ),
                            "openai_estimated_cost_usd": (
                                extraction.payload.telemetry.estimated_cost
                                if extraction_charge_available
                                else 0.0
                            )
                            + (
                                solve_stage.telemetry.estimated_cost
                                if config.solution_engine == "direct_llm"
                                and solve_stage is not None
                                else 0.0
                            ),
                            "openai_input_tokens": (
                                extraction.payload.telemetry.openai_input_tokens
                                if extraction_charge_available
                                else 0
                            )
                            + (
                                solve_stage.telemetry.openai_input_tokens
                                if config.solution_engine == "direct_llm"
                                and solve_stage is not None
                                else 0
                            ),
                            "openai_output_tokens": (
                                extraction.payload.telemetry.openai_output_tokens
                                if extraction_charge_available
                                else 0
                            )
                            + (
                                solve_stage.telemetry.openai_output_tokens
                                if config.solution_engine == "direct_llm"
                                and solve_stage is not None
                                else 0
                            ),
                            "jev_input_tokens": (
                                jev_telemetry.jev_input_tokens
                                if config.use_jev
                                and jev_charge_available
                                and jev_telemetry
                                else 0
                            ),
                            "new_stage_latency_s": (
                                extraction.payload.telemetry.latency_total_s
                                if extraction_charge_available
                                else 0.0
                            )
                            + (
                                jev_telemetry.latency_total_s
                                if config.use_jev
                                and jev_charge_available
                                and jev_telemetry
                                else 0.0
                            )
                            + (
                                solve_stage.telemetry.latency_total_s
                                if solve_stage is not None
                                else 0.0
                            ),
                        }
                        record = self._success_record(
                            case=case,
                            config=config,
                            extraction=extraction,
                            jev_stage=jev_stage if config.use_jev else None,
                            final_problem=final_problem,
                            decisions=decisions,
                            resolved_jev_model=resolved_jev_model,
                            solve_stage=solve_stage,
                            evaluation=evaluation.model_dump(mode="json"),
                            telemetry=combined,
                            actual_paid=actual_paid,
                        )
                        self._append_result(record)
                        self.successful_keys.add(record["result_key"])
                        counts["succeeded"] += 1
                        extraction_charge_available = False
                        if config.use_jev:
                            jev_charge_available = False
                    except Exception as error:
                        self._append_failure(
                            case,
                            config,
                            "solve_or_evaluate",
                            error,
                            actual_paid={
                                "openai_call_attempts": (
                                    (
                                        extraction.payload.telemetry.n_model_calls
                                        if extraction_charge_available
                                        else 0
                                    )
                                    + int(config.solution_engine == "direct_llm")
                                ),
                                "jev_call_attempts": (
                                    jev_stage.payload["telemetry"].n_jev_calls
                                    if config.use_jev
                                    and jev_charge_available
                                    and jev_stage is not None
                                    else 0
                                ),
                            },
                        )
                        extraction_charge_available = False
                        if config.use_jev:
                            jev_charge_available = False
                        counts["failed"] += 1
                self._write_summary()
        self._write_summary()
        return counts

    def preview(
        self,
        cases: Sequence[BenchmarkCase],
        configs: Sequence[ExperimentConfig],
    ) -> Dict[str, Any]:
        """Estimate pending cells and paid calls without invoking either API."""

        pending = [
            (case, config)
            for case in cases
            for config in configs
            if result_key(case.case_id, config) not in self.successful_keys
        ]
        pairs = {
            (case.case_id, config.model)
            for case, config in pending
        }
        extraction_calls_min = 0
        extraction_calls_max = 0
        jev_calls = 0
        estimated_jev_questions = 0
        for case_id, model in pairs:
            case = next(item for item in cases if item.case_id == case_id)
            extraction_fingerprint = _extraction_fingerprint(case, model)
            extraction_path = self._cache_path(case, model, "extraction")
            extraction_cached = _valid_cache(extraction_path, extraction_fingerprint)
            if not extraction_cached:
                extraction_calls_min += 1
                extraction_calls_max += 1 + int(bool(case.clarification))
            pair_pending = [
                config
                for pending_case, config in pending
                if pending_case.case_id == case_id and config.model == model
            ]
            if any(config.use_jev for config in pair_pending):
                jev_fingerprint = _jev_fingerprint(
                    extraction_fingerprint,
                    self.jev_batch_size,
                    jev_apply_threshold(),
                    typesafe_model(),
                )
                jev_path = self._cache_path(case, model, "jev")
                if not _valid_cache(jev_path, jev_fingerprint):
                    question_count = len(
                        build_jev_questions(case.prompt, case.canonical)
                    )
                    estimated_jev_questions += question_count
                    jev_calls += math.ceil(question_count / self.jev_batch_size)
        direct_calls = sum(
            config.solution_engine == "direct_llm" for _, config in pending
        )
        return {
            "cases": len(cases),
            "architectures": len(configs),
            "total_cells": len(cases) * len(configs),
            "pending_cells": len(pending),
            "completed_cells": len(cases) * len(configs) - len(pending),
            "paired_case_model_extractions": len(pairs),
            "expected_paid_calls": {
                "openai_min": extraction_calls_min + direct_calls,
                "openai_max": extraction_calls_max + direct_calls,
                "typesafe_jev": jev_calls,
                "total_min": extraction_calls_min + direct_calls + jev_calls,
                "total_max": extraction_calls_max + direct_calls + jev_calls,
            },
            "estimated_jev_questions_from_canonical_specs": estimated_jev_questions,
            "notes": [
                "The OpenAI range reflects conditional clarification calls.",
                "Standalone row telemetry includes reused extraction/Jev stages; actual paid-call attempts are recorded separately.",
                "Jev question counts use canonical formulations only for dry-run estimation and are never sent to GPT or Jev.",
            ],
        }

    def _load_or_extract(self, case: BenchmarkCase, model: str) -> _LoadedStage:
        fingerprint = _extraction_fingerprint(case, model)
        path = self._cache_path(case, model, "extraction")
        cached = _read_cache(path, fingerprint)
        if cached is not None:
            problem = _problem_from_json(case.domain, cached.get("problem"))
            telemetry = _telemetry_from_dict(cached["telemetry"])
            return _LoadedStage(
                ExtractionStage(problem, cached["missing_info"], telemetry),
                True,
                fingerprint,
            )

        telemetry = RunTelemetry.start(
            ExperimentConfig(model=model, use_jev=False, solution_engine="cp_sat"),
            case.domain,
            case.case_id,
        )
        try:
            stage = self.backend.extract(
                domain=case.domain,
                prompt=case.prompt,
                clarification=case.clarification,
                model=model,
                telemetry=telemetry,
                request_timeout_seconds=self.request_timeout_seconds,
            )
        except Exception as error:
            telemetry.finish(error)
            raise
        _atomic_json(
            path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage": "extraction",
                "fingerprint": fingerprint,
                "case_id": case.case_id,
                "model": model,
                "problem": None
                if stage.problem is None
                else stage.problem.model_dump(mode="json"),
                "missing_info": stage.missing_info,
                "telemetry": stage.telemetry.to_dict(),
            },
        )
        return _LoadedStage(stage, False, fingerprint)

    def _load_or_decide(
        self, case: BenchmarkCase, model: str, extraction: _LoadedStage
    ) -> _LoadedStage:
        fingerprint = _jev_fingerprint(
            extraction.fingerprint,
            self.jev_batch_size,
            jev_apply_threshold(),
            typesafe_model(),
        )
        path = self._cache_path(case, model, "jev")
        cached = _read_cache(path, fingerprint)
        if cached is not None:
            payload = {
                "problem": _problem_from_json(case.domain, cached["problem"]),
                "decisions": [
                    JevDecision.model_validate(item) for item in cached["decisions"]
                ],
                "resolved_model": cached.get("resolved_jev_model"),
                "telemetry": _telemetry_from_dict(cached["telemetry"]),
            }
            self._record_jev_decisions(case, model, fingerprint, payload)
            return _LoadedStage(payload, True, fingerprint)

        problem = extraction.payload.problem
        assert problem is not None
        telemetry = RunTelemetry.start(
            ExperimentConfig(model=model, use_jev=True, solution_engine="cp_sat"),
            case.domain,
            case.case_id,
        )
        try:
            result = self.backend.decide(
                prompt=case.prompt,
                problem=problem.model_copy(deep=True),
                telemetry=telemetry,
                request_timeout_seconds=self.request_timeout_seconds,
                jev_batch_size=self.jev_batch_size,
                api_retries=self.api_retries,
            )
        except Exception as error:
            telemetry.finish(error)
            raise
        payload = {
            "problem": result.problem,
            "decisions": result.decisions,
            "resolved_model": result.resolved_model,
            "telemetry": telemetry,
        }
        _atomic_json(
            path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage": "jev",
                "fingerprint": fingerprint,
                "case_id": case.case_id,
                "model": model,
                "requested_jev_model": typesafe_model(),
                "resolved_jev_model": result.resolved_model,
                "problem": result.problem.model_dump(mode="json"),
                "decisions": [
                    item.model_dump(mode="json") for item in result.decisions
                ],
                "telemetry": telemetry.to_dict(),
            },
        )
        self._record_jev_decisions(case, model, fingerprint, payload)
        return _LoadedStage(payload, False, fingerprint)

    def _success_record(self, **values: Any) -> Dict[str, Any]:
        case: BenchmarkCase = values["case"]
        config: ExperimentConfig = values["config"]
        extraction: _LoadedStage = values["extraction"]
        jev_stage: Optional[_LoadedStage] = values["jev_stage"]
        solve_stage: Optional[SolveStage] = values["solve_stage"]
        final_problem: Optional[Problem] = values["final_problem"]
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "result_key": result_key(case.case_id, config),
            "status": "success",
            "recorded_at": _utc_now(),
            "case_id": case.case_id,
            "domain": case.domain,
            "source": case.source,
            "architecture": config.label(),
            "configuration": config.model_dump(mode="json"),
            "runtime_configuration": {
                "request_timeout_seconds": self.request_timeout_seconds,
                "solver_timeout_seconds": self.solver_timeout_seconds,
                "api_retries": self.api_retries,
                "jev_batch_size": self.jev_batch_size,
                "jev_apply_threshold": jev_apply_threshold(),
                "requested_jev_model": typesafe_model()
                if config.use_jev
                else None,
            },
            "model": config.model,
            "resolved_jev_model": values["resolved_jev_model"],
            "intermediate_reuse": {
                "extraction_fingerprint": extraction.fingerprint,
                "extraction_cache_hit": extraction.cache_hit,
                "jev_fingerprint": None if jev_stage is None else jev_stage.fingerprint,
                "jev_cache_hit": None if jev_stage is None else jev_stage.cache_hit,
            },
            "actual_paid_call_attempts_created_for_this_row": values["actual_paid"],
            "standalone_architecture_telemetry_estimate": values[
                "telemetry"
            ].to_dict(),
            "extraction": {
                "problem": None
                if extraction.payload.problem is None
                else extraction.payload.problem.model_dump(mode="json"),
                "missing_info": extraction.payload.missing_info,
            },
            "final_problem": None
            if final_problem is None
            else final_problem.model_dump(mode="json"),
            "solution": None
            if solve_stage is None
            else solve_stage.solution.model_dump(mode="json"),
            "direct_output": None
            if solve_stage is None
            else solve_stage.direct_output,
            "jev_decisions": [
                item.model_dump(mode="json") for item in values["decisions"]
            ],
            "canonical_evaluation": values["evaluation"],
            "error": None,
        }

    def _append_failure(
        self,
        case: BenchmarkCase,
        config: ExperimentConfig,
        stage: str,
        error: Exception,
        actual_paid: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._append_result(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "result_key": result_key(case.case_id, config),
                "status": "failure",
                "recorded_at": _utc_now(),
                "case_id": case.case_id,
                "domain": case.domain,
                "source": case.source,
                "architecture": config.label(),
                "configuration": config.model_dump(mode="json"),
                "model": config.model,
                "resolved_jev_model": None,
                "failure_stage": stage,
                "actual_paid_call_attempts_created_for_this_row": actual_paid
                or {"openai_call_attempts": 0, "jev_call_attempts": 0},
                "canonical_evaluation": None,
                "jev_decisions": [],
                "error": "%s: %s" % (type(error).__name__, error),
            }
        )

    def _append_result(self, record: Dict[str, Any]) -> None:
        _append_jsonl(self.results_path, record)

    def _record_jev_decisions(
        self,
        case: BenchmarkCase,
        model: str,
        fingerprint: str,
        payload: Dict[str, Any],
    ) -> None:
        for index, decision in enumerate(payload["decisions"]):
            decision_id = hashlib.sha256(
                ("%s|%s|%s|%d" % (case.case_id, model, fingerprint, index)).encode()
            ).hexdigest()
            if decision_id in self.decision_ids:
                continue
            _append_jsonl(
                self.decisions_path,
                {
                    "schema_version": 1,
                    "decision_id": decision_id,
                    "case_id": case.case_id,
                    "domain": case.domain,
                    "model": model,
                    "requested_jev_model": typesafe_model(),
                    "resolved_jev_model": payload["resolved_model"],
                    "jev_fingerprint": fingerprint,
                    **decision.model_dump(mode="json"),
                },
            )
            self.decision_ids.add(decision_id)

    def _cache_path(self, case: BenchmarkCase, model: str, stage: str) -> Path:
        slug = hashlib.sha256(case.case_id.encode()).hexdigest()[:12]
        return self.cache_dir / ("%s_%s_%s.json" % (slug, model, stage))

    def _successful_result_keys(self) -> set[str]:
        return {
            item["result_key"]
            for item in _read_jsonl(self.results_path)
            if item.get("status") == "success"
        }

    def _existing_decision_ids(self) -> set[str]:
        return {
            item["decision_id"]
            for item in _read_jsonl(self.decisions_path)
            if item.get("decision_id")
        }

    def _write_summary(self) -> None:
        latest: Dict[str, Dict[str, Any]] = {}
        for item in _read_jsonl(self.results_path):
            key = item.get("result_key")
            if key:
                if key not in latest or item.get("status") == "success":
                    latest[key] = item
        fieldnames = [
            "case_id",
            "domain",
            "architecture",
            "model",
            "use_jev",
            "solution_engine",
            "status",
            "valid_output",
            "canonical_validation_valid",
            "hard_constraints_satisfied",
            "hard_constraints_total",
            "required_completed",
            "formulation_match",
            "objective_gap",
            "standalone_estimated_cost",
            "actual_openai_call_attempts_created_for_row",
            "actual_jev_call_attempts_created_for_row",
            "actual_openai_estimated_cost_created_for_row",
            "error",
        ]
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in sorted(
                latest.values(), key=lambda item: (item["case_id"], item["architecture"])
            ):
                evaluation = record.get("canonical_evaluation") or {}
                config = record.get("configuration") or {}
                telemetry = record.get("standalone_architecture_telemetry_estimate") or {}
                paid = record.get("actual_paid_call_attempts_created_for_this_row") or {}
                writer.writerow(
                    {
                        "case_id": record["case_id"],
                        "domain": record["domain"],
                        "architecture": record["architecture"],
                        "model": record["model"],
                        "use_jev": config.get("use_jev"),
                        "solution_engine": config.get("solution_engine"),
                        "status": record["status"],
                        "valid_output": evaluation.get("valid_output"),
                        "canonical_validation_valid": evaluation.get(
                            "canonical_validation_valid"
                        ),
                        "hard_constraints_satisfied": evaluation.get(
                            "hard_constraints_satisfied"
                        ),
                        "hard_constraints_total": evaluation.get(
                            "hard_constraints_total"
                        ),
                        "required_completed": evaluation.get("required_completed"),
                        "formulation_match": evaluation.get("formulation_match"),
                        "objective_gap": evaluation.get("objective_gap"),
                        "standalone_estimated_cost": telemetry.get("estimated_cost"),
                        "actual_openai_call_attempts_created_for_row": paid.get(
                            "openai_call_attempts"
                        ),
                        "actual_jev_call_attempts_created_for_row": paid.get(
                            "jev_call_attempts"
                        ),
                        "actual_openai_estimated_cost_created_for_row": paid.get(
                            "openai_estimated_cost_usd"
                        ),
                        "error": record.get("error"),
                    }
                )
        temporary.replace(self.summary_path)


def result_key(case_id: str, config: ExperimentConfig) -> str:
    return "%s|%s|jev=%s|engine=%s" % (
        case_id,
        config.model,
        str(config.use_jev).lower(),
        config.solution_engine,
    )


def _evaluate(
    case: BenchmarkCase,
    solution: Optional[Solution],
    final_problem: Optional[Problem],
    telemetry: RunTelemetry,
) -> Any:
    if case.domain == "dayplan":
        assert isinstance(case.canonical, DayPlan)
        assert solution is None or isinstance(solution, DayPlanSolution)
        assert final_problem is None or isinstance(final_problem, DayPlan)
        return evaluate_dayplan(
            case.canonical,
            solution,
            arm_spec=final_problem,
            telemetry=telemetry,
        )
    assert isinstance(case.canonical, ShiftSchedule)
    assert solution is None or isinstance(solution, ShiftScheduleSolution)
    assert final_problem is None or isinstance(final_problem, ShiftSchedule)
    return evaluate_shift_schedule(
        case.canonical,
        solution,
        arm_spec=final_problem,
        telemetry=telemetry,
    )


def _combine_telemetry(
    config: ExperimentConfig,
    case: BenchmarkCase,
    *stages: Optional[RunTelemetry],
) -> RunTelemetry:
    combined = RunTelemetry.start(config, case.domain, case.case_id)
    fields = (
        "latency_total_s",
        "latency_llm_s",
        "latency_jev_s",
        "latency_solver_s",
        "openai_input_tokens",
        "openai_output_tokens",
        "n_model_calls",
        "jev_input_tokens",
        "n_jev_questions",
        "n_jev_calls",
        "estimated_cost",
    )
    for stage in stages:
        if stage is None:
            continue
        for field in fields:
            setattr(combined, field, getattr(combined, field) + getattr(stage, field))
    combined.success = True
    combined.error = None
    return combined


def _extraction_fingerprint(case: BenchmarkCase, model: str) -> str:
    return _fingerprint(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "extraction",
            "domain": case.domain,
            "prompt": case.prompt,
            "clarification": case.clarification,
            "model": model,
        }
    )


def _jev_fingerprint(
    extraction_fingerprint: str,
    batch_size: int,
    threshold: float,
    requested_model: str,
) -> str:
    return _fingerprint(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "jev",
            "extraction_fingerprint": extraction_fingerprint,
            "batch_size": batch_size,
            "threshold": threshold,
            "requested_model": requested_model,
        }
    )


def _fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _problem_from_json(domain: Domain, payload: Any) -> Optional[Problem]:
    if payload is None:
        return None
    return (
        DayPlan.model_validate(payload)
        if domain == "dayplan"
        else ShiftSchedule.model_validate(payload)
    )


def _telemetry_from_dict(payload: Dict[str, Any]) -> RunTelemetry:
    allowed = {
        name
        for name in RunTelemetry.__dataclass_fields__
        if name != "_started_at"
    }
    return RunTelemetry(**{key: value for key, value in payload.items() if key in allowed})


def _read_cache(path: Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("fingerprint") == fingerprint else None


def _valid_cache(path: Path, fingerprint: str) -> bool:
    return _read_cache(path, fingerprint) is not None


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A final torn line after process termination is ignored; every
            # previous fsynced line remains resumable.
            continue
    return records


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
