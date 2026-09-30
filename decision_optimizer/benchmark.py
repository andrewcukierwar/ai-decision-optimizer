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

from .failures import CachedTerminalFailure, failure_category, terminal_failure
from .jev_alignment import indexed_questions
from .config import jev_apply_threshold, typesafe_model
from .dayplan import DayPlan, DayPlanSolution, solve_day_plan
from .direct_solver import solve_day_plan_direct, solve_shift_schedule_direct
from .evaluation import evaluate_dayplan, evaluate_shift_schedule
from .experiment import ExperimentConfig
from .jev import DEFAULT_JEV_BATCH_SIZE, JevDecision, JevResult, apply_jev, build_jev_questions, final_question_value
from .parsing.dayplan import parse_dayplan
from .parsing.shift_schedule import parse_shift_schedule
from .shift_schedule import ShiftSchedule, ShiftScheduleSolution, solve_shift_schedule
from .telemetry import RunTelemetry


Domain = Literal["dayplan", "shift_schedule"]
Problem = DayPlan | ShiftSchedule
Solution = DayPlanSolution | ShiftScheduleSolution
CACHE_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 2


ALL_EXPERIMENT_CONFIGS: tuple[ExperimentConfig, ...] = tuple(
    ExperimentConfig(model=model, use_jev=use_jev, solution_engine=engine)
    for model in ("gpt-6-luna", "gpt-6.1-sol")
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
        if api_retries != 0:
            raise ValueError("Benchmark policy requires api_retries=0; transport failures may be retried on explicit resume")
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
        """Run pending cells; completed output failures/timeouts are terminal."""

        counts = {"succeeded": 0, "failed": 0, "skipped": 0}
        for case in cases:
            for model in dict.fromkeys(config.model for config in configs):
                pair_configs = [config for config in configs if config.model == model]
                pending = []
                for config in pair_configs:
                    key = self._result_key(case, config)
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
                            actual_paid=_stage_accounting(
                                (getattr(error, "telemetry", None), "openai", 1)
                                if index == 0 and not getattr(error, "cache_hit", False) else None,
                            ),
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
                            extraction=extraction,
                            actual_paid=_stage_accounting(
                                (extraction.payload.telemetry, "openai", 0) if extraction_charge_available else None,
                                (getattr(jev_error, "telemetry", None), "jev", 1)
                                if jev_failure_charge_available and not getattr(jev_error, "cache_hit", False) else None,
                            ),
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
                        actual_paid = _stage_accounting(
                            (extraction.payload.telemetry, "openai", 0) if extraction_charge_available else None,
                            (jev_telemetry, "jev", 0) if config.use_jev and jev_charge_available else None,
                            (solve_stage.telemetry, "openai", 0) if solve_stage is not None else None,
                        )
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
                        if final_problem is not None:
                            solve_telemetry.finish(error)
                        self._append_failure(
                            case,
                            config,
                            "solve_or_evaluate",
                            error,
                            extraction=extraction,
                            final_problem=final_problem,
                            jev_stage=jev_stage if config.use_jev else None,
                            telemetry=_combine_telemetry(config, case, extraction.payload.telemetry, jev_telemetry, solve_telemetry if final_problem is not None else None),
                            actual_paid=_stage_accounting(
                                (extraction.payload.telemetry, "openai", 0) if extraction_charge_available else None,
                                (jev_telemetry, "jev", 0) if config.use_jev and jev_charge_available else None,
                                (solve_telemetry, "openai", int(config.solution_engine == "direct_llm")) if final_problem is not None else None,
                            ),
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
            if self._result_key(case, config) not in self.successful_keys
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
            extraction_fingerprint = _extraction_fingerprint(case, model, self._runtime_config())
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
                    problem_hash=(_read_cache(extraction_path, extraction_fingerprint) or {}).get("problem_hash"),
                    runtime=self._runtime_config(),
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
        fingerprint = _extraction_fingerprint(case, model, self._runtime_config())
        path = self._cache_path(case, model, "extraction")
        cached = _read_cache(path, fingerprint)
        if cached is not None:
            _raise_cached_failure(cached)
            problem = _problem_from_json(case.domain, cached.get("problem"))
            if cached.get("problem_hash") != _problem_hash(problem):
                raise ValueError("Extraction cache problem hash mismatch")
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
            if terminal_failure(error):
                _atomic_json(path, {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "stage": "extraction",
                    "fingerprint": fingerprint,
                    "runtime_configuration": self._runtime_config(),
                    "case_id": case.case_id, "model": model,
                    "terminal_error": str(error), "failure_category": failure_category(error),
                    "telemetry": telemetry.to_dict(),
                })
            error.telemetry = telemetry
            raise
        _atomic_json(
            path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage": "extraction",
                "runtime_configuration": self._runtime_config(),
                "problem_hash": _problem_hash(stage.problem),
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
            problem_hash=_problem_hash(extraction.payload.problem),
            runtime=self._runtime_config(),
        )
        path = self._cache_path(case, model, "jev")
        cached = _read_cache(path, fingerprint)
        if cached is not None:
            if "terminal_error" in cached:
                self._record_jev_decisions(case, model, fingerprint, {"decisions": [JevDecision.model_validate(d) for d in cached.get("decisions", [])], "resolved_model": cached.get("resolved_jev_model")}, extraction.payload.problem)
            _raise_cached_failure(cached)
            payload = {
                "problem": _problem_from_json(case.domain, cached["problem"]),
                "decisions": [
                    JevDecision.model_validate(item) for item in cached["decisions"]
                ],
                "resolved_model": cached.get("resolved_jev_model"),
                "telemetry": _telemetry_from_dict(cached["telemetry"]),
            }
            self._record_jev_decisions(case, model, fingerprint, payload, extraction.payload.problem)
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
            if terminal_failure(error):
                _atomic_json(path, {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "stage": "jev",
                    "jev_batch_size": self.jev_batch_size,
                    "jev_apply_threshold": jev_apply_threshold(),
                    "requested_jev_model": typesafe_model(),
                    "extraction_fingerprint": extraction.fingerprint,
                    "extraction_problem_hash": _problem_hash(problem),
                    "decisions": [d.model_dump(mode="json") for d in getattr(error, "observed_decisions", [])],
                    "invalid_questions": getattr(error, "invalid_questions", []),
                    "missing_question_names": getattr(error, "missing_question_names", []),
                    "resolved_jev_model": getattr(error, "resolved_jev_model", None),
                    "fingerprint": fingerprint,
                    "runtime_configuration": self._runtime_config(),
                    "case_id": case.case_id, "model": model,
                    "terminal_error": str(error), "failure_category": failure_category(error),
                    "telemetry": telemetry.to_dict(),
                })
            error.telemetry = telemetry
            self._record_jev_decisions(case, model, fingerprint, {"decisions": getattr(error, "observed_decisions", []), "resolved_model": getattr(error, "resolved_jev_model", None)}, problem)
            raise
        questions = build_jev_questions(case.prompt, problem)
        question_by_item = {(q.question_type.key, q.target.item): (identity, q) for identity, q in indexed_questions(case.domain, questions, problem, problem)}
        for decision in result.decisions:
            matched = question_by_item.get((decision.question_type, decision.item))
            if matched:
                identity, question = matched
                decision.question_name = question.name
                decision.semantic_question_id = identity
                decision.locator = dict(question.target.locator)
                decision.context = dict(question.target.context)
                decision.final_value = final_question_value(result.problem, question)
                decision.final_value_status = "represented" if decision.final_value is not None else "not_represented"
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
                "jev_batch_size": self.jev_batch_size,
                "jev_apply_threshold": jev_apply_threshold(),
                "runtime_configuration": self._runtime_config(),
                "extraction_fingerprint": extraction.fingerprint,
                "extraction_problem_hash": _problem_hash(extraction.payload.problem),
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
        self._record_jev_decisions(case, model, fingerprint, payload, extraction.payload.problem)
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
            "result_key": self._result_key(case, config),
            "status": "success",
            "recorded_at": _utc_now(),
            "case_id": case.case_id,
            "domain": case.domain,
            "source": case.source,
            "architecture": config.label(),
            "configuration": config.model_dump(mode="json"),
            "runtime_configuration": {
                **self._runtime_config(),
                "jev_batch_size": self.jev_batch_size,
                "jev_apply_threshold": jev_apply_threshold(),
                "requested_jev_model": typesafe_model()
                if config.use_jev
                else None,
            },
            "model": config.model,
            "resolved_openai_models": values["telemetry"].resolved_openai_models,
            "extraction_problem_hash": _problem_hash(extraction.payload.problem),
            "question_alignment": question_observations(case, extraction.payload.problem, values["decisions"]),
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
                "problem_hash": _problem_hash(extraction.payload.problem),
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
        self, case: BenchmarkCase, config: ExperimentConfig, stage: str,
        error: Exception, actual_paid: Optional[Dict[str, Any]] = None,
        extraction: Optional[_LoadedStage] = None,
        final_problem: Optional[Problem] = None,
        telemetry: Optional[RunTelemetry] = None,
        jev_stage: Optional[_LoadedStage] = None,
    ) -> None:
        decisions = (jev_stage.payload["decisions"] if jev_stage is not None
                     else getattr(error, "observed_decisions", []))
        resolved_jev_model = (jev_stage.payload["resolved_model"] if jev_stage is not None
                              else getattr(error, "resolved_jev_model", None))
        terminal = terminal_failure(error)
        if telemetry is None and extraction is not None:
            telemetry = _combine_telemetry(config, case, extraction.payload.telemetry, getattr(error, "telemetry", None))
        telemetry = telemetry or getattr(error, "telemetry", None)
        if telemetry is not None:
            telemetry.success = False
            telemetry.error = str(error)
        evaluation = _evaluate(case, None, final_problem, telemetry).model_dump(mode="json") if terminal else None
        record = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "result_key": self._result_key(case, config),
            "status": "terminal_failure" if terminal else "failure",
            "failure_category": failure_category(error), "retryable": not terminal,
            "recorded_at": _utc_now(), "case_id": case.case_id,
            "domain": case.domain, "source": case.source,
            "architecture": config.label(), "configuration": config.model_dump(mode="json"),
            "model": config.model, "resolved_jev_model": resolved_jev_model,
            "runtime_configuration": self._runtime_config(),
            "failure_stage": stage,
            "actual_paid_call_attempts_created_for_this_row": actual_paid or {"openai_call_attempts": 0, "jev_call_attempts": 0},
            "standalone_architecture_telemetry_estimate": None if telemetry is None else telemetry.to_dict(),
            "canonical_evaluation": evaluation,
            "jev_decisions": [d.model_dump(mode="json") for d in decisions],
            "final_problem": None if final_problem is None else final_problem.model_dump(mode="json"),
            "extraction": None if extraction is None else {
                "problem": None if extraction.payload.problem is None else extraction.payload.problem.model_dump(mode="json"),
                "missing_info": extraction.payload.missing_info,
                "problem_hash": _problem_hash(extraction.payload.problem),
            },
            "jev_invalid_questions": getattr(error, "invalid_questions", []),
            "jev_missing_question_names": getattr(error, "missing_question_names", []),
            "extraction_problem_hash": None if extraction is None else _problem_hash(extraction.payload.problem),
            "question_alignment": question_observations(case, None if extraction is None else extraction.payload.problem, decisions),
            "resolved_openai_models": [] if telemetry is None else telemetry.resolved_openai_models,
            "intermediate_reuse": None if extraction is None else {"extraction_fingerprint": extraction.fingerprint,
                "jev_fingerprint": None if jev_stage is None else jev_stage.fingerprint},
            "error": "%s: %s" % (type(error).__name__, error),
        }
        self._append_result(record)
        if terminal:
            self.successful_keys.add(record["result_key"])

    def _runtime_config(self) -> Dict[str, Any]:
        return {"request_timeout_seconds": self.request_timeout_seconds,
                "solver_timeout_seconds": self.solver_timeout_seconds,
                "api_retries": self.api_retries,
                "timeout_policy": "terminal_no_retry_v1"}

    def _result_key(self, case: BenchmarkCase, config: ExperimentConfig) -> str:
        return result_key(case.case_id, config) + "|" + _fingerprint({
            "schema": RESULT_SCHEMA_VERSION, "case_prompt": case.prompt,
            "extraction_fingerprint": _extraction_fingerprint(case, config.model, self._runtime_config()),
            "canonical": case.canonical.model_dump(mode="json"), "clarification": case.clarification,
            "runtime": self._runtime_config(), "threshold": jev_apply_threshold(),
            "jev_model": typesafe_model(), "batch_size": self.jev_batch_size,
            "implementation": _implementation_hash(),
        })

    def _append_result(self, record: Dict[str, Any]) -> None:
        # Compare hashes, never assume a common model/prompt implies identical extraction.
        reuse = record.get("intermediate_reuse") or {}
        fingerprint = reuse.get("extraction_fingerprint")
        if fingerprint and record.get("extraction_problem_hash"):
            for previous in _read_jsonl(self.results_path):
                previous_reuse = previous.get("intermediate_reuse") or {}
                if previous_reuse.get("extraction_fingerprint") == fingerprint and previous.get("extraction_problem_hash") != record["extraction_problem_hash"]:
                    raise ValueError("Paired architecture rows have different extraction outputs")
        _append_jsonl(self.results_path, record)

    def _record_jev_decisions(
        self,
        case: BenchmarkCase,
        model: str,
        fingerprint: str,
        payload: Dict[str, Any],
        extracted: Problem,
    ) -> None:
        observations = question_observations(case, extracted, payload["decisions"])
        by_item = {(item["question_type"], item["generated_item"]): item for item in observations["observed_questions"]}
        for index, decision in enumerate(payload["decisions"]):
            decision_id = hashlib.sha256(
                ("%s|%s|%s|%d" % (case.case_id, model, fingerprint, index)).encode()
            ).hexdigest()
            if decision_id in self.decision_ids:
                continue
            _append_jsonl(
                self.decisions_path,
                {
                    "schema_version": 2,
                    "decision_id": decision_id,
                    "case_id": case.case_id,
                    "domain": case.domain,
                    "model": model,
                    "requested_jev_model": typesafe_model(),
                    "resolved_jev_model": payload["resolved_model"],
                    "jev_fingerprint": fingerprint,
                    **decision.model_dump(mode="json"),
                    "answer_key_alignment": by_item.get((decision.question_type, decision.item), {"alignment_status": "unaligned_observed_decision", "canonical_question_id": None}),
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
            if item.get("status") in {"success", "terminal_failure"}
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
            "formulation_match_structural",
            "feasible_correctly_reported",
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
                        "formulation_match_structural": evaluation.get("formulation_match_structural"),
                        "feasible_correctly_reported": evaluation.get("feasible_correctly_reported"),
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
    telemetry: Optional[RunTelemetry],
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
        "n_model_call_attempts",
        "jev_input_tokens",
        "n_jev_questions",
        "n_jev_calls",
        "n_jev_call_attempts",
        "estimated_cost",
    )
    for stage in stages:
        if stage is None:
            continue
        for field in fields:
            setattr(combined, field, getattr(combined, field) + getattr(stage, field))
        for model in stage.resolved_openai_models:
            if model not in combined.resolved_openai_models:
                combined.resolved_openai_models.append(model)
    combined.success = True
    combined.error = None
    return combined


def _extraction_fingerprint(case: BenchmarkCase, model: str, runtime: Optional[Dict[str, Any]] = None) -> str:
    from .parsing.dayplan import DAYPLAN_EXTRACTION_INSTRUCTIONS
    from .parsing.shift_schedule import SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS
    return _fingerprint(
        {
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
            "instructions": DAYPLAN_EXTRACTION_INSTRUCTIONS if case.domain == "dayplan" else SHIFTSCHEDULE_EXTRACTION_INSTRUCTIONS,
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "extraction",
            "implementation": _implementation_hash(),
            "runtime": runtime,
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
    *, problem_hash: Optional[str] = None, runtime: Optional[Dict[str, Any]] = None,
) -> str:
    return _fingerprint(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "stage": "jev",
            "typesafe_base_url": os.getenv("TYPESAFE_BASE_URL"),
            "implementation": _implementation_hash(),
            "problem_hash": problem_hash,
            "runtime": runtime,
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
    return payload if payload.get("schema_version") == CACHE_SCHEMA_VERSION and payload.get("fingerprint") == fingerprint else None


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


def _problem_hash(problem: Optional[Problem]) -> str:
    return _fingerprint({"problem": None if problem is None else problem.model_dump(mode="json")})


def _implementation_hash() -> str:
    """Source and dependency provenance; no manual version bump can be forgotten."""
    from importlib.metadata import version
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for dependency in ("openai", "typesafe-sdk", "pydantic", "ortools"):
        digest.update((dependency + version(dependency)).encode())
    return digest.hexdigest()


def _raise_cached_failure(payload: Dict[str, Any]) -> None:
    if "terminal_error" in payload:
        error = CachedTerminalFailure(payload["terminal_error"], payload["failure_category"])
        error.cache_hit = True
        error.telemetry = _telemetry_from_dict(payload["telemetry"])
        error.observed_decisions = [JevDecision.model_validate(d) for d in payload.get("decisions", [])]
        error.invalid_questions = payload.get("invalid_questions", [])
        error.missing_question_names = payload.get("missing_question_names", [])
        error.resolved_jev_model = payload.get("resolved_jev_model")
        raise error


def question_observations(
    case: BenchmarkCase, extracted: Optional[Problem], decisions: Sequence[JevDecision],
) -> Dict[str, Any]:
    """Join the actual shared extraction/Jev sample to proposed canonical questions.

    Canonical context is used only for post-response evaluation, never requests.
    Unaligned observations and canonical omissions both remain explicit.
    """
    canonical_questions = build_jev_questions(case.prompt, case.canonical)
    canonical_ids = {identity for identity, _ in indexed_questions(case.domain, canonical_questions, case.canonical, case.canonical)}
    questions = [] if extracted is None else build_jev_questions(case.prompt, extracted)
    indexed = [] if extracted is None else indexed_questions(case.domain, questions, extracted, case.canonical)
    decisions_by_item = {(d.question_type, d.item): d for d in decisions}
    matched = set()
    observed = []
    for candidate_id, question in indexed:
        aligned = candidate_id in canonical_ids and candidate_id not in matched
        if aligned:
            matched.add(candidate_id)
        decision = decisions_by_item.get((question.question_type.key, question.target.item))
        gpt_value = question.question_type.gpt_value(extracted, question.target)
        observed.append({
            "canonical_question_id": candidate_id if aligned else None,
            "semantic_candidate_id": candidate_id,
            "question_type": question.question_type.key,
            "generated_item": question.target.item,
            "locator": dict(question.target.locator), "context": dict(question.target.context),
            "alignment_status": "aligned" if aligned else "unaligned_generated",
            "gpt_baseline_answer": gpt_value,
            "jev_answer": None if decision is None else decision.jev_value,
            "jev_probability": None if decision is None else decision.confidence,
            "probabilities": {} if decision is None else decision.probabilities,
            "applied": False if decision is None else decision.applied,
            "changed": False if decision is None else decision.changed,
            "final_applied_answer": decision.final_value if decision is not None else gpt_value,
            "final_value_status": "gpt_baseline" if decision is None else decision.final_value_status,
            "jev_observation_status": "not_observed" if decision is None else "observed",
        })
    unaligned_decisions = [d.model_dump(mode="json") for d in decisions if (d.question_type, d.item) not in {(q.question_type.key, q.target.item) for q in questions}]
    missing = sorted(canonical_ids - matched)
    return {
        "canonical_questions": len(canonical_ids), "aligned_questions": len(matched),
        "not_generated": len(missing), "missing_canonical_question_ids": missing,
        "unaligned_generated": sum(item["alignment_status"] != "aligned" for item in observed),
        "unaligned_jev_decisions": unaligned_decisions,
        "observed_questions": observed,
    }


def _attempts(telemetry: Optional[RunTelemetry], provider: str, *, minimum: int = 0) -> int:
    if telemetry is None:
        return minimum
    if provider == "openai":
        return max(minimum, telemetry.n_model_call_attempts, telemetry.n_model_calls)
    return max(minimum, telemetry.n_jev_call_attempts, telemetry.n_jev_calls)


def _stage_accounting(*stages: Optional[tuple[Optional[RunTelemetry], str, int]]) -> Dict[str, Any]:
    """Charge only new stages, retaining known usage and explicitly unknown calls.

    Zero recorded tokens is never a billing assertion for an unanswered request.
    The minimum attempt is used at a failed provider boundary only.
    """
    paid = {"openai_call_attempts": 0, "jev_call_attempts": 0,
            "openai_responses_with_metadata": 0, "jev_completed_calls": 0,
            "openai_input_tokens": 0, "openai_output_tokens": 0,
            "openai_estimated_cost_usd": 0.0, "jev_input_tokens": 0,
            "jev_questions": 0, "new_stage_latency_s": 0.0,
            "openai_usage_unknown_calls": 0, "jev_usage_unknown_calls": 0,
            "unknown_usage_billing_usd": None}
    for stage in stages:
        if stage is None:
            continue
        telemetry, provider, minimum = stage
        attempts = _attempts(telemetry, provider, minimum=minimum)
        paid[provider + "_call_attempts"] += attempts
        responses = 0 if telemetry is None else (telemetry.n_model_calls if provider == "openai" else telemetry.n_jev_calls)
        paid[provider + "_usage_unknown_calls"] += max(0, attempts - responses)
        if telemetry is None:
            continue
        paid["openai_responses_with_metadata"] += telemetry.n_model_calls
        paid["jev_completed_calls"] += telemetry.n_jev_calls
        paid["openai_input_tokens"] += telemetry.openai_input_tokens
        paid["openai_output_tokens"] += telemetry.openai_output_tokens
        paid["openai_estimated_cost_usd"] += telemetry.estimated_cost
        paid["jev_input_tokens"] += telemetry.jev_input_tokens
        paid["jev_questions"] += telemetry.n_jev_questions
        paid["new_stage_latency_s"] += telemetry.latency_total_s
    return paid
