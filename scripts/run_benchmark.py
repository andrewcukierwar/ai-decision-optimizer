"""Run the paired, resumable eight-architecture research benchmark.

Examples:
  .venv/bin/python scripts/run_benchmark.py --all-configurations --dry-run
  .venv/bin/python scripts/run_benchmark.py --model gpt-6-luna --jev on \
      --engine direct_llm --case active_and_passive --allow-paid

Live execution is refused unless ``--allow-paid`` is explicit. Results append
incrementally under ``benchmark_results/pilot_sol61_v2/``. Successful cells and
terminal output failures/timeouts are skipped on resume; transport failures can retry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from decision_optimizer import config as _config  # noqa: F401 - loads .env
from decision_optimizer.benchmark import (
    ALL_EXPERIMENT_CONFIGS,
    BenchmarkCase,
    BenchmarkRunner,
    LiveBenchmarkBackend,
)
from decision_optimizer.config import openai_api_key, typesafe_api_key
from decision_optimizer.experiment import ExperimentConfig
try:
    from scripts.label_jev_fixtures import (
        DEFAULT_SELECTION,
        canonical_problem,
        load_selected_cases,
    )
except ModuleNotFoundError:  # direct ``python scripts/run_benchmark.py`` execution
    from label_jev_fixtures import (  # type: ignore[no-redef]
        DEFAULT_SELECTION,
        canonical_problem,
        load_selected_cases,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "benchmark_results" / "pilot_sol61_v2"


def load_benchmark_cases(
    selection: Path = DEFAULT_SELECTION,
    selected_names: List[str] | None = None,
) -> List[BenchmarkCase]:
    raw_cases = load_selected_cases(selection)
    if selected_names:
        available = {case["name"] for case in raw_cases}
        unknown = set(selected_names) - available
        if unknown:
            raise ValueError("Unknown case(s): " + ", ".join(sorted(unknown)))
        selected = set(selected_names)
        raw_cases = [case for case in raw_cases if case["name"] in selected]
    return [
        BenchmarkCase(
            case_id=case["name"],
            domain=case["domain"],
            prompt=case["prompt"],
            canonical=canonical_problem(case),
            source=case["source"],
            clarification=case.get("clarification"),
        )
        for case in raw_cases
    ]


def selected_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.all_configurations:
        return list(ALL_EXPERIMENT_CONFIGS)
    missing = [
        flag
        for flag, value in (
            ("--model", args.model),
            ("--jev", args.jev),
            ("--engine", args.engine),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Specify --all-configurations or all of " + ", ".join(missing)
        )
    return [
        ExperimentConfig(
            model=args.model,
            use_jev=args.jev == "on",
            solution_engine=args.engine,
        )
    ]


def build_live_backend(args: argparse.Namespace) -> LiveBenchmarkBackend:
    if not openai_api_key():
        raise ValueError("OPENAI_API_KEY is required for live benchmark calls")
    if any(config.use_jev for config in selected_configs(args)) and not typesafe_api_key():
        raise ValueError("TYPESAFE_API_KEY is required for Jev-on benchmark calls")
    try:
        from openai import OpenAI
        from typesafe_sdk import RetryPolicy, TypeSafeClient
    except ImportError as error:
        raise ValueError("Benchmark API dependencies are not installed") from error

    openai_client = OpenAI(
        api_key=openai_api_key(),
        timeout=args.request_timeout,
        max_retries=0,
    )
    jev_client = None
    if any(config.use_jev for config in selected_configs(args)):
        jev_client = TypeSafeClient(
            api_key=typesafe_api_key(),
            timeout=args.request_timeout,
            retry=RetryPolicy(max_retries=0),
        )
    return LiveBenchmarkBackend(
        openai_client=openai_client,
        jev_client=jev_client,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--all-configurations", action="store_true")
    parser.add_argument("--model", choices=("gpt-6-luna", "gpt-6.1-sol"))
    parser.add_argument("--jev", choices=("on", "off"))
    parser.add_argument("--engine", choices=("direct_llm", "cp_sat"))
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case id to run; repeat for multiple. Omit for all 14 cases.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--solver-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, choices=(0,), default=0, help="Fixed benchmark policy: SDK retries disabled; explicitly resume transport failures.")
    parser.add_argument("--jev-batch-size", type=int, default=64)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.all_configurations and any(
        value is not None for value in (args.model, args.jev, args.engine)
    ):
        parser.error(
            "--all-configurations cannot be combined with --model, --jev, or --engine"
        )
    try:
        configs = selected_configs(args)
        cases = load_benchmark_cases(args.selection, args.cases)
    except ValueError as error:
        parser.error(str(error))

    preview_runner = BenchmarkRunner(
        args.results_dir,
        LiveBenchmarkBackend(),
        request_timeout_seconds=args.request_timeout,
        solver_timeout_seconds=args.solver_timeout,
        jev_batch_size=args.jev_batch_size,
        api_retries=args.api_retries,
    )
    preview = preview_runner.preview(cases, configs)
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return
    if not args.allow_paid:
        parser.error(
            "Live benchmark calls require --allow-paid. Use --dry-run to preview."
        )

    try:
        backend = build_live_backend(args)
    except ValueError as error:
        parser.error(str(error))
    runner = BenchmarkRunner(
        args.results_dir,
        backend,
        request_timeout_seconds=args.request_timeout,
        solver_timeout_seconds=args.solver_timeout,
        jev_batch_size=args.jev_batch_size,
        api_retries=args.api_retries,
    )
    counts = runner.run(cases, configs)
    print(
        json.dumps(
            {"preview_before_run": preview, "completed_this_run": counts},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
