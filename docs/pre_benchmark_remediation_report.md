# Pre-benchmark remediation report — 2026-09-30

All requested P0/P1 corrections are implemented. No paid API calls or live full
benchmark execution were performed. The runner is ready for a paid pilot;
answer-key approval is still required before accuracy/calibration claims.

## Resolutions

| Item | Resolution |
| --- | --- |
| P0 1 — Output failures | Specific extraction/refusal, Direct LLM output, and Jev answer-contract failures are terminal, canonically scored with `valid_output=False`, and skipped on repeat/resume. Shared extraction/Jev terminal failures are cached. Arbitrary Pydantic errors retain infrastructure/implementation attribution. Default SDK timeout is 180 seconds per HTTP operation/inactivity, terminal with a distinct timeout category; automatic SDK retries are disabled. Transport failures are separate, append-only, and retryable on explicit resume. Response metadata and valid partial Jev observations survive SDK parsing failures. |
| P0 2 — Infeasible evaluation | Both domains emit null hard-constraint counts and required completion on canonical infeasibility. Correctness uses `feasible_correctly_reported`; UNKNOWN is not infeasibility. Feasible cases use binary canonical validity; violations and counts are diagnostics. Empty Workforce assignments cannot earn a valid headline outcome. |
| P1 3 — Score gating | Score uses modal level, SDK 0–4 maps to weight 1–5, and selected probability controls application. Lowest level wins ties. Expected score, SDK confidence, and the complete distribution remain separate. |
| P1 4 — Jev leakage | Actual SDK state excludes GPT answers, weights, task requirement/mode classes, hardness locators, and availability-selection roles. Instructions and context use neutral descriptions. Internal GPT answers and locators remain available for evaluation/logging. Existing question targeting is preserved. |
| P1 5 — Structural accuracy | Added primary `formulation_match_structural`, preserving meaningful structure while ignoring preference/objective weights. Full formulation matching remains. Methodology specifies separate ordinal weight error/agreement. |
| P1 6 — Fixture defects | Added a benchmark-manifest prompt override with 2026-10-05 and all four shift IDs. The original regression fixture is unchanged. Deduplicated the generator collision for Harper and regenerated Workforce fixtures. Identical duplicate unavailability entries compare idempotently. |
| P1 7 — Direct assignments | DayPlan Direct LLM explicitly returns task assignments only and excludes fixed events. Unknown/fixed-event extras remain visible and canonically invalid. |
| P1 8 — Migration/provenance | Active models are Luna 6 and Sol 6.1 only. Defaults, labels, CLI choices, tests, README/plan, and ignored local `.env` updated. Official dated prices are nonzero; unknown model pricing raises. Actual resolved response models are logged when available. Version-2 source/dependency/prompt/config fingerprints prevent incompatible cache/result reuse; actual extraction hashes are persisted and paired equality enforced. Pilot defaults use a separate directory; final instructions require a fresh isolated directory. |
| P1 9 — Decisions/answer key | Shared semantic identities, locators, context, distributions, proposed/applied/final values, and canonical alignment are logged. Missing/unaligned questions remain explicit. The labeling CLI attaches actual verified benchmark caches without another GPT/Jev sample. Failed chunks retain their valid observations without claiming they were applied. |
| P1 10 — Numeric IDs | Name alignment rejects conflicting numeric identities, including adjacent dates and different shift numbers; equivalent date formatting and harmless wording changes still align. |

## Labeling and subsequent analysis

The regenerated JSON/TSV review queue has **128 proposals**: **57** fixture
proposals marked `needs_human_review`, and **71** generator-derived proposals
marked `generator_derived_not_human_reviewed`. **Zero** were human-approved.
The three specified subjective weight labels remain documented context and
are excluded from primary exact-answer accuracy/calibration until a defensible
rubric is agreed upon. The other 125 proposals still require review before
primary label-based claims.

Methodology now specifies separate Choice/binary and Noul Brier scores from
five-class Score Brier; accuracy conditional on alignment; end-to-end accuracy
including GPT omissions; final represented decision accuracy; and coverage,
missing, invalid, and unmatched counts. No Phase 10 analysis script was built.

## Verification

- Complete deterministic suite: **243 passed**, zero failures. The previous
  196-test suite is retained, with **47 additional parametrized regression
  cases** in `tests/test_benchmark_remediation.py`; existing model and Jev mock
  tests were updated to reflect the corrected semantics.
- Full mocked runner: **14 cases × 8 architectures = 112 successful cells**,
  zero failed cells, **28 shared extractions**, **28 shared Jev stages**, and
  **112 solves**. All paired extraction hashes agree. Repeat skips all 112.
- Real OpenAI and TypeSafe SDK HTTP paths use mock transports in tests, including
  invalid output, timeout, metadata retention, and serialized Jev request checks.
- Dry run: **112 pending cells**, **28 case/model extraction pairs**, estimated
  **84 OpenAI calls + 28 Jev calls = 112 calls**, **256 Jev questions** estimated
  from canonical specs across both models. These are estimates, not executed
  paid calls.
- Active architecture models: `gpt-6-luna` and `gpt-6.1-sol`. Local ignored
  `OPENAI_MODEL` verified as `gpt-6.1-sol`; credentials were untouched.
- `git diff --check` passes.

Standard short-context pricing was checked on 2026-09-30 against official
[Sol 6.1 model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
and [Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
The estimator uses $2/$10 and $0.10/$0.50 input/output per million tokens,
respectively, for the requested base model; resolved response model provenance
is separately recorded. Historical README MVP results remain labeled as
pre-migration results, never as Sol 6.1 benchmark observations.

## Remaining work and pilot readiness

There is no known implementation issue preventing a paid pilot. Provider
account/model access has not been tested with paid requests, as instructed.
Humans must review the answer queue, confirm generator-derived labels, and
approve a defensible strength rubric before including the three subjective
weights in primary accuracy/calibration. Collection for a paid pilot can
proceed while labels remain pending; final accuracy/calibration claims cannot.

Use `benchmark_results/pilot_sol61_v2/` for the pilot. The subsequent final
benchmark should start with a fresh explicit
`--results-dir benchmark_results/final_sol61_v2`; do not copy development,
old Sol, or pilot artifacts there.

## Files changed
- `.env.example`
- `README.md`
- `app.py`
- `decision_optimizer/benchmark.py`
- `decision_optimizer/config.py`
- `decision_optimizer/direct_solver.py`
- `decision_optimizer/evaluation.py`
- `decision_optimizer/experiment.py`
- `decision_optimizer/failures.py`
- `decision_optimizer/jev.py`
- `decision_optimizer/jev_alignment.py`
- `decision_optimizer/parsing/dayplan.py`
- `decision_optimizer/parsing/shift_schedule.py`
- `decision_optimizer/telemetry.py`
- `docs/benchmark_methodology.md`
- `docs/pre_benchmark_remediation_report.md`
- `project_plan.md`
- `scripts/generate_workforce_cases.py`
- `scripts/label_jev_fixtures.py`
- `scripts/run_benchmark.py`
- `tests/fixtures/generated_workforce_cases.json`
- `tests/fixtures/jev_benchmark_selection.json`
- `tests/fixtures/jev_expected_draft.json`
- `tests/fixtures/jev_expected_review.tsv`
- `tests/test_benchmark.py`
- `tests/test_benchmark_remediation.py`
- `tests/test_direct_solver.py`
- `tests/test_experiment_telemetry.py`
- `tests/test_jev.py`
- Local ignored `.env`: model setting only; no credential changes.
