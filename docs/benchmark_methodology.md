# Benchmark methodology notes

## Workforce availability question selection

Availability questions are generated only for employees whose extracted typed
record contains at least one `unavailable` entry. A preferred shift is a soft
objective signal and is never treated as evidence that an availability
statement exists.

For each such employee, the selector includes:

1. every candidate shift matched by an explicit unavailable shift ID;
2. every candidate shift on an unavailable day or overlapping an unavailable
   time window; and
3. at most two deterministic negative controls, preferring same-day temporal
   contrasts and then the chronologically closest unmatched shifts.

The controls make false/negative answers observable without recreating the
employee-by-all-shifts cross product. The cap applies only to controls, not to
positive matches. A broad unavailable window can therefore still generate many
questions when many shifts truly overlap it.

This method depends on GPT having extracted an unavailability statement. It
cannot ask about a statement omitted from the typed formulation. It also tests
shift-level effects, not whether the extracted window boundaries themselves
are exact.

TypeSafe's public System One documentation encourages batching multiple
questions and currently states no maximum number of questions per request.
The runner therefore uses deterministic chunks of 64 by default as an
operational guard rather than as a claimed API limit. The chunk size is
configurable with `--jev-batch-size`. Token, question, call, and latency
telemetry accumulates across all chunks.

## Answer-key status

The draft answer key keeps three concepts separate:

- `canonical_expected_answer`: a proposal derived from fixture intent or the
  generator specification;
- `gpt_baseline_answer`: an optional observation from an actual named-model
  extraction; and
- `jev_answer`: an optional observation from Jev on that extraction.

Semantic question IDs align minor task, employee, and shift naming differences
and normalize equivalent soft-timing and hard-timing representations. A
canonical question that a model never generated is marked `not_generated`.
Questions that cannot align are retained under `unaligned_observed_questions`.

Labels from the generated workforce templates are marked
`generator_derived_not_human_reviewed`. They are deterministic because the
template explicitly renders the canonical unavailable shift IDs and numeric
objective weights. Other fixture proposals remain `needs_human_review`, with
ambiguous language and subjective preference strength called out separately.
No draft status represents human approval.

## Paired benchmark execution

Within each `(case, base model)` pair, all four variants share one original GPT
extraction. The two Jev-on variants also share one Jev-adjusted formulation.
Only the Direct LLM solve differs by making another model call; CP-SAT remains
local. Canonical specifications are used after solving for deterministic
evaluation and are never included in GPT or Jev prompts.

Each successful result records a standalone architecture estimate that includes
the shared stages it would need in isolation. It separately records paid-call
attempts newly created while producing that row, so summing standalone rows
does not masquerade as actual expenditure under reuse.

Intermediate extraction and Jev artifacts use prompt/configuration
fingerprints and atomic writes. Result rows are appended and synced after each
cell. Successful cells and terminal output failures/timeouts are skipped on resume;
transport failures remain retryable on explicit resume. Source and dependency
hashes additionally invalidate incompatible cached stages automatically. See
the fixed failure policy and version-2 provenance below.

## Pre-pilot integrity policy (2026-09-30)

The only active base models are `gpt-6-luna` (Luna 6) and `gpt-6.1-sol`
(Sol 6.1). There are eight architectures. Sol 6 artifacts are historical and
cannot represent Sol 6.1 observations. Standard short-context token estimates,
verified on 2026-09-30, use $0.10 input / $0.50 output per million tokens for
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) and $2 input /
$10 output for [Sol 6.1](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
These are token estimates, excluding cached-input discounts and other tiers.
Unknown model IDs raise an explicit pricing error. Requested model IDs and
actual response model IDs are recorded separately when supplied by the API.

### Failed outputs and fixed deadlines

A completed refused, malformed, absent, or schema-invalid model output is a
`terminal_failure`, categorized `invalid_model_output`, with a canonical
scoring record and `valid_output=False`. It remains in the experiment's
denominator. Output-schema validation errors are classified at the parser or
answer boundary; arbitrary Pydantic errors from local configuration, request
construction, cache loading, or evaluation are not model errors.

The predefined SDK request timeout is 180 seconds by default (per HTTP
operation/inactivity, not a total wall-clock runtime limit). Request timeouts
are terminal failures under the same timeout policy for every arm, with
`failure_category=timeout` and no model-output attribution. A timeout receives
`valid_output=False`; it is never silently replaced on resume. The timeout is
recorded and fingerprinted; a changed timeout defines a new experimental
configuration. Both SDKs have automatic retries disabled in benchmark runs
(`--api-retries` must be zero).

Transport/provider HTTP failures are separately recorded as retryable
`failure` rows with no claimed canonical score. An explicit rerun can retry
these infrastructure failures, preserving the previous append-only attempt.
Report infrastructure failure counts and the resulting coverage separately;
never silently treat them as model successes or silently drop terminal
failures. A shared extraction failure or timeout is cached and affects all
paired variants, including variants first requested on a later invocation.
Jev stage failures are likewise shared by both Jev engines. Valid Jev answers
observed before a failed chunk/stage remain logged as observations with no
applied decision.

### Primary outcome and formulation metrics

For **feasible canonical cases**, the primary validity outcome is binary
`canonical_validation_valid`: all canonical constraints must hold. Required
completion, individual constraint violations, and k/n counts are secondary
diagnostics. Do not average raw k/n fractions as the headline comparison:
absent assignments can vacuously satisfy many pairwise and capacity checks.

For **infeasible canonical cases**, `hard_constraints_satisfied`,
`hard_constraints_total`, and `required_completed` are null (N/A). The primary
outcome is `feasible_correctly_reported`; UNKNOWN is not an infeasible verdict.
A fabricated partial schedule cannot earn a better headline outcome than a
correct infeasibility report. Do not compare objective quality on infeasible
cases. Canonical solver UNKNOWN prevents scoring rather than establishing
infeasibility. Report feasible validity and infeasibility-report accuracy
separately, or combine one binary correctness observation per case with clearly
stated denominators and domain/case weighting.

`formulation_match_structural` is the primary formulation-correctness metric.
It preserves tasks, requirements, modes, timing, hard versus soft structure,
preferences, shifts, staffing, employees, availability, eligibility, capacity,
and rules, while excluding preference and objective weights. Full
`formulation_match` remains available. Identical duplicate unavailability
entries are semantically idempotent. Weight agreement is reported separately
using ordinal absolute error (1–5 weights), exact agreement, and optionally
ordinal weighted agreement; a two-level weight error is not equivalent to a
hard/soft error. For unavailable/omitted weight targets after a rewrite, retain
`final_value_status=not_represented` rather than claiming the accepted proposed
weight exists in the final formulation.

### Jev inputs and probability gating

Question targeting remains dependent on GPT's extracted items. GPT baseline
answers, internal locators, hardness identifiers, task requirement/mode
classifications, preference weights, and availability-selection roles stay in
internal logs. Jev receives the original request and neutral task/timing,
objective-description, or employee/candidate-shift facts. The original natural
language can of course contain explicit requirements and numeric weights.
Tests inspect real SDK-serialized HTTP payloads with mock transports.

Choice uses the probability of the SDK-selected answer. Noul chooses yes at
p≥0.5, otherwise no, and gates on that selected answer's probability. Score
chooses the modal probability level, maps SDK levels 0–4 to optimizer weights
1–5, and applies only when that probability meets `JEV_APPLY_THRESHOLD`.
Score ties choose the lowest level deterministically. The full five-class
distribution, SDK expected score, and SDK confidence are retained separately.

### Cache provenance and isolated final run

Cache and result schema version 2 fingerprints include source hashes of the
active implementation (including extraction prompts, Jev question construction
and schemas), dependency versions, original prompt/clarification, requested
models, endpoint configuration, fixed timeout/retry policy, batch size and
threshold where relevant. Jev cache fingerprints additionally include the
actual extracted problem hash. Result identities also include the canonical
specification and runtime configuration, so incompatible old successful rows
cannot suppress new cells. Each paired row records the original extraction
hash; differing hashes under the same extraction fingerprint are rejected.

The default output directory is `benchmark_results/pilot_sol61_v2/`. Keep
mock/development outputs in temporary directories. A subsequent final run must
use a fresh explicit `--results-dir benchmark_results/final_sol61_v2`; do not
copy pilot caches/results there. This remediation made no paid calls.

### Answer keys and later Phase 10 analysis

The review artifacts contain 128 proposals. Generator provenance remains
`generator_derived_not_human_reviewed`; the other labels remain
`needs_human_review`. No label has been automatically human-approved. The
following three weight labels are retained as context with
`primary_accuracy_calibration_eligible=False` until a defensible strength
rubric is agreed upon: `workout_before_four_soft` preference weight,
`soft_finish_preference` preference weight, and
`preferences_and_availability` preference penalty.

Use `scripts/label_jev_fixtures.py --baseline benchmark
--benchmark-results-dir <actual run directory>` to attach the actual, verified
extraction/Jev cache observations. This path makes no API calls. An independent
development sample is explicitly marked as such and must not be substituted
for a benchmark observation. Logs persist semantic question IDs, locators,
context, GPT answers, observed distributions, applied/changed flags, final
represented values, and canonical alignment IDs. Missing caches, missing
canonical questions, unaligned generated questions, and unaligned Jev
observations remain explicit.

The later analysis must report:

- Choice/binary Brier scores and Noul Brier scores separately from five-class
  Score Brier scores. State the normalization (binary positive-class squared
  error; multiclass sum of squared class errors) and eligible reviewed labels.
- GPT and Jev decision accuracy conditional on canonical question alignment.
- End-to-end accuracy using every eligible canonical item as denominator;
  GPT omissions count as unsuccessful, with infrastructure coverage disclosed.
- Final applied decision accuracy using final represented values, including
  threshold fallbacks and targets removed by prior rewrites. Keep missing
  final targets explicit rather than counting a proposal as applied truth.
- Coverage, not-generated canonical item counts, unmatched question counts,
  invalid/missing API answers, and failures, by case/model/question type.

No complete Phase 10 analysis script is included in this remediation.
