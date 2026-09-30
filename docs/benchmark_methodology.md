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
cell. A successful cell is skipped on resume; a failure remains retryable on a
later invocation. Cache schema versions are explicit but manually versioned, so
methodology changes that alter a cached stage must bump the cache version.
