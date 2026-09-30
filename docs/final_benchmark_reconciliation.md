# Final benchmark offline reconciliation

`benchmark_results/final_sol61_v2/reconciled.jsonl` is the authoritative input
for Phase 10. The original observations remain in `latest.jsonl`, the 52
original stage caches, `jev_decisions.jsonl`, and `latest_summary.csv`.
All 55 SHA-256 hashes must match the integrity audit before and after replay.
No benchmark backend or provider is called by reconciliation.

Run from the repository root with the existing local environment:

```bash
.venv/bin/python -m scripts.reconcile_benchmark
```

The script writes only `reconciled.jsonl`, `reconciled_summary.csv`, and
`reconciliation_manifest.json`. Use `--output-dir <directory>` to inspect a
separate copy. Unchanged inputs and source produce byte-identical outputs on
repeat runs, including the retained creation timestamp. New source versions
are recorded explicitly; they do not invalidate or regenerate paid observations.
The manifest includes raw and auxiliary hashes, source and schema versions,
each affected result key, coverage, a unique-stage accounting ledger, and
before/after counts. Each derived row retains its identity and backs up every
superseded raw field under `reconciliation.raw_values_superseded_for_analysis`.

Only the following rules supersede raw fields for analysis:

- Both formulation flags become true for the Jev-off Direct LLM and CP-SAT
  rows under both Luna and Sol for `straightforward_staffing` and
  `infeasible_staffing_is_extracted_not_repaired`: eight rows, sixteen flags.
  The normalized JSON/Pydantic replay must support every correction.
- Incremental accounting is reconstructed from unique shared stage caches and
  Direct solve telemetry after subtracting shared stages. Shared stages are
  charged exactly once, using the original row's attempt allocation. Completed
  usage omitted from failure rows is restored. Standalone architecture
  estimates remain preserved and must never be summed as actual expenditure.
- Luna Jev Direct on `workout_before_four_hard` receives the three completed
  cached/logged decisions, resolved model `jev-1.13.0`, shared final formulation,
  and corresponding observation alignment. This is upstream evidence; the
  exact timed-out Direct request body was not persisted. No solution is inferred.

All 112 rows are replayed against canonical specifications. Any additional
metric, evidence, pairing, provenance, coverage, or accounting discrepancy
stops reconciliation before writing outputs. Outcomes remain 103 execution
successes and nine terminal failures, including eight execution-success rows
with `valid_output=false`. Terminal failures remain four Luna soft-workout
extraction timeouts, four Luna impossible-workout invalid extractions, and one
Luna hard-workout Jev Direct solve timeout.

Recorded experiment totals are 76 OpenAI attempts, 74 responses with metadata,
86,583 input tokens, 46,461 output tokens, and $0.2874103 estimated cost.
The two unanswered timeout requests have unknown usage and billing; the
recorded token subtotals exclude that unknown usage. Jev has 24 attempts and
24 completed calls, 50,946 input tokens, and 230 questions. Jev dollar cost is
unassigned. These are recorded token estimates, not provider invoices.

Labels remain 54 human-approved fixture labels, 71 generator-derived labels,
and three excluded subjective labels. Across two models there are 250 primary
eligible instances, 223 aligned, and 27 missing. Overall coverage is 256 label
instances, 228 aligned, 28 missing, two unaligned generated questions, and zero
unattached observations. Join canonical labels by the explicit canonical
alignment ID. Missing and unaligned questions retain their status.

This boundary establishes trustworthy Phase 10 inputs only. It computes no
architecture rankings, answer accuracy, Brier scores, or calibration plots.
