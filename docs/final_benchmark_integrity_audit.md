# Final benchmark integrity audit — 2026-09-30

**Determination: not ready for Phase 10 as-is.** Collection is complete and the observations are recoverable. Phase 10 needs offline reconciliation of eight formulation records, failure-stage usage, and one failure row's Jev observations. No paid rerun is needed. No repository code or raw experimental artifacts were changed. No accuracy, Brier, calibration, architecture rankings, or README claims were calculated.

## Completion and failure classification

- Exactly **112 unique cells**, covering the current **14 cases × eight architectures**; no missing, unexpected, duplicate, or resampled result keys.
- **103 execution-success rows + nine terminal-failure rows**. The summary selects exactly these 112 final states and agrees with every persisted summary metric.
- Terminal failures: Luna extraction timeout on `workout_before_four_soft` affects four paired cells; Luna invalid extraction (duplicate task names) on `impossible_hard_workout_window` affects four paired cells; Luna Jev Direct solve timeout on `workout_before_four_hard` affects one cell. Thus nine cells reflect **two timeout requests and one invalid-output request**.
- **Zero transport/provider failure rows**. All terminal rows are non-retryable, scored with `valid_output=false`, and retained in the denominator. No terminal cell has a subsequent row. SDK retries are zero and request timeout is 240 seconds throughout.
- Eight `success` rows have no usable formulation/output: four Luna `wfh_laundry_lift_groceries` arms and four Sol `optional_walk_if_time` arms. Their extraction caches preserve `missing_info`; `valid_output=false`. They are completed clarification/abstention outcomes, not successful schedules or transport failures. There is no supplied clarification for these cases, and no follow-up call was made.
- Final row timestamps: **2026-09-30 14:16:50 EDT through 14:36:32 EDT**. These follow all pilot rows.

## Paid attempts, responses, and accounting

Each of 28 extraction pairs has exactly one attempt. The 24 pairs with usable formulations generate 48 Direct solve attempts and 24 Jev calls. Two OpenAI requests time out without response usage; the invalid-output response retains its model and usage metadata.

| Measure | Reconciled from unique stages | Sum of row incremental fields |
|---|---:|---:|
| OpenAI attempts | 76 | 76 |
| OpenAI responses with metadata | 74 | — |
| OpenAI input tokens | 86,583 | 85,099 |
| OpenAI output tokens | 46,461 | 45,989 |
| Recorded OpenAI estimated cost | $0.2874103 | $0.2870259 |
| Jev attempts / completed calls | 24 / 24 | 24 attempts |
| Jev input tokens | 50,946 | 50,302 |
| Jev questions / observations | 230 | — |

Reconciliation counts each extraction cache once, each Jev cache once, and each Direct solve by subtracting its extraction telemetry from its standalone row telemetry. Every stage cost agrees with the configured token estimator. Luna contributes 36 attempts / 34 responses, 40,988 input / 28,683 output tokens, and $0.0184403. Sol contributes 40 attempts / 40 responses, 45,595 input / 17,778 output tokens, and $0.2689700.

The raw row totals omit **1,484 input tokens, 472 output tokens and $0.0003844** from the invalid Luna extraction, plus **644 Jev tokens** charged to the timed-out Luna Jev Direct cell. Failure branches persist attempt counts but omit incremental token/cost fields. These values remain recoverable from caches. Summed standalone cost is **$0.6507532** and duplicates shared stages; it is not actual incremental spend. Recorded token estimates are not provider invoices; the two timed-out requests have unknown provider billing. No authoritative TypeSafe dollar pricing is available, so no Jev dollar estimate is assigned.

## Models, provenance, and pairing

Requested OpenAI configurations are exclusively `gpt-6-luna` and `gpt-6.1-sol`. Every recorded resolved OpenAI model matches its requested model; the Luna extraction timeout has no resolved response. OpenAI provenance is an aggregated model list, not a retained raw response/ID per call. All 24 Jev caches and 230 decision logs request `jev-latest` and resolve **`jev-1.13.0`**. The timed-out Jev Direct row incorrectly records no Jev version, although its completed cache/log records it.

All **52 cache fingerprints** and **112 result identities** match current source/dependency, prompt, canonical, requested-model, and runtime provenance. All cache/result schemas are version 2. No stale Sol 6 models or cache names exist. All available cache-hit flags are false. Overlapping pilot cache contents and fingerprints differ, pilot/final result identities are disjoint, and final timestamps are later. These checks show no persisted pilot reuse; separate raw HTTP response IDs are unavailable for a provider-level audit.

All four rows within each case/model pair share the expected original extraction hash when an extraction exists. The two completed null-formulation pairs share their null-problem hash consistently. Both terminal extraction pairs propagate the same cached terminal failure and have null extraction hashes. Jev fingerprints bind the matching extraction fingerprint and hash.

For 24 completed Jev stages, every available Jev-on final formulation matches its shared cache. Direct/CP-SAT formulations agree in all 23 pairs with both formulations persisted. The remaining Luna `workout_before_four_hard` Direct timeout row omits its final formulation; its CP-SAT row and Jev cache preserve the shared formulation, and the runner's common stage path is verified. Its exact Direct request body is not persisted, so raw-row equality cannot be asserted for that failed cell.

## Canonical evaluation

Offline replay checked all 112 cells against current canonical specifications, reconstructing the failed solve's final formulation from its Jev cache. Validity, feasibility correctness, completion, constraint counts/violations, objective/optimum/gap, and telemetry agree with the persisted values. The only metric disagreements are **eight Workforce rows / 16 full-and-structural formulation flags**, listed below.

All **16 infeasible-case rows** have null hard-constraint numerator/denominator, required completion, and objective metrics. They preserve `feasible_correctly_reported` as the correctness outcome. Every non-null objective gap occurs only for a canonically valid feasible output and equals objective minus optimum. Full formulation match never occurs without structural match. Hard-constraint counts remain diagnostics, including for absent output; they must not substitute for the binary canonical-validity outcome.

## Finalized labels and join coverage

JSON, TSV, and manifest metadata agree: **54 human-reviewed approved fixture labels + 71 generator-derived deterministic labels = 125 primary labels**, with three subjective weights retained and excluded. Canonical expected answers are unchanged from the independently generated fixture proposals. The exclusion gate rejects the subjective labels even if their eligibility flag or review status is accidentally toggled. The earlier methodology/remediation documents describe the historical pending-review state; the finalized artifacts and explicit human decisions supersede that status.

The excluded labels are `soft_finish_preference/pref_weight=1`, `workout_before_four_soft/pref_weight=3`, and `preferences_and_availability/pref_weight=1`.

Count once per case/model extraction/Jev stage, not once per solution engine:

| Coverage measure | Count |
|---|---:|
| Canonical label instances across two models | 256 |
| Aligned observed questions | 228 |
| Missing canonical questions | 28 |
| Unaligned generated questions | 2 |
| Observations not attached to a generated question | 0 |
| Primary eligible label instances | 250 |
| Primary aligned observations | 223 |
| Primary missing instances | 27 |

The 28 missing items are 15 Luna WFH clarification items, four Luna soft-workout timeout items, three Luna impossible-workout invalid-extraction items, four Sol optional-walk clarification items, and two Luna optional-walk omissions. One missing soft-workout item is subjective, leaving 27 primary missing instances. Both `active_and_passive` extractions generate one additional unaligned question; both observations remain explicit in the decision log. They must not acquire labels by position or fuzzy fallback.

All **230 decision logs** are unique and match their caches, including complete normalized distributions, selected probability, threshold application, semantic identity, locator/context, and represented/not-represented final value. Five aligned observed instances are subjective exclusions. Missing-label IDs and unaligned questions remain explicit. Canonical joins must use `(case_id, answer_key_alignment.canonical_question_id)`; raw semantic IDs may preserve model wording/case and are not direct canonical IDs.

## Suspicious records and smallest proposed fixes

1. **Representation-sensitive Workforce formulation comparison.** Jev-off Direct and CP-SAT rows for both models on `straightforward_staffing` and `infeasible_staffing_is_extracted_not_repaired` record full/structural=false; replay of their persisted formulations gives true/true. Raw unavailability time equality is sensitive to timezone metadata, while serializers and solvers use local clock minutes. Adding UTC metadata deterministically reproduces false before a validation roundtrip and true afterward. Original provider output is unavailable, so the precise live suffix cannot be recovered. Smallest source fix: normalize both ShiftSchedule inputs through their existing JSON representation before formulation comparison, mirroring the existing DayPlan fix; add a timezone-roundtrip regression. For this dataset, preserve raw flags and record an explicit derived correction for these eight rows.
2. **Failure-row stage accounting.** The extraction/solve failure branches only record incremental attempts. Smallest fix: use the same stage charge flags to persist token and cost deltas on failed rows. For existing observations, derive totals from unique stage caches plus Direct solve deltas rather than summing incomplete row fields. Preserve missing timeout usage as unknown billing.
3. **Successful Jev evidence lost from a failed solve row.** Luna Jev Direct on `workout_before_four_hard` reports no Jev version/decisions despite one completed Jev call and three cached/logged observations. `_append_failure` takes observations only from the thrown solve error rather than the completed Jev stage. Smallest fix: pass the existing stage's model/decisions/final formulation to the failure writer. For Phase 10, join that row to its existing shared cache; do not treat its three decisions as missing or sample replacement answers.

The underlying observations are sufficient to repair these derived fields offline. **Do not use the CSV or row-level accounting as the sole Phase 10 input.** Establish the small reconciliation/normalization layer and deterministic regressions before reporting results. No experimental rerun, relabeling, or methodological redesign is required.

## Preservation and verification

Read all 55 persisted final files. SHA-256 snapshots in `final_benchmark_integrity_audit_details.json` verified every original artifact remained unchanged. The audit ran local assertions and canonical replay only; no API calls, benchmark execution, or repository implementation changes occurred. No full test-suite rerun was needed because no code changed. No headline performance or label-answer scores were computed.
