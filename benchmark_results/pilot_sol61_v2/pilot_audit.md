# Pilot audit — 2026-09-30

No additional paid calls were made during this audit. Scope remains exactly workout_before_four_soft and regional_week across eight architectures. Raw results, CSV summary, caches, and Jev logs were not altered.

Preflight from the original invocation: clean working tree, expected models, credentials present in .env without exposure, 243 deterministic tests passed, dry-run 16 pending cells with four paired extractions / 12 expected OpenAI calls / four expected Jev calls. Timeout 240, automatic retries zero.

## Execution history and final state

latest.jsonl has 32 rows and 16 distinct result keys. Each key has one original sandbox extraction transport failure and one subsequent success. Original failures attempted four OpenAI calls total, received no responses, and made no Jev calls. The later run attempted and completed 12 OpenAI calls (four extractions and eight Direct solves) and four Jev calls. All final execution statuses are success; there are zero terminal model-output failures and zero timeouts. One successful structured output fails canonical validation. No invalid output was resampled. Cumulative persisted attempts: 16 OpenAI and four Jev; successful live collection: 12 OpenAI and four Jev.

The CSV has exactly 16 rows. Every status and evaluation field agrees with the last raw row for its result key, with no original transport-failure row selected. The summary inherits four erroneous structural-match flags; see the offline correction below.

## Sixteen-cell results

Every final cell has valid_output=true, feasible_correctly_reported=true, required_completed=true, and canonical_feasible=true. Success means execution completed, not that its schedule is canonically valid. Structural/full are the recorded values. All workout structural=false values marked * re-evaluate to true offline. Latency/cost/call columns are standalone architecture estimates including shared stages.

| Case | Model | Engine | Jev | Canonical valid | Structural/full | Constraints k/n | Violations | Objective/optimum/gap | Latency s | OpenAI estimate USD | OpenAI/Jev calls |
|---|---|---|---|---|---|---|---|---|---:|---:|---|
| Workout | Luna | direct_llm | False | True | False*/False | 3/3 | 0 | 0/0/0 | 8.807 | 0.0003792 | 2/0 |
| Workout | Luna | cp_sat | False | True | False*/False | 3/3 | 0 | 0/0/0 | 4.800 | 0.0002688 | 1/0 |
| Workout | Luna | direct_llm | True | True | True/False | 3/3 | 0 | 0/0/0 | 11.008 | 0.0003982 | 2/1 |
| Workout | Luna | cp_sat | True | True | True/False | 3/3 | 0 | 0/0/0 | 5.008 | 0.0002688 | 1/1 |
| Workout | Sol | direct_llm | False | True | False*/False | 3/3 | 0 | 0/0/0 | 10.717 | 0.0064940 | 2/0 |
| Workout | Sol | cp_sat | False | True | False*/False | 3/3 | 0 | 0/0/0 | 5.726 | 0.0048460 | 1/0 |
| Workout | Sol | direct_llm | True | True | True/False | 3/3 | 0 | 0/0/0 | 9.402 | 0.0064240 | 2/1 |
| Workout | Sol | cp_sat | True | True | True/False | 3/3 | 0 | 0/0/0 | 5.922 | 0.0048460 | 1/1 |
| Regional | Luna | direct_llm | False | False | True/True | 215/218 | 3 | 960/840/None | 33.325 | 0.0024271 | 2/0 |
| Regional | Luna | cp_sat | False | True | True/True | 218/218 | 0 | 840/840/0 | 12.682 | 0.0010126 | 1/0 |
| Regional | Luna | direct_llm | True | True | False/False | 218/218 | 0 | 1080/840/240 | 57.499 | 0.0037489 | 2/1 |
| Regional | Luna | cp_sat | True | True | False/False | 218/218 | 0 | 840/840/0 | 12.921 | 0.0010126 | 1/1 |
| Regional | Sol | direct_llm | False | True | True/True | 218/218 | 0 | 840/840/0 | 61.911 | 0.0423020 | 2/0 |
| Regional | Sol | cp_sat | False | True | True/True | 218/218 | 0 | 840/840/0 | 23.582 | 0.0213120 | 1/0 |
| Regional | Sol | direct_llm | True | True | False/False | 218/218 | 0 | 840/840/0 | 78.540 | 0.0470660 | 2/1 |
| Regional | Sol | cp_sat | True | True | False/False | 218/218 | 0 | 840/840/0 | 23.862 | 0.0213120 | 1/1 |

Luna Direct without Jev on regional_week violates Gray location eligibility for 20261005_morning, Casey required day off on 2026-10-07, and Harper required day off on 2026-10-10. Its objective 960 is recorded diagnostically; objective gap is null because canonical validation fails. Luna Direct with Jev is canonically valid at 1080 versus optimum 840 (gap 240). All CP-SAT regional schedules are canonically valid at 840, as are both Sol Direct regional schedules. These are observations from one sample per arm, not generalized rankings.

Both workout extractions interpret the 16:00 timing as soft: finish_before preference, latest_end=null, task required=true, preference weight=1. Jev retains soft with probability .99 in both calls and does not change either workout formulation. All eight schedules are canonically valid with objective/optimum/gap 0/0/0. Full matching fails because the fixture has weight 3. No label accuracy or calibration claim is made.

## Model provenance and exact calls

All four extraction caches request and resolve gpt-6-luna or gpt-6.1-sol as expected. Each Direct row records the same singleton resolved model after extraction and solve. Response metadata are aggregated into a deduplicated model list; separate response IDs/raw HTTP bodies are not retained. The list supports the observed model provenance, but is not a separate raw response record for each solve. Every Sol request configuration uses gpt-6.1-sol. All cache fingerprints were reconstructed and matched against pre-fix source/dependency provenance and requested models. No stale Sol 6 stage is present.

| Case | Requested/resolved model | OpenAI stage | Input tokens | Output tokens | Estimated USD |
|---|---|---|---:|---:|---:|
| workout_before_four_soft | gpt-6-luna | extraction | 1488 | 240 | 0.0002688 |
| workout_before_four_soft | gpt-6-luna | direct_jev_false | 449 | 131 | 0.0001104 |
| workout_before_four_soft | gpt-6-luna | direct_jev_true | 449 | 169 | 0.0001294 |
| workout_before_four_soft | gpt-6.1-sol | extraction | 1488 | 187 | 0.0048460 |
| workout_before_four_soft | gpt-6.1-sol | direct_jev_false | 449 | 75 | 0.0016480 |
| workout_before_four_soft | gpt-6.1-sol | direct_jev_true | 449 | 68 | 0.0015780 |
| regional_week | gpt-6-luna | extraction | 2236 | 1578 | 0.0010126 |
| regional_week | gpt-6-luna | direct_jev_false | 2500 | 2329 | 0.0014145 |
| regional_week | gpt-6-luna | direct_jev_true | 2568 | 4959 | 0.0027363 |
| regional_week | gpt-6.1-sol | extraction | 2236 | 1684 | 0.0213120 |
| regional_week | gpt-6.1-sol | direct_jev_false | 2500 | 1599 | 0.0209900 |
| regional_week | gpt-6.1-sol | direct_jev_true | 2602 | 2055 | 0.0257540 |

Direct-stage tokens/cost above are derived by subtracting the corresponding extraction telemetry from the combined row telemetry. Each extraction reports one completed call/attempt; each Direct row reports two, consisting of extraction plus one solve.

## Jev observations and paired integrity

All four Jev calls completed successfully with requested model jev-latest and resolved version jev-1.13.0. This demonstrates TypeSafe authentication succeeded. All 76 decisions are present in caches and the unique decision log, with semantic ID, locator/context, normalized probability distribution, GPT baseline, Jev proposal, confidence, applied/changed flags, final value and represented status. Missing/unaligned fields are explicit and zero in final Jev rows. Original transport rows preserve missing canonical questions.

| Case/model | Questions | Applied | Changed | Jev input tokens |
|---|---:|---:|---:|---:|
| workout_before_four_soft/gpt-6-luna | 4 | 3 | 0 | 825 |
| workout_before_four_soft/gpt-6.1-sol | 4 | 3 | 0 | 825 |
| regional_week/gpt-6-luna | 34 | 28 | 2 | 7719 |
| regional_week/gpt-6.1-sol | 34 | 30 | 3 | 7719 |

Question totals: six Choice (two each hardness/requirement/mode), six Score (two workout weights plus four Workforce objective weights), 64 Noul availability questions. Applied count 64, actual changed count five. Workforce has 32 availability + two weight questions per model. Workout has four per model.

Workout proposed weight 2 is rejected at selected modal probability .30 (Luna) and .32 (Sol), below .70; final weight remains 1. Luna modal tie resolves to the lower level. Expected SDK scores 1.41/1.24 and model confidence .17/.29 remain distinct from selected probability.

Regional Luna adds unavailability Casey/20261007_morning at .96 and Finley/20261006_morning at .74. Regional Sol adds those at .97/.71 plus Blake/20261007_morning at .74. Other differing proposals below threshold remain unapplied and their final baseline values are retained. These additions make recorded full and structural formulations differ from the canonical specification. This reports formulation differences without claiming answer-key accuracy.

For each case/model, all four row extraction hashes agree with the cache hash, and the Jev cache binds the same extraction fingerprint/hash. Both Jev engines have identical final_problem and Jev fingerprint, matching the cache. Workout hash: 1369c22b0079db916a89209e6a18f6fafabcb34e32b39c81d2918475aa24f1c7. Regional hash: de0022980a25540858124b42d5badefc53786a38968dbb9f65cd86a57ef28463. Each hash happens to agree across the two models too.

Reconstructed requests from the actual cached extractions use neutral contexts without GPT baseline values, weight, task requirement/mode, hardness locator, or availability selection role. Internal log context deliberately retains these fields. Original HTTP bodies are not persisted, so this is code-path/reconstruction verification plus passing deterministic SDK serialization tests, not a captured-wire audit.

## Accounting

Successful live collection: 19,414 OpenAI input tokens, 15,074 output tokens, 17,088 Jev input tokens. Luna: 9,690 input / 9,406 output / $0.005672. Sol: 9,724 input / 5,668 output / $0.076128. Actual incremental recorded OpenAI estimate under stage sharing: $0.081800. Summing standalone architecture estimates gives $0.1641182 and duplicates shared stages; it is not actual spend. All four initial transport attempts have zero recorded usage and no confirmed provider billing. No authoritative Jev dollar pricing is configured, so no dollar estimate is assigned.

Standalone latency includes reused stages, and incremental new-stage latency sums to 224.483 seconds for the later run; this is stage accounting, not a separately measured invocation wall time.

## Implementation defect and deterministic fix

Four workout Jev-off rows record structural=false while all eight persisted final formulations are identical and Jev reports no workout changes. Offline re-evaluation of every persisted formulation gives structural=true for all eight; full remains false. The CSV correctly selects the final attempts but inherits these incorrect recorded flags.

Root cause demonstrated deterministically: DayPlan formulation comparison used raw datetime.time equality for horizon/preferences/events while serializers and operational solvers use local clock minutes. A timezone suffix can make a fresh arm compare differently from the same formulation after persistence or Jev validation. Adding UTC timezone metadata reproduces false before the roundtrip and true afterward. Raw provider output is not retained, so the precise live suffix cannot be recovered. The confirmed defect is representation-sensitive comparison, with an observed live scoring inconsistency.

Small fix: normalize canonical and arm through existing DayPlan JSON serialization before formulation comparison. A regression checks timezone serialization stability, weight-sensitive full matching, and preservation of real horizon differences. Full deterministic suite after fix: 244 passed. No paid rerun occurred. Original results and summary remain untouched; offline_audit_details.json lists the four metric corrections. Changing source invalidates old fingerprints; these artifacts remain pre-fix evidence and must not be resumed to spend more automatically.

## Readiness

After this deterministic fix, the system is technically ready to collect the full 112-cell benchmark from an environment with outbound API access, using a fresh isolated final directory. Model access, TypeSafe authentication, call sharing, persisted coverage, and accounting were exercised successfully. This small pilot does not establish statistical performance or eliminate risk of later provider/output failures. The Codex sandbox still has an observed connectivity limitation. Human labels remain unapproved; no answer-key accuracy, calibration, or Brier claims are authorized. No full benchmark was run.
