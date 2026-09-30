# Live pilot audit — 2026-09-30

Preflight: initially clean working tree; active architecture models gpt-6-luna and gpt-6.1-sol; local parser model gpt-6.1-sol; both credentials present in .env (values never printed); 243 deterministic tests passed. Dry run: 2 cases, 8 architectures, 16 pending cells, 4 paired extractions, 12 expected OpenAI calls and 4 expected Jev calls. uv cache redirected to /private/tmp/codex-pilot-uv-cache due to filesystem sandbox restriction.

Exactly one live runner invocation used the specified cases, isolated directory, 240-second request timeout, zero retries, and --allow-paid. No paid rerun occurred.

## Persisted coverage

| Case | Architecture | Status/category | Shared extraction latency (s) | Incremental OpenAI/Jev attempts | Recorded standalone OpenAI estimate |
|---|---|---|---:|---:|---:|
| workout_before_four_soft | Luna 6 · Direct LLM | failure / transport_failure | 0.163855 | 1/0 | $0.000000 |
| workout_before_four_soft | Luna 6 · CP-SAT | failure / transport_failure | 0.163855 | 0/0 | $0.000000 |
| workout_before_four_soft | Luna 6 · Jev · Direct LLM | failure / transport_failure | 0.163855 | 0/0 | $0.000000 |
| workout_before_four_soft | Luna 6 · Jev · CP-SAT | failure / transport_failure | 0.163855 | 0/0 | $0.000000 |
| workout_before_four_soft | Sol 6.1 · Direct LLM | failure / transport_failure | 0.002481 | 1/0 | $0.000000 |
| workout_before_four_soft | Sol 6.1 · CP-SAT | failure / transport_failure | 0.002481 | 0/0 | $0.000000 |
| workout_before_four_soft | Sol 6.1 · Jev · Direct LLM | failure / transport_failure | 0.002481 | 0/0 | $0.000000 |
| workout_before_four_soft | Sol 6.1 · Jev · CP-SAT | failure / transport_failure | 0.002481 | 0/0 | $0.000000 |
| regional_week | Luna 6 · Direct LLM | failure / transport_failure | 0.002001 | 1/0 | $0.000000 |
| regional_week | Luna 6 · CP-SAT | failure / transport_failure | 0.002001 | 0/0 | $0.000000 |
| regional_week | Luna 6 · Jev · Direct LLM | failure / transport_failure | 0.002001 | 0/0 | $0.000000 |
| regional_week | Luna 6 · Jev · CP-SAT | failure / transport_failure | 0.002001 | 0/0 | $0.000000 |
| regional_week | Sol 6.1 · Direct LLM | failure / transport_failure | 0.001909 | 1/0 | $0.000000 |
| regional_week | Sol 6.1 · CP-SAT | failure / transport_failure | 0.001909 | 0/0 | $0.000000 |
| regional_week | Sol 6.1 · Jev · Direct LLM | failure / transport_failure | 0.001909 | 0/0 | $0.000000 |
| regional_week | Sol 6.1 · Jev · CP-SAT | failure / transport_failure | 0.001909 | 0/0 | $0.000000 |

All 16 unique requested cells are represented exactly once. All failures are at extraction; all rows are retryable infrastructure failures, not terminal model-output failures or timeouts. DayPlan rows retain DayPlanAPIError and Workforce rows retain ShiftScheduleAPIError, both reporting Connection error. The persisted errors do not retain an underlying network cause; sandbox/network restriction is possible but not established by these artifacts. No repeated invocation or automatic retry occurred.

For every cell: no output was obtained; canonical evaluation is null. valid_output, canonical_validation_valid, feasible_correctly_reported, structural/full formulation match, required completion, constraint violations/counts, objective/optimal objective/gap are unavailable, not scored false. Direct solve and CP-SAT never ran. Shared failed extraction latency is repeated in standalone telemetry and must not be summed as actual runtime. No soft/hard interpretation of the workout or scheduling differences on regional_week can be assessed.

## Provenance and Jev integrity

Four attempted OpenAI requests: workout_before_four_soft/gpt-6-luna, workout_before_four_soft/gpt-6.1-sol, regional_week/gpt-6-luna, regional_week/gpt-6.1-sol. Requested model is recorded in every row; resolved_openai_models is empty everywhere because no response was received. No resolved Luna/Sol version can be confirmed. The directory did not exist before preflight and contains no extraction or Jev cache artifacts, so no stale Sol 6 cache/result was reused.

All extraction hashes are null. There are no Jev-adjusted formulations; paired extraction hash equality and shared Jev formulation integrity cannot be verified on this failed run. There was no output to resample; four attempts account for the four case/model groups and all 16 rows.

Configured Jev request model is jev-latest, but no Jev request occurred and all resolved_jev_model values are null. Authentication, resolved version, distributions, confidence, and GPT/proposed/applied/final values cannot be checked live. No Jev question payload was sent, so no live leakage assessment is available; deterministic serialization regression checks passed in preflight. Decisions and observed_questions are empty. Alignment explicitly retains 4 missing canonical questions per workout/model group and 34 per regional/model group: 76 unique case/model canonical questions unavailable. This is missing coverage, not evidence of answer error. Zero Jev observations or changes.

## Accounting and readiness

Actual attempts: 4 OpenAI extraction attempts, 0 Direct solve attempts, 0 Jev attempts. Recorded successful provider responses: 0. Recorded OpenAI input/output tokens and Jev input tokens: 0; these denote absent usage metadata, not verified provider billing. Recorded standalone estimated OpenAI cost is $0 for every row. Failure-row incremental accounting persists attempts only, without cost/token fields; no actual incremental dollar cost is available. No TypeSafe dollar cost is assigned. Planned standalone architecture call counts would have duplicated shared stages, but none of those successful stages materialized.

No genuine implementation defect was established and no source code or tests were modified. Original latest.jsonl and latest_summary.csv are preserved. They and this audit are new untracked artifacts. The deterministic implementation preflight passes, but live technical readiness for the 112-cell benchmark is not established: connectivity/provider access, resolved models, TypeSafe authentication, paired successful stages, and canonical live outcomes remain unverified. Do not start the full benchmark based on this pilot. Human label approval remains pending; no accuracy, calibration, or Brier claims were made.
