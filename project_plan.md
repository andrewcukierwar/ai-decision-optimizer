# AI Decision Optimizer — Research MVP Plan (v2)

## 0. Before writing any code

**Send the recruiter your resume today.** Use a project bullet that is true of the current live version. The upgraded repo will be in place long before a hiring manager clicks through. Lead the Raycon bullets with measured business outcomes; for this role, that experience is the primary match and the project is supporting evidence.

---

## 1. Thesis

**Research question:**

> How do model capability, a specialized decision model, and deterministic optimization affect the reliability, quality, latency, and cost of AI systems solving constrained decision problems?

The application becomes an interactive evaluation environment built around a three-layer division of labor:

| Layer | Responsibility |
|---|---|
| GPT-6 Luna / Sol | Extract open-ended values: names, durations, times, shifts, employees |
| TypeSafe Jev | Make every closed-set judgment about what the language means, with calibrated probabilities |
| OR-Tools CP-SAT (or the LLM, in the baseline arm) | Decide the schedule |

Deterministic validators judge every architecture against canonical ground truth. They are evaluation infrastructure, not an experimental toggle.

### Experimental dimensions

| Dimension | Options |
|---|---|
| Base model | `gpt-6-luna` / `gpt-6.1-sol` |
| Jev | Off / On |
| Solution engine | Direct LLM / OR-Tools CP-SAT |

**8 architectures.** Canonical labels:

```text
Luna 6 · Direct LLM
Luna 6 · Jev · Direct LLM
Luna 6 · CP-SAT
Luna 6 · Jev · CP-SAT
Sol 6.1 · Direct LLM
Sol 6.1 · Jev · Direct LLM
Sol 6.1 · CP-SAT
Sol 6.1 · Jev · CP-SAT
```

### The target resume bullet

Everything in this plan exists to fill in this sentence with real numbers:

> Benchmarked 8 LLM / decision-model / solver architectures on N constrained-scheduling tasks with independent validation against ground truth. Hybrid LLM + CP-SAT reached X% hard-constraint validity vs Y% for direct LLM solving; Jev's calibrated decisions [improved / did not change] formulation accuracy (Brier = B) at +Z ms per request.

If a task doesn't move a number in that sentence or make the demo clearer, it waits.

---

## 2. Preserve what already works

Do **not** rewrite:

- `DayPlan` / `ShiftSchedule` schemas
- CP-SAT compilers and solvers
- independent validators
- infeasibility diagnostics
- explanations / presentation helpers
- existing parsing tests and NL evaluation fixtures

```text
decision_optimizer/
├── dayplan/          schema, compiler, solver, validator
├── shift_schedule/   schema, compiler, solver, validator
├── parsing/
├── application.py
├── diagnostics.py
├── explanations.py
└── presentation.py
```

The CP-SAT pipeline becomes the hybrid arm under test.

---

## 3. Build order

The comparison (Direct LLM vs CP-SAT, scored against ground truth) is the core result, so it comes before Jev. Without it, the project is "added a classifier," not a research platform.

1. `.env` bootstrap + `ExperimentConfig` + telemetry
2. Luna / Sol selector
3. Canonical evaluator + name alignment
4. Direct LLM solver: Day Planner, then Workforce
5. Larger generated Workforce instances
6. Jev framework
7. Jev question types, in value order
8. Fixture labels (answer key) via labeling helper
9. Benchmark runner
10. Run benchmark + calibration analysis
11. Research UI + Jev decisions panel + Compare mode (precomputed)
12. README + screenshots + deploy
13. Update resume bullet with real numbers

---

# Phase 1 — Config, experiment config, telemetry

### `.env` loading

Add `python-dotenv`. Create `decision_optimizer/config.py` as the single bootstrap that calls `load_dotenv()`; `app.py`, benchmark scripts, and eval scripts all import it rather than loading `.env` themselves.

```dotenv
# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-6.1-sol

# TypeSafe / Jev
TYPESAFE_API_KEY=
TYPESAFE_MODEL=jev-latest
JEV_APPLY_THRESHOLD=0.7
```

`.env` stays gitignored. **Streamlit Cloud:** `load_dotenv()` does nothing there; add `TYPESAFE_API_KEY` to the app's secrets before deploying.

### `ExperimentConfig`

```python
ExperimentConfig(
    model="gpt-6.1-sol",  # gpt-6-luna | gpt-6.1-sol
    use_jev=False,
    solution_engine="cp_sat",  # direct_llm | cp_sat
)
```

Plus one canonical `label()` for the strings above.

### Telemetry

One small reusable record per run (`telemetry.py`):

```text
architecture, problem_type, case_id
model, use_jev, solution_engine
latency_total_s, latency_llm_s, latency_jev_s, latency_solver_s
openai_input_tokens, openai_output_tokens, n_model_calls
jev_input_tokens, n_jev_questions
success, error
```

Keep pricing in one dated config dict; compute cost from tokens in one place.

**Done when:** Sol + CP-SAT behaves exactly as before, now represented as an `ExperimentConfig` with telemetry recorded.

---

# Phase 2 — Luna / Sol selector

The parsers already accept `model`. Route it explicitly:

```text
Streamlit → ExperimentConfig → parse_*_request(..., model=config.model)
```

**Done when:** both models run the existing CP-SAT pipeline from the UI.

---

# Phase 3 — Canonical evaluator

This is the most important methodological piece.

### Rule: score every arm against the fixture's canonical spec, never against the arm's own parse

If results are validated against the arm's own interpretation, every CP-SAT arm scores 100% valid by construction, even when the parse was wrong, and Jev's effect is invisible, because Jev acts at the interpretation step.

For each benchmark run, `evaluation.py` produces:

```text
valid_output                  structured output parsed and materialized
canonical_validation_valid    binary primary validity, checked against CANONICAL spec
hard_constraints_satisfied    k / n secondary diagnostic; N/A for infeasible canonical cases
required_completed            all required tasks / shifts covered (canonical); N/A if infeasible
feasible_correctly_reported   arm's feasible/infeasible verdict matches canonical
objective_value               computed under CANONICAL spec
optimal_objective             CP-SAT solve of the CANONICAL spec
objective_gap
formulation_match_structural  primary structural comparison, ignoring weights
formulation_match             full semantic comparison including weights
latency, tokens, estimated_cost
```

### Name alignment

An arm's task/employee/shift names won't always match canonical names ("groceries" vs "grocery run"). Before validating, map arm names onto canonical names, reusing whatever the existing NL eval scripts do for semantic comparison. Unmatched items count as missing, not as crashes. Unit-test the alignment.

**Done when:** a canonical spec plus any arm's materialized result produces the metric record above, with deterministic tests and no API calls.

---

# Phase 4 — Direct LLM solver

The baseline where the model itself makes the scheduling decisions.

### Input

The direct solver receives the same final typed problem the CP-SAT arm would (after GPT extraction, and after Jev if enabled). The two engines differ only in who solves.

### Output schemas (structured outputs, no free-form)

```text
DirectDayPlanSolution
- status: feasible | infeasible
- assignments[]: task, start, end
- unscheduled_tasks[]
- explanation

DirectShiftSolution
- status: feasible | infeasible
- assignments[]: shift_id, employee
- explanation
```

Convert these into the representation the existing validators inspect.

### Prompt principle

Tell the model to solve the supplied problem, satisfy all hard constraints, optimize the stated preferences/objective, and return the schema. Do **not** give it CP-SAT pseudocode or the solving strategy.

**Done when:** Luna and Sol each produce Day Planner and Workforce solutions without CP-SAT, and the evaluator scores them.

---

# Phase 5 — Larger generated Workforce instances

Direct LLM will likely handle a 6-task day fine; the headline gap between engines will show up on larger instances. The existing Workforce fixtures are small (3 cases).

Write a small generator that:

1. builds a canonical `ShiftSchedule` spec first (~8–10 employees, ~1–2 weeks, rest / consecutive-day / fairness rules, a few availability statements);
2. renders the natural-language request from a template;
3. saves both as a fixture.

Generate 2–3 instances. Ground truth comes free, since the spec came first.

---

# Phase 6 — Jev framework

### What Jev does and doesn't do

Jev makes **closed-set semantic judgments** about the request. It does not generate schedules, extract names/times, write prose, or replace CP-SAT.

### One mechanism, many question types

Each question type is a registry entry in `jev.py`:

```python
JevQuestionType(
    key="pref_weight",
    primitive="score",  # choice | score | noul
    domains={"dayplan"},
    applies_to=...,  # items in the problem this question is asked about
    prompt=...,  # question text for one item
    gpt_value=...,  # what GPT's extraction already says (Jev-off baseline)
    apply=...,  # deterministic rewrite of the typed problem
)
```

One function then:

1. builds every applicable question for the request;
2. sends them in a **single Jev call** (state = original request + relevant extracted context; questions = map of typed questions);
3. applies each answer only when Jev's selected probability ≥ `JEV_APPLY_THRESHOLD`, otherwise keeps GPT's value; Score selects the modal level (lowest level on ties), maps 0–4 to weights 1–5, and keeps the expected score separately;
4. logs, per question: key, item, GPT's value, Jev's answer, probability, applied yes/no.

The rewritten problem must still pass the normal Pydantic boundary.

### Jev-off baseline

With Jev off, GPT's extraction already makes each of these calls implicitly (weights, modes, hard vs soft structure, availability entries, completeness). The comparison is **who makes the call**, not whether it gets made. `gpt_value` records GPT's answer to each question so both can be scored against the same label.

### Scaling guard

Only generate Workforce availability questions for employees GPT flagged with any availability or preference statement, so the question count doesn't become employees × all shifts. Check TypeSafe's per-request question limit and chunk if needed.

---

# Phase 7 — Jev question types

Build in this order. Each type = prompt + `apply` rewrite + unit test (no API).

| # | Key | Domain | Primitive | Question (per item) | Rewrite |
|---|---|---|---|---|---|
| 1 | `constraint_hardness` | Day Planner | Choice: hard / soft | Is this timing statement a firm requirement or a preference? | soft `finish_before` ↔ hard latest-finish bound; `preferred_window` ↔ hard earliest-start / latest-finish |
| 2 | `task_requirement` | Day Planner | Choice: required / optional | Must this task happen today? | set task required / optional |
| 3 | `task_mode` | Day Planner | Choice: active / passive | Does this task need the person's attention the whole time? | set task mode |
| 4 | `pref_weight` | Both | Score (1–5) | How strongly does the user want this preference satisfied? | set preference `weight` (currently all 1) |
| 5 | `availability_applies` | Workforce | Noul | Does this employee's statement rule out / concern this shift? | add / remove shift from employee's availability entries |
| 6 | `availability_hardness` | Workforce | Choice: unavailable / prefers not | Is the employee unable to work it, or would they rather not? | hard unavailability ↔ soft preference penalty |
| 7 | `solvable_as_stated` | Both | Noul | Does the request contain enough information to solve? | proceed vs trigger the clarification round |
| 8 | `relaxation_choice` | Both (infeasible only) | Choice over candidates | Given the request, which of these relaxations would the user most plausibly accept? | reorder the diagnostics' suggested relaxations |

Notes:

- **#4** is the only type that changes what CP-SAT optimizes rather than just what's feasible. It's the most interesting for the results.
- **#7** turns Jev's probability into an abstain-or-proceed policy. The Jev-off baseline is the existing GPT completeness check.
- **#8** candidates come from the existing deterministic diagnostics; Jev only ranks them. The Jev-off baseline is the diagnostics' default order. It fires only on infeasible cases, so expect a demo feature, not a meaningful metric.
- Preference *type* (`finish_before` vs `preferred_window`) stays with GPT; it's extraction, not a judgment call.

**Cut order if behind schedule:** #8 → #7 → #6. Types 1–5 are the core.

**Done when:** toggling Jev on produces a visible set of typed decisions with probabilities, some of which change the formulation, and the result still validates.

---

# Phase 8 — Benchmark dataset + answer key

Reuse `tests/fixtures/dayplan_nl_eval.json` and `tests/fixtures/shift_schedule_nl_eval.json`. Select ~12–14 cases:

| Slice | Count |
|---|---|
| Existing Day Planner cases (mix of precedence, passive tasks, timing, preferences) | ~6 |
| New Day Planner cases with **genuinely ambiguous** hard/soft or strength phrasing | 3–4 |
| Existing Workforce cases | ~3 |
| Generated large Workforce instances (Phase 5) | 2–3 |
| Infeasible cases (at least one per domain, for #8) | 1–2, may overlap |
| Needs-clarification cases (for #7) | 1–2, may overlap |

Example ambiguity pair:

> "I'd really like to work out before four, but it's okay if I can't."
>
> "I absolutely have to finish my workout before four."

### Answer key

Each fixture gets an `expected_jev` block: the correct answer to every applicable Jev question.

**Labeling helper** (`scripts/label_jev_fixtures.py`):

1. run GPT extraction once per fixture;
2. list every applicable Jev question;
3. pre-fill the expected answer with GPT's value;
4. you review and correct.

Labeling becomes reviewing rather than writing. Expect roughly 100 labeled decisions across the set.

---

# Phase 9 — Benchmark runner

Keep `scripts/evaluate_dayplan_nl.py` and `scripts/evaluate_shift_schedule_nl.py` as regression utilities.

Add `scripts/run_benchmark.py`:

```bash
uv run python scripts/run_benchmark.py --model gpt-6-luna --jev on --engine direct_llm
uv run python scripts/run_benchmark.py --all-configurations
```

Output:

```text
benchmark_results/
├── latest.jsonl            one line per (case, architecture) run
├── jev_decisions.jsonl     one line per Jev question asked
└── latest_summary.csv
```

- **Resumable:** skip completed compatible cells, including terminal invalid outputs and timeouts. Transport failures remain visible and can retry on explicit resume; automatic SDK retries are disabled.
- Auto-loads `.env` via `config.py`.
- No database.

~13 cases × 8 architectures ≈ 104 runs. One run per cell; note the unmeasured variance in Limitations.

---

# Phase 10 — Analysis (`scripts/analyze_benchmark.py`)

Produce:

1. **Architecture table:** validity, hard constraints, required completion, objective gap, latency, cost for all 8 arms.
2. **Jev vs GPT decision accuracy** per question type, against the answer key.
3. **Calibration:** reliability diagram (5 bins); report Choice/binary and Noul Brier separately from five-class Score Brier, using approved eligible labels. GPT has accuracy only. Report aligned accuracy, end-to-end accuracy including omissions, final applied accuracy, and coverage/unmatched counts (see `docs/benchmark_methodology.md`).
4. **Jev effect on outcomes:** for Jev-on vs Jev-off pairs, how often Jev changed the formulation and whether that fixed or broke validity / objective.

Save the plot to `docs/images/calibration.png` for the README.

A null Jev result is fine to report honestly, as long as the ambiguous cases were there to give it a chance.

---

# Phase 11 — Research UI

### Configuration panel

```text
Problem            [Day Planner ▾]
Base model         [GPT-6.1 Sol ▾]
Decision layer     [x] TypeSafe Jev
Solution engine    ( ) Direct LLM   (o) OR-Tools CP-SAT
```

### Results header

```text
Architecture      Sol 6.1 · Jev · CP-SAT
Valid             ✓
Hard constraints  12 / 12
Objective         4
Optimal           Yes
Latency           2.81 s
Cost              $0.00xx
```

Schedule / assignment tables stay underneath.

### Jev decisions panel (new, and the best demo element)

When Jev is on, show a table: question, item, GPT's value, Jev's answer, probability, applied?

### Compare mode

For curated benchmark cases, display the **precomputed** 8-architecture table from `benchmark_results/`. Don't run 8 architectures live from the public demo: it's slow, and the server-side keys pay for every click.

Don't claim rigorous scoring for custom prompts; there's no ground truth for them.

---

# Phase 12 — README

New opening:

> **AI Decision Optimizer is an interactive research platform for evaluating AI decision-making architectures on constrained scheduling problems.**

Sections:

```text
Research question
Architectures (three-layer diagram)
How Jev is used (question-type table)
Evaluation methodology (canonical scoring, name alignment)
Benchmark dataset
Results (architecture table + calibration plot)
Interactive demo
Deterministic validation
Limitations
Running locally
Testing
```

State plainly that this is a **curated project benchmark**, not a general benchmark of model intelligence.

Limitations should include: single run per cell; small N; generated Workforce instances use templated language; custom prompts are not scored; Jev availability questions only cover employees GPT flagged.

Setup:

```bash
cp .env.example .env
uv sync --extra test
uv run streamlit run app.py
```

No manual export step.

---

# Phase 13 — Done / stop gate

The Resume MVP is done when:

- [ ] Luna and Sol are selectable
- [ ] Direct LLM works for both domains
- [ ] CP-SAT still works; existing deterministic tests pass
- [ ] All 8 architectures run
- [ ] Evaluator scores against canonical specs with name alignment
- [ ] Jev question types 1–5 work (6–8 if time allows), with probabilities logged
- [ ] ~12–14 cases labeled and run across all 8 architectures
- [ ] Calibration plot and Brier score produced
- [ ] UI shows architecture controls, Jev decisions panel, precomputed Compare mode
- [ ] README has methodology and the results table
- [ ] Live deployment works (Jev key in Streamlit secrets)
- [ ] Resume bullet updated with real numbers

**Then stop.** Do not add:

- a third model / GPT-5.6 / Astra
- the Luna → Jev-verify → Sol escalation cascade (good follow-up, but a ninth architecture)
- new optimization domains
- multi-agent orchestration, provider registries, generalized pipelines
- a database, AWS, experiment-tracking tools
- repeated runs, significance testing, a 100-case benchmark

Those can happen while interviewing.

---

## Rough schedule

| Block | Work |
|---|---|
| Day 1 AM | Phases 1–3 (config, telemetry, selector, canonical evaluator) |
| Day 1 PM | Phase 4 (Direct LLM, both domains), Phase 5 (generator) |
| Day 1 evening | Phase 6 + question types 1–4 |
| Day 2 AM | Question types 5–8, labeling helper, answer key |
| Day 2 midday | Run benchmark, analysis, calibration plot |
| Day 2 PM | UI, README, deploy, resume bullet |

If Day 2 midday arrives without all eight question types, cut per Phase 7 and run the benchmark with what exists.

---

## Likely new files

```text
decision_optimizer/
├── config.py          load_dotenv + env configuration
├── experiment.py      ExperimentConfig + labels
├── telemetry.py       run records + cost calc
├── jev.py             question registry, batched call, apply + log
├── direct_solver.py   direct LLM solution schemas + conversion
└── evaluation.py      canonical scoring + name alignment

scripts/
├── generate_workforce_cases.py
├── label_jev_fixtures.py
├── run_benchmark.py
└── analyze_benchmark.py

tests/
├── test_config.py
├── test_experiment.py
├── test_jev.py           one test per question type's rewrite (no API)
├── test_direct_solver.py
└── test_evaluation.py    incl. name alignment
```

## Guiding rule

Every change must strengthen one of: the experimental comparison, measured correctness, the interactive demo, or the resume bullet.
