# AI Decision Optimizer — Project Plan (v2)

## Thesis

A natural-language system that turns goals, constraints and preferences into a mathematical optimization model, solves it with a deterministic solver, independently verifies the result, and explains the decision.

`Natural language → typed problem → user confirms interpretation → deterministic compiler → OR-Tools CP-SAT → independent validator → grounded explanation`

**The LLM is not the optimizer.** The LLM fills in a typed schema. Your code compiles the schema into a model. CP-SAT solves it. A separate validator checks the solution against the schema. The LLM explains *only facts produced by the solver and validator*.

---

## Phase 1 — Résumé-ready MVP

**Ship date: ______ (fixed).** Whatever state the project is in on this date, applications start. The project goes on the résumé when the definition of done is met.

### Scope: scheduling only, two demos

1. **Personal Day Planner** (flagship): meetings, work window, lift, groceries, laundry (wash → transfer → dry), deadlines, soft timing preferences.
2. **Workforce Scheduler**: employees, shifts, coverage, availability, max hours, eligibility, fairness and preference penalties.

Both are CP-SAT-native. Resource/budget allocation moves to Phase 2 (it wants an LP/convex solver, not CP-SAT).

### Design: two concrete schemas, one shared pipeline

Do **not** build one general schema. Build two small, typed (Pydantic) schemas, each with its own compiler and validator. They share the pipeline: LLM extraction → confirmation → compile → solve → validate → explain. In the MVP, the user picks the problem type in the UI, so no router is needed.

**DayPlan (sketch)**

```json
{
  "horizon": {"start": "08:00", "end": "21:00"},
  "work_window": {"start": "09:00", "end": "18:00"},
  "fixed_events": [{"name": "standup", "start": "10:00", "end": "10:30"}],
  "tasks": [
    {"name": "lift", "duration_min": 60, "mode": "active", "earliest_start": null, "latest_end": "19:00", "required": true},
    {"name": "wash", "duration_min": 30, "mode": "passive"},
    {"name": "transfer", "duration_min": 5, "mode": "active"},
    {"name": "dry", "duration_min": 45, "mode": "passive"}
  ],
  "precedences": [
    {"before": "wash", "after": "transfer", "min_gap_min": 0, "max_gap_min": 15},
    {"before": "transfer", "after": "dry", "min_gap_min": 0, "max_gap_min": 0}
  ],
  "preferences": [
    {"type": "finish_before", "task": "lift", "time": "17:00", "weight": 5},
    {"type": "minimize_work_interruptions", "weight": 3}
  ],
  "missing_info": []
}
```

- Time is integer minutes internally.
- `active` tasks can't overlap each other or fixed events (NoOverlap on the person). `passive` tasks (machines running) can overlap work but not their own dependent steps.
- **Define "work interruption" precisely.** Count the separate break blocks inside the work window. If merging adjacent tasks into one block gets fiddly, start with a simpler count: active tasks placed inside the work window.

**ShiftSchedule (sketch):** `shifts[{id, day, start, end, location, required_staff}]`, `employees[{name, max_hours, unavailable, eligible_locations, preferred_shifts}]`, `rules[{type: enum, ...}]`, `objective_weights{labor_cost, fairness, preferences}`. Keep the rule types a closed enum: min rest, max consecutive days, days off, and so on.

### Behaviors required in the MVP

- **Show the interpretation before solving.** Render the parsed tasks, constraints and preferences as an editable table. The user confirms or edits, then solves. This is the main safeguard against a mathematically optimal answer to the wrong problem.
- **One round of clarification.** The LLM fills `missing_info` (e.g., "most convenient" is undefined). If that list isn't empty, ask one question before solving.
- **Infeasibility is a first-class result.** Real day plans will often be infeasible. Detect it, and identify which hard constraints conflict, using CP-SAT assumption literals (`SufficientAssumptionsForInfeasibility`) or by relaxing constraints one at a time. Then say which ones to relax.
- **Independent validator.** Check the solution against the *schema* with plain Python, not against the CP model. It must not reuse compiler code.
- **Grounded explanation.** Pass the LLM only structured facts: objective breakdown, which preferences were violated and by how much, and which constraints were tight. Then it writes the explanation. No free-form reasoning about the schedule.

### Build order

Build the deterministic core first, with no LLM, then put the language layer on top.

| Day | Goal | Done when |
|---|---|---|
| 1 | DayPlan schema + compiler + CP-SAT + validator | 5 hand-written JSON cases (incl. the laundry day and 1 infeasible case) pass pytest |
| 2 | NL → DayPlan via LLM structured output; `missing_info` clarification | 10 NL test prompts parse into the expected key fields; end-to-end solve works from text |
| 3 | ShiftSchedule schema + compiler + validator; reuse NL path | 3 hand cases + 3 NL prompts pass |
| 4 | Explanations, infeasibility reporting, Streamlit UI (both demos, editable interpretation table) | Full demo flow works locally for both problem types |
| 5 | README, architecture diagram, test-results table, demo GIF, deploy, résumé bullet | Repo is public and linkable |

**If behind schedule, cut in this order:** (1) the workforce demo (ship the day planner alone), (2) clarification, (3) deploying (link a GIF instead). Never cut the validator, infeasibility handling or the tests.

### Test suite as a mini-benchmark

The Day 1–3 tests double as a small benchmark: ~15–20 NL prompts, each with the expected parse and a known optimal objective. Report one line in the README and on the résumé, e.g. "N/M problems formulated correctly and solved to proven optimality." This isn't the Phase 5 benchmark. It's just tests with a number attached.

### Definition of done

You can type the WFH laundry/lift/groceries problem, confirm the interpretation, and get a validated optimal schedule with an explanation grounded in solver output. An over-constrained version returns a clear infeasibility explanation. All tests pass.

### Out of scope for Phase 1

Multi-agent orchestration, solver router, budget/resource allocation, general or universal schema, benchmark phase, RL, calendar/maps integrations, vacation planning, AWS, database/auth, MLOps, Gurobi, stochastic or nonlinear optimization, Jev.

---

## Later phases (build while interviewing)

- **Phase 2 — More problem classes.** Budget/resource allocation with diminishing returns via CVXPY or LP (ties to real marketing-budget allocation experience), assignment, knapsack. Add a classifier/router once there are ≥3 problem types.
- **Phase 3 — Personal copilot.** Google Calendar import, travel times, recurring constraints, itinerary planning (orienteering/routing).
- **Phase 4 — What-if analysis.** "What if lift must be before 3?" → modify the model, re-solve, diff the results, report binding constraints.
- **Phase 5 — Benchmark.** Expand the test suite. Optionally evaluate on a public NL-to-optimization dataset. Metrics: parse accuracy, constraint recall, feasibility rate, optimal rate, robustness to paraphrase, cost, latency.
- **Phase 6 — Jev / multi-agent experiments.** Jev for bounded decisions (hard vs. soft classification, clarify-or-solve, routing) benchmarked against the LLM. Add a separate verifier or critic agent only if the Phase 5 metrics show a gain.

---

## Résumé positioning after Phase 1

**AI Decision Optimizer | Python, LLMs, OR-Tools CP-SAT, Streamlit**
- Built a natural-language optimization system that translates goals and hard/soft constraints into typed constraint-programming models, solved with OR-Tools and verified by an independent validator.
- Supports personal and workforce scheduling with user-confirmed interpretations, infeasibility diagnosis and explanations grounded in solver output; [N/M] test problems formulated correctly and solved to proven optimality.

## Guiding rule

Every feature must improve the demo, problem coverage or measured correctness. Phase 1 exists to get on the résumé. Everything after it happens while interviewing.
