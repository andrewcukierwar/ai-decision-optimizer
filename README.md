# AI Decision Optimizer

Turn natural-language planning problems into validated, optimized recommendations using LLM-based interpretation and deterministic mathematical optimization.

> **The LLM interprets the problem; OR-Tools chooses the solution.**

Python · GPT-6 Sol · OpenAI Structured Outputs · Pydantic · OR-Tools CP-SAT · Streamlit · pytest · uv

[Live Demo](LIVE_DEMO_URL_HERE) · [GitHub Repository](https://github.com/andrewcukierwar/ai-decision-optimizer)

<!-- Hero screenshot placeholder. Add the image after deployment:
![AI Decision Optimizer Day Planner](docs/images/day-planner.png)
-->

## Why this exists

Planning requests are usually expressed in language: goals, deadlines, fixed events, preferences, and exceptions. LLMs are useful for turning that messy input into a structured problem. They are not the right component to guarantee that every hard constraint is satisfied or that the resulting plan is optimal.

AI Decision Optimizer separates those responsibilities:

- language understanding happens at a typed extraction boundary;
- deterministic code compiles the confirmed formulation into a mathematical model;
- OR-Tools CP-SAT determines feasibility and optimized assignments;
- an independent validator checks the materialized result against the confirmed schema;
- the application presents the schedule, diagnostics, and explanation from structured facts.

The result is a small, inspectable pipeline for decision support rather than a language model that simply proposes a plausible-looking schedule.

## Architecture

```mermaid
flowchart LR
    U[User request] --> E

    subgraph LLM["LLM interpretation"]
        E[GPT-6 Sol<br/>OpenAI Structured Outputs] --> X{Complete extraction?}
        X -- No --> C[One clarification round]
        C --> E
        X -- Yes --> P[Typed Pydantic<br/>problem schema]
    end

    P --> I[Human-readable interpretation]
    I --> K[User confirms<br/>or edits structured JSON]

    subgraph DET["Deterministic optimization and validation"]
        K --> COMP[Deterministic compiler]
        COMP --> CP[OR-Tools CP-SAT]
        CP --> R[Materialized result]
        R --> V[Independent plain-Python validator]
        V --> F{Feasible and valid?}
        F -- Yes --> G[Grounded result<br/>and explanation]
        F -- No --> D[Infeasibility diagnosis]
    end

    G --> S[Streamlit presentation]
    D --> S

    classDef llm fill:#eef6ff,stroke:#4776a8,color:#17324d;
    classDef deterministic fill:#eef8f0,stroke:#43865a,color:#173d24;
    classDef human fill:#fff7e6,stroke:#c4862b,color:#4d3512;
    class E,X,C,P llm;
    class COMP,CP,R,V,F,G,D deterministic;
    class U,I,K,S human;
```

- GPT-6 Sol performs semantic extraction into one of the typed problem schemas.
- CP-SAT determines feasibility and the optimized decisions; it does not receive a free-form request.
- The independent validator checks the materialized result against the confirmed schema and recomputes the relevant hard-constraint and objective semantics.
- Explanations use structured solver and validator facts. They do not claim formal verification or rely on hidden reasoning about what the optimizer “wanted.”

## What it can solve

### Day Planner

The Day Planner models a single planning horizon with:

- a planning horizon and optional work window;
- fixed events such as meetings;
- required and optional tasks;
- active tasks that consume the person’s attention;
- passive tasks that can run while the person does other work when constraints allow;
- hard timing bounds such as earliest start and latest finish;
- precedence and process chains with minimum or maximum gaps, including immediate handoffs;
- soft `finish_before`, `preferred_window`, and work-interruption preferences;
- infeasibility diagnosis using deterministic evidence and bounded relaxation analysis.

Active tasks and fixed events share the person’s capacity. Passive tasks remain part of the timing and precedence model but do not consume that active-person resource.

### Workforce Scheduler

The Workforce Scheduler models dated shifts and employees with:

- shift start/end times, locations, and required staffing levels;
- employee maximum hours;
- location eligibility;
- unavailable shifts, time windows, and full days;
- preferred shifts as soft preferences;
- minimum rest, maximum consecutive days, and required days off;
- preference-violation penalties and workload fairness;
- infeasibility diagnosis using deterministic evidence and bounded relaxation analysis.

The MVP uses two concrete schemas with a deliberately closed rule set. It is not a general-purpose workforce scheduling DSL.

## Example workflow

### Natural-language request

> Plan my work-from-home day from 09:00 to 18:00. Work hours bound the plan. I have a meeting from 10:30 to 11:00. I need 30 minutes for groceries, a laundry chain of 30-minute wash, 5-minute transfer, and 45-minute dry, plus a one-hour lift and a 30-minute lunch. The machines can run unattended, but I must be free for the transfer. Prefer groceries before lunch, lunch between 12:00 and 13:00, and lifting in the early afternoon, finished before 16:00.

### Interpreted formulation

The request becomes a typed `DayPlan` containing a fixed event, active and passive tasks, immediate process dependencies, hard timing facts, and soft preferences. A shortened view looks like this:

```json
{
  "horizon": {"start": "09:00", "end": "18:00"},
  "work_window": {"start": "09:00", "end": "18:00"},
  "fixed_events": [
    {"name": "meeting", "start": "10:30", "end": "11:00"}
  ],
  "tasks": [
    {"name": "wash", "duration_min": 30, "mode": "passive"},
    {"name": "transfer", "duration_min": 5, "mode": "active"},
    {"name": "dry", "duration_min": 45, "mode": "passive"},
    {"name": "groceries", "duration_min": 30, "mode": "active"},
    {"name": "lift", "duration_min": 60, "mode": "active"},
    {"name": "lunch", "duration_min": 30, "mode": "active"}
  ],
  "precedences": [
    {"before": "wash", "after": "transfer", "max_gap_min": 0},
    {"before": "transfer", "after": "dry", "max_gap_min": 0}
  ],
  "preferences": [
    {"type": "finish_before", "task": "groceries", "time": "12:00", "weight": 1},
    {"type": "preferred_window", "task": "lunch", "start": "12:00", "end": "13:00", "weight": 1},
    {"type": "preferred_window", "task": "lift", "start": "12:00", "end": "15:00", "weight": 1}
  ]
}
```

### Optimized result

After confirmation, CP-SAT materializes task assignments while respecting hard constraints and minimizing the encoded preference penalties. The independent validator then checks the assignments and penalty calculations against the confirmed `DayPlan`. The Streamlit UI presents the schedule, preference outcomes, validation status, and a grounded explanation—or an infeasibility diagnosis when no hard-constraint-satisfying schedule exists.

## Reliability by design

### Typed structured extraction

The model must return one of the typed extraction schemas: `DayPlanExtraction` or `ShiftScheduleExtraction`. Pydantic validation rejects malformed structures and unsupported fields at the schema boundary.

### User-confirmed interpretation

The UI shows a readable interpretation before solving. Users can confirm it or edit the structured JSON; edited input is validated against the corresponding schema before it reaches the solver.

### Deterministic optimization

The compiler translates the confirmed schema into OR-Tools CP-SAT constraints and objective terms. CP-SAT determines feasibility and assignments with deterministic settings in the solver wrapper.

### Independent validation

A solver result is not treated as trustworthy merely because CP-SAT returned it. The materialized output is checked against the confirmed problem schema by separate plain-Python validators that do not import the compiler or inspect CP-SAT variables.

### Infeasibility handling

The system can return no feasible solution. For infeasible inputs, the application reports deterministic diagnostic findings and tests bounded relaxations such as timing bounds, fixed events, precedence, active-person capacity, eligibility, availability, hours, and supported hard rules where applicable. It does not force a schedule.

### Grounded explanations

Explanation text is rendered from structured assignments, objective values, preference penalties, workload facts, and validator results. It is not a free-form account of hidden solver reasoning.

Additional engineering safeguards include a no-API deterministic test path, rejection of unknown fields in edited structured input, invalidation of stale Streamlit results when the request or formulation changes, and environment-based secret handling (`.env` is ignored by Git).

## Evaluation

The following are curated MVP evaluation cases designed to exercise the supported scheduling semantics. They are not an industry benchmark, public benchmark, research benchmark, state-of-the-art claim, or claim of general natural-language optimization accuracy.

| Problem type | Formulations matched expected key fields | Formulation-valid cases solved successfully | Solved cases passed independent validation |
| --- | ---: | ---: | ---: |
| Day Planner | 13 / 13 | 11 / 11 | 11 / 11 |
| Workforce Scheduler | 3 / 3 | 2 / 2 | 2 / 2 |
| **Combined** | **16 / 16** | **13 / 13** | **13 / 13** |

The live NL evaluation uses the configured OpenAI model and API. It is separate from the deterministic pytest suite, which currently reports **128 tests passing** and requires no API access.

## Running locally

### Install

```bash
git clone https://github.com/andrewcukierwar/ai-decision-optimizer.git
cd ai-decision-optimizer
uv sync --extra test
```

### Configure the model API

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The parser defaults to `gpt-6-sol`; set `OPENAI_MODEL` if you need to override it. The application reads these values from the environment, and `.env` is intentionally ignored by Git.

### Run the Streamlit app

```bash
uv run streamlit run app.py
```

The app provides both the Day Planner and Workforce Scheduler flows. Each flow interprets the request, supports one clarification round when required, shows the typed interpretation, allows structured edits, and then confirms, solves, validates, and explains the result.

### Run the deterministic test suite

```bash
uv run pytest
```

### Run the opt-in live NL evaluations

These scripts make paid API requests and are intentionally not collected by pytest:

```bash
uv run python scripts/evaluate_dayplan_nl.py
uv run python scripts/evaluate_shift_schedule_nl.py
```

Use `--model MODEL_NAME` to override `OPENAI_MODEL` for an evaluation run. The checked-in fixtures live in `tests/fixtures/`.

## Limitations and scope

- The MVP contains two concrete schemas: Day Planner and Workforce Scheduler; it does not attempt to represent arbitrary optimization problems.
- Day Planner times use minute precision within one calendar-day horizon.
- Workforce shifts use dated, same-day start and end times and the supported closed set of hard rules.
- The workforce objective covers preference penalties and workload fairness; labor-cost optimization is not implemented.
- Calendar integration, routing, breaks, skills, overnight shifts, and arbitrary user-defined rule languages are outside the current repository scope.
- Live natural-language behavior depends on the configured OpenAI model and API. The deterministic compiler, solvers, validators, diagnostics, explanations, and tests can be exercised without an API key.

## Repository map

```text
app.py                              Streamlit application
decision_optimizer/
  parsing/                          Structured natural-language extraction
  dayplan/                          Day Planner schema, compiler, solver, validator
  shift_schedule/                   Workforce schema, compiler, solver, validator
  diagnostics.py                    Infeasibility findings and bounded relaxations
  explanations.py                   Structured, grounded explanation facts
scripts/                            Opt-in live NL evaluation runners
tests/                              Deterministic tests, cases, and NL fixtures
project_plan.md                     MVP scope and design rationale
```

## License

No license file is currently included. Until one is added, the repository should be treated as source-available rather than as granting broad permission to reuse or redistribute the code.
