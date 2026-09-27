"""Small Streamlit MVP for the AI Decision Optimizer."""

import json
from typing import Any, Dict, Optional

from decision_optimizer.application import (
    DayPlanRun,
    ShiftScheduleRun,
    clarify_dayplan_request,
    clarify_shift_schedule_request,
    parse_dayplan_request,
    parse_shift_schedule_request,
    solve_confirmed_dayplan,
    solve_confirmed_shift_schedule,
    validate_edited_dayplan_json,
    validate_edited_shift_schedule_json,
)
from decision_optimizer.diagnostics import InfeasibilityDiagnostic
from decision_optimizer.explanations import (
    render_dayplan_explanation,
    render_shift_schedule_explanation,
)
from decision_optimizer.parsing.dayplan import DayPlanError
from decision_optimizer.parsing.shift_schedule import ShiftScheduleError


PROBLEM_TYPES = ("Day Planner", "Workforce Scheduler")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AI Decision Optimizer", page_icon="🧭", layout="wide")
    st.title("AI Decision Optimizer")
    st.caption(
        "Interpretation → deterministic optimization → independent validation → grounded explanation"
    )

    problem_type = st.selectbox("Problem type", PROBLEM_TYPES, key="problem_type")
    _reset_for_problem_change(st, problem_type)
    prefix = _prefix(problem_type)

    request = st.text_area(
        "Natural-language problem",
        key=prefix + "_request",
        height=150,
        placeholder=(
            "Example: Plan my day from 08:00 to 18:00 with a 60-minute active workout..."
            if problem_type == "Day Planner"
            else "Example: Staff the 09:00-13:00 store shift on 2026-10-05..."
        ),
    )
    if st.button("Interpret", type="primary", key=prefix + "_interpret"):
        _interpret(st, problem_type, request)

    extraction = st.session_state.get(prefix + "_extraction")
    if extraction is None:
        error = st.session_state.get(prefix + "_error")
        if error:
            st.error(error)
        else:
            _show_api_hint(st)
        return

    if problem_type == "Day Planner":
        _render_dayplan_flow(st, request, prefix, extraction)
    else:
        _render_shift_schedule_flow(st, request, prefix, extraction)


def _interpret(st: Any, problem_type: str, request: str) -> None:
    prefix = _prefix(problem_type)
    try:
        extraction = (
            parse_dayplan_request(request)
            if problem_type == "Day Planner"
            else parse_shift_schedule_request(request)
        )
    except (DayPlanError, ShiftScheduleError) as error:
        st.session_state[prefix + "_error"] = str(error)
        st.session_state.pop(prefix + "_extraction", None)
        return
    st.session_state.pop(prefix + "_error", None)
    _store_extraction(st, prefix, extraction, clarification_used=False)


def _render_dayplan_flow(st: Any, request: str, prefix: str, extraction: Any) -> None:
    error = st.session_state.get(prefix + "_error")
    if error:
        st.error(error)

    if extraction.plan is None:
        _render_clarification(st, prefix, extraction.missing_info)
        if not st.session_state.get(prefix + "_clarification_used", False):
            answer = st.text_input("One clarification response", key=prefix + "_clarification")
            if st.button("Submit clarification", key=prefix + "_clarify"):
                try:
                    extraction = clarify_dayplan_request(
                        request, extraction, answer
                    )
                    _store_extraction(st, prefix, extraction, clarification_used=True)
                except (DayPlanError, ShiftScheduleError) as error:
                    st.error(str(error))
                    return
        if extraction.plan is None:
            return

    _render_dayplan_interpretation(st, request, prefix, extraction.plan)


def _render_shift_schedule_flow(
    st: Any, request: str, prefix: str, extraction: Any
) -> None:
    error = st.session_state.get(prefix + "_error")
    if error:
        st.error(error)

    if extraction.schedule is None:
        _render_clarification(st, prefix, extraction.missing_info)
        if not st.session_state.get(prefix + "_clarification_used", False):
            answer = st.text_input("One clarification response", key=prefix + "_clarification")
            if st.button("Submit clarification", key=prefix + "_clarify"):
                try:
                    extraction = clarify_shift_schedule_request(
                        request, extraction, answer
                    )
                    _store_extraction(st, prefix, extraction, clarification_used=True)
                except (DayPlanError, ShiftScheduleError) as error:
                    st.error(str(error))
                    return
        if extraction.schedule is None:
            return

    _render_shift_schedule_interpretation(st, request, prefix, extraction.schedule)


def _render_dayplan_interpretation(st: Any, request: str, prefix: str, plan: Any) -> None:
    st.subheader("Interpretation")
    st.caption("LLM-produced structured formulation — edit it before solving.")
    edited_json = st.text_area(
        "Editable DayPlan JSON",
        key=prefix + "_edited_json",
        height=420,
    )
    if st.button("Confirm & Solve", type="primary", key=prefix + "_solve"):
        try:
            confirmed_plan = validate_edited_dayplan_json(edited_json)
        except ValueError as error:
            st.error("Edited interpretation is invalid: " + str(error))
            return
        run = solve_confirmed_dayplan(confirmed_plan)
        st.session_state[prefix + "_run"] = run

    run = st.session_state.get(prefix + "_run")
    if run is not None:
        _render_dayplan_run(st, run)


def _render_shift_schedule_interpretation(
    st: Any, request: str, prefix: str, schedule: Any
) -> None:
    st.subheader("Interpretation")
    st.caption("LLM-produced structured formulation — edit it before solving.")
    edited_json = st.text_area(
        "Editable ShiftSchedule JSON",
        key=prefix + "_edited_json",
        height=420,
    )
    if st.button("Confirm & Solve", type="primary", key=prefix + "_solve"):
        try:
            confirmed_schedule = validate_edited_shift_schedule_json(edited_json)
        except ValueError as error:
            st.error("Edited interpretation is invalid: " + str(error))
            return
        run = solve_confirmed_shift_schedule(confirmed_schedule)
        st.session_state[prefix + "_run"] = run

    run = st.session_state.get(prefix + "_run")
    if run is not None:
        _render_shift_schedule_run(st, run)


def _render_dayplan_run(st: Any, run: DayPlanRun) -> None:
    solution = run.solution
    st.subheader("Optimization")
    st.caption("Deterministic OR-Tools CP-SAT result — the LLM does not choose the schedule.")
    st.write(
        "Solve status: **%s** · Optimal: **%s** · Objective value: **%d**"
        % (solution.status.value, "yes" if solution.optimal else "no", solution.objective_value)
    )

    st.subheader("Validation")
    if not run.validation.valid:
        st.error("Independent validation failed. The solver output is not trustworthy.")
        for error in run.validation.errors:
            st.write("- " + error)
        return
    st.success("Independent plain-Python validation passed.")

    if solution.status.value == "infeasible":
        _render_diagnostic(st, run.diagnosis)
        return
    if solution.status.value not in {"optimal", "feasible"}:
        st.warning(solution.message or "The solver did not determine feasibility.")
        return

    task_by_name = {task.name: task for task in run.plan.tasks}
    rows = []
    for assignment in sorted(run.solution.assignments, key=lambda item: item.start):
        task = task_by_name[assignment.name]
        rows.append(
            {
                "task": assignment.name,
                "start": assignment.start.strftime("%H:%M"),
                "end": assignment.end.strftime("%H:%M"),
                "duration_min": task.duration_min,
                "mode": task.mode.value,
            }
        )
    st.markdown("**Task schedule**")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    penalty_rows = [
        {
            "preference": penalty.preference_index,
            "type": penalty.preference_type.value,
            "amount": penalty.amount,
            "weighted_penalty": penalty.weighted_penalty,
        }
        for penalty in run.solution.preference_penalties
    ]
    st.markdown("**Preference / objective breakdown**")
    if penalty_rows:
        st.dataframe(penalty_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No preference penalties recorded.")

    if run.explanation_facts is not None:
        st.subheader("Grounded explanation")
        st.write(render_dayplan_explanation(run.explanation_facts))


def _render_shift_schedule_run(st: Any, run: ShiftScheduleRun) -> None:
    solution = run.solution
    st.subheader("Optimization")
    st.caption("Deterministic OR-Tools CP-SAT result — the LLM does not choose assignments.")
    st.write(
        "Solve status: **%s** · Optimal: **%s** · Objective value: **%d**"
        % (solution.status.value, "yes" if solution.optimal else "no", solution.objective_value)
    )

    st.subheader("Validation")
    if not run.validation.valid:
        st.error("Independent validation failed. The solver output is not trustworthy.")
        for error in run.validation.errors:
            st.write("- " + error)
        return
    st.success("Independent plain-Python validation passed.")

    if solution.status.value == "infeasible":
        _render_diagnostic(st, run.diagnosis)
        return
    if solution.status.value not in {"optimal", "feasible"}:
        st.warning(solution.message or "The solver did not determine feasibility.")
        return

    shifts_by_id = {shift.id: shift for shift in run.schedule.shifts}
    employees_by_shift: Dict[str, list] = {shift.id: [] for shift in run.schedule.shifts}
    for assignment in solution.assignments:
        employees_by_shift[assignment.shift_id].append(assignment.employee_name)
    rows = []
    for shift in sorted(
        run.schedule.shifts,
        key=lambda item: (item.day, item.start, item.id),
    ):
        rows.append(
            {
                "shift": shift.id,
                "day": shift.day.isoformat(),
                "start": shift.start.strftime("%H:%M"),
                "end": shift.end.strftime("%H:%M"),
                "location": shift.location,
                "assigned_employees": ", ".join(sorted(employees_by_shift[shift.id])),
            }
        )
    st.markdown("**Assignments grouped by shift**")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**Employee hours**")
    st.dataframe(
        [
            {
                "employee": load.employee_name,
                "hours": load.hours,
                "assigned_shifts": ", ".join(load.assigned_shift_ids),
            }
            for load in solution.employee_hours
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Preference / fairness objective breakdown**")
    st.dataframe(
        [
            {"component": "preference penalties", "value": solution.objective_breakdown.preference_penalty},
            {"component": "weighted preference penalty", "value": solution.objective_breakdown.weighted_preference_penalty},
            {"component": "fairness spread (minutes)", "value": solution.objective_breakdown.fairness_minutes_spread},
            {"component": "weighted fairness", "value": solution.objective_breakdown.weighted_fairness},
            {"component": "total", "value": solution.objective_breakdown.total},
        ],
        use_container_width=True,
        hide_index=True,
    )
    if run.solution.preference_penalties:
        st.markdown("**Missed preferences**")
        st.dataframe(
            [
                {
                    "employee": item.employee_name,
                    "shift": item.shift_id,
                    "amount": item.amount,
                    "weighted_penalty": item.weighted_penalty,
                }
                for item in run.solution.preference_penalties
            ],
            use_container_width=True,
            hide_index=True,
        )

    if run.explanation_facts is not None:
        st.subheader("Grounded explanation")
        st.write(render_shift_schedule_explanation(run.explanation_facts))


def _render_clarification(st: Any, prefix: str, missing_info: Any) -> None:
    st.subheader("Clarification needed")
    st.warning("The interpretation is incomplete. Answer one clarification question before solving.")
    for item in missing_info:
        st.write("- " + item)
    if st.session_state.get(prefix + "_clarification_used", False):
        st.info("The single clarification round has been used; edit the request and interpret again if needed.")


def _render_diagnostic(st: Any, diagnostic: Optional[InfeasibilityDiagnostic]) -> None:
    st.error("No feasible solution found")
    if diagnostic is None:
        st.warning("No bounded diagnostic was available.")
        return
    st.markdown("**Likely conflicts and deterministic evidence**")
    for finding in diagnostic.findings:
        st.markdown("**%s** — %s" % (finding.summary, finding.evidence))
        st.write("Suggested relaxation: " + finding.suggestion)


def _store_extraction(st: Any, prefix: str, extraction: Any, clarification_used: bool) -> None:
    st.session_state[prefix + "_extraction"] = extraction
    st.session_state[prefix + "_clarification_used"] = clarification_used
    st.session_state.pop(prefix + "_run", None)
    if extraction.plan is not None:
        st.session_state[prefix + "_edited_json"] = json.dumps(
            extraction.plan.model_dump(mode="json"), indent=2
        )
    elif extraction.schedule is not None:
        st.session_state[prefix + "_edited_json"] = json.dumps(
            extraction.schedule.model_dump(mode="json"), indent=2
        )
    else:
        st.session_state.pop(prefix + "_edited_json", None)


def _reset_for_problem_change(st: Any, problem_type: str) -> None:
    previous = st.session_state.get("_active_problem_type")
    if previous == problem_type:
        return
    for prefix in ("dayplan", "shift_schedule"):
        for suffix in ("request", "extraction", "clarification", "edited_json", "run", "error"):
            st.session_state.pop(prefix + "_" + suffix, None)
        st.session_state.pop(prefix + "_clarification_used", None)
    st.session_state["_active_problem_type"] = problem_type


def _prefix(problem_type: str) -> str:
    return "dayplan" if problem_type == "Day Planner" else "shift_schedule"


def _show_api_hint(st: Any) -> None:
    st.info("Enter a request and select Interpret. Live natural-language interpretation requires OPENAI_API_KEY.")


if __name__ == "__main__":
    main()
