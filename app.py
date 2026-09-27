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
from decision_optimizer.parsing.dayplan import DayPlanError, DayPlanExtraction
from decision_optimizer.parsing.shift_schedule import (
    ShiftScheduleError,
    ShiftScheduleExtraction,
)
from decision_optimizer.presentation import (
    build_dayplan_schedule_rows,
    dayplan_explanation_lines,
    dayplan_preference_statements,
    dayplan_infeasibility_summary,
    format_date,
    format_duration,
    format_minutes_duration,
    format_time,
    format_time_range,
    format_time_window,
    shift_explanation_lines,
    shift_preference_statements,
)


PROBLEM_TYPES = ("Day Planner", "Workforce Scheduler")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="AI Decision Optimizer", page_icon="🧭", layout="wide"
    )
    st.title("AI Decision Optimizer")
    st.caption(
        "Turn a natural-language planning problem into a validated, optimized recommendation."
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
            answer = st.text_input(
                "One clarification response", key=prefix + "_clarification"
            )
            if st.button("Submit clarification", key=prefix + "_clarify"):
                try:
                    extraction = clarify_dayplan_request(request, extraction, answer)
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
            answer = st.text_input(
                "One clarification response", key=prefix + "_clarification"
            )
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


def _render_dayplan_interpretation(
    st: Any, request: str, prefix: str, plan: Any
) -> None:
    st.subheader("Interpretation")
    st.caption("A readable summary of the confirmed planning problem.")
    _render_dayplan_interpretation_summary(st, plan)
    with st.expander("Advanced: Edit structured interpretation"):
        st.caption(
            "This typed formulation is sent to the optimizer. "
            "Edits are validated against the DayPlan schema before solving."
        )
        edited_json = st.text_area(
            "Editable DayPlan JSON",
            key=prefix + "_edited_json",
            height=360,
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
    st.caption("A readable summary of the confirmed workforce problem.")
    _render_shift_schedule_interpretation_summary(st, schedule)
    with st.expander("Advanced: Edit structured interpretation"):
        st.caption(
            "This typed formulation is sent to the optimizer. "
            "Edits are validated against the ShiftSchedule schema before solving."
        )
        edited_json = st.text_area(
            "Editable ShiftSchedule JSON",
            key=prefix + "_edited_json",
            height=360,
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
    if solution.status.value == "infeasible":
        _render_dayplan_result_summary(st, run)
        if not run.validation.valid:
            st.error("Independent validation reported an issue with the infeasible result.")
            for error in run.validation.errors:
                st.write("- " + error)
        _render_dayplan_diagnostic(st, run.diagnosis)
        return

    st.subheader("Recommended schedule")
    _render_dayplan_result_summary(st, run)
    if not run.validation.valid:
        st.error("Independent validation failed. The solver output is not trustworthy.")
        for error in run.validation.errors:
            st.write("- " + error)
        return

    if solution.status.value not in {"optimal", "feasible"}:
        st.warning(solution.message or "The solver did not determine feasibility.")
        return

    st.markdown("**Schedule**")
    st.dataframe(
        build_dayplan_schedule_rows(run.plan, solution.assignments),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Preferences")
    if run.explanation_facts is not None:
        preference_statements = dayplan_preference_statements(run.explanation_facts)
        if preference_statements:
            for statement in preference_statements:
                st.write(statement)
        else:
            st.caption("No soft preferences were encoded for this plan.")

    if run.explanation_facts is not None:
        st.subheader("Why this schedule?")
        for line in dayplan_explanation_lines(run.explanation_facts):
            st.write(line)

    _render_dayplan_advanced_details(st, run)


def _render_shift_schedule_run(st: Any, run: ShiftScheduleRun) -> None:
    solution = run.solution
    st.subheader("Recommended assignments")
    _render_shift_result_summary(st, run)
    if not run.validation.valid:
        st.error("Independent validation failed. The solver output is not trustworthy.")
        for error in run.validation.errors:
            st.write("- " + error)
        return

    if solution.status.value == "infeasible":
        _render_diagnostic(st, run.diagnosis)
        return
    if solution.status.value not in {"optimal", "feasible"}:
        st.warning(solution.message or "The solver did not determine feasibility.")
        return

    st.markdown("**Shift coverage**")
    st.dataframe(
        _shift_assignment_rows(run.schedule, solution),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Employee workload**")
    st.dataframe(
        [
            {
                "Employee": load.employee_name,
                "Hours": f"{load.hours:g} hr",
                "Assigned shifts": ", ".join(load.assigned_shift_ids) or "—",
            }
            for load in solution.employee_hours
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Preferences")
    if run.explanation_facts is not None:
        for statement in shift_preference_statements(
            run.schedule, run.explanation_facts
        ):
            st.write(statement)

    if run.explanation_facts is not None:
        st.subheader("Why these assignments?")
        for line in shift_explanation_lines(run.explanation_facts):
            st.write(line)

    _render_shift_advanced_details(st, run)


def _render_dayplan_interpretation_summary(st: Any, plan: Any) -> None:
    window_left, window_right = st.columns(2)
    with window_left:
        st.markdown("**Planning window**")
        st.write(format_time_window(plan.horizon.start, plan.horizon.end))
    with window_right:
        st.markdown("**Work window**")
        if plan.work_window is None:
            st.write("Not specified")
        else:
            st.write(format_time_window(plan.work_window.start, plan.work_window.end))

    st.markdown("**Fixed events**")
    if plan.fixed_events:
        st.dataframe(
            [
                {
                    "Event": event.name,
                    "Time": format_time_range(event.start, event.end),
                    "Duration": format_duration(_duration_minutes(event.start, event.end)),
                }
                for event in sorted(plan.fixed_events, key=lambda item: item.start)
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No fixed events")

    st.markdown("**Activities**")
    if plan.tasks:
        st.dataframe(
            [
                {
                    "Activity": task.name,
                    "Duration": format_duration(task.duration_min),
                    "Mode": task.mode.value.capitalize(),
                    "Required": "Required" if task.required else "Optional",
                }
                for task in plan.tasks
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No activities")

    st.markdown("**Dependencies**")
    if plan.precedences:
        st.dataframe(
            [
                {
                    "Dependency": f"{item.before} → {item.after}",
                    "Timing": _dependency_timing(item),
                }
                for item in plan.precedences
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No dependencies")

    st.markdown("**Preferences**")
    if plan.preferences:
        st.dataframe(
            [{"Preference": _dayplan_preference_label(item)} for item in plan.preferences],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No soft preferences")


def _render_shift_schedule_interpretation_summary(st: Any, schedule: Any) -> None:
    st.markdown("**Shifts & required staffing**")
    st.dataframe(
        [
            {
                "Shift": shift.id,
                "Date": format_date(shift.day),
                "Time": format_time_range(shift.start, shift.end),
                "Location": shift.location,
                "Required staff": shift.required_staff,
            }
            for shift in sorted(schedule.shifts, key=lambda item: (item.day, item.start, item.id))
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Employees & maximum hours**")
    st.dataframe(
        [
            {
                "Employee": employee.name,
                "Max hours": f"{employee.max_hours} hr",
            }
            for employee in schedule.employees
        ],
        use_container_width=True,
        hide_index=True,
    )

    eligibility, availability = st.columns(2)
    with eligibility:
        st.markdown("**Eligibility**")
        st.dataframe(
            [
                {
                    "Employee": employee.name,
                    "Eligible locations": ", ".join(employee.eligible_locations) or "None",
                }
                for employee in schedule.employees
            ],
            use_container_width=True,
            hide_index=True,
        )
    with availability:
        st.markdown("**Availability**")
        availability_rows = []
        for employee in schedule.employees:
            for item in employee.unavailable:
                availability_rows.append(
                    {
                        "Employee": employee.name,
                        "Unavailable": _unavailability_label(item),
                    }
                )
        if availability_rows:
            st.dataframe(availability_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No unavailable windows")

    preferences, rules = st.columns(2)
    with preferences:
        st.markdown("**Preferences**")
        preference_rows = [
            {
                "Employee": employee.name,
                "Preferred shifts": ", ".join(employee.preferred_shifts) or "None",
            }
            for employee in schedule.employees
        ]
        st.dataframe(preference_rows, use_container_width=True, hide_index=True)
    with rules:
        st.markdown("**Configured rules**")
        if schedule.rules:
            st.dataframe(
                [{"Rule": _shift_rule_label(rule)} for rule in schedule.rules],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No additional rules")

    st.markdown("**Objective weights**")
    st.dataframe(
        [
            {
                "Component": "Preference penalty",
                "Weight": schedule.objective_weights.preference_penalty,
            },
            {"Component": "Fairness", "Weight": schedule.objective_weights.fairness},
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_dayplan_result_summary(st: Any, run: DayPlanRun) -> None:
    solution = run.solution
    if solution.status.value == "infeasible":
        summary = dayplan_infeasibility_summary(run.diagnosis)
        st.error(summary[0])
        for line in summary[1:]:
            st.write(line)
        return

    columns = st.columns(3)
    columns[0].metric("Status", _status_label(solution.status.value))
    columns[1].metric("Validation", "Passed" if run.validation.valid else "Failed")
    columns[2].metric("Preference penalty", solution.objective_value)
    st.caption(
        "Schedule produced by deterministic OR-Tools CP-SAT and independently validated."
    )


def _render_shift_result_summary(st: Any, run: ShiftScheduleRun) -> None:
    solution = run.solution
    columns = st.columns(4)
    columns[0].metric("Status", _status_label(solution.status.value))
    columns[1].metric("Validation", "Passed" if run.validation.valid else "Failed")
    columns[2].metric(
        "Preference penalty", solution.objective_breakdown.weighted_preference_penalty
    )
    columns[3].metric(
        "Fairness spread",
        format_minutes_duration(solution.objective_breakdown.fairness_minutes_spread),
    )
    st.caption(
        "Assignments produced by deterministic OR-Tools CP-SAT and independently validated."
    )


def _render_dayplan_advanced_details(st: Any, run: DayPlanRun) -> None:
    solution = run.solution
    with st.expander("Advanced: Optimization details"):
        st.write(
            "Raw solve status: **%s** · Optimal: **%s** · Weighted preference penalty: **%d**"
            % (
                solution.status.value,
                "yes" if solution.optimal else "no",
                solution.objective_value,
            )
        )
        st.markdown("**Solver assignments**")
        st.dataframe(
            [
                {
                    "Task": assignment.name,
                    "Start": format_time(assignment.start),
                    "End": format_time(assignment.end),
                    "Duration": format_duration(
                        _duration_minutes(assignment.start, assignment.end)
                    ),
                }
                for assignment in sorted(solution.assignments, key=lambda item: item.start)
            ],
            use_container_width=True,
            hide_index=True,
        )
        penalty_rows = _dayplan_penalty_rows(run)
        st.markdown("**Technical preference penalties**")
        if penalty_rows:
            st.dataframe(penalty_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No preference penalties recorded")


def _render_shift_advanced_details(st: Any, run: ShiftScheduleRun) -> None:
    solution = run.solution
    with st.expander("Advanced: Optimization details"):
        breakdown = solution.objective_breakdown
        st.write(
            "Raw solve status: **%s** · Optimal: **%s** · Objective value: **%d**"
            % (
                solution.status.value,
                "yes" if solution.optimal else "no",
                solution.objective_value,
            )
        )
        st.dataframe(
            [
                {"Component": "Preference penalties", "Value": breakdown.preference_penalty},
                {
                    "Component": "Weighted preference penalty",
                    "Value": breakdown.weighted_preference_penalty,
                },
                {
                    "Component": "Fairness spread (minutes)",
                    "Value": breakdown.fairness_minutes_spread,
                },
                {"Component": "Weighted fairness", "Value": breakdown.weighted_fairness},
                {"Component": "Total", "Value": breakdown.total},
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Technical preference penalties**")
        if solution.preference_penalties:
            st.dataframe(
                [
                    {
                        "Employee": item.employee_name,
                        "Shift": item.shift_id,
                        "Amount": item.amount,
                        "Weighted penalty": item.weighted_penalty,
                    }
                    for item in solution.preference_penalties
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No preference penalties recorded")
        st.caption("Independent validation: passed")


def _shift_assignment_rows(schedule: Any, solution: Any) -> list:
    assigned_by_shift: Dict[str, list] = {shift.id: [] for shift in schedule.shifts}
    for assignment in solution.assignments:
        assigned_by_shift[assignment.shift_id].append(assignment.employee_name)
    return [
        {
            "Shift": shift.id,
            "Date": format_date(shift.day),
            "Time": format_time_range(shift.start, shift.end),
            "Location": shift.location,
            "Assigned staff": ", ".join(sorted(assigned_by_shift[shift.id])) or "Unfilled",
        }
        for shift in sorted(schedule.shifts, key=lambda item: (item.day, item.start, item.id))
    ]


def _dayplan_penalty_rows(run: DayPlanRun) -> list:
    preferences_by_index = {
        index: preference for index, preference in enumerate(run.plan.preferences)
    }
    rows = []
    for penalty in run.solution.preference_penalties:
        preference = preferences_by_index[penalty.preference_index]
        if preference.time is not None:
            target = format_time(preference.time)
        elif preference.start is not None and preference.end is not None:
            target = format_time_range(preference.start, preference.end)
        else:
            target = ""
        rows.append(
            {
                "Preference": penalty.preference_index,
                "Type": penalty.preference_type.value,
                "Task": preference.task or "",
                "Target": target,
                "Penalty amount": penalty.amount,
                "Weighted penalty": penalty.weighted_penalty,
            }
        )
    return rows


def _dayplan_preference_label(preference: Any) -> str:
    if preference.type.value == "finish_before":
        return f"{preference.task} · finish before {format_time(preference.time)}"
    if preference.type.value == "preferred_window":
        return (
            f"{preference.task} · preferred "
            f"{format_time_range(preference.start, preference.end)}"
        )
    return "Minimize work interruptions"


def _dependency_timing(precedence: Any) -> str:
    if precedence.min_gap_min == 0 and precedence.max_gap_min == 0:
        return "immediately"
    if precedence.max_gap_min is None:
        return f"after at least {precedence.min_gap_min} min"
    if precedence.min_gap_min == precedence.max_gap_min:
        return f"after {precedence.min_gap_min} min"
    return f"after {precedence.min_gap_min}–{precedence.max_gap_min} min"


def _unavailability_label(item: Any) -> str:
    if item.shift_id is not None:
        return f"Shift {item.shift_id}"
    if item.day is None:
        return "Unspecified"
    if item.start is None or item.end is None:
        return f"{format_date(item.day)} · all day"
    return f"{format_date(item.day)} · {format_time_range(item.start, item.end)}"


def _shift_rule_label(rule: Any) -> str:
    rule_type = getattr(rule, "type", "")
    if rule_type == "minimum_rest":
        return f"Minimum rest · {rule.min_rest_hours} hr"
    if rule_type == "maximum_consecutive_days":
        return f"Maximum consecutive days · {rule.max_days}"
    if rule_type == "required_days_off":
        days = ", ".join(format_date(day) for day in rule.days)
        return f"Required days off · {rule.employee_name}: {days}"
    return str(rule_type)


def _duration_minutes(start: Any, end: Any) -> int:
    return (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)


def _status_label(status: str) -> str:
    return status.replace("_", " ").title()


def _render_clarification(st: Any, prefix: str, missing_info: Any) -> None:
    st.subheader("Clarification needed")
    st.warning(
        "The interpretation is incomplete. Answer one clarification question before solving."
    )
    for item in missing_info:
        st.write("- " + item)
    if st.session_state.get(prefix + "_clarification_used", False):
        st.info(
            "The single clarification round has been used; edit the request and interpret again if needed."
        )


def _render_diagnostic(st: Any, diagnostic: Optional[InfeasibilityDiagnostic]) -> None:
    st.error("No feasible solution found")
    if diagnostic is None:
        st.warning("No bounded diagnostic was available.")
        return
    st.markdown("**Likely conflicts and deterministic evidence**")
    for finding in diagnostic.findings:
        st.markdown("**%s** — %s" % (finding.summary, finding.evidence))
        st.write("Suggested relaxation: " + finding.suggestion)


def _render_dayplan_diagnostic(
    st: Any, diagnostic: Optional[InfeasibilityDiagnostic]
) -> None:
    if diagnostic is None:
        st.warning("No bounded diagnostic was available.")
        return

    if diagnostic.suggestions:
        st.markdown("**Practical relaxation suggestions**")
        for suggestion in diagnostic.suggestions:
            st.write("- " + suggestion)

    relaxation_codes = {
        "timing_bounds_binding",
        "fixed_event_conflict",
        "precedence_constraints_binding",
        "active_person_capacity_binding",
    }
    primary_findings = [
        finding
        for finding in diagnostic.findings
        if finding.code not in relaxation_codes
    ]
    advanced_findings = [
        finding
        for finding in diagnostic.findings
        if finding.code in relaxation_codes
    ]

    if primary_findings:
        st.markdown("**Most useful deterministic diagnosis**")
        for finding in primary_findings:
            st.markdown("**%s** — %s" % (finding.summary, finding.evidence))

    if advanced_findings or diagnostic.tested_relaxations:
        with st.expander("Advanced diagnostic details"):
            if diagnostic.tested_relaxations:
                st.write(
                    "Bounded relaxation tests: "
                    + ", ".join(diagnostic.tested_relaxations)
                )
            for finding in advanced_findings:
                st.markdown("**%s** — %s" % (finding.summary, finding.evidence))


def _store_extraction(
    st: Any, prefix: str, extraction: Any, clarification_used: bool
) -> None:
    st.session_state[prefix + "_extraction"] = extraction
    st.session_state[prefix + "_clarification_used"] = clarification_used
    st.session_state.pop(prefix + "_run", None)

    if isinstance(extraction, DayPlanExtraction):
        if extraction.plan is None:
            st.session_state.pop(prefix + "_edited_json", None)
            return
        st.session_state[prefix + "_edited_json"] = json.dumps(
            extraction.plan.model_dump(mode="json"), indent=2
        )
        return

    if isinstance(extraction, ShiftScheduleExtraction):
        if extraction.schedule is None:
            st.session_state.pop(prefix + "_edited_json", None)
            return
        st.session_state[prefix + "_edited_json"] = json.dumps(
            extraction.schedule.model_dump(mode="json"), indent=2
        )
        return

    raise TypeError("unsupported extraction type: " + type(extraction).__name__)


def _reset_for_problem_change(st: Any, problem_type: str) -> None:
    previous = st.session_state.get("_active_problem_type")
    if previous == problem_type:
        return
    for prefix in ("dayplan", "shift_schedule"):
        for suffix in (
            "request",
            "extraction",
            "clarification",
            "edited_json",
            "run",
            "error",
        ):
            st.session_state.pop(prefix + "_" + suffix, None)
        st.session_state.pop(prefix + "_clarification_used", None)
    st.session_state["_active_problem_type"] = problem_type


def _prefix(problem_type: str) -> str:
    return "dayplan" if problem_type == "Day Planner" else "shift_schedule"


def _show_api_hint(st: Any) -> None:
    st.info("Enter a scheduling problem above, then select Interpret.")


if __name__ == "__main__":
    main()
