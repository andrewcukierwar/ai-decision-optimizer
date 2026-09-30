"""Small Streamlit MVP for the AI Decision Optimizer."""

import json
from typing import Any, Dict, Optional

from decision_optimizer import config as _config  # noqa: F401 - loads local .env
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
from decision_optimizer.direct_solver import DirectSolverError
from decision_optimizer.experiment import ExperimentConfig
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
    dayplan_preference_summary,
    format_dayplan_text_names,
    format_entity_name,
    format_date,
    format_duration,
    format_minutes_duration,
    format_shift_label,
    format_shift_location,
    format_shift_references,
    shift_infeasibility_summary,
    shift_preference_violation_count,
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

    controls = st.columns(2)
    with controls[0]:
        model = st.selectbox(
            "Base model",
            ("gpt-6-luna", "gpt-6.1-sol"),
            index=1,
            format_func=lambda value: (
                "GPT-6 Luna" if value == "gpt-6-luna" else "GPT-6.1 Sol"
            ),
        )
    with controls[1]:
        solution_engine = st.radio(
            "Solution engine",
            ("cp_sat", "direct_llm"),
            horizontal=True,
            format_func=lambda value: (
                "OR-Tools CP-SAT" if value == "cp_sat" else "Direct LLM"
            ),
        )
    experiment = ExperimentConfig(
        model=model, use_jev=False, solution_engine=solution_engine
    )
    _reset_for_experiment_change(st, prefix, experiment)
    st.caption("Architecture: " + experiment.label())

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
    _reset_for_request_change(st, prefix, request)
    if st.button("Interpret", type="primary", key=prefix + "_interpret"):
        _interpret(st, problem_type, request, experiment)

    extraction = st.session_state.get(prefix + "_extraction")
    if extraction is None:
        error = st.session_state.get(prefix + "_error")
        if error:
            st.error(error)
        else:
            _show_api_hint(st)
        return

    if problem_type == "Day Planner":
        _render_dayplan_flow(st, request, prefix, extraction, experiment)
    else:
        _render_shift_schedule_flow(st, request, prefix, extraction, experiment)


def _interpret(
    st: Any, problem_type: str, request: str, config: ExperimentConfig
) -> None:
    prefix = _prefix(problem_type)
    _clear_interpretation_state(st, prefix)
    st.session_state[prefix + "_source_request"] = request
    try:
        extraction = (
            parse_dayplan_request(request, config=config)
            if problem_type == "Day Planner"
            else parse_shift_schedule_request(request, config=config)
        )
    except (DayPlanError, ShiftScheduleError) as error:
        st.session_state[prefix + "_error"] = str(error)
        st.session_state.pop(prefix + "_extraction", None)
        return
    st.session_state.pop(prefix + "_error", None)
    _store_extraction(st, prefix, extraction, clarification_used=False)


def _render_dayplan_flow(
    st: Any,
    request: str,
    prefix: str,
    extraction: Any,
    config: ExperimentConfig,
) -> None:
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
                    extraction = clarify_dayplan_request(
                        request, extraction, answer, config=config
                    )
                    _store_extraction(st, prefix, extraction, clarification_used=True)
                except (DayPlanError, ShiftScheduleError) as error:
                    st.error(str(error))
                    return
        if extraction.plan is None:
            return

    _render_dayplan_interpretation(st, request, prefix, extraction.plan, config)


def _render_shift_schedule_flow(
    st: Any,
    request: str,
    prefix: str,
    extraction: Any,
    config: ExperimentConfig,
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
                        request, extraction, answer, config=config
                    )
                    _store_extraction(st, prefix, extraction, clarification_used=True)
                except (DayPlanError, ShiftScheduleError) as error:
                    st.error(str(error))
                    return
        if extraction.schedule is None:
            return

    _render_shift_schedule_interpretation(
        st, request, prefix, extraction.schedule, config
    )


def _render_dayplan_interpretation(
    st: Any,
    request: str,
    prefix: str,
    plan: Any,
    config: ExperimentConfig,
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
    _discard_run_if_json_changed(st, prefix, edited_json)
    if st.button("Confirm & Solve", type="primary", key=prefix + "_solve"):
        try:
            confirmed_plan = validate_edited_dayplan_json(edited_json)
        except ValueError as error:
            st.error("Edited interpretation is invalid: " + str(error))
            return
        try:
            run = solve_confirmed_dayplan(confirmed_plan, config=config)
        except DirectSolverError as error:
            st.error(str(error))
            return
        st.session_state[prefix + "_run"] = run
        st.session_state[prefix + "_solved_json"] = edited_json

    run = st.session_state.get(prefix + "_run")
    if run is not None:
        _render_dayplan_run(st, run)


def _render_shift_schedule_interpretation(
    st: Any,
    request: str,
    prefix: str,
    schedule: Any,
    config: ExperimentConfig,
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
    _discard_run_if_json_changed(st, prefix, edited_json)
    if st.button("Confirm & Solve", type="primary", key=prefix + "_solve"):
        try:
            confirmed_schedule = validate_edited_shift_schedule_json(edited_json)
        except ValueError as error:
            st.error("Edited interpretation is invalid: " + str(error))
            return
        try:
            run = solve_confirmed_shift_schedule(confirmed_schedule, config=config)
        except DirectSolverError as error:
            st.error(str(error))
            return
        st.session_state[prefix + "_run"] = run
        st.session_state[prefix + "_solved_json"] = edited_json

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
        _render_dayplan_diagnostic(st, run.diagnosis, run.plan)
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
    if solution.status.value == "infeasible":
        _render_shift_infeasible_result(st, run)
        return

    st.subheader("Recommended assignments")
    _render_shift_result_summary(st, run)
    if not run.validation.valid:
        st.error("Independent validation failed. The solver output is not trustworthy.")
        for error in run.validation.errors:
            st.write("- " + error)
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
                "Assigned shifts": format_shift_references(
                    run.schedule, load.assigned_shift_ids
                )
                or "—",
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
                    "Event": format_entity_name(event.name),
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
                    "Activity": format_entity_name(task.name),
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
                    "Dependency": (
                        f"{format_entity_name(item.before)} → "
                        f"{format_entity_name(item.after)}"
                    ),
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
                "Shift": format_shift_label(shift),
                "Date": format_date(shift.day),
                "Time": format_time_range(shift.start, shift.end),
                "Location": format_shift_location(shift.location),
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
                    "Eligible locations": ", ".join(
                        format_shift_location(location)
                        for location in employee.eligible_locations
                    )
                    or "None",
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
                "Preferred shifts": format_shift_references(
                    schedule, employee.preferred_shifts
                )
                or "None",
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
    columns[2].metric(
        "Preferences",
        dayplan_preference_summary(solution.preference_penalties),
    )
    st.caption(
        "Schedule produced by %s and independently validated."
        % (run.telemetry.architecture if run.telemetry else "the selected architecture")
    )


def _render_shift_result_summary(st: Any, run: ShiftScheduleRun) -> None:
    solution = run.solution
    columns = st.columns(4)
    columns[0].metric("Status", _status_label(solution.status.value))
    columns[1].metric("Validation", "Passed" if run.validation.valid else "Failed")
    columns[2].metric(
        "Preference violations",
        shift_preference_violation_count(solution.preference_penalties),
    )
    columns[3].metric(
        "Fairness spread",
        format_minutes_duration(solution.objective_breakdown.fairness_minutes_spread),
    )
    st.caption(
        "Assignments produced by %s and independently validated."
        % (run.telemetry.architecture if run.telemetry else "the selected architecture")
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
                    "Task": format_entity_name(assignment.name),
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
                    "Component": "Weighted preference contribution (raw count × weight)",
                    "Value": breakdown.weighted_preference_penalty,
                },
                {
                    "Component": "Normalized preference contribution (60 min per violation)",
                    "Value": breakdown.normalized_preference_penalty,
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
            "Shift": format_shift_label(shift),
            "Date": format_date(shift.day),
            "Time": format_time_range(shift.start, shift.end),
            "Location": format_shift_location(shift.location),
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
                "Task": format_entity_name(preference.task) if preference.task else "",
                "Target": target,
                "Penalty amount": penalty.amount,
                "Weighted penalty": penalty.weighted_penalty,
            }
        )
    return rows


def _dayplan_preference_label(preference: Any) -> str:
    if preference.type.value == "finish_before":
        return f"{format_entity_name(preference.task)} · finish before {format_time(preference.time)}"
    if preference.type.value == "preferred_window":
        return (
            f"{format_entity_name(preference.task)} · preferred "
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


def _render_shift_infeasible_result(st: Any, run: ShiftScheduleRun) -> None:
    summary = shift_infeasibility_summary(run.diagnosis)
    st.error(summary[0])
    for line in summary[1:]:
        st.write(line)
    if not run.validation.valid:
        st.error("Independent validation reported an issue with the infeasible result.")
        for error in run.validation.errors:
            st.write("- " + error)
    _render_shift_diagnostic(st, run.diagnosis)


def _render_shift_diagnostic(
    st: Any, diagnostic: Optional[InfeasibilityDiagnostic]
) -> None:
    if diagnostic is None:
        st.warning("No bounded diagnostic was available.")
        return

    suggestions = list(diagnostic.suggestions)
    if any(
        finding.code
        in {
            "coverage_eligibility_or_availability",
            "total_max_hours_too_small",
            "location_capacity_too_small",
        }
        for finding in diagnostic.findings
    ):
        suggestions.append("Reduce staffing requirements.")
    if suggestions:
        st.markdown("**Practical next steps**")
        for suggestion in dict.fromkeys(suggestions):
            st.write("- " + suggestion)

    primary_findings = [
        finding
        for finding in diagnostic.findings
        if not finding.code.endswith("_binding")
    ]
    if primary_findings:
        st.markdown("**Deterministic conflicts**")
    for finding in primary_findings:
        st.markdown("**%s** — %s" % (finding.summary, finding.evidence))

    advanced_findings = [
        finding
        for finding in diagnostic.findings
        if finding.code.endswith("_binding")
    ]
    if advanced_findings or diagnostic.tested_relaxations:
        with st.expander("Advanced diagnostic details"):
            for finding in advanced_findings:
                st.markdown("**%s** — %s" % (finding.summary, finding.evidence))
            if diagnostic.tested_relaxations:
                st.caption(
                    "Bounded relaxation checks: "
                    + ", ".join(diagnostic.tested_relaxations)
                )


def _render_dayplan_diagnostic(
    st: Any, diagnostic: Optional[InfeasibilityDiagnostic], plan: Any
) -> None:
    if diagnostic is None:
        st.warning("No bounded diagnostic was available.")
        return

    if diagnostic.suggestions:
        st.markdown("**Practical relaxation suggestions**")
        for suggestion in diagnostic.suggestions:
            st.write(
                "- "
                + format_dayplan_text_names(
                    suggestion,
                    [task.name for task in plan.tasks]
                    + [event.name for event in plan.fixed_events],
                )
            )

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
            st.markdown(
                "**%s** — %s"
                % (
                    finding.summary,
                    format_dayplan_text_names(
                        finding.evidence,
                        [task.name for task in plan.tasks]
                        + [event.name for event in plan.fixed_events],
                    ),
                )
            )

    if advanced_findings or diagnostic.tested_relaxations:
        with st.expander("Advanced diagnostic details"):
            if diagnostic.tested_relaxations:
                st.write(
                    "Bounded relaxation tests: "
                    + ", ".join(diagnostic.tested_relaxations)
                )
            for finding in advanced_findings:
                st.markdown(
                    "**%s** — %s"
                    % (
                        finding.summary,
                        format_dayplan_text_names(
                            finding.evidence,
                            [task.name for task in plan.tasks]
                            + [event.name for event in plan.fixed_events],
                        ),
                    )
                )


def _store_extraction(
    st: Any, prefix: str, extraction: Any, clarification_used: bool
) -> None:
    st.session_state[prefix + "_extraction"] = extraction
    st.session_state[prefix + "_clarification_used"] = clarification_used
    st.session_state.pop(prefix + "_run", None)
    st.session_state.pop(prefix + "_solved_json", None)

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
            "solved_json",
            "source_request",
            "error",
            "experiment_config",
        ):
            st.session_state.pop(prefix + "_" + suffix, None)
        st.session_state.pop(prefix + "_clarification_used", None)
    st.session_state["_active_problem_type"] = problem_type


def _reset_for_request_change(st: Any, prefix: str, request: str) -> None:
    """Discard all derived state when the natural-language source changes."""

    source_request = st.session_state.get(prefix + "_source_request")
    if source_request is None or source_request == request:
        return
    _clear_interpretation_state(st, prefix)
    st.session_state.pop(prefix + "_source_request", None)


def _reset_for_experiment_change(
    st: Any, prefix: str, config: ExperimentConfig
) -> None:
    """Invalidate derived state when the model or solution engine changes."""

    key = prefix + "_experiment_config"
    previous = st.session_state.get(key)
    current = config.model_dump()
    if previous is not None and previous != current:
        if previous.get("model") != current["model"]:
            _clear_interpretation_state(st, prefix)
            st.session_state.pop(prefix + "_source_request", None)
        else:
            st.session_state.pop(prefix + "_run", None)
            st.session_state.pop(prefix + "_solved_json", None)
    st.session_state[key] = current


def _clear_interpretation_state(st: Any, prefix: str) -> None:
    for suffix in (
        "extraction",
        "clarification",
        "clarification_used",
        "edited_json",
        "run",
        "solved_json",
        "error",
    ):
        st.session_state.pop(prefix + "_" + suffix, None)


def _discard_run_if_json_changed(st: Any, prefix: str, edited_json: str) -> None:
    """Prevent a run for an older JSON formulation from being redisplayed."""

    if prefix + "_run" not in st.session_state:
        return
    if st.session_state.get(prefix + "_solved_json") == edited_json:
        return
    st.session_state.pop(prefix + "_run", None)
    st.session_state.pop(prefix + "_solved_json", None)


def _prefix(problem_type: str) -> str:
    return "dayplan" if problem_type == "Day Planner" else "shift_schedule"


def _show_api_hint(st: Any) -> None:
    st.info("Enter a scheduling problem above, then select Interpret.")


if __name__ == "__main__":
    main()
