"""Stable semantic question identities for cached observations and labels."""
import json
from typing import Any, Dict, List, Mapping, Sequence
from .dayplan import DayPlan
from .shift_schedule import ShiftSchedule
from .evaluation import align_names
from .jev import JevQuestion

def indexed_questions(
    domain: str,
    questions: Sequence[JevQuestion],
    problem: DayPlan | ShiftSchedule,
    canonical: DayPlan | ShiftSchedule,
) -> List[tuple[str, JevQuestion]]:
    task_mapping: Dict[str, str] = {}
    employee_mapping: Dict[str, str] = {}
    shift_mapping: Dict[str, str] = {}
    if domain == "dayplan":
        assert isinstance(problem, DayPlan) and isinstance(canonical, DayPlan)
        task_mapping = align_names(
            [item.name for item in problem.tasks],
            [item.name for item in canonical.tasks],
        ).mapping
    else:
        assert isinstance(problem, ShiftSchedule) and isinstance(
            canonical, ShiftSchedule
        )
        employee_mapping = align_names(
            [item.name for item in problem.employees],
            [item.name for item in canonical.employees],
        ).mapping
        shift_mapping = align_names(
            [item.id for item in problem.shifts],
            [item.id for item in canonical.shifts],
        ).mapping

    occurrences: Dict[str, int] = {}
    indexed = []
    for question in questions:
        base = _semantic_question_identity(
            question,
            task_mapping=task_mapping,
            employee_mapping=employee_mapping,
            shift_mapping=shift_mapping,
        )
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        indexed.append(("%s#%d" % (base, occurrence), question))
    return indexed


def _semantic_question_identity(
    question: JevQuestion,
    *,
    task_mapping: Mapping[str, str],
    employee_mapping: Mapping[str, str],
    shift_mapping: Mapping[str, str],
) -> str:
    key = question.question_type.key
    locator = question.target.locator
    context = question.target.context
    identity: Dict[str, Any]
    if key in {"task_requirement", "task_mode"}:
        task = str(locator["task"])
        identity = {"task": task_mapping.get(task, task)}
    elif key == "constraint_hardness":
        if locator["kind"] == "soft_preference":
            timing = locator["signature"]
            task = str(timing.get("task"))
            identity = {
                "task": task_mapping.get(task, task),
                "start": timing.get("start"),
                "end": timing.get("end") or timing.get("time"),
            }
        else:
            task = str(locator["task"])
            identity = {
                "task": task_mapping.get(task, task),
                "start": context.get("earliest_start"),
                "end": context.get("latest_end"),
            }
    elif key == "pref_weight" and locator["domain"] == "dayplan":
        preference = locator["signature"]
        task = preference.get("task")
        identity = {
            "type": preference["type"],
            "task": task_mapping.get(str(task), task) if task is not None else None,
            "time": preference.get("time"),
            "start": preference.get("start"),
            "end": preference.get("end"),
        }
    elif key == "pref_weight":
        identity = {"objective": locator["field"]}
    else:
        employee = str(locator["employee"])
        shift = str(locator["shift"])
        identity = {
            "employee": employee_mapping.get(employee, employee),
            "shift": shift_mapping.get(shift, shift),
        }
    return "%s:%s" % (
        key,
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )


