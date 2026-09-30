"""Focused TypeSafe Jev decisions at the typed interpretation boundary.

Jev only revisits closed-set semantic judgments already represented by a
``DayPlan`` or ``ShiftSchedule``.  It does not extract names, times, or
durations and it never schedules work.  All applicable questions are sent in
one ``system_one`` request and confidence-gated rewrites are validated through
the existing Pydantic model before reaching either solution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as datetime_time
import math
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

from .config import jev_apply_threshold, typesafe_api_key, typesafe_model
from .dayplan import DayPlan, Preference, PreferenceType, TaskMode
from .shift_schedule import Shift, ShiftSchedule, Unavailability, time_to_minutes
from .telemetry import RunTelemetry


Domain = Literal["dayplan", "shift_schedule"]
Primitive = Literal["choice", "score", "noul"]
Problem = Union[DayPlan, ShiftSchedule]


class JevError(Exception):
    """The Jev decision request could not produce a usable typed result."""


class MissingTypeSafeAPIKeyError(JevError):
    """A live Jev call was requested without ``TYPESAFE_API_KEY``."""


class JevDecision(BaseModel):
    """Serializable evidence for one closed-set Jev judgment."""

    model_config = ConfigDict(extra="forbid")

    question_type: str
    item: str
    primitive: Primitive
    gpt_value: Union[str, int, bool]
    jev_value: Union[str, int, bool]
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    yes_probability: Optional[float] = Field(default=None, ge=0, le=1)
    expected_score: Optional[float] = None
    applied: bool
    changed: bool


@dataclass(frozen=True)
class JevQuestionTarget:
    """One item to which a registry question type applies."""

    item: str
    locator: Mapping[str, Any]
    context: Mapping[str, Any]


@dataclass(frozen=True)
class JevQuestionType:
    """Registry abstraction for a closed-set semantic question."""

    key: str
    primitive: Primitive
    domains: FrozenSet[Domain]
    applies_to: Callable[[str, Problem], Sequence[JevQuestionTarget]]
    prompt: Callable[[JevQuestionTarget], str]
    gpt_value: Callable[[Problem, JevQuestionTarget], Union[str, int, bool]]
    apply: Callable[[Problem, JevQuestionTarget, Union[str, int, bool]], Problem]


@dataclass(frozen=True)
class JevQuestion:
    """A concrete registry question ready for the SDK batch."""

    name: str
    question_type: JevQuestionType
    target: JevQuestionTarget
    sdk_question: Union[Choice, Score, Noul]


@dataclass(frozen=True)
class JevResult:
    problem: Problem
    decisions: List[JevDecision]
    resolved_model: Optional[str] = None


# TypeSafe documents and encourages multi-question batching, but its public API
# reference does not currently publish a maximum question count.  Keep requests
# comfortably bounded and deterministic instead of depending on an undocumented
# server limit.  Callers can lower this for operational constraints.
DEFAULT_JEV_BATCH_SIZE = 64
AVAILABILITY_CONTROLS_PER_EMPLOYEE = 2


PREFERENCE_STRENGTH_RUBRIC = (
    "Weight 1: a mild preference; it is fine to miss",
    "Weight 2: a modest preference",
    "Weight 3: an important preference",
    "Weight 4: a very strong preference",
    "Weight 5: the strongest soft preference, short of a hard requirement",
)


def score_to_weight(expected_score: float) -> int:
    """Map SDK score levels 0..4 to integer optimizer weights 1..5.

    The SDK returns the probability-weighted expected zero-based rubric level.
    We add one and round half upward (not Python's banker rounding), then clamp
    to the schema's five supported weights.  The unrounded expected score and
    full distribution remain in ``JevDecision`` for later calibration.
    """

    return max(1, min(5, int(math.floor(float(expected_score) + 1.5))))


def build_jev_questions(
    request: str,
    problem: Problem,
    *,
    domain: Optional[Domain] = None,
) -> List[JevQuestion]:
    """Derive applicable questions without making an API call."""

    resolved_domain: Domain = domain or (
        "dayplan" if isinstance(problem, DayPlan) else "shift_schedule"
    )
    questions: List[JevQuestion] = []
    for definition in JEV_QUESTION_TYPES:
        if resolved_domain not in definition.domains:
            continue
        for target in definition.applies_to(request, problem):
            questions.append(
                JevQuestion(
                    name="q%03d_%s" % (len(questions), definition.key),
                    question_type=definition,
                    target=target,
                    sdk_question=_sdk_question(definition, target),
                )
            )
    return questions


def apply_jev(
    request: str,
    problem: Problem,
    *,
    client: Any = None,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
    telemetry: Optional[RunTelemetry] = None,
    max_questions_per_request: int = DEFAULT_JEV_BATCH_SIZE,
    request_timeout_seconds: Optional[float] = None,
    max_retries: int = 0,
) -> JevResult:
    """Batch applicable questions, confidence-gate rewrites, and revalidate."""

    questions = build_jev_questions(request, problem)
    if not questions:
        return JevResult(problem=problem, decisions=[])
    if max_questions_per_request < 1:
        raise ValueError("max_questions_per_request must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")

    selected_model = model or typesafe_model()
    selected_threshold = jev_apply_threshold() if threshold is None else threshold
    if not 0 <= selected_threshold <= 1:
        raise ValueError("Jev apply threshold must be between 0 and 1")

    owns_client = client is None
    if client is None:
        api_key = typesafe_api_key()
        if not api_key:
            raise MissingTypeSafeAPIKeyError(
                "TYPESAFE_API_KEY is required when Jev is enabled"
            )
        client = TypeSafeClient(
            api_key=api_key,
            timeout=request_timeout_seconds,
            retry=RetryPolicy(max_retries=max_retries),
        )

    answers_by_name: Dict[str, Any] = {}
    resolved_models: List[str] = []
    try:
        for start in range(0, len(questions), max_questions_per_request):
            chunk = questions[start : start + max_questions_per_request]
            state = {
                "original_request": request,
                "extracted_context": [
                    {
                        "question": item.question_type.key,
                        "item": item.target.item,
                        "gpt_value": item.question_type.gpt_value(problem, item.target),
                        **dict(item.target.context),
                    }
                    for item in chunk
                ],
            }
            try:
                call_kwargs = {
                    "state": state,
                    "questions": {
                        item.name: item.sdk_question for item in chunk
                    },
                    "model": selected_model,
                    "retry": RetryPolicy(max_retries=max_retries),
                }
                if request_timeout_seconds is not None:
                    call_kwargs["timeout"] = request_timeout_seconds
                if telemetry is None:
                    response = client.system_one(**call_kwargs)
                else:
                    with telemetry.track("jev"):
                        response = client.system_one(**call_kwargs)
                    telemetry.record_jev_response(response, len(chunk))
            except JevError:
                raise
            except Exception as exc:
                raise JevError("TypeSafe System One request failed: %s" % exc) from exc

            answers = getattr(response, "answers", None)
            if not isinstance(answers, Mapping):
                raise JevError("TypeSafe response did not contain an answers mapping")
            for question in chunk:
                if question.name not in answers:
                    raise JevError("TypeSafe response omitted answer %s" % question.name)
                answers_by_name[question.name] = answers[question.name]
            response_model = getattr(response, "model", None)
            if response_model and str(response_model) not in resolved_models:
                resolved_models.append(str(response_model))
    finally:
        if owns_client and client is not None:
            client.close()

    adjusted: Problem = problem.model_copy(deep=True)
    decisions: List[JevDecision] = []
    for question in questions:
        answer = answers_by_name[question.name]
        gpt_value = question.question_type.gpt_value(problem, question.target)
        jev_value, confidence, detail = _read_answer(
            question.question_type.primitive, answer
        )
        accepted = confidence >= selected_threshold
        if accepted:
            before = adjusted.model_dump(mode="json")
            candidate = question.question_type.apply(
                adjusted, question.target, jev_value
            )
            changed = candidate.model_dump(mode="json") != before
            adjusted = candidate
        else:
            changed = False
        decisions.append(
            JevDecision(
                question_type=question.question_type.key,
                item=question.target.item,
                primitive=question.question_type.primitive,
                gpt_value=gpt_value,
                jev_value=jev_value,
                confidence=confidence,
                probabilities=detail["probabilities"],
                model_confidence=detail.get("model_confidence"),
                yes_probability=detail.get("yes_probability"),
                expected_score=detail.get("expected_score"),
                applied=accepted,
                changed=changed,
            )
        )

    # This is the only output boundary: no parallel Jev planning schema.
    validated = type(problem).model_validate(adjusted.model_dump(mode="python"))
    return JevResult(
        problem=validated,
        decisions=decisions,
        resolved_model=",".join(resolved_models) or None,
    )


def _sdk_question(
    definition: JevQuestionType, target: JevQuestionTarget
) -> Union[Choice, Score, Noul]:
    instructions = definition.prompt(target)
    if definition.key == "constraint_hardness":
        return Choice(
            instructions=instructions,
            criteria={
                "hard": "A firm requirement that the schedule must satisfy",
                "soft": "A preference that may be violated at an objective cost",
            },
        )
    if definition.key == "task_requirement":
        return Choice(
            instructions=instructions,
            criteria={
                "required": "The task must happen in this plan",
                "optional": "The task may be omitted if useful or necessary",
            },
        )
    if definition.key == "task_mode":
        return Choice(
            instructions=instructions,
            criteria={
                "active": "Needs the person's attention for its full duration",
                "passive": "Can run unattended after any setup",
            },
        )
    if definition.key == "pref_weight":
        return Score(instructions=instructions, criteria=PREFERENCE_STRENGTH_RUBRIC)
    if definition.key == "availability_applies":
        return Noul(
            instructions=instructions,
            criteria={
                "true": "The stated unavailability rules out this candidate shift",
                "false": "The stated unavailability does not rule out this shift",
            },
        )
    raise JevError("No SDK question constructor for " + definition.key)


def _read_answer(
    primitive: Primitive, answer: Any
) -> tuple[Union[str, int, bool], float, Dict[str, Any]]:
    if primitive == "choice":
        selected = str(_answer_value(answer, "choice"))
        probabilities = {
            str(key): float(value)
            for key, value in dict(_answer_value(answer, "probabilities")).items()
        }
        confidence = probabilities.get(
            selected, float(_answer_value(answer, "confidence"))
        )
        return selected, confidence, {
            "probabilities": probabilities,
            "model_confidence": float(_answer_value(answer, "confidence")),
        }
    if primitive == "noul":
        yes_probability = float(_answer_value(answer, "noul"))
        return yes_probability >= 0.5, max(yes_probability, 1 - yes_probability), {
            "probabilities": {
                "no": 1 - yes_probability,
                "yes": yes_probability,
            },
            "yes_probability": yes_probability,
        }
    expected_score = float(_answer_value(answer, "score"))
    probabilities = {
        str(key): float(value)
        for key, value in dict(_answer_value(answer, "probabilities")).items()
    }
    return score_to_weight(expected_score), float(
        _answer_value(answer, "confidence")
    ), {
        "probabilities": probabilities,
        "model_confidence": float(_answer_value(answer, "confidence")),
        "expected_score": expected_score,
    }


def _answer_value(answer: Any, key: str) -> Any:
    if isinstance(answer, Mapping):
        return answer[key]
    return getattr(answer, key)


def _constraint_targets(_: str, problem: Problem) -> Sequence[JevQuestionTarget]:
    if not isinstance(problem, DayPlan):
        return []
    targets: List[JevQuestionTarget] = []
    for index, preference in enumerate(problem.preferences):
        if preference.type not in {
            PreferenceType.FINISH_BEFORE,
            PreferenceType.PREFERRED_WINDOW,
        }:
            continue
        targets.append(
            JevQuestionTarget(
                item="preference:%d:%s" % (index, preference.task),
                locator={
                    "kind": "soft_preference",
                    "signature": preference.model_dump(mode="json"),
                },
                context={"timing": preference.model_dump(mode="json")},
            )
        )
    for task in problem.tasks:
        if task.latest_end is None:
            continue
        kind = "hard_window" if task.earliest_start is not None else "hard_latest"
        targets.append(
            JevQuestionTarget(
                item="task:%s:%s" % (task.name, kind),
                locator={"kind": kind, "task": task.name},
                context={
                    "task": task.name,
                    "earliest_start": _time_text(task.earliest_start),
                    "latest_end": _time_text(task.latest_end),
                },
            )
        )
    return targets


def _constraint_prompt(target: JevQuestionTarget) -> str:
    return (
        "For %s, is the user's timing language a firm requirement or a soft preference?"
        % target.item
    )


def _constraint_gpt_value(
    _: Problem, target: JevQuestionTarget
) -> Union[str, int, bool]:
    return "soft" if target.locator["kind"] == "soft_preference" else "hard"


def _apply_constraint(
    problem: Problem, target: JevQuestionTarget, value: Union[str, int, bool]
) -> Problem:
    assert isinstance(problem, DayPlan)
    kind = target.locator["kind"]
    if kind == "soft_preference":
        if value == "soft":
            return problem
        index = _find_preference(problem, target.locator["signature"])
        if index is None:
            return problem
        preference = problem.preferences[index]
        tasks = list(problem.tasks)
        task_index = next(
            index for index, task in enumerate(tasks) if task.name == preference.task
        )
        task = tasks[task_index]
        if preference.type == PreferenceType.FINISH_BEFORE:
            task = task.model_copy(update={"latest_end": preference.time})
        else:
            task = task.model_copy(
                update={
                    "earliest_start": preference.start,
                    "latest_end": preference.end,
                }
            )
        tasks[task_index] = task
        preferences = list(problem.preferences)
        preferences.pop(index)
        return problem.model_copy(update={"tasks": tasks, "preferences": preferences})

    if value == "hard":
        return problem
    tasks = list(problem.tasks)
    task_index = next(
        index for index, task in enumerate(tasks) if task.name == target.locator["task"]
    )
    task = tasks[task_index]
    preferences = list(problem.preferences)
    if kind == "hard_window":
        preferences.append(
            Preference(
                type=PreferenceType.PREFERRED_WINDOW,
                task=task.name,
                start=task.earliest_start,
                end=task.latest_end,
                weight=1,
            )
        )
        task = task.model_copy(update={"earliest_start": None, "latest_end": None})
    else:
        preferences.append(
            Preference(
                type=PreferenceType.FINISH_BEFORE,
                task=task.name,
                time=task.latest_end,
                weight=1,
            )
        )
        task = task.model_copy(update={"latest_end": None})
    tasks[task_index] = task
    return problem.model_copy(update={"tasks": tasks, "preferences": preferences})


def _task_targets(_: str, problem: Problem) -> Sequence[JevQuestionTarget]:
    if not isinstance(problem, DayPlan):
        return []
    return [
        JevQuestionTarget(
            item="task:%s" % task.name,
            locator={"task": task.name},
            context={
                "task": task.name,
                "duration_min": task.duration_min,
                "mode": task.mode.value,
                "required": task.required,
            },
        )
        for task in problem.tasks
    ]


def _requirement_prompt(target: JevQuestionTarget) -> str:
    return "Must %s happen as part of this plan?" % target.locator["task"]


def _requirement_gpt_value(
    problem: Problem, target: JevQuestionTarget
) -> Union[str, int, bool]:
    assert isinstance(problem, DayPlan)
    task = _task(problem, str(target.locator["task"]))
    return "required" if task.required else "optional"


def _apply_requirement(
    problem: Problem, target: JevQuestionTarget, value: Union[str, int, bool]
) -> Problem:
    assert isinstance(problem, DayPlan)
    return _replace_task(
        problem,
        str(target.locator["task"]),
        required=value == "required",
    )


def _mode_prompt(target: JevQuestionTarget) -> str:
    return (
        "Does %s need the person's attention for the whole duration, or can it run unattended?"
        % target.locator["task"]
    )


def _mode_gpt_value(
    problem: Problem, target: JevQuestionTarget
) -> Union[str, int, bool]:
    assert isinstance(problem, DayPlan)
    return _task(problem, str(target.locator["task"])).mode.value


def _apply_mode(
    problem: Problem, target: JevQuestionTarget, value: Union[str, int, bool]
) -> Problem:
    assert isinstance(problem, DayPlan)
    return _replace_task(
        problem, str(target.locator["task"]), mode=TaskMode(str(value))
    )


def _preference_targets(request: str, problem: Problem) -> Sequence[JevQuestionTarget]:
    if isinstance(problem, DayPlan):
        return [
            JevQuestionTarget(
                item="preference:%d:%s" % (index, preference.type.value),
                locator={
                    "domain": "dayplan",
                    "signature": preference.model_dump(mode="json"),
                },
                context={"preference": preference.model_dump(mode="json")},
            )
            for index, preference in enumerate(problem.preferences)
        ]

    targets: List[JevQuestionTarget] = []
    if any(employee.preferred_shifts for employee in problem.employees) or (
        problem.objective_weights.preference_penalty != 1
    ):
        targets.append(
            JevQuestionTarget(
                item="objective:preference_penalty",
                locator={"domain": "shift_schedule", "field": "preference_penalty"},
                context={
                    "objective": "employee shift preferences",
                    "weight": problem.objective_weights.preference_penalty,
                },
            )
        )
    request_lower = request.lower()
    if problem.objective_weights.fairness != 1 or any(
        token in request_lower for token in ("fair", "balance", "evenly")
    ):
        targets.append(
            JevQuestionTarget(
                item="objective:fairness",
                locator={"domain": "shift_schedule", "field": "fairness"},
                context={
                    "objective": "fair distribution of work",
                    "weight": problem.objective_weights.fairness,
                },
            )
        )
    return targets


def _preference_prompt(target: JevQuestionTarget) -> str:
    return "How strongly does the user want %s satisfied?" % target.item


def _preference_gpt_value(
    problem: Problem, target: JevQuestionTarget
) -> Union[str, int, bool]:
    if isinstance(problem, DayPlan):
        index = _find_preference(problem, target.locator["signature"])
        if index is None:
            # A hardness rewrite may remove it later, but baseline is always
            # read against the original problem before application.
            return int(target.locator["signature"]["weight"])
        return problem.preferences[index].weight
    return int(getattr(problem.objective_weights, str(target.locator["field"])))


def _apply_preference(
    problem: Problem, target: JevQuestionTarget, value: Union[str, int, bool]
) -> Problem:
    weight = int(value)
    if isinstance(problem, DayPlan):
        index = _find_preference(problem, target.locator["signature"])
        if index is None:
            return problem
        preferences = list(problem.preferences)
        preferences[index] = preferences[index].model_copy(update={"weight": weight})
        return problem.model_copy(update={"preferences": preferences})
    weights = problem.objective_weights.model_copy(
        update={str(target.locator["field"]): weight}
    )
    return problem.model_copy(update={"objective_weights": weights})


def _availability_targets(_: str, problem: Problem) -> Sequence[JevQuestionTarget]:
    """Select positive availability candidates plus bounded negative controls.

    Every shift actually covered by an extracted explicit shift id, unavailable
    day, or overlapping unavailable window is included.  Per employee, up to two
    non-covered shifts are added as controls, preferring same-day temporal
    contrasts and then the closest chronological shifts.  Preferred shifts do
    not make an employee eligible for availability questions.

    This intentionally tests the consequences of extracted unavailability; it
    cannot recover an unavailability statement that GPT omitted entirely, and a
    very broad statement may still yield many positive questions.
    """

    if not isinstance(problem, ShiftSchedule):
        return []
    targets: List[JevQuestionTarget] = []
    ordered_shifts = sorted(
        problem.shifts,
        key=lambda shift: (
            shift.day,
            time_to_minutes(shift.start),
            time_to_minutes(shift.end),
            shift.id,
        ),
    )
    for employee in problem.employees:
        if not employee.unavailable:
            continue
        positive_ids = {
            shift.id
            for shift in ordered_shifts
            if _is_unavailable(employee.unavailable, shift)
        }
        negative_shifts = [
            shift for shift in ordered_shifts if shift.id not in positive_ids
        ]
        control_ids = _availability_control_ids(
            employee.unavailable,
            ordered_shifts,
            negative_shifts,
        )
        selected_ids = positive_ids | control_ids
        for shift in ordered_shifts:
            if shift.id not in selected_ids:
                continue
            targets.append(
                JevQuestionTarget(
                    item="employee:%s:shift:%s" % (employee.name, shift.id),
                    locator={"employee": employee.name, "shift": shift.id},
                    context={
                        "employee": employee.name,
                        "unavailable": [
                            item.model_dump(mode="json") for item in employee.unavailable
                        ],
                        "candidate_shift": shift.model_dump(mode="json"),
                        "selection_role": (
                            "covered_by_extracted_unavailability"
                            if shift.id in positive_ids
                            else "negative_control"
                        ),
                    },
                )
            )
    return targets


def _availability_control_ids(
    periods: Sequence[Unavailability],
    ordered_shifts: Sequence[Shift],
    negative_shifts: Sequence[Shift],
) -> set[str]:
    if not negative_shifts:
        return set()

    ranked: List[tuple[int, int, int, str]] = []
    for shift_index, shift in enumerate(ordered_shifts):
        if shift not in negative_shifts:
            continue
        same_day = any(period.day == shift.day for period in periods if period.day)
        closest_positive = min(
            (
                abs(shift_index - positive_index)
                for positive_index, candidate in enumerate(ordered_shifts)
                if _is_unavailable(periods, candidate)
            ),
            default=len(ordered_shifts),
        )
        ranked.append(
            (
                0 if same_day else 1,
                closest_positive,
                shift_index,
                shift.id,
            )
        )
    ranked.sort()
    return {
        shift_id
        for _, _, _, shift_id in ranked[:AVAILABILITY_CONTROLS_PER_EMPLOYEE]
    }


def _availability_prompt(target: JevQuestionTarget) -> str:
    return (
        "Does %s's unavailability statement rule out candidate shift %s?"
        % (target.locator["employee"], target.locator["shift"])
    )


def _availability_gpt_value(
    problem: Problem, target: JevQuestionTarget
) -> Union[str, int, bool]:
    assert isinstance(problem, ShiftSchedule)
    employee = next(
        item for item in problem.employees if item.name == target.locator["employee"]
    )
    shift = next(item for item in problem.shifts if item.id == target.locator["shift"])
    return _is_unavailable(employee.unavailable, shift)


def _apply_availability(
    problem: Problem, target: JevQuestionTarget, value: Union[str, int, bool]
) -> Problem:
    assert isinstance(problem, ShiftSchedule)
    employees = list(problem.employees)
    employee_index = next(
        index
        for index, employee in enumerate(employees)
        if employee.name == target.locator["employee"]
    )
    employee = employees[employee_index]
    target_shift = next(
        shift for shift in problem.shifts if shift.id == target.locator["shift"]
    )
    currently_applies = _is_unavailable(employee.unavailable, target_shift)
    if bool(value) == currently_applies:
        return problem
    if bool(value):
        unavailable = list(employee.unavailable) + [Unavailability(shift_id=target_shift.id)]
    else:
        unavailable = []
        for period in employee.unavailable:
            if not _period_applies(period, target_shift):
                unavailable.append(period)
                continue
            # Preserve every other effect of a broad day/window statement as
            # explicit shift ids while removing only this candidate shift.
            for shift in problem.shifts:
                if shift.id != target_shift.id and _period_applies(period, shift):
                    unavailable.append(Unavailability(shift_id=shift.id))
        unavailable = _unique_unavailability(unavailable)
    employees[employee_index] = employee.model_copy(update={"unavailable": unavailable})
    return problem.model_copy(update={"employees": employees})


def _task(problem: DayPlan, name: str) -> Any:
    return next(task for task in problem.tasks if task.name == name)


def _replace_task(problem: DayPlan, name: str, **updates: Any) -> DayPlan:
    tasks = list(problem.tasks)
    index = next(index for index, task in enumerate(tasks) if task.name == name)
    tasks[index] = tasks[index].model_copy(update=updates)
    return problem.model_copy(update={"tasks": tasks})


def _find_preference(problem: DayPlan, signature: Mapping[str, Any]) -> Optional[int]:
    for index, preference in enumerate(problem.preferences):
        if preference.model_dump(mode="json") == dict(signature):
            return index
    return None


def _time_text(value: Optional[datetime_time]) -> Optional[str]:
    return None if value is None else value.strftime("%H:%M")


def _is_unavailable(periods: Sequence[Unavailability], shift: Shift) -> bool:
    return any(_period_applies(period, shift) for period in periods)


def _period_applies(period: Unavailability, shift: Shift) -> bool:
    if period.shift_id is not None:
        return period.shift_id == shift.id
    if period.day != shift.day:
        return False
    if period.start is None:
        return True
    assert period.end is not None
    return (
        time_to_minutes(shift.start) < time_to_minutes(period.end)
        and time_to_minutes(period.start) < time_to_minutes(shift.end)
    )


def _unique_unavailability(items: Sequence[Unavailability]) -> List[Unavailability]:
    result: List[Unavailability] = []
    seen = set()
    for item in items:
        key = tuple(sorted(item.model_dump(mode="json").items()))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


JEV_QUESTION_TYPES: tuple[JevQuestionType, ...] = (
    JevQuestionType(
        key="constraint_hardness",
        primitive="choice",
        domains=frozenset({"dayplan"}),
        applies_to=_constraint_targets,
        prompt=_constraint_prompt,
        gpt_value=_constraint_gpt_value,
        apply=_apply_constraint,
    ),
    JevQuestionType(
        key="task_requirement",
        primitive="choice",
        domains=frozenset({"dayplan"}),
        applies_to=_task_targets,
        prompt=_requirement_prompt,
        gpt_value=_requirement_gpt_value,
        apply=_apply_requirement,
    ),
    JevQuestionType(
        key="task_mode",
        primitive="choice",
        domains=frozenset({"dayplan"}),
        applies_to=_task_targets,
        prompt=_mode_prompt,
        gpt_value=_mode_gpt_value,
        apply=_apply_mode,
    ),
    JevQuestionType(
        key="pref_weight",
        primitive="score",
        domains=frozenset({"dayplan", "shift_schedule"}),
        applies_to=_preference_targets,
        prompt=_preference_prompt,
        gpt_value=_preference_gpt_value,
        apply=_apply_preference,
    ),
    JevQuestionType(
        key="availability_applies",
        primitive="noul",
        domains=frozenset({"shift_schedule"}),
        applies_to=_availability_targets,
        prompt=_availability_prompt,
        gpt_value=_availability_gpt_value,
        apply=_apply_availability,
    ),
)
