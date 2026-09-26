"""Typed input and output models for the DayPlan problem.

The public representation uses ``datetime.time`` because it is a standard,
typed Pydantic value and serializes naturally from ``HH:MM`` JSON.  The
compiler uses integer minutes through the conversion helpers below.
"""

from datetime import time as datetime_time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer, model_validator


class TaskMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


class PreferenceType(str, Enum):
    FINISH_BEFORE = "finish_before"
    MINIMIZE_WORK_INTERRUPTIONS = "minimize_work_interruptions"


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


def time_to_minutes(value: datetime_time) -> int:
    """Convert a time-of-day to whole minutes after midnight."""

    if value.second or value.microsecond:
        raise ValueError("DayPlan times must have minute precision")
    return value.hour * 60 + value.minute


def minutes_to_time(minutes: int) -> datetime_time:
    """Convert whole minutes after midnight back to a ``datetime.time``."""

    if not 0 <= minutes < 24 * 60:
        raise ValueError("minutes must be within one calendar day")
    return datetime_time(hour=minutes // 60, minute=minutes % 60)


class _TimeModel(BaseModel):
    """Shared compact ``HH:MM`` JSON serialization for time-bearing models."""

    @field_serializer(
        "start",
        "end",
        "earliest_start",
        "latest_end",
        "time",
        check_fields=False,
    )
    def serialize_time(self, value: Optional[datetime_time]) -> Optional[str]:
        if value is None:
            return None
        return value.strftime("%H:%M")


class TimeWindow(_TimeModel):
    start: datetime_time
    end: datetime_time

    @model_validator(mode="after")
    def validate_order(self) -> "TimeWindow":
        if time_to_minutes(self.end) <= time_to_minutes(self.start):
            raise ValueError("end must be after start")
        return self


class Horizon(TimeWindow):
    pass


class FixedEvent(_TimeModel):
    name: str = Field(min_length=1)
    start: datetime_time
    end: datetime_time

    @model_validator(mode="after")
    def validate_event(self) -> "FixedEvent":
        if time_to_minutes(self.end) <= time_to_minutes(self.start):
            raise ValueError("fixed event end must be after start")
        return self


class Task(_TimeModel):
    name: str = Field(min_length=1)
    duration_min: int = Field(gt=0)
    mode: TaskMode
    earliest_start: Optional[datetime_time] = None
    latest_end: Optional[datetime_time] = None
    required: bool = True

    @model_validator(mode="after")
    def validate_task_window(self) -> "Task":
        if self.earliest_start:
            time_to_minutes(self.earliest_start)
        if self.latest_end:
            time_to_minutes(self.latest_end)
        if self.earliest_start and self.latest_end:
            if time_to_minutes(self.latest_end) <= time_to_minutes(self.earliest_start):
                raise ValueError("latest_end must be after earliest_start")
        return self


class Precedence(BaseModel):
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)
    min_gap_min: int = Field(default=0, ge=0)
    max_gap_min: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_gap(self) -> "Precedence":
        if self.before == self.after:
            raise ValueError("a task cannot precede itself")
        if self.max_gap_min is not None and self.max_gap_min < self.min_gap_min:
            raise ValueError("max_gap_min must be at least min_gap_min")
        return self


class Preference(_TimeModel):
    type: PreferenceType
    task: Optional[str] = None
    time: Optional[datetime_time] = None
    weight: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_preference(self) -> "Preference":
        if self.time:
            time_to_minutes(self.time)
        if self.type == PreferenceType.FINISH_BEFORE:
            if self.task is None or self.time is None:
                raise ValueError("finish_before requires task and time")
        elif self.task is not None or self.time is not None:
            raise ValueError("minimize_work_interruptions takes no task or time")
        return self


class DayPlan(BaseModel):
    # A missing horizon is allowed only for an interpretation that has
    # ``missing_info``.  The deterministic compiler still requires one.
    horizon: Optional[Horizon] = None
    work_window: Optional[TimeWindow] = None
    fixed_events: List[FixedEvent] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    precedences: List[Precedence] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references_and_windows(self) -> "DayPlan":
        horizon_start = horizon_end = None
        if self.horizon:
            horizon_start = time_to_minutes(self.horizon.start)
            horizon_end = time_to_minutes(self.horizon.end)
        elif not self.missing_info:
            raise ValueError("horizon is required unless missing_info is populated")
        task_names = [task.name for task in self.tasks]
        fixed_names = [event.name for event in self.fixed_events]

        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique")
        if len(set(fixed_names)) != len(fixed_names):
            raise ValueError("fixed event names must be unique")
        if set(task_names).intersection(fixed_names):
            raise ValueError("task and fixed event names must be disjoint")

        if self.work_window:
            work_start = time_to_minutes(self.work_window.start)
            work_end = time_to_minutes(self.work_window.end)
            if self.horizon and (work_start < horizon_start or work_end > horizon_end):
                raise ValueError("work_window must be inside horizon")

        for event in self.fixed_events:
            event_start = time_to_minutes(event.start)
            event_end = time_to_minutes(event.end)
            if self.horizon and (event_start < horizon_start or event_end > horizon_end):
                raise ValueError("fixed events must be inside horizon")

        task_name_set = set(task_names)
        for precedence in self.precedences:
            if precedence.before not in task_name_set or precedence.after not in task_name_set:
                raise ValueError("precedence references an unknown task")

        for preference in self.preferences:
            if preference.type == PreferenceType.FINISH_BEFORE:
                if preference.task not in task_name_set:
                    raise ValueError("finish_before references an unknown task")
        return self


class TaskAssignment(_TimeModel):
    name: str = Field(min_length=1)
    start: datetime_time
    end: datetime_time


class PreferencePenalty(BaseModel):
    preference_index: int = Field(ge=0)
    preference_type: PreferenceType
    amount: int = Field(ge=0)
    weighted_penalty: int = Field(ge=0)


class DayPlanSolution(BaseModel):
    status: SolveStatus
    assignments: List[TaskAssignment] = Field(default_factory=list)
    objective_value: int = Field(default=0, ge=0)
    preference_penalties: List[PreferencePenalty] = Field(default_factory=list)
    optimal: bool = False
    message: Optional[str] = None
