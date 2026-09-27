"""Typed input and output models for the workforce ShiftSchedule problem.

This is deliberately a small, concrete schema.  It models dated, same-day
shifts and a closed set of rules rather than attempting to be a general
workforce scheduling DSL.
"""

from datetime import date as datetime_date
from datetime import time as datetime_time
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_serializer, model_validator


class RuleType(str, Enum):
    MINIMUM_REST = "minimum_rest"
    MAXIMUM_CONSECUTIVE_DAYS = "maximum_consecutive_days"
    REQUIRED_DAYS_OFF = "required_days_off"


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


def time_to_minutes(value: datetime_time) -> int:
    """Convert a minute-precision time of day to minutes after midnight."""

    if value.second or value.microsecond:
        raise ValueError("ShiftSchedule times must have minute precision")
    return value.hour * 60 + value.minute


def minutes_to_time(minutes: int) -> datetime_time:
    """Convert minutes after midnight back to a time of day."""

    if not 0 <= minutes < 24 * 60:
        raise ValueError("minutes must be within one calendar day")
    return datetime_time(hour=minutes // 60, minute=minutes % 60)


class _ShiftTimeModel(BaseModel):
    """Compact HH:MM JSON serialization for time-bearing models."""

    @field_serializer("start", "end", check_fields=False)
    def serialize_time(self, value: Optional[datetime_time]) -> Optional[str]:
        if value is None:
            return None
        return value.strftime("%H:%M")


class Shift(_ShiftTimeModel):
    id: str = Field(min_length=1)
    day: datetime_date
    start: datetime_time
    end: datetime_time
    location: str = Field(min_length=1)
    required_staff: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_shift(self) -> "Shift":
        if time_to_minutes(self.end) <= time_to_minutes(self.start):
            raise ValueError("shift end must be after shift start")
        return self

    @property
    def duration_minutes(self) -> int:
        return time_to_minutes(self.end) - time_to_minutes(self.start)


class Unavailability(_ShiftTimeModel):
    """An unavailable shift, dated window, or entire calendar day.

    Use ``shift_id`` for a specific shift.  Use ``day`` with ``start`` and
    ``end`` for a time window, or just ``day`` for a full day off.
    """

    shift_id: Optional[str] = None
    day: Optional[datetime_date] = None
    start: Optional[datetime_time] = None
    end: Optional[datetime_time] = None

    @model_validator(mode="after")
    def validate_unavailability(self) -> "Unavailability":
        if self.shift_id is not None:
            if any(value is not None for value in (self.day, self.start, self.end)):
                raise ValueError("shift_id unavailability cannot also specify a day or time window")
            return self

        if self.day is None:
            raise ValueError("unavailability requires shift_id or day")
        if (self.start is None) != (self.end is None):
            raise ValueError("unavailability start and end must be supplied together")
        if self.start is not None and time_to_minutes(self.end) <= time_to_minutes(self.start):
            raise ValueError("unavailability end must be after start")
        return self


class Employee(BaseModel):
    name: str = Field(min_length=1)
    max_hours: int = Field(ge=0)
    unavailable: List[Unavailability] = Field(default_factory=list)
    eligible_locations: List[str] = Field(default_factory=list)
    preferred_shifts: List[str] = Field(default_factory=list)


class MinimumRestRule(BaseModel):
    type: Literal["minimum_rest"] = "minimum_rest"
    min_rest_hours: int = Field(ge=0)


class MaximumConsecutiveDaysRule(BaseModel):
    type: Literal["maximum_consecutive_days"] = "maximum_consecutive_days"
    max_days: int = Field(ge=1)


class RequiredDaysOffRule(BaseModel):
    type: Literal["required_days_off"] = "required_days_off"
    employee_name: str = Field(min_length=1)
    days: List[datetime_date] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_days(self) -> "RequiredDaysOffRule":
        if len(set(self.days)) != len(self.days):
            raise ValueError("required days off must be unique")
        return self


# Keep this as a regular union rather than a discriminated union.  Pydantic's
# discriminated form emits JSON Schema ``oneOf``, which is rejected by the
# Structured Outputs subset used by the Responses API.  The Literal ``type``
# field on each branch still makes model validation select the right concrete
# rule class, while the regular union emits ``anyOf``.
Rule = Union[MinimumRestRule, MaximumConsecutiveDaysRule, RequiredDaysOffRule]


class ObjectiveWeights(BaseModel):
    """Integer weights for the two intentionally small objective components."""

    preference_penalty: int = Field(default=1, ge=0)
    fairness: int = Field(default=1, ge=0)


# The normalized optimization objective treats one outside-preference assignment
# as approximately one hour of workload spread.  Materialized fairness remains
# exact in minutes; this constant only defines the primary objective's units.
PREFERENCE_NORMALIZATION_MINUTES = 60


class ShiftSchedule(BaseModel):
    shifts: List[Shift] = Field(min_length=1)
    employees: List[Employee] = Field(min_length=1)
    rules: List[Rule] = Field(default_factory=list)
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)

    @model_validator(mode="after")
    def validate_references(self) -> "ShiftSchedule":
        shift_ids = [shift.id for shift in self.shifts]
        employee_names = [employee.name for employee in self.employees]
        if len(set(shift_ids)) != len(shift_ids):
            raise ValueError("shift ids must be unique")
        if len(set(employee_names)) != len(employee_names):
            raise ValueError("employee names must be unique")

        shift_id_set = set(shift_ids)
        employee_name_set = set(employee_names)
        for employee in self.employees:
            unknown_preferences = set(employee.preferred_shifts) - shift_id_set
            if unknown_preferences:
                raise ValueError(
                    "employee %s prefers unknown shift(s): %s"
                    % (employee.name, ", ".join(sorted(unknown_preferences)))
                )
            for unavailable in employee.unavailable:
                if unavailable.shift_id is not None and unavailable.shift_id not in shift_id_set:
                    raise ValueError(
                        "employee %s is unavailable for unknown shift: %s"
                        % (employee.name, unavailable.shift_id)
                    )

        for rule in self.rules:
            if isinstance(rule, RequiredDaysOffRule) and rule.employee_name not in employee_name_set:
                raise ValueError(
                    "required_days_off references unknown employee: " + rule.employee_name
                )
        return self


class ShiftAssignment(BaseModel):
    shift_id: str = Field(min_length=1)
    employee_name: str = Field(min_length=1)


class EmployeeLoad(BaseModel):
    employee_name: str = Field(min_length=1)
    hours: float = Field(ge=0, allow_inf_nan=False)
    assigned_shift_ids: List[str] = Field(default_factory=list)


class PreferencePenalty(BaseModel):
    employee_name: str = Field(min_length=1)
    shift_id: str = Field(min_length=1)
    amount: int = Field(ge=0)
    weighted_penalty: int = Field(ge=0)


class ObjectiveBreakdown(BaseModel):
    """Materialized objective values.

    ``weighted_preference_penalty`` is the raw weighted count.  One violation
    is normalized to 60 minutes in ``normalized_preference_penalty`` so it is
    comparable to the exact minute-based fairness spread.  ``total`` is the
    normalized primary optimization score.  The CP-SAT model uses raw
    preference violations only as a secondary tie-break after this primary
    score, so that tie-break is intentionally not included in ``total``.
    """

    preference_penalty: int = Field(default=0, ge=0)
    weighted_preference_penalty: int = Field(default=0, ge=0)
    normalized_preference_penalty: int = Field(default=0, ge=0)
    fairness_minutes_spread: int = Field(default=0, ge=0)
    weighted_fairness: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)


class ShiftScheduleSolution(BaseModel):
    status: SolveStatus
    assignments: List[ShiftAssignment] = Field(default_factory=list)
    employee_hours: List[EmployeeLoad] = Field(default_factory=list)
    preference_penalties: List[PreferencePenalty] = Field(default_factory=list)
    objective_breakdown: ObjectiveBreakdown = Field(default_factory=ObjectiveBreakdown)
    objective_value: int = Field(default=0, ge=0)
    optimal: bool = False
    message: Optional[str] = None
