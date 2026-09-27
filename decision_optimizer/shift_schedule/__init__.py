"""Small, deterministic workforce ShiftSchedule scheduling core."""

from .compiler import CompiledShiftSchedule, compile_shift_schedule
from .schema import (
    Employee,
    EmployeeLoad,
    MaximumConsecutiveDaysRule,
    MinimumRestRule,
    ObjectiveBreakdown,
    ObjectiveWeights,
    PREFERENCE_NORMALIZATION_MINUTES,
    PreferencePenalty,
    RequiredDaysOffRule,
    Rule,
    RuleType,
    Shift,
    ShiftAssignment,
    ShiftSchedule,
    ShiftScheduleSolution,
    SolveStatus,
    Unavailability,
    minutes_to_time,
    time_to_minutes,
)
from .solver import solve_shift_schedule
from .validator import ValidationReport, validate_solution

__all__ = [
    "CompiledShiftSchedule",
    "Employee",
    "EmployeeLoad",
    "MaximumConsecutiveDaysRule",
    "MinimumRestRule",
    "ObjectiveBreakdown",
    "ObjectiveWeights",
    "PREFERENCE_NORMALIZATION_MINUTES",
    "PreferencePenalty",
    "RequiredDaysOffRule",
    "Rule",
    "RuleType",
    "Shift",
    "ShiftAssignment",
    "ShiftSchedule",
    "ShiftScheduleSolution",
    "SolveStatus",
    "Unavailability",
    "ValidationReport",
    "compile_shift_schedule",
    "minutes_to_time",
    "solve_shift_schedule",
    "time_to_minutes",
    "validate_solution",
]
