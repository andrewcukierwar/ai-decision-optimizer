"""Boundary markers and the benchmark's fixed failure policy."""

class ModelOutputFailure(Exception):
    """A completed response failed the requested output contract."""


class CachedTerminalFailure(Exception):
    def __init__(self, message: str, category: str):
        super().__init__(message)
        self.category = category


def failure_category(error: Exception) -> str:
    if isinstance(error, CachedTerminalFailure):
        return error.category
    if isinstance(error, ModelOutputFailure):
        return "invalid_model_output"
    # API boundary wrappers preserve their causes. No generic Pydantic rule.
    from httpx import TimeoutException, TransportError
    from openai import APIConnectionError, APIStatusError
    from typesafe_sdk import TypeSafeAPIError, TypeSafeAPIConnectionError, TypeSafeAPITimeoutError
    current = error
    while current is not None:
        if isinstance(current, (TimeoutError, TimeoutException, TypeSafeAPITimeoutError)) or type(current).__name__ == "APITimeoutError":
            return "timeout"
        if isinstance(current, (TransportError, APIConnectionError, APIStatusError, TypeSafeAPIError, TypeSafeAPIConnectionError)):
            return "transport_failure"
        current = current.__cause__
    return "implementation_or_configuration_error"


def terminal_failure(error: Exception) -> bool:
    return failure_category(error) in {"invalid_model_output", "timeout"}
