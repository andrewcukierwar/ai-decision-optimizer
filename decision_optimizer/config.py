"""Environment bootstrap and small runtime configuration helpers.

This is the only module in the project that loads ``.env``.  Callers import
configuration values from here so local ``uv`` commands work without manual
shell exports while deployed environments can continue to provide variables
normally.
"""

import os
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-6-sol"
DEFAULT_TYPESAFE_MODEL = "jev-latest"
DEFAULT_JEV_APPLY_THRESHOLD = 0.7


def openai_model() -> str:
    """Return the configured parser model, preserving the existing default."""

    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or None


def typesafe_model() -> str:
    return os.getenv("TYPESAFE_MODEL", DEFAULT_TYPESAFE_MODEL)


def typesafe_api_key() -> Optional[str]:
    return os.getenv("TYPESAFE_API_KEY") or None


def jev_apply_threshold() -> float:
    raw_value = os.getenv("JEV_APPLY_THRESHOLD")
    if raw_value is None:
        return DEFAULT_JEV_APPLY_THRESHOLD
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("JEV_APPLY_THRESHOLD must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError("JEV_APPLY_THRESHOLD must be between 0 and 1")
    return value
