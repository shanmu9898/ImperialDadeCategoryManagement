"""Single source of truth for input validation across pipeline stages.

Replaces the duplicated `ValidationError` and `validate_*` helpers that lived
in Classification_functions.py, Load_Isolate_functions.py, Exclusion_functions.py,
Matching_functions.py, etc.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


class ValidationError(Exception):
    """Raised when an input fails a pipeline-stage precondition."""


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValidationError(f"{name} cannot be empty")


def validate_string_input(value: str, name: str) -> None:
    if not value or not isinstance(value, str):
        raise ValidationError(f"{name} must be a non-empty string")


def validate_list_input(value: Sequence[str], name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValidationError(f"All items in {name} must be strings")


def validate_columns_exist(
    df: pd.DataFrame, required_columns: Sequence[str], df_name: str
) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required columns in {df_name}: {', '.join(missing)}"
        )
