# src/tightening_project/analysis/features/labels.py  (o label.py)
from __future__ import annotations
import pandas as pd

from .base import require_columns


def make_label_current_nok(df: pd.DataFrame, *, torque_result_col: str = "Torque_Result") -> pd.DataFrame:
    require_columns(df, [torque_result_col], ctx="make_label_current_nok")
    out = df.copy()
    out["label"] = (out[torque_result_col] == "NOK").astype("int8")
    return out


def make_label_next_nok(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    torque_result_col: str = "Torque_Result",
) -> pd.DataFrame:
    require_columns(df, key_cols + [torque_result_col], ctx="make_label_next_nok")
    out = df.copy()
    out["label"] = (
        (out.groupby(key_cols, sort=False)[torque_result_col].shift(-1) == "NOK")
        .astype("int8")
    )
    return out
