# src/tightening_project/analysis/features/base.py
from __future__ import annotations
import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str], ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{ctx}] " if ctx else ""
        raise KeyError(f"{prefix}Faltan columnas: {missing}")


def ensure_sorted(df: pd.DataFrame, *, key_cols: list[str], time_col: str) -> pd.DataFrame:
    require_columns(df, key_cols + [time_col], ctx="ensure_sorted")
    return df.sort_values(by=key_cols + [time_col], kind="mergesort").reset_index(drop=True)
