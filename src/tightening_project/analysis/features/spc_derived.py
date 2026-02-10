# src/tightening_project/analysis/features/spc_derived.py
from __future__ import annotations
import pandas as pd

from .base import require_columns


def add_spc_distance_features(
    df_points: pd.DataFrame,
    df_limits: pd.DataFrame,
    *,
    key_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    """
    df_limits debe traer por grupo: mean, sigma, ucl, lcl.
    """
    require_columns(df_points, key_cols + [value_col], ctx="add_spc_distance_features(points)")
    require_columns(df_limits, key_cols + ["mean", "sigma", "ucl", "lcl"], ctx="add_spc_distance_features(limits)")

    out = df_points.copy()
    out = out.merge(df_limits[key_cols + ["mean", "sigma", "ucl", "lcl"]], on=key_cols, how="left")

    sigma = out["sigma"].where(out["sigma"] != 0)
    out["z_baseline"] = (out[value_col] - out["mean"]) / sigma
    out["dist_to_ucl"] = out["ucl"] - out[value_col]
    out["dist_to_lcl"] = out[value_col] - out["lcl"]
    out["near_ucl"] = (out["z_baseline"].abs() > 2).astype("int8")

    return out
