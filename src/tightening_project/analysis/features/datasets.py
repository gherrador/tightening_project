from __future__ import annotations
import pandas as pd

from tightening_project.analysis.features.labels import make_label_current_nok, make_label_next_nok
from tightening_project.analysis.features.spc_derived import add_spc_distance_features
from tightening_project.analysis.features import temporal


def build_features_tabular(
    df_silver: pd.DataFrame,
    *,
    key_cols: list[str],
    time_col: str,
    torque_col: str = "FinalTorque",
    target_col: str = "TorqueTarget",
    lsl_col: str = "Torque_LSL",
    usl_col: str = "Torque_USL",
    label_mode: str = "current_nok",   # "current_nok" | "next_nok"
    lags: list[int] = [1, 2, 3],
    rolling_windows: list[int] = [10, 30, 100],
) -> pd.DataFrame:
    """
    Features v1: tabular y reproducibles (sin IO).
    """
    df = df_silver.copy()

    # --- checks mínimos
    required = key_cols + [time_col, torque_col, target_col, lsl_col, usl_col, "Torque_Result"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para features: {missing}")

    # --- orden temporal estable
    df = df.sort_values(by=key_cols + [time_col], kind="mergesort").reset_index(drop=True)

    # --- label
    if label_mode == "current_nok":
        df = make_label_current_nok(df)
    elif label_mode == "next_nok":
        df = make_label_next_nok(df, key_cols=key_cols)
    else:
        raise ValueError(f"label_mode inválido: {label_mode}")
    
    df = temporal.add_rolling_event_rate(df, key_cols=key_cols, event_col="label", windows=rolling_windows)  
    
    # --- core engineering (error / margins)
    df["torque_error"] = df[torque_col] - df[target_col]
    df["torque_error_abs"] = df["torque_error"].abs()

    spec_width = (df[usl_col] - df[lsl_col]).replace(0, pd.NA)
    df["error_norm_spec"] = df["torque_error"] / spec_width
    df["abs_error_norm_spec"] = df["error_norm_spec"].abs()

    df["margin_to_usl"] = df[usl_col] - df[torque_col]
    df["margin_to_lsl"] = df[torque_col] - df[lsl_col]
    df["in_spec"] = ((df["margin_to_usl"] >= 0) & (df["margin_to_lsl"] >= 0)).astype("int8")

    # --- temporal features (sobre torque_error por defecto)
    df = temporal.add_lags(df, key_cols=key_cols, value_col="torque_error", lags=lags)
    df = temporal.add_rolling_stats(df, key_cols=key_cols, value_col="torque_error", windows=rolling_windows, min_periods=5)
    df = temporal.add_rolling_trend(df, key_cols=key_cols, value_col="torque_error", windows=[10, 30], min_periods=5)
    df = temporal.add_rows_since_last_event(df, key_cols=key_cols, event_col="label", event_value=1)
    return df



