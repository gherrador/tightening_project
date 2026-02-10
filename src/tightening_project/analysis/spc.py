from __future__ import annotations

import pandas as pd

# Constantes I-MR para MR de 2
D2_MR2 = 1.128
D3_MR2 = 0.0
D4_MR2 = 3.267


def compute_imr_limits(
    df_baseline: pd.DataFrame,
    *,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
    value_col: str = "FinalTorque",
    min_points: int = 200,
) -> pd.DataFrame:
    """
    Calcula límites I-MR por STEP_ID usando baseline.

    Output:
      [STEP_ID, N, Xbar, MRbar, Sigma, UCL, LCL, UCL_MR, LCL_MR]
    """
    if df_baseline is None or df_baseline.empty:
        return pd.DataFrame(
            columns=[key_col, "N", "Xbar", "MRbar", "Sigma", "UCL", "LCL", "UCL_MR", "LCL_MR"]
        )

    need = [key_col, time_col, value_col]
    missing = [c for c in need if c not in df_baseline.columns]
    if missing:
        raise KeyError(f"compute_imr_limits: faltan columnas {missing}")

    x = df_baseline[[key_col, time_col, value_col]].dropna().copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])

    # Orden estable para MR
    x = x.sort_values([key_col, time_col], kind="mergesort")

    # MR(t) = |x(t) - x(t-1)|
    x["MR"] = x.groupby(key_col, sort=False)[value_col].diff().abs()

    g = x.groupby(key_col, sort=False)
    limits = g.agg(
        N=(value_col, "size"),
        Xbar=(value_col, "mean"),
        MRbar=("MR", "mean"),
    ).reset_index()

    # Rechaza grupos con pocos puntos (sigma se vuelve basura)
    limits = limits[limits["N"] >= min_points].copy()

    # sigma estimada desde MRbar
    limits["Sigma"] = limits["MRbar"] / D2_MR2

    # Límites Individuals
    limits["UCL"] = limits["Xbar"] + 3.0 * limits["Sigma"]
    limits["LCL"] = limits["Xbar"] - 3.0 * limits["Sigma"]

    # Límites MR
    limits["UCL_MR"] = D4_MR2 * limits["MRbar"]
    limits["LCL_MR"] = D3_MR2 * limits["MRbar"]  # 0

    return limits


def score_imr_points(
    df_eval: pd.DataFrame,
    df_limits: pd.DataFrame,
    *,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
    value_col: str = "FinalTorque",
) -> pd.DataFrame:
    """
    Scorea puntos del mes contra límites I-MR.

    Agrega:
      - MR (moving range)
      - z  (Individuals z-score)
      - is_ooc_i, is_ooc_mr
      - rule: OK | I_3SIGMA | MR_3SIGMA
      - is_alert (union de I o MR)
    """
    if df_eval is None or df_eval.empty:
        return pd.DataFrame()
    if df_limits is None or df_limits.empty:
        return pd.DataFrame()

    need = [key_col, time_col, value_col]
    missing = [c for c in need if c not in df_eval.columns]
    if missing:
        raise KeyError(f"score_imr_points: faltan columnas {missing}")

    pts = df_eval[[key_col, time_col, value_col]].dropna().copy()
    pts[time_col] = pd.to_datetime(pts[time_col], errors="coerce")
    pts = pts.dropna(subset=[time_col])

    pts = pts.sort_values([key_col, time_col], kind="mergesort")

    join_cols = [key_col, "N", "Xbar", "Sigma", "UCL", "LCL", "MRbar", "UCL_MR", "LCL_MR"]
    pts = pts.merge(df_limits[join_cols], on=key_col, how="inner")

    pts["MR"] = pts.groupby(key_col, sort=False)[value_col].diff().abs()
    pts["z"] = (pts[value_col] - pts["Xbar"]) / pts["Sigma"].replace(0, pd.NA)

    pts["is_ooc_i"] = ((pts[value_col] > pts["UCL"]) | (pts[value_col] < pts["LCL"])).astype("int8")
    pts["is_ooc_mr"] = (pts["MR"] > pts["UCL_MR"]).fillna(False).astype("int8")

    # Regla final (prioridad a MR porque suele ser causa especial mecánica)
    pts["rule"] = "OK"
    pts.loc[pts["is_ooc_mr"] == 1, "rule"] = "MR_3SIGMA"
    pts.loc[(pts["is_ooc_i"] == 1) & (pts["rule"] == "OK"), "rule"] = "I_3SIGMA"

    pts["is_alert"] = ((pts["is_ooc_i"] == 1) | (pts["is_ooc_mr"] == 1)).astype("int8")

    return pts


def build_spc_alerts_from_points(
    spc_points: pd.DataFrame,
    *,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
) -> pd.DataFrame:
    """
    Agrega alertas por STEP_ID desde puntos scoreados.
    Requiere columnas: is_alert, rule, DateTime.
    Devuelve:
      STEP_ID, n_points, n_alerts, first_alert, last_alert, I_3SIGMA, MR_3SIGMA
    """
    if spc_points is None or spc_points.empty:
        return pd.DataFrame(columns=[key_col, "n_points", "n_alerts", "first_alert", "last_alert"])

    need = [key_col, time_col, "is_alert", "rule"]
    missing = [c for c in need if c not in spc_points.columns]
    if missing:
        raise KeyError(f"build_spc_alerts_from_points: faltan columnas {missing}")

    df = spc_points.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    agg = (
        df.groupby(key_col, sort=False)
        .agg(
            n_points=(time_col, "size"),
            n_alerts=("is_alert", "sum"),
            first_alert=(time_col, lambda s: df.loc[s.index][df.loc[s.index, "is_alert"] == 1][time_col].min()),
            last_alert=(time_col, lambda s: df.loc[s.index][df.loc[s.index, "is_alert"] == 1][time_col].max()),
        )
        .reset_index()
    )
    agg = agg[agg["n_alerts"] > 0].copy()

    # Breakdown por tipo de regla
    breakdown = (
        df[df["is_alert"] == 1]
        .pivot_table(index=key_col, columns="rule", values=time_col, aggfunc="size", fill_value=0)
        .reset_index()
    )

    out = agg.merge(breakdown, on=key_col, how="left")

    # Asegura columnas (por si un mes no hubo de un tipo)
    if "I_3SIGMA" not in out.columns:
        out["I_3SIGMA"] = 0
    if "MR_3SIGMA" not in out.columns:
        out["MR_3SIGMA"] = 0

    return out
