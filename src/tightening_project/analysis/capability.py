from __future__ import annotations

import pandas as pd

D2_MR2 = 1.128  # para MR de 2 (I-MR)


def compute_capability_v1(
    df_baseline: pd.DataFrame,
    *,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
    value_col: str = "FinalTorque",
    lsl_col: str = "TorqueMinTolerance",
    usl_col: str = "TorqueMaxTolerance",
    min_points: int = 200,
    min_mr: int | None = None,
) -> pd.DataFrame:
    """
    Capability v2 por STEP_ID usando baseline (I-MR).

    - Overall sigma: std(ddof=1) de todo el baseline (Pp/Ppk).
    - Within sigma: MRbar/d2 (Cp/Cpk), consistente con I-MR/SPC.

    Manejo tolerancias:
    - Si LSL/USL varían dentro del STEP, tomamos mediana (robusto) y
      devolvemos flags + conteos de tolerancias distintas.

    Mejoras vs v1:
    - Calcula n_mr (conteo de MR válidos) y filtra por min_mr.
    - Protege divisiones por 0/NaN: si std_overall<=0 o sigma_within<=0 => NA.
    - Protege tolerancias inválidas (tol_span<=0) => NA.

    Output principal:
      STEP_ID, N, n_mr, mean, std_overall, mrbar, sigma_within,
      lsl, usl, tol_span,
      Pp, Ppk, Cp, Cpk,
      tol_variants_lsl, tol_variants_usl, tol_inconsistent
    """
    if df_baseline is None or df_baseline.empty:
        return pd.DataFrame()

    need = [key_col, time_col, value_col, lsl_col, usl_col]
    missing = [c for c in need if c not in df_baseline.columns]
    if missing:
        raise KeyError(f"compute_capability_v2: faltan columnas {missing}")

    # min_mr por defecto: N-1 para I-MR (si pides 200 puntos, esperas 199 MR)
    if min_mr is None:
        min_mr = max(min_points - 1, 1)

    x = df_baseline[[key_col, time_col, value_col, lsl_col, usl_col]].dropna().copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])

    # Orden estable para MR
    x = x.sort_values([key_col, time_col], kind="mergesort")

    # MR por STEP (I-MR: rango móvil entre observaciones consecutivas)
    x["MR"] = x.groupby(key_col, sort=False)[value_col].diff().abs()

    g = x.groupby(key_col, sort=False)

    stats = g.agg(
        N=(value_col, "size"),
        mean=(value_col, "mean"),
        std_overall=(value_col, lambda s: s.std(ddof=1)),
        mrbar=("MR", "mean"),          # mean ignora NaN del primer diff
        n_mr=("MR", "count"),          # count ignora NaN -> #MR válidos
        lsl_med=(lsl_col, "median"),
        usl_med=(usl_col, "median"),
        tol_variants_lsl=(lsl_col, "nunique"),
        tol_variants_usl=(usl_col, "nunique"),
    ).reset_index()

    # Filtra mínimos (puntos y MR válidos)
    stats = stats[(stats["N"] >= min_points) & (stats["n_mr"] >= min_mr)].copy()

    # tolerancias finales
    stats["lsl"] = stats["lsl_med"]
    stats["usl"] = stats["usl_med"]
    stats["tol_span"] = stats["usl"] - stats["lsl"]

    # flags de tolerancias
    stats["tol_inconsistent"] = (
        (stats["tol_variants_lsl"] > 1) | (stats["tol_variants_usl"] > 1)
    ).astype("int8")

    # sigma within (I-MR)
    stats["sigma_within"] = stats["mrbar"] / D2_MR2

    # Inicializa outputs como NA
    for c in ["Pp", "Ppk", "Cp", "Cpk"]:
        stats[c] = pd.NA

    # --- Máscaras de validez ---
    tol_ok = stats["tol_span"] > 0
    overall_ok = stats["std_overall"].notna() & (stats["std_overall"] > 0)
    within_ok = stats["sigma_within"].notna() & (stats["sigma_within"] > 0)

    # --- Overall capability (Pp/Ppk) ---
    m_overall = tol_ok & overall_ok
    if m_overall.any():
        denom_overall = 3.0 * stats.loc[m_overall, "std_overall"]
        stats.loc[m_overall, "Pp"] = stats.loc[m_overall, "tol_span"] / (6.0 * stats.loc[m_overall, "std_overall"])
        ppu = (stats.loc[m_overall, "usl"] - stats.loc[m_overall, "mean"]) / denom_overall
        ppl = (stats.loc[m_overall, "mean"] - stats.loc[m_overall, "lsl"]) / denom_overall
        stats.loc[m_overall, "Ppk"] = pd.concat([ppu, ppl], axis=1).min(axis=1)

    # --- Within capability (Cp/Cpk) ---
    m_within = tol_ok & within_ok
    if m_within.any():
        denom_within = 3.0 * stats.loc[m_within, "sigma_within"]
        stats.loc[m_within, "Cp"] = stats.loc[m_within, "tol_span"] / (6.0 * stats.loc[m_within, "sigma_within"])
        cpu = (stats.loc[m_within, "usl"] - stats.loc[m_within, "mean"]) / denom_within
        cpl = (stats.loc[m_within, "mean"] - stats.loc[m_within, "lsl"]) / denom_within
        stats.loc[m_within, "Cpk"] = pd.concat([cpu, cpl], axis=1).min(axis=1)

    out_cols = [
        key_col, "N", "n_mr", "mean",
        "std_overall", "mrbar", "sigma_within",
        "lsl", "usl", "tol_span",
        "Pp", "Ppk", "Cp", "Cpk",
        "tol_variants_lsl", "tol_variants_usl", "tol_inconsistent",
    ]
    return stats[out_cols].copy()
