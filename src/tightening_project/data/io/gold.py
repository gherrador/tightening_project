from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import List, Dict, Optional
from tightening_project.analysis.capability import compute_capability_v1

import pandas as pd

from tightening_project.analysis.spc import (
    compute_imr_limits,
    score_imr_points,
    build_spc_alerts_from_points,
)


def _asof_str(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _tier_validate(tier: str) -> str:
    t = (tier or "").strip().lower()
    if t not in {"core", "recurring"}:
        raise ValueError(f"tier inválido: {tier}. Usa 'core' o 'recurring'.")
    return t


# -------------------------
# Paths GOLD (layout con tier)
# -------------------------

def gold_spc_points_path(lake_root: Path, tier: str, year: int, month: int) -> Path:
    tier = _tier_validate(tier)
    return lake_root / "gold" / "spc_points" / f"tier={tier}" / f"year={year}" / f"month={month:02d}" / "part-0000.parquet"


def gold_spc_alerts_path(lake_root: Path, tier: str, year: int, month: int) -> Path:
    tier = _tier_validate(tier)
    return lake_root / "gold" / "spc_alerts" / f"tier={tier}" / f"year={year}" / f"month={month:02d}" / "part-0000.parquet"


def gold_spc_limits_path(lake_root: Path, tier: str, baseline_window: str, asof: str) -> Path:
    tier = _tier_validate(tier)
    return lake_root / "gold" / "spc_limits" / f"tier={tier}" / f"baseline_window={baseline_window}" / f"asof={asof}" / "part-0000.parquet"


def gold_meta_path(lake_root: Path, tier: str, year: int, month: int) -> Path:
    tier = _tier_validate(tier)
    return lake_root / "gold" / "_meta" / f"tier={tier}" / f"asof={_asof_str(year, month)}" / "gold_build_meta.json"


# -------------------------
# Lectura SILVER (mensual)
# -------------------------

def _silver_month_parquet_path(silver_month_dir: Path) -> Path:
    return silver_month_dir / "part-0000.parquet"


def _read_silver_month_dir(silver_month_dir: Path, *, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    p = _silver_month_parquet_path(silver_month_dir)
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=columns)


def _concat_silver_dirs(silver_dirs: List[Path], *, columns: Optional[List[str]] = None) -> pd.DataFrame:
    dfs = []
    for d in silver_dirs:
        df = _read_silver_month_dir(d, columns=columns)
        if df is not None and len(df) > 0:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -------------------------
# Build GOLD SPC v2 (I-MR)
# -------------------------

def build_spc_v2_for_month(
    *,
    lake_root: Path,
    year: int,
    month: int,
    tier: str,
    baseline_silver_dirs: List[Path],
    step_ids: Optional[List[int]] = None,
    baseline_window: str = "12m",
    force: bool = False,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
    value_col: str = "FinalTorque",
    min_points_limits: int = 200,
) -> Dict[str, Path]:
    """
    Construye GOLD SPC v2 (I-MR) para (year, month) y un tier:
      - limits: baseline (meses previos)
      - points: mes actual scoreado (join límites + flags)
      - alerts: agregado de OOC por STEP_ID
      - meta
    """
    tier = _tier_validate(tier)
    asof = _asof_str(year, month)

    out_points = gold_spc_points_path(lake_root, tier, year, month)
    out_alerts = gold_spc_alerts_path(lake_root, tier, year, month)
    out_limits = gold_spc_limits_path(lake_root, tier, baseline_window, asof)
    out_meta = gold_meta_path(lake_root, tier, year, month)

    # Idempotencia
    if all(p.exists() for p in [out_points, out_alerts, out_limits, out_meta]) and not force:
        return {"spc_points": out_points, "spc_alerts": out_alerts, "spc_limits": out_limits, "meta": out_meta}

    cols = [key_col, time_col, value_col]

    # Eval (mes actual)
    silver_month_dir = lake_root / "silver" / f"year={year}" / f"month={month:02d}"
    df_eval = _read_silver_month_dir(silver_month_dir, columns=cols)
    if df_eval is None or df_eval.empty:
        raise FileNotFoundError(f"No existe SILVER del mes o vacío: {silver_month_dir}")

    # Baseline (meses previos)
    df_base = _concat_silver_dirs(baseline_silver_dirs, columns=cols)

    # Filtrado por STEP_ID (CORE/RECURRING)
    if step_ids is not None:
        df_eval = df_eval[df_eval[key_col].isin(step_ids)]
        df_base = df_base[df_base[key_col].isin(step_ids)]

    # Limits y scoring
    df_limits = compute_imr_limits(
        df_base,
        key_col=key_col,
        time_col=time_col,
        value_col=value_col,
        min_points=min_points_limits,
    )

    df_points = score_imr_points(
    df_eval,
    df_limits,
    key_col=key_col,
    time_col=time_col,
    value_col=value_col 
)
    df_alerts = build_spc_alerts_from_points(df_points, key_col=key_col, time_col=time_col)

    # Escribir outputs
    for p in [out_points, out_alerts, out_limits, out_meta]:
        p.parent.mkdir(parents=True, exist_ok=True)

    df_limits.to_parquet(out_limits, index=False)
    df_points.to_parquet(out_points, index=False)
    df_alerts.to_parquet(out_alerts, index=False)

    meta = {
        "layer": "gold",
        "version": "spc_v2_imr",
        "tier": tier,
        "asof": asof,
        "baseline_window": baseline_window,
        "year": year,
        "month": month,
        "key_col": key_col,
        "time_col": time_col,
        "value_col": value_col,
        "min_points_limits": min_points_limits,
        "filters": {
            "step_ids_provided": bool(step_ids is not None),
            "step_ids_count": int(len(step_ids)) if step_ids is not None else None,
        },
        "inputs": {
            "silver_month_dir": str(silver_month_dir.resolve()),
            "baseline_silver_dirs": [str(p.resolve()) for p in baseline_silver_dirs],
        },
        "outputs": {
            "spc_points": str(out_points.resolve()),
            "spc_alerts": str(out_alerts.resolve()),
            "spc_limits": str(out_limits.resolve()),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "eval_rows": int(df_eval.shape[0]),
            "baseline_rows": int(df_base.shape[0]) if df_base is not None else 0,
            "limits_rows": int(df_limits.shape[0]),
            "points_rows": int(df_points.shape[0]) if df_points is not None else 0,
            "alerts_rows": int(df_alerts.shape[0]) if df_alerts is not None else 0,
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"spc_points": out_points, "spc_alerts": out_alerts, "spc_limits": out_limits, "meta": out_meta}


def gold_capability_path(lake_root: Path, tier: str, baseline_window: str, asof: str) -> Path:
    tier = _tier_validate(tier)
    return (
        lake_root / "gold" / "capability"
        / f"tier={tier}" / f"baseline_window={baseline_window}" / f"asof={asof}"
        / "part-0000.parquet"
    )


def build_capability_v1_for_month(
    *,
    lake_root: Path,
    year: int,
    month: int,
    tier: str,
    baseline_silver_dirs: List[Path],
    step_ids: Optional[List[int]] = None,
    baseline_window: str = "12m",
    force: bool = False,
    key_col: str = "STEP_ID",
    time_col: str = "DateTime",
    value_col: str = "FinalTorque",
    lsl_col: str = "TorqueMinTolerance",
    usl_col: str = "TorqueMaxTolerance",
    min_points: int = 200,
) -> Dict[str, Path]:
    """
    Capability v1 (baseline only) para asof=YYYY-MM.
    """
    tier = _tier_validate(tier)
    asof = _asof_str(year, month)

    out_cap = gold_capability_path(lake_root, tier, baseline_window, asof)
    out_meta = gold_meta_path(lake_root, tier, year, month)  # reutiliza meta por asof/tier

    if out_cap.exists() and out_meta.exists() and not force:
        return {"capability": out_cap, "meta": out_meta}

    cols = [key_col, time_col, value_col, lsl_col, usl_col]
    df_base = _concat_silver_dirs(baseline_silver_dirs, columns=cols)

    if step_ids is not None:
        df_base = df_base[df_base[key_col].isin(step_ids)]

    df_cap = compute_capability_v1(
        df_base,
        key_col=key_col,
        time_col=time_col,
        value_col=value_col,
        lsl_col=lsl_col,
        usl_col=usl_col,
        min_points=min_points,
    )

    out_cap.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    df_cap.to_parquet(out_cap, index=False)

    # Meta mínimo (no pisa el SPC meta si ya lo usas; si quieres separar, hacemos otro meta file)
    meta = {
        "layer": "gold",
        "version": "capability_v1",
        "tier": tier,
        "asof": asof,
        "baseline_window": baseline_window,
        "year": year,
        "month": month,
        "key_col": key_col,
        "time_col": time_col,
        "value_col": value_col,
        "lsl_col": lsl_col,
        "usl_col": usl_col,
        "min_points": min_points,
        "inputs": {
            "baseline_silver_dirs": [str(p.resolve()) for p in baseline_silver_dirs],
        },
        "outputs": {
            "capability": str(out_cap.resolve()),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "baseline_rows": int(df_base.shape[0]),
            "capability_rows": int(df_cap.shape[0]) if df_cap is not None else 0,
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"capability": out_cap, "meta": out_meta}
