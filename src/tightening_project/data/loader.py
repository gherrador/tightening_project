from __future__ import annotations
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Tuple, Dict
from tightening_project.data.io.gold import build_capability_v1_for_month 

from tightening_project.paths import RAW_ROOT, LAKE_ROOT

from tightening_project.data.io.discovery import (
    available_batches_in_raw,
    latest_available_batch,
    latest_raw_csv_auto,
    latest_raw_csv_for_month,
    months_between
)
from tightening_project.data.io.bronze import ingest_csv_to_bronze
from tightening_project.data.io.naming import parse_source_stem
from tightening_project.data.io.silver import build_silver_from_bronze, load_silver_range
from tightening_project.data.io.gold import build_spc_v2_for_month


# ----------------------------
# Helpers internos (privados)
# ----------------------------

@dataclass(frozen=True)
class Batch:
    year: int
    month: int


def _prev_month(year: int, month: int) -> Batch:
    if month == 1:
        return Batch(year=year - 1, month=12)
    return Batch(year=year, month=month - 1)


def _last_n_months(year: int, month: int, n: int, *, include_current: bool) -> List[Batch]:
    """
    Devuelve lista de Batch hacia atrás. Orden: desde el más reciente al más antiguo.
    """
    out: List[Batch] = []
    cur = Batch(year=year, month=month)
    if not include_current:
        cur = _prev_month(cur.year, cur.month)

    for _ in range(n):
        out.append(cur)
        cur = _prev_month(cur.year, cur.month)

    return out


def _silver_month_dir(lake_root: Path, year: int, month: int) -> Path:
    return lake_root / "silver" / f"year={year}" / f"month={month:02d}"


def _read_concat_baseline(baseline_dirs: List[Path], *, columns: Optional[List[str]] = None) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for d in baseline_dirs:
        p = d / "part-0000.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p, columns=columns))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _compute_presence_classes(
    df_base: pd.DataFrame,
    *,
    time_col: str = "DateTime",
    step_col: str = "STEP_ID",
    min_points_per_month: int = 30,
    min_total_points: int = 300,
    recurring_months: int = 6,
    core_months: int = 12,
) -> pd.DataFrame:
    """
    Clasificación disjunta:
    CORE: n_months >= core_months y total_points >= min_total_points
    RECURRING: recurring_months <= n_months < core_months y total_points >= min_total_points
    INTERMITTENT: n_months >= 2
    TEMPORARY: resto
    """
    if df_base is None or df_base.empty:
        return pd.DataFrame(columns=[step_col, "n_months", "total_points", "presence_class"])

    x = df_base[[step_col, time_col]].dropna().copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col])

    x["month"] = x[time_col].dt.to_period("M")

    monthly_counts = (
        x.groupby([step_col, "month"])
        .size()
        .reset_index(name="n_points")
    )

    monthly_valid = monthly_counts[monthly_counts["n_points"] >= min_points_per_month].copy()

    summary = (
        monthly_valid.groupby(step_col)
        .agg(
            n_months=("month", "nunique"),
            total_points=("n_points", "sum"),
        )
        .reset_index()
    )

    def classify_presence(row) -> str:
        if row["n_months"] >= core_months and row["total_points"] >= min_total_points:
            return "CORE"
        if row["n_months"] >= recurring_months and row["n_months"] < core_months and row["total_points"] >= min_total_points:
            return "RECURRING"
        if row["n_months"] >= 2:
            return "INTERMITTENT"
        return "TEMPORARY"

    summary["presence_class"] = summary.apply(classify_presence, axis=1)
    return summary


# ----------------------------
# API pública estable (SILVER) - SE MANTIENE
# ----------------------------

def ingest_bronze(
    csv_path: Path,
    *,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
) -> Path:
    return ingest_csv_to_bronze(csv_path=csv_path, lake_root=lake_root, force=force)


def build_silver_from_csv(
    csv_path: Path,
    *,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
) -> Path:
    info = parse_source_stem(csv_path.stem)
    bronze_path = ingest_csv_to_bronze(csv_path=csv_path, lake_root=lake_root, force=force)
    silver_path = build_silver_from_bronze(bronze_parquet=bronze_path, lake_root=lake_root, info=info, force=force)
    return silver_path


def build_silver_for_month(
    year: int,
    month: int,
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
) -> Path:
    csv_path = latest_raw_csv_for_month(raw_root, year, month)
    return build_silver_from_csv(csv_path, lake_root=lake_root, force=force)


def build_silver_auto(raw_root: Path = RAW_ROOT, lake_root: Path = LAKE_ROOT, force: bool = False) -> Path:
    csv_path = latest_raw_csv_auto(raw_root)
    return build_silver_from_csv(csv_path, lake_root=lake_root, force=force)


def build_silver_backfill(
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    months: Optional[Iterable[Tuple[int, int]]] = None,
    last_n: Optional[int] = None,
    start: Optional[Tuple[int, int]] = None,
    end: Optional[Tuple[int, int]] = None,
    force: bool = False,
) -> List[Path]:
    modes_used = sum(x is not None for x in [months, last_n, start or end])
    if modes_used > 1:
        raise ValueError("Usa solo uno: months, last_n o start/end.")

    if start or end:
        if not (start and end):
            raise ValueError("Debes especificar start y end.")
        months_list = months_between(start, end)
    elif months is not None:
        months_list = list(months)
    elif last_n is not None:
        y_latest, m_latest = latest_available_batch(raw_root)
        target_batches = _last_n_months(y_latest, m_latest, last_n, include_current=True)
        target_set = {(b.year, b.month) for b in target_batches}
        months_list = [ym for ym in available_batches_in_raw(raw_root) if ym in target_set]
    else:
        months_list = available_batches_in_raw(raw_root)

    outputs: List[Path] = []
    for y, m in months_list:
        outputs.append(build_silver_for_month(y, m, raw_root=raw_root, lake_root=lake_root, force=force))
    return outputs


def load_silver_range_df(
    start: Tuple[int, int],
    end: Tuple[int, int],
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    ensure_built: bool = False,
    require_all: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    if ensure_built:
        months = []
        sy, sm = start
        ey, em = end
        y, m = sy, sm
        while (y, m) <= (ey, em):
            months.append((y, m))
            m += 1
            if m == 13:
                m = 1
                y += 1
        build_silver_backfill(raw_root=raw_root, lake_root=lake_root, months=months, force=force)

    return load_silver_range(lake_root, start=start, end=end, require_all=require_all)


# ----------------------------
# SPC v2 - CORE / RECURRING
# ----------------------------

def build_spc_v2_core_for_month(
    year: int,
    month: int,
    *,
    baseline_months: int = 12,
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
    # presence params
    min_points_per_month: int = 30,
    min_total_points: int = 300,
    recurring_months: int = 6,
    core_months: int = 12,
    # spc params
    min_points_limits: int = 200,
) -> Dict[str, Path]:
    # Asegura SILVER mes evaluado
    build_silver_for_month(year, month, raw_root=raw_root, lake_root=lake_root, force=force)

    # Asegura baseline SILVER (meses previos)
    prev = _last_n_months(year, month, baseline_months, include_current=False)
    for b in prev:
        build_silver_for_month(b.year, b.month, raw_root=raw_root, lake_root=lake_root, force=False)

    baseline_dirs = [_silver_month_dir(lake_root, b.year, b.month) for b in prev]

    # Presencia desde baseline (solo STEP_ID + DateTime)
    df_base = _read_concat_baseline(baseline_dirs, columns=["STEP_ID", "DateTime"])
    presence = _compute_presence_classes(
        df_base,
        min_points_per_month=min_points_per_month,
        min_total_points=min_total_points,
        recurring_months=recurring_months,
        core_months=core_months,
    )
    core_step_ids = presence.loc[presence["presence_class"] == "CORE", "STEP_ID"].tolist()

    return build_spc_v2_for_month(
        lake_root=lake_root,
        year=year,
        month=month,
        tier="core",
        baseline_silver_dirs=baseline_dirs,
        step_ids=core_step_ids,
        baseline_window=f"{baseline_months}m",
        force=force,
        min_points_limits=min_points_limits,
    )


def build_spc_v2_recurring_for_month(
    year: int,
    month: int,
    *,
    baseline_months: int = 12,
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
    # presence params
    min_points_per_month: int = 30,
    min_total_points: int = 300,
    recurring_months: int = 6,
    core_months: int = 11,
    # spc params
    min_points_limits: int = 200,
) -> Dict[str, Path]:
    build_silver_for_month(year, month, raw_root=raw_root, lake_root=lake_root, force=force)

    prev = _last_n_months(year, month, baseline_months, include_current=False)
    for b in prev:
        build_silver_for_month(b.year, b.month, raw_root=raw_root, lake_root=lake_root, force=False)

    baseline_dirs = [_silver_month_dir(lake_root, b.year, b.month) for b in prev]

    df_base = _read_concat_baseline(baseline_dirs, columns=["STEP_ID", "DateTime"])
    presence = _compute_presence_classes(
        df_base,
        min_points_per_month=min_points_per_month,
        min_total_points=min_total_points,
        recurring_months=recurring_months,
        core_months=core_months,
    )
    recurring_step_ids = presence.loc[presence["presence_class"] == "RECURRING", "STEP_ID"].tolist()

    return build_spc_v2_for_month(
        lake_root=lake_root,
        year=year,
        month=month,
        tier="recurring",
        baseline_silver_dirs=baseline_dirs,
        step_ids=recurring_step_ids,
        baseline_window=f"{baseline_months}m",
        force=force,
        min_points_limits=min_points_limits,
    )


def build_capability_v1_core_for_month(
    year: int,
    month: int,
    *,
    baseline_months: int = 12,
    raw_root: Path = RAW_ROOT,
    lake_root: Path = LAKE_ROOT,
    force: bool = False,
    # presence params (igual que SPC)
    min_points_per_month: int = 30,
    min_total_points: int = 300,
    recurring_months: int = 6,
    core_months: int = 11,
    # capability params
    min_points_cap: int = 200,
) -> Dict[str, Path]:
    # asegurar SILVER mes evaluado (no lo usamos en baseline, pero mantiene consistencia)
    build_silver_for_month(year, month, raw_root=raw_root, lake_root=lake_root, force=force)

    # baseline meses previos
    prev = _last_n_months(year, month, baseline_months, include_current=False)
    for b in prev:
        build_silver_for_month(b.year, b.month, raw_root=raw_root, lake_root=lake_root, force=False)

    baseline_dirs = [_silver_month_dir(lake_root, b.year, b.month) for b in prev]

    # presence desde baseline
    df_base_presence = _read_concat_baseline(baseline_dirs, columns=["STEP_ID", "DateTime"])
    presence = _compute_presence_classes(
        df_base_presence,
        min_points_per_month=min_points_per_month,
        min_total_points=min_total_points,
        recurring_months=recurring_months,
        core_months=core_months,
    )
    core_step_ids = presence.loc[presence["presence_class"] == "CORE", "STEP_ID"].tolist()

    return build_capability_v1_for_month(
        lake_root=lake_root,
        year=year,
        month=month,
        tier="core",
        baseline_silver_dirs=baseline_dirs,
        step_ids=core_step_ids,
        baseline_window=f"{baseline_months}m",
        force=force,
        min_points=min_points_cap,
    )
