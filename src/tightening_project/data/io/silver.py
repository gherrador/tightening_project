from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

from .naming import FileInfo
from tightening_project.data.cleaning.clean_torque import clean_tightening_df

from tightening_project.paths import LAKE_ROOT

from typing import List, Tuple
import pandas as pd


SILVER_ROOT = Path(LAKE_ROOT) / "silver"


def silver_output_dir(lake_root: Path, info: FileInfo) -> Path:    
    return lake_root / "silver" / f"year={info.year}" / f"month={info.month:02d}"

def silver_parquet_path(lake_root: Path, info: FileInfo) -> Path:
    return silver_output_dir(lake_root, info) / "part-0000.parquet"

def silver_report_path(lake_root: Path, info: FileInfo) -> Path:
    return (lake_root/ "reports"/ "silver_cleaning"/ f"year={info.year}"/ f"month={info.month:02d}"/ "cleaning_report.json")

def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def build_silver_from_bronze(bronze_parquet: Path,lake_root: Path, info: FileInfo,force: bool = False) -> Path:
    """
    BRONZE -> SILVER:
    - lee parquet bronze
    - aplica cleaning definitivo (clean_tightening_df)
    - escribe parquet silver por year/month
    - escribe cleaning_report.json
    """
    if not bronze_parquet.exists():
        raise FileNotFoundError(f"No existe BRONZE parquet: {bronze_parquet}")

    out_parquet = silver_parquet_path(lake_root, info)
    out_report = silver_report_path(lake_root, info)

    # Idempotente
    if out_parquet.exists() and out_report.exists() and not force:
        return out_parquet

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_parquet(bronze_parquet)

    df_clean, report = clean_tightening_df(df_raw)

    df_clean.to_parquet(out_parquet, index=False)

    # Serialización segura del report
    if hasattr(report, "to_dict"):
        report_dict = report.to_dict()
    elif hasattr(report, "__dict__"):
        report_dict = dict(report.__dict__)
    else:
        report_dict = {"report": str(report)}

    # Añadimos metadata útil del build (muy recomendado)
    report_payload = {
        "layer": "silver",
        "dataset_year": info.year,
        "dataset_month": info.month,
        "source_stem": info.source_stem,
        "export_ts": info.export_ts,
        "bronze_parquet": str(bronze_parquet.resolve()),
        "silver_parquet": str(out_parquet.resolve()),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "cleaning_report": report_dict,
        "rows_out": int(df_clean.shape[0]),
        "cols_out": int(df_clean.shape[1]),
    }

    _write_json(out_report, report_payload)

    return out_parquet


def load_silver(silver_parquet: Path) -> pd.DataFrame:
    if not silver_parquet.exists():
        raise FileNotFoundError(f"No existe SILVER parquet: {silver_parquet}")
    return pd.read_parquet(silver_parquet)


def silver_month_dir(lake_root: Path, year: int, month: int) -> Path:
    return lake_root / "silver" / f"year={year}" / f"month={month:02d}"


def silver_month_parquet(lake_root: Path, year: int, month: int) -> Path:
    return silver_month_dir(lake_root, year, month) / "part-0000.parquet"


def load_silver_month(lake_root: Path, year: int, month: int) -> pd.DataFrame:
    p = silver_month_parquet(lake_root, year, month)
    if not p.exists():
        raise FileNotFoundError(f"No existe SILVER parquet esperado: {p}")
    return pd.read_parquet(p)


def _iter_months_inclusive(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    sy, sm = start
    ey, em = end
    if (sy, sm) > (ey, em):
        raise ValueError(f"Rango inválido: start={start} es mayor que end={end}")

    out: List[Tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def load_silver_range(
    lake_root: Path,    
    start: Tuple[int, int],
    end: Tuple[int, int],
    require_all: bool = False,
) -> pd.DataFrame:
    """
    Une SILVER entre start y end (inclusive).
    - start/end: (year, month)
    - require_all=False: si falta un mes, se salta
    - require_all=True: si falta un mes, error
    """
    months = _iter_months_inclusive(start, end)

    dfs = []
    missing = []
    for y, m in months:
        p = silver_month_parquet(lake_root, y, m)
        if not p.exists():
            missing.append((y, m, str(p)))
            if require_all:
                raise FileNotFoundError(f"Falta SILVER para {y}-{m:02d}. Esperado: {p}")
            continue
        dfs.append(pd.read_parquet(p))

    if not dfs:
        if missing:
            raise FileNotFoundError(f"No se encontró ningún SILVER en rango {start}..{end}. Ejemplo faltante: {missing[0][2]}")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
