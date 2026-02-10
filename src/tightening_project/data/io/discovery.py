from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import re

from .naming import parse_source_stem, FileInfo

_EXPORT_TS_RE = re.compile(r"_(\d{12})_")


def _export_ts_from_stem(stem: str) -> Optional[str]:
    m = _EXPORT_TS_RE.search(stem)
    return m.group(1) if m else None


def iter_raw_csvs(raw_root: Path) -> List[Path]:
    """
    Devuelve todos los CSV en raw_root (solo nivel 1).
    """
    return sorted(raw_root.glob("*.csv"))


def available_batches_in_raw(raw_root: Path) -> List[Tuple[int, int]]:    
    """
    Devuelve lista ordenada de (year, month) detectados en raw por MonthNameYY.
    """
    batches = set()
    for p in iter_raw_csvs(raw_root):
        try:
            info = parse_source_stem(p.stem)
        except ValueError:
            continue
        batches.add((info.year, info.month))
    return sorted(batches)


def latest_available_batch(raw_root: Path) -> Tuple[int, int]:
    """
    Devuelve el (year, month) más reciente encontrado en raw.
    """
    batches = available_batches_in_raw(raw_root)
    if not batches:
        raise FileNotFoundError(f"No se detectaron CSVs válidos en {raw_root}")
    return batches[-1]


def raw_csvs_for_month(raw_root: Path, year: int, month: int) -> List[Path]:
    """
    Devuelve los CSVs de raw que pertenecen al dataset (year, month) según MonthNameYY.
    """
    out: List[Path] = []
    for p in iter_raw_csvs(raw_root):
        try:
            info = parse_source_stem(p.stem)
        except ValueError:
            continue
        if info.year == year and info.month == month:
            out.append(p)
    return sorted(out)


def latest_raw_csv_for_month(raw_root: Path, year: int, month: int) -> Path:
    """
    Selecciona el 'último' CSV para un (year, month).

    Regla:
    1) Si hay export_ts (12 dígitos) en el nombre, elige el mayor.
    2) Si no hay export_ts en ninguno, usa fecha de modificación (mtime).
    """
    candidates = raw_csvs_for_month(raw_root, year, month)
    if not candidates:
        raise FileNotFoundError(f"No hay CSVs para {year}-{month:02d} en {raw_root}")

    with_ts = []
    without_ts = []

    for p in candidates:
        ts = _export_ts_from_stem(p.stem)
        if ts:
            with_ts.append((ts, p))
        else:
            without_ts.append(p)

    if with_ts:
        # Orden lexicográfico funciona para YYMMDDHHMMSS
        with_ts.sort(key=lambda t: t[0])
        return with_ts[-1][1]

    # Fallback: mtime
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def latest_raw_csv_auto(raw_root: Path) -> Path:
    """
    AUTO:
    - detecta el mes más reciente disponible en raw
    - devuelve el último CSV de ese mes
    """
    year, month = latest_available_batch(raw_root)
    return latest_raw_csv_for_month(raw_root, year, month)


def selected_file_info_for_month(raw_root: Path, year: int, month: int) -> FileInfo:
    """
    Devuelve FileInfo del CSV elegido como 'latest' para ese mes.
    """
    p = latest_raw_csv_for_month(raw_root, year, month)
    return parse_source_stem(p.stem)


def months_between(
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    start_year, start_month = start
    end_year, end_month = end

    if (start_year, start_month) > (end_year, end_month):
        raise ValueError("start debe ser <= end")

    months = []
    y, m = start_year, start_month

    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

    return months
