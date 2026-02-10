from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

from .naming import FileInfo, parse_source_stem


def bronze_output_dir(lake_root: Path, info: FileInfo) -> Path:
    """
    BRONZE layout:
      lake_root/bronze/year=YYYY/month=MM/source_file=<source_stem>/
    """
    return (
        lake_root
        / "bronze"
        / f"year={info.year}"
        / f"month={info.month:02d}"
        / f"source_file={info.source_stem}"
    )


def bronze_parquet_path(lake_root: Path, info: FileInfo) -> Path:
    return bronze_output_dir(lake_root, info) / "part-0000.parquet"


def bronze_meta_path(lake_root: Path, info: FileInfo) -> Path:
    return bronze_output_dir(lake_root, info) / "meta.json"


def _write_meta(meta_file: Path, meta: dict) -> None:
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_csv_to_bronze(csv_path: Path, lake_root: Path, force: bool = False) -> Path:
    """
    CSV -> BRONZE Parquet (sin cleaning).
    También genera meta.json con info útil para auditoría / discovery.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    info = parse_source_stem(csv_path.stem)

    out_dir = bronze_output_dir(lake_root, info)
    out_parquet = bronze_parquet_path(lake_root, info)
    out_meta = bronze_meta_path(lake_root, info)

    # Idempotencia: si existe el parquet y no forzamos, no rehacemos
    if out_parquet.exists() and not force:
        return out_parquet

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Leer CSV sin transformaciones
    df = pd.read_csv(csv_path, sep=";")  

    # 2) Escribir Parquet
    df.to_parquet(out_parquet, index=False)

    # 3) Escribir metadata (auditoría)
    meta = {
        "layer": "bronze",
        "dataset_year": info.year,
        "dataset_month": info.month,
        "source_stem": info.source_stem,
        "source_csv_name": csv_path.name,
        "source_csv_path": str(csv_path.resolve()),
        "export_ts": info.export_ts,  # puede ser None
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    _write_meta(out_meta, meta)

    return out_parquet


def load_bronze(bronze_parquet: Path) -> pd.DataFrame:
    """
    Utilidad simple para leer BRONZE parquet.
    """
    if not bronze_parquet.exists():
        raise FileNotFoundError(f"No existe BRONZE parquet: {bronze_parquet}")
    return pd.read_parquet(bronze_parquet)
