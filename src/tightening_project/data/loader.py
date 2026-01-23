from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv

import pandas as pd

from tightening_project.cache import CacheArtifact, DataFingerprint


@dataclass
class LoadResult:
    df: pd.DataFrame
    parquet_path: Optional[Path]
    used_cache: bool


class TighteningLoader:
    def __init__(self, cache: CacheArtifact, app_namespace: str = "tightening"):
        self.cache = cache
        self.app_namespace = app_namespace

    def _detect_delimiter(self, csv_path: Path) -> str:
        # Lee un poquito del archivo y trata de inferir el delimitador
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except Exception:
            # fallback razonable para tu caso
            return ";"

    def load_csv_cached(self, csv_path: Path, force: bool = False) -> LoadResult:
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"No existe: {csv_path}")

        fp = DataFingerprint.from_file(csv_path).to_key()
        parquet_name = f"{self.app_namespace}_clean_{fp}"
        parquet_path = self.cache.path_for(parquet_name, ".parquet")

        if parquet_path.is_file() and not force:
            df = pd.read_parquet(parquet_path)
            return LoadResult(df=df, parquet_path=parquet_path, used_cache=True)

        sep = self._detect_delimiter(csv_path)
        df = pd.read_csv(csv_path, sep=sep)

        return LoadResult(df=df, parquet_path=parquet_path, used_cache=False)
