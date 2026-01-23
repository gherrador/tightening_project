from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


class CacheError(Exception):
    pass


CACHE_BASE_DIR = Path.home() / ".my_cache"


@dataclass(frozen=True)
class DataFingerprint:
    """Huella del archivo para invalidar caché si el CSV cambia."""
    path: str
    size: int
    mtime_ns: int

    @staticmethod
    def from_file(file_path: Path) -> "DataFingerprint":
        st = file_path.stat()
        return DataFingerprint(
            path=str(file_path.resolve()),
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
        )

    def to_key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.md5(payload).hexdigest()


class Cache:
    def __init__(self, app_name: str, obsolescence_days: int = 7, cache_dir: Optional[Path] = None):
        if not app_name:
            raise CacheError("app_name no puede estar vacío")

        self._app_name = app_name
        self._obsolescence_days = obsolescence_days
        base_dir = cache_dir if cache_dir is not None else CACHE_BASE_DIR
        self._cache_dir = base_dir / app_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _get_filepath(self, name: str) -> Path:
        return self.cache_dir / name

    def exists(self, name: str) -> bool:
        return self._get_filepath(name).is_file()

    def set_text(self, name: str, data: str) -> None:
        fp = self._get_filepath(name)
        fp.write_text(data, encoding="utf-8")

    def load_text(self, name: str) -> str:
        fp = self._get_filepath(name)
        if not fp.is_file():
            raise CacheError(f"No existe en caché: {name}")
        return fp.read_text(encoding="utf-8")

    def how_old_ms(self, name: str) -> float:
        fp = self._get_filepath(name)
        if not fp.is_file():
            raise CacheError(f"No existe en caché: {name}")
        mtime_ms = fp.stat().st_mtime * 1000
        return (time.time() * 1000) - mtime_ms

    def is_obsolete(self, name: str) -> bool:
        if not self.exists(name):
            return True
        return self.how_old_ms(name) > self._obsolescence_days * 24 * 60 * 60 * 1000

    def delete(self, name: str) -> None:
        fp = self._get_filepath(name)
        if fp.is_file():
            fp.unlink()

    def clear(self) -> None:
        for item in self.cache_dir.iterdir():
            if item.is_file():
                item.unlink()


class CacheURL(Cache):
    """Cachea contenido de URLs por hash MD5 de la URL."""

    def _hash_url(self, url: str) -> str:
        return hashlib.md5(url.encode("utf-8")).hexdigest()

    def _name_for_url(self, url: str) -> str:
        return self._hash_url(url) + ".txt"

    def get(self, url: str, timeout: int = 15, force: bool = False, headers: Optional[dict] = None) -> str:
        """
        Devuelve el contenido de una URL usando caché.
        - Si existe y no está obsoleto -> devuelve caché
        - Si no -> descarga, guarda, devuelve
        """
        name = self._name_for_url(url)

        if not force and self.exists(name) and not self.is_obsolete(name):
            return self.load_text(name)

        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        text = resp.text
        self.set_text(name, text)
        return text


class CacheArtifact(Cache):
    """
    Caché para artefactos (parquet, modelos, features, métricas).
    Maneja paths directamente.
    """

    def path_for(self, name: str, suffix: str) -> Path:
        safe = name.replace("/", "_")
        return self.cache_dir / f"{safe}{suffix}"

    def exists_path(self, path: Path) -> bool:
        return path.is_file()
