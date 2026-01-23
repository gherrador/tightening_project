from pathlib import Path

# Raíz del repo: .../tightening_project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PREFIX = "TighteningProduct_"
CSV_EXTENSION = ".csv"


def find_latest_csv(data_dir: Path = DATA_DIR) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")

    candidates = sorted(
        (
            p for p in data_dir.iterdir()
            if p.is_file()
            and p.name.startswith(CSV_PREFIX)
            and p.suffix.lower() == CSV_EXTENSION
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No se encontraron CSV con prefijo '{CSV_PREFIX}' en {data_dir}"
        )

    return candidates[0]
