from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
LAKE_ROOT = DATA_ROOT / "lake"