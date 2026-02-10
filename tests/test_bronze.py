from pathlib import Path
import pandas as pd
from tightening_project.data.io.bronze import ingest_csv_to_bronze

def test_bronze_idempotent(tmp_path):
    csv = tmp_path / "TighteningProduct_January25_250101120000_1.csv"
    df = pd.DataFrame({"A": [1, 2]})
    df.to_csv(csv, index=False)

    lake = tmp_path / "lake"

    p1 = ingest_csv_to_bronze(csv, lake)
    p2 = ingest_csv_to_bronze(csv, lake)

    assert p1 == p2
    assert p1.exists()
