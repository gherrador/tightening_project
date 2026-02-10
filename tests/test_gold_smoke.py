import pandas as pd
from pathlib import Path
from tightening_project.data.io.gold import build_gold_for_month

def test_gold_smoke(tmp_path):
    lake = tmp_path / "lake"
    silver = lake / "silver" / "year=2025" / "month=01"
    silver.mkdir(parents=True)

    df = pd.DataFrame({
        "STEP_ID": [1, 1, 1],
        "FinalTorque": [10, 11, 9],
        "TorqueMinTolerance": [8, 8, 8],
        "TorqueMaxTolerance": [12, 12, 12],
        "Quality_Status": ["OK", "OK", "OK"],
    })
    df.to_parquet(silver / "part-0000.parquet")

    outs = build_gold_for_month(
        lake_root=lake,
        year=2025,
        month=1,
        baseline_silver_dirs=[silver],
        key_cols=["STEP_ID"],
    )

    assert outs["spc_limits"].exists()
    assert outs["capability"].exists()
