import pandas as pd
from tightening_project.data.cleaning.clean_torque import clean_tightening_df

def test_swap_tolerances():
    df = pd.DataFrame({
        "STEP_ID": [1],
        "STEP_TorqueTarget": [10.0],
        "STEP_TorqueMinTolerance": [12.0],
        "STEP_TorqueMaxTolerance": [8.0],
        "RES_FinalTorque": [10.0],
    })

    clean, _ = clean_tightening_df(df)

    assert clean["TorqueMinTolerance"].iloc[0] == 8.0
    assert clean["TorqueMaxTolerance"].iloc[0] == 12.0
