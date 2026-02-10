import pandas as pd
import pytest
from tightening_project.data.schema_silver import finalize_silver_df

def test_missing_required_column_fails():
    df = pd.DataFrame({
        "STEP_ID": [1],
        "FinalTorque": [10.0],
    })

    with pytest.raises(ValueError):
        finalize_silver_df(df)
