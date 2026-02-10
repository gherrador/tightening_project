from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict
import pandas as pd

from tightening_project.data.cleaning import helpers
from tightening_project.data.schema_raw import (TorqueColumns, CleaningReport)
from tightening_project.data.cleaning.constant import (COLUMNS_TO_DROP, FINAL_COLUMNS_ORDER, RENAME_MAP)
from tightening_project.data.schema_silver import finalize_silver_df

csv = Path("data/raw/TighteningProduct_November25_260115114110_1.csv")
lake = Path("data/lake")

def clean_tightening_df(
    df_raw: pd.DataFrame,
    cols: TorqueColumns = TorqueColumns(),
    essential_cols: Optional[Sequence[str]] = None,
    drop_zero_block: bool = True,
    add_ok_nok: bool = True,
) -> Tuple[pd.DataFrame, CleaningReport]:   

    df = df_raw.copy()
    rows_in = len(df)
    issues: Dict[str, int] = {}

    required_columns = [cols.step_id, cols.target, cols.tol_min, cols.tol_max, cols.final]
    helpers._validate_required_columns(df, required_columns)  

    essential_cols = helpers._resolve_essential_cols(cols, essential_cols)
    helpers._validate_essential_cols_exist(df, essential_cols) 

    # 1) Swap min > max 
    df,swapped = helpers._swap_min_gt_max(df, cols.tol_min, cols.tol_max)  
    issues["min_gt_max_swapped"] = swapped  

    # 2) Drop zero-block
    dropped_zero = 0
    if drop_zero_block:
        df, dropped_zero = helpers._drop_zero_block_rows(df, cols.target, cols.tol_min)
    issues["zero_block_dropped"] = dropped_zero

    # 3) Parse datetime
    df = helpers._parse_datetime_column(df, cols.dt)

    # 4) OK/NOK por tolerancias
    ok_count = nok_count = 0
    if add_ok_nok:
        df, ok_count, nok_count = helpers._add_ok_nok_column(df, cols.final, cols.tol_min, cols.tol_max)

    # 5) Quality_Status
    df, qs_issues, review_needed = helpers._add_quality_status(df, essential_cols, cols.tol_min, cols.tol_max)
    issues.update(qs_issues)

    # 6) Drop filas con esenciales missing    
    df, dropped_missing = helpers._drop_missing_essentials(df, essential_cols)
    
    # 7-9) Finalize
    df = helpers._finalize_dataframe(
        df,
        columns_to_drop=COLUMNS_TO_DROP,
        rename_map=RENAME_MAP,
        final_order=FINAL_COLUMNS_ORDER,
        round_decimals=2,
    )

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(df),
        swapped_min_max=swapped,
        dropped_zero_block=dropped_zero,
        dropped_missing_essentials=dropped_missing,
        ok_count=ok_count,
        nok_count=nok_count,
        review_needed=review_needed,
        issue_counts=issues,
    )
    df = finalize_silver_df(df)

    return df, report
   
