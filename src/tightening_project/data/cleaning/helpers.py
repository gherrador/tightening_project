from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]    
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")


def _resolve_essential_cols(cols, essential_cols: Optional[Sequence[str]]) -> Sequence[str]:  
    if essential_cols is None:
        return [cols.final, cols.target, cols.tol_min, cols.tol_max]

    if not isinstance(essential_cols, (list, tuple)):
        raise TypeError("essential_cols must be a list/tuple of column names (str).")

    if not essential_cols:
        raise ValueError("essential_cols cannot be empty.")

    non_str = [c for c in essential_cols if not isinstance(c, str)]
    if non_str:
        raise TypeError(f"essential_cols must contain only strings. Invalid: {non_str}")

    must_include = {cols.final, cols.tol_min, cols.tol_max}
    missing_must = [c for c in must_include if c not in essential_cols]
    if missing_must:
        raise ValueError(
            f"essential_cols must include {sorted(must_include)} because the pipeline uses them. "
            f"Missing: {missing_must}"
        )

    return list(essential_cols)


def _validate_essential_cols_exist(df: pd.DataFrame, essential_cols: Sequence[str]) -> None:
    missing = [c for c in essential_cols if c not in df.columns]
    if missing:
        raise ValueError(f"essential_cols columns are missing in df: {missing}")


def _swap_min_gt_max(df: pd.DataFrame, tol_min: str, tol_max: str) -> Tuple[pd.DataFrame, int]:   
    swap_mask = df[tol_min] > df[tol_max]
    swapped = int(swap_mask.sum())
    if swapped:
        mn = df.loc[swap_mask, tol_min].copy()
        df.loc[swap_mask, tol_min] = df.loc[swap_mask, tol_max].values
        df.loc[swap_mask, tol_max] = mn.values
    return df,swapped

def _drop_zero_block_rows(df: pd.DataFrame, target: str, tol_min: str) -> Tuple[pd.DataFrame, int]:
    zero_mask = (df[target] == 0) & (df[tol_min] == 0)
    dropped = int(zero_mask.sum())
    df = df.loc[~zero_mask].copy()
    return df, dropped

def _parse_datetime_column(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    if dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    return df

def _add_ok_nok_column(
    df: pd.DataFrame,
    final: str,
    tol_min: str,
    tol_max: str,
    out_col: str = "Torque_Result",
) -> Tuple[pd.DataFrame, int, int]:
    ok_mask = (df[final] >= df[tol_min]) & (df[final] <= df[tol_max])
    df[out_col] = np.where(ok_mask, "OK", "NOK")
    ok_count = int((df[out_col] == "OK").sum())
    nok_count = int((df[out_col] == "NOK").sum())
    return df, ok_count, nok_count


def _add_quality_status(
    df: pd.DataFrame,
    essential_cols: Sequence[str],
    tol_min: str,
    tol_max: str,
    out_col: str = "Quality_Status",
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    issues: Dict[str, int] = {}

    missing_ess = df[list(essential_cols)].isna().any(axis=1)
    zero_tol = df[tol_min] == df[tol_max]

    issues["missing_essentials"] = int(missing_ess.sum())
    issues["zero_tolerance_range"] = int(zero_tol.sum())

    df[out_col] = np.where(missing_ess | zero_tol, "REVIEW_NEEDED", "OK")
    review_needed = int((df[out_col] == "REVIEW_NEEDED").sum())

    return df, issues, review_needed

def _drop_missing_essentials(df: pd.DataFrame, essential_cols: Sequence[str]) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=list(essential_cols))
    dropped = int(before - len(df))
    return df, dropped

def _finalize_dataframe(
    df: pd.DataFrame,
    columns_to_drop: Sequence[str],
    rename_map: Dict[str, str],
    final_order: Sequence[str],
    round_decimals: int = 2,
) -> pd.DataFrame:
    # Drop + rename
    df = df.drop(columns=list(columns_to_drop), errors="ignore")
    df = df.rename(columns=rename_map)

    # Reorder
    existing_order = [c for c in final_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_order]
    df = df[existing_order + remaining]

    # Round
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].round(round_decimals)

    return df
