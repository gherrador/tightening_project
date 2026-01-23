from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tightening_project.data.schema import (
    TORQUE_COLS,
    ANGLE_COLS,
    is_torque_col,
    is_angle_col,
    fix_row_with_scale,
    fix_target_with_limits,
    fix_final_value_with_range,
)


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    datetime_parsed: bool
    fixed_torque_rows: int
    fixed_angle_rows: int
    dropped_rows_missing_essentials: int


def _fix_torque_block(df: pd.DataFrame, decimals: int = 2) -> tuple[pd.DataFrame, int]:
    """
    Repara TorqueTarget/Min/Max/Final en coherencia por fila.
    Devuelve df y cantidad de filas donde aplicó escala !=1 (aprox proxy de corrección).
    """
    d = df.copy()

    needed = ["TorqueTarget", "TorqueMinTolerance", "TorqueMaxTolerance", "RES_FinalTorque"]
    if not all(c in d.columns for c in needed):
        return d, 0

    fixed = [
        fix_row_with_scale(t_raw, mn_raw, mx_raw, f_raw, scales=(1, 10, 100, 1000), decimals=decimals)
        for t_raw, mn_raw, mx_raw, f_raw in zip(
            d["TorqueTarget"], d["TorqueMinTolerance"], d["TorqueMaxTolerance"], d["RES_FinalTorque"]
        )
    ]

    t = pd.Series([x[0] for x in fixed], index=d.index)
    mn = pd.Series([x[1] for x in fixed], index=d.index)
    mx = pd.Series([x[2] for x in fixed], index=d.index)
    final = pd.Series([x[3] for x in fixed], index=d.index)
    scale = pd.Series([x[4] for x in fixed], index=d.index).astype(int)

    # Ajuste target con rango (4->40, etc.)
    t2 = [fix_target_with_limits(tt, mnn, mxx) for tt, mnn, mxx in zip(t, mn, mx)]
    t = pd.Series(t2, index=d.index)

    # Recalcular final si target se ajustó
    final2 = [
        fix_final_value_with_range(raw_f, tt, mnn, mxx, decimals=decimals)
        for raw_f, tt, mnn, mxx in zip(d["RES_FinalTorque"], t, mn, mx)
    ]
    final = pd.Series(final2, index=d.index)

    d["TorqueTarget"] = t.astype(float).round(decimals)
    d["TorqueMinTolerance"] = mn.astype(float).round(decimals)
    d["TorqueMaxTolerance"] = mx.astype(float).round(decimals)
    d["RES_FinalTorque"] = final.astype(float).round(decimals)

    # opcional: guardar escala aplicada para auditoría
    d["TorqueScaleApplied"] = scale

    fixed_rows = int((scale != 1).sum())
    return d, fixed_rows


def _fix_angle_block(df: pd.DataFrame, decimals: int = 2) -> tuple[pd.DataFrame, int]:
    """
    (Opcional) Mismo enfoque row-wise para ángulos.
    Si tus ángulos también vienen corruptos, aplicamos el mismo patrón.
    """
    d = df.copy()
    needed = ["STEP_AngleTarget", "STEP_AngleMinTolerance", "STEP_AngleMaxTolerance", "RES_FinalAngle"]
    if not all(c in d.columns for c in needed):
        return d, 0

    fixed = [
        fix_row_with_scale(t_raw, mn_raw, mx_raw, f_raw, scales=(1, 10, 100, 1000), decimals=decimals)
        for t_raw, mn_raw, mx_raw, f_raw in zip(
            d["STEP_AngleTarget"], d["STEP_AngleMinTolerance"], d["STEP_AngleMaxTolerance"], d["RES_FinalAngle"]
        )
    ]

    t = pd.Series([x[0] for x in fixed], index=d.index)
    mn = pd.Series([x[1] for x in fixed], index=d.index)
    mx = pd.Series([x[2] for x in fixed], index=d.index)
    final = pd.Series([x[3] for x in fixed], index=d.index)
    scale = pd.Series([x[4] for x in fixed], index=d.index).astype(int)

    t = pd.Series([fix_target_with_limits(tt, mnn, mxx) for tt, mnn, mxx in zip(t, mn, mx)], index=d.index)
    final = pd.Series(
        [fix_final_value_with_range(raw_f, tt, mnn, mxx, decimals=decimals) for raw_f, tt, mnn, mxx in zip(d["RES_FinalAngle"], t, mn, mx)],
        index=d.index
    )

    d["STEP_AngleTarget"] = t.astype(float).round(decimals)
    d["STEP_AngleMinTolerance"] = mn.astype(float).round(decimals)
    d["STEP_AngleMaxTolerance"] = mx.astype(float).round(decimals)
    d["RES_FinalAngle"] = final.astype(float).round(decimals)

    d["AngleScaleApplied"] = scale

    fixed_rows = int((scale != 1).sum())
    return d, fixed_rows


def clean_tightening_df(
    df: pd.DataFrame,
    datetime_col: str = "RES_DateTime",
    decimals: int = 2,
    essential_cols: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    d0 = df.copy()
    rows_in = len(d0)

    # Torque row-wise
    d1, fixed_torque_rows = _fix_torque_block(d0, decimals=decimals)

    # Angle row-wise (si aplica)
    d2, fixed_angle_rows = _fix_angle_block(d1, decimals=decimals)

    # Datetime
    dt_ok = False
    if datetime_col in d2.columns:
        d2[datetime_col] = pd.to_datetime(d2[datetime_col], errors="coerce", utc=True)
        dt_ok = True

    # Esenciales
    if essential_cols is None:
        essential_cols = [c for c in ["RES_FinalTorque", "RES_FinalAngle"] if c in d2.columns]

    before = len(d2)
    if essential_cols:
        d2 = d2.dropna(subset=list(essential_cols))
    dropped = before - len(d2)

    rep = CleaningReport(
        rows_in=rows_in,
        rows_out=len(d2),
        datetime_parsed=dt_ok,
        fixed_torque_rows=fixed_torque_rows,
        fixed_angle_rows=fixed_angle_rows,
        dropped_rows_missing_essentials=dropped,
    )
    return d2, rep
