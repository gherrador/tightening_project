from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


_NULL_LITERALS = {"", "nan", "none", "null", "nat"}

# Whitelist de columnas físicas (NO tocar IDs)
TORQUE_COLS = {
    "TorqueTarget",
    "TorqueMinTolerance",
    "TorqueMaxTolerance",
    "RES_FinalTorque",
}
ANGLE_COLS = {
    "STEP_AngleTarget",
    "STEP_AngleMinTolerance",
    "STEP_AngleMaxTolerance",
    "RES_FinalAngle",
}

def is_torque_col(c: str) -> bool:
    return c in TORQUE_COLS

def is_angle_col(c: str) -> bool:
    return c in ANGLE_COLS


# --------- parsing helpers ---------

_num_simple = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")

def to_float_normal(x: Any) -> float:
    """Parsea números normales tipo '12.34' o '12,34'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in _NULL_LITERALS:
        return np.nan
    if _num_simple.match(s):
        return float(s.replace(",", "."))
    return np.nan

def digits_only(x: Any) -> str:
    """Extrae solo dígitos (para reconstrucción tipo '12.999.999...' -> '12999999...')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return re.sub(r"[^0-9]", "", str(x))

def int_digits_from_value(v: float, fallback: int = 2) -> int:
    """Número de dígitos enteros de un valor ya parseado."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    try:
        return max(1, len(str(int(abs(float(v))))))
    except Exception:
        return fallback


# --------- reconstrucción "weird" ---------

def fix_weird_value_using_int_digits(x: Any, int_digits: int, decimals: int = 2) -> float:
    """
    Reconstruye un valor cuando viene con puntos raros.
    Ej:
      x="14.101.351.737.976", int_digits=2, decimals=2 -> 14.10
      x="500.846...", int_digits=1 -> 5.00
    """
    s = str(x).strip()
    if not s or s.lower() in _NULL_LITERALS:
        return np.nan

    v = to_float_normal(s)
    if not np.isnan(v):
        return float(np.round(v, decimals))

    d = digits_only(s)
    if not d:
        return np.nan

    left = d[:int_digits] if len(d) >= int_digits else d
    right = d[int_digits:int_digits + decimals] if len(d) > int_digits else ""
    right = right.ljust(decimals, "0")

    if not left:
        return np.nan

    return float(f"{int(left)}.{right}")


def fix_target_with_limits(target: float, mn: float, mx: float) -> float:
    """Arregla target tipo 4->40 usando el rango [mn, mx]."""
    if np.isnan(target) or np.isnan(mn) or np.isnan(mx):
        return target

    lo, hi = (mn, mx) if mn <= mx else (mx, mn)
    if lo <= target <= hi:
        return target

    for k in (1, 2, 3):
        cand = target * (10 ** k)
        if lo <= cand <= hi:
            return cand

    for k in (1, 2, 3):
        cand = target / (10 ** k)
        if lo <= cand <= hi:
            return cand

    return target


def fix_final_value_with_range(raw_final: Any, target: float, mn: float, mx: float, decimals: int = 2) -> float:
    """
    Repara final (torque/angle) usando int_digits del target y validando con rango.
    Si no cae en rango, prueba alternativas de int_digits (1..4) y elige la más cercana al target.
    """
    int_digits = int_digits_from_value(target, fallback=2)
    v = fix_weird_value_using_int_digits(raw_final, int_digits=int_digits, decimals=decimals)

    if np.isnan(v) or np.isnan(mn) or np.isnan(mx):
        return v

    lo, hi = (mn, mx) if mn <= mx else (mx, mn)
    if lo - 1e-6 <= v <= hi + 1e-6:
        return v

    candidates = []
    for idg in (1, 2, 3, 4):
        cand = fix_weird_value_using_int_digits(raw_final, int_digits=idg, decimals=decimals)
        if not np.isnan(cand):
            candidates.append(cand)

    in_range = [c for c in candidates if lo - 1e-6 <= c <= hi + 1e-6]
    if in_range:
        return min(in_range, key=lambda c: abs(c - target))

    return v


def fix_row_with_scale(
    t_raw: Any,
    mn_raw: Any,
    mx_raw: Any,
    f_raw: Any,
    scales=(1, 10, 100, 1000),
    decimals: int = 2,
) -> tuple[float, float, float, float, int]:
    """
    Decide la mejor escala s para que target y final caigan dentro de [min,max].
    Devuelve (t, mn, mx, f, s). Si no encuentra, devuelve lo parseable sin escala.
    """
    t0 = to_float_normal(t_raw)
    mn0 = to_float_normal(mn_raw)
    mx0 = to_float_normal(mx_raw)
    f0 = to_float_normal(f_raw)

    best = None  # (score, ts, mns, mxs, fs, s)

    for s in scales:
        ts = (t0 * s) if not np.isnan(t0) else (fix_weird_value_using_int_digits(t_raw, 2, decimals) * s)
        int_digits = int_digits_from_value(ts, fallback=2)

        mns = (mn0 * s) if not np.isnan(mn0) else fix_weird_value_using_int_digits(mn_raw, int_digits, decimals)
        mxs = (mx0 * s) if not np.isnan(mx0) else fix_weird_value_using_int_digits(mx_raw, int_digits, decimals)
        fs = (f0 * s) if not np.isnan(f0) else fix_weird_value_using_int_digits(f_raw, int_digits, decimals)

        if any(np.isnan(v) for v in (ts, mns, mxs, fs)):
            continue

        lo, hi = (mns, mxs) if mns <= mxs else (mxs, mns)
        if not (lo - 1e-6 <= ts <= hi + 1e-6):
            continue
        if not (lo - 1e-6 <= fs <= hi + 1e-6):
            continue

        score = min(abs(ts - round(ts)), abs(ts - round(ts, 1)) + 0.01) + 0.001 * (s - 1)

        cand = (score, ts, mns, mxs, fs, s)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return (t0, mn0, mx0, f0, 1)

    _, ts, mns, mxs, fs, s = best
    return (float(np.round(ts, decimals)),
            float(np.round(mns, decimals)),
            float(np.round(mxs, decimals)),
            float(np.round(fs, decimals)),
            int(s))
