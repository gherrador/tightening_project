from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Toma DF limpio y devuelve DF con features derivadas.
    No lee ni escribe archivos.
    """
    d = df.copy()

    # ---------- Labels de tolerancia ----------
    if {"RES_FinalTorque", "TorqueMinTolerance", "TorqueMaxTolerance"}.issubset(d.columns):
        d["torque_ok"] = (d["RES_FinalTorque"] >= d["TorqueMinTolerance"]) & (d["RES_FinalTorque"] <= d["TorqueMaxTolerance"])

    if {"RES_FinalAngle", "STEP_AngleMinTolerance", "STEP_AngleMaxTolerance"}.issubset(d.columns):
        d["angle_ok"] = (d["RES_FinalAngle"] >= d["STEP_AngleMinTolerance"]) & (d["RES_FinalAngle"] <= d["STEP_AngleMaxTolerance"])

    if {"torque_ok", "angle_ok"}.issubset(d.columns):
        d["tightening_ok"] = d["torque_ok"] & d["angle_ok"]
        d["tightening_ok"] = d["tightening_ok"].astype("int8")

    # ---------- Errores respecto a target ----------
    if {"RES_FinalTorque", "TorqueTarget"}.issubset(d.columns):
        d["torque_error"] = d["RES_FinalTorque"] - d["TorqueTarget"]

    if {"RES_FinalAngle", "STEP_AngleTarget"}.issubset(d.columns):
        d["angle_error"] = d["RES_FinalAngle"] - d["STEP_AngleTarget"]

    # ---------- Normalización por tolerancia (comparabilidad) ----------
    if {"torque_error", "TorqueTarget", "TorqueMaxTolerance"}.issubset(d.columns):
        denom = (d["TorqueMaxTolerance"] - d["TorqueTarget"]).replace(0, np.nan)
        d["torque_error_norm"] = d["torque_error"] / denom

    if {"angle_error", "STEP_AngleTarget", "STEP_AngleMaxTolerance"}.issubset(d.columns):
        denom = (d["STEP_AngleMaxTolerance"] - d["STEP_AngleTarget"]).replace(0, np.nan)
        d["angle_error_norm"] = d["angle_error"] / denom

    # ---------- Distancia al límite más cercano (fragilidad) ----------
    if {"RES_FinalTorque", "TorqueMinTolerance", "TorqueMaxTolerance"}.issubset(d.columns):
        d["torque_dist_to_limit"] = np.minimum(
            d["TorqueMaxTolerance"] - d["RES_FinalTorque"],
            d["RES_FinalTorque"] - d["TorqueMinTolerance"],
        )

    if {"RES_FinalAngle", "STEP_AngleMinTolerance", "STEP_AngleMaxTolerance"}.issubset(d.columns):
        d["angle_dist_to_limit"] = np.minimum(
            d["STEP_AngleMaxTolerance"] - d["RES_FinalAngle"],
            d["RES_FinalAngle"] - d["STEP_AngleMinTolerance"],
        )

    # ---------- Features temporales (si hay timestamp) ----------
    if "RES_DateTime" in d.columns and is_datetime64_any_dtype(d["RES_DateTime"]):
        d["hour"] = d["RES_DateTime"].dt.hour.astype("int8")
        d["dayofweek"] = d["RES_DateTime"].dt.dayofweek.astype("int8")

    return d
