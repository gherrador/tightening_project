from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import pandas as pd


# ----------------------------
# Contrato SILVER (verdad única)
# ----------------------------

@dataclass(frozen=True)
class SilverContract:
    """
    Define el contrato de SILVER:
    - columnas requeridas
    - tipos esperados (dtypes)
    - orden recomendado
    - columnas opcionales habituales
    """

    # Columnas mínimas para SPC/Capability/ML
    required_cols: Set[str] = frozenset({
        "STEP_ID",
        "FinalTorque",
        "TorqueTarget",
        "TorqueMinTolerance",
        "TorqueMaxTolerance",
        "Torque_Result",
        "Quality_Status",
    })

    # Opcionales típicas (no obligatorias)
    optional_cols: Set[str] = frozenset({
        "DateTime",
        "Tool_ID",
        "ToolId",
        "ToolID",
        "Part_ID",
        "Product_ID",
        "Batch_ID",
        "Station_ID",
        "Program_ID",
        "Operator_ID",
    })

    # Tipos recomendados (no todos se pueden forzar al 100% sin perder NaNs)
    # Nota: Usamos "pandas nullable" cuando conviene.
    dtype_map: Dict[str, str] = None  # se setea en __post_init__ (truco)

    # Orden recomendado en SILVER (si existen)
    preferred_order: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, "dtype_map", {
            "STEP_ID": "Int64",  # nullable int
            "FinalTorque": "float64",
            "TorqueTarget": "float64",
            "TorqueMinTolerance": "float64",
            "TorqueMaxTolerance": "float64",
            # resultados/categorías
            "Torque_Result": "string",     # "OK"/"NOK"
            "Quality_Status": "string",    # "OK"/"REVIEW_NEEDED"
            # opcionales
            "Tool_ID": "string",
            "DateTime": "datetime64[ns]",
        })

        object.__setattr__(self, "preferred_order", [
            "DateTime",
            "STEP_ID",
            "Tool_ID",
            "FinalTorque",
            "TorqueTarget",
            "TorqueMinTolerance",
            "TorqueMaxTolerance",
            "Torque_Result",
            "Quality_Status",
        ])


SILVER = SilverContract()


# ----------------------------
# Validación / Enforcements
# ----------------------------

def validate_silver_df(df: pd.DataFrame, *, strict: bool = True) -> None:
    """
    Valida que un DF cumple el contrato de SILVER.
    - strict=True: error si faltan required cols
    - strict=False: warning-like (pero aquí lanzamos ValueError igual si faltan)
    """
    missing = SILVER.required_cols - set(df.columns)
    if missing:
        raise ValueError(f"SILVER inválido: faltan columnas requeridas: {sorted(missing)}")

    # Validaciones semánticas mínimas (baratas pero muy útiles)
    # 1) tolerancias coherentes (permitimos iguales, pero eso debe estar marcado en Quality_Status)
    if {"TorqueMinTolerance", "TorqueMaxTolerance"}.issubset(df.columns):
        bad = (df["TorqueMinTolerance"].notna() & df["TorqueMaxTolerance"].notna()
               & (df["TorqueMinTolerance"] > df["TorqueMaxTolerance"]))
        if bad.any():
            n = int(bad.sum())
            raise ValueError(f"SILVER inválido: {n} filas con TorqueMinTolerance > TorqueMaxTolerance (swap debió ocurrir en cleaning).")

    # 2) Torque_Result en valores permitidos
    if "Torque_Result" in df.columns:
        allowed = {"OK", "NOK"}
        bad_vals = set(df["Torque_Result"].dropna().unique()) - allowed
        if bad_vals:
            raise ValueError(f"SILVER inválido: Torque_Result contiene valores no permitidos: {sorted(bad_vals)}")

    # 3) Quality_Status valores permitidos
    if "Quality_Status" in df.columns:
        allowed = {"OK", "REVIEW_NEEDED"}
        bad_vals = set(df["Quality_Status"].dropna().unique()) - allowed
        if bad_vals:
            raise ValueError(f"SILVER inválido: Quality_Status contiene valores no permitidos: {sorted(bad_vals)}")


def enforce_silver_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta alinear dtypes a lo esperado sin romper NaNs.
    Devuelve un DF nuevo (no modifica in-place).
    """
    out = df.copy()

    # Numéricos (si ya vienen bien, esto no cambia nada)
    for col in ["FinalTorque", "TorqueTarget", "TorqueMinTolerance", "TorqueMaxTolerance"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # STEP_ID nullable int
    if "STEP_ID" in out.columns:
        out["STEP_ID"] = pd.to_numeric(out["STEP_ID"], errors="coerce").astype("Int64")

    # DateTime si existe
    if "DateTime" in out.columns:
        out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")

    # Strings categóricos (string dtype)
    for col in ["Torque_Result", "Quality_Status", "Tool_ID"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    return out


def enforce_silver_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena columnas: primero las preferidas (si existen), luego el resto en orden estable.
    """
    cols = list(df.columns)
    head = [c for c in SILVER.preferred_order if c in cols]
    tail = [c for c in cols if c not in head]
    return df[head + tail]


def finalize_silver_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso final recomendado al final del cleaning:
    - enforce dtypes
    - enforce order
    - validate contrato
    """
    out = enforce_silver_dtypes(df)
    out = enforce_silver_order(out)
    validate_silver_df(out, strict=True)
    return out
