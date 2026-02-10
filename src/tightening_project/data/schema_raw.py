from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TorqueColumns:
    step_id: str = "STEP_ID"
    target: str = "STEP_TorqueTarget"
    tol_min: str = "STEP_TorqueMinTolerance"
    tol_max: str = "STEP_TorqueMaxTolerance"
    final: str = "RES_FinalTorque"
    dt: str = "RES_DateTime"


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    swapped_min_max: int
    dropped_zero_block: int
    dropped_missing_essentials: int
    ok_count: int
    nok_count: int
    review_needed: int
    issue_counts: Dict[str, int]


