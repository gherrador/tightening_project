from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re


MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_PATTERN = "|".join(m.capitalize() for m in MONTH_MAP.keys())
_MONTHYY_RE = re.compile(rf"({MONTH_PATTERN})(\d{{2}})", flags=re.IGNORECASE)
_EXPORT_TS_RE = re.compile(r"_(\d{12})_")


@dataclass(frozen=True)
class FileInfo:
    year: int
    month: int
    source_stem: str
    export_ts: Optional[str] = None


def parse_source_stem(source_stem: str) -> FileInfo:    
    m_time = _MONTHYY_RE.search(source_stem)
    if not m_time:
        raise ValueError(f"No pude parsear MonthNameYY de: {source_stem}")

    month_name = m_time.group(1).lower()
    yy = int(m_time.group(2))

    year = 2000 + yy
    month = MONTH_MAP[month_name]

    m_ts = _EXPORT_TS_RE.search(source_stem)
    export_ts = m_ts.group(1) if m_ts else None

    return FileInfo(
        year=year,
        month=month,
        source_stem=source_stem,
        export_ts=export_ts,
    )
