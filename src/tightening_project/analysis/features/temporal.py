from __future__ import annotations
import pandas as pd


def add_rolling_trend(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    value_col: str,
    windows: list[int],
    min_periods: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(key_cols, sort=False)

    for w in windows:
        r = g[value_col].rolling(window=w, min_periods=min_periods)
        mean = r.mean().reset_index(level=key_cols, drop=True)
        out[f"{value_col}_trend_roll{w}"] = out[value_col] - mean

    return out


def add_lags(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    value_col: str,
    lags: list[int],
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(key_cols, sort=False)
    for k in lags:
        out[f"{value_col}_lag{k}"] = g[value_col].shift(k)
    return out


def add_rolling_stats(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    value_col: str,
    windows: list[int],
    min_periods: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(key_cols, sort=False)

    for w in windows:
        r = g[value_col].rolling(window=w, min_periods=min_periods)
        out[f"{value_col}_roll{w}_mean"] = r.mean().reset_index(level=key_cols, drop=True)
        out[f"{value_col}_roll{w}_std"]  = r.std(ddof=1).reset_index(level=key_cols, drop=True)

    return out


def add_rows_since_last_event(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    event_col: str,
    event_value: int = 1,
) -> pd.DataFrame:
    out = df.copy()

    def _since_last(s: pd.Series) -> pd.Series:
        last = None
        res = []
        for i, v in enumerate(s.to_numpy()):
            if v == event_value:
                last = i
                res.append(0)
            else:
                res.append(i - last if last is not None else pd.NA)
        return pd.Series(res, index=s.index)

    out[f"rows_since_last_{event_col}{event_value}"] = (
        out.groupby(key_cols, sort=False)[event_col]
           .apply(_since_last)
           .reset_index(level=key_cols, drop=True)
    )
    return out

def add_rolling_event_rate(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    event_col: str,
    windows: list[int],
    min_periods: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(key_cols, sort=False)

    for w in windows:
        r = g[event_col].rolling(window=w, min_periods=min_periods)
        out[f"{event_col}_rate_roll{w}"] = (
            r.mean().reset_index(level=key_cols, drop=True)
        )

    return out



