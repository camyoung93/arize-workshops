"""Timestamp generation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def random_timestamps(
    n: int,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> pd.Series:
    """Return n timestamps uniformly distributed across [start, end]."""
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    span_seconds = (end - start).total_seconds()
    offsets = rng.uniform(0, span_seconds, size=n)
    timestamps = [start + timedelta(seconds=float(s)) for s in offsets]
    return pd.Series(timestamps, name="timestamp")
