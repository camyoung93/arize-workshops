"""Spike registry + apply pipeline.

Each spike is a pure function ``(df, ctx) -> df`` that mutates a copy of the
dataframe to introduce a specific demoable issue. Spikes are applied in the
order requested.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from arize_demo_data.spikes.feature_drift import apply_feature_drift
from arize_demo_data.spikes.missing_values import apply_missing_values
from arize_demo_data.spikes.schema_regression import apply_schema_regression


@dataclass
class SpikeContext:
    """Per-run state passed to every spike."""

    rng: np.random.Generator
    flavor_key: str


SpikeFn = Callable[[pd.DataFrame, SpikeContext], pd.DataFrame]


SPIKES: dict[str, SpikeFn] = {
    "feature_drift": apply_feature_drift,
    "missing_values": apply_missing_values,
    "schema_regression": apply_schema_regression,
}


def get_spike(key: str) -> SpikeFn:
    if key not in SPIKES:
        available = ", ".join(sorted(SPIKES))
        raise KeyError(f"Unknown spike '{key}'. Available: {available}")
    return SPIKES[key]


def apply_spikes(
    df: pd.DataFrame,
    spike_keys: Iterable[str],
    ctx: SpikeContext,
) -> pd.DataFrame:
    """Apply the requested spikes in order, returning the resulting dataframe."""
    out = df.copy()
    for key in spike_keys:
        fn = get_spike(key)
        out = fn(out, ctx)
    return out
