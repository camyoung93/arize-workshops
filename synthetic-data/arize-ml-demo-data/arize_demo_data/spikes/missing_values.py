"""Missing-value injection.

Defaults to nulling ~30% of ``urban_rural`` over the spike window, but the
target columns and rate are configurable per flavor by overriding this
function later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

if False:
    from arize_demo_data.spikes.registry import SpikeContext

_DEFAULT_TARGETS: dict[str, float] = {
    "urban_rural": 0.30,
}


def apply_missing_values(df: pd.DataFrame, ctx: "SpikeContext") -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    for col, rate in _DEFAULT_TARGETS.items():
        if col not in out.columns:
            continue
        mask = ctx.rng.random(n) < rate
        if mask.any():
            out.loc[mask, col] = np.nan
    return out
