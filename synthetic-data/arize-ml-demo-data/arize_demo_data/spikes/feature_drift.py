"""Feature distribution drift.

Pushes ``transaction_amount`` higher (favoring the $2k–$5k tail) and skews
``transaction_type`` / ``entry_method`` toward mobile, so the drift tab in
Arize shows a clear shift over the spike window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arize_demo_data.core import bands

if False:
    from arize_demo_data.spikes.registry import SpikeContext


def apply_feature_drift(df: pd.DataFrame, ctx: "SpikeContext") -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    out["transaction_amount"] = np.round(
        ctx.rng.uniform(low=2000.0, high=5000.0, size=n), 2
    )
    if "amount_band" in out.columns:
        out["amount_band"] = bands.amount_band(out["transaction_amount"])

    if "transaction_type" in out.columns:
        out["transaction_type"] = "mobile_payment"
    if "entry_method" in out.columns:
        out["entry_method"] = "mobile"

    return out
