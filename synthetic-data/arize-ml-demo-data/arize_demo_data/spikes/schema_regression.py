"""Schema/format regression.

Simulates a validation regression upstream by mixing two-letter province
codes with their full names (e.g. "ON" alongside "Ontario") for ~70% of the
spike window. Surfaces in Arize as cardinality + DQ changes.
"""

from __future__ import annotations

import pandas as pd

if False:
    from arize_demo_data.spikes.registry import SpikeContext

_PROVINCE_FULL_NAMES = {
    "ON": "Ontario",
    "BC": "British Columbia",
    "AB": "Alberta",
    "QC": "Quebec",
    "NS": "Nova Scotia",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
}

_FULL_NAME_RATE = 0.70


def apply_schema_regression(df: pd.DataFrame, ctx: "SpikeContext") -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n == 0 or "province" not in out.columns:
        return out

    flip = ctx.rng.random(n) < _FULL_NAME_RATE
    if not flip.any():
        return out

    mapped = out.loc[flip, "province"].map(_PROVINCE_FULL_NAMES)
    mapped = mapped.fillna(out.loc[flip, "province"])
    out.loc[flip, "province"] = mapped
    return out
