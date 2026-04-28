"""Helpers for converting numeric values into ordinal band categoricals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def to_band(values: Sequence[float], bins: Sequence[float], labels: Sequence[int]) -> pd.Series:
    """Bin values into integer-labelled ordinal bands.

    ``bins`` should already include sentinel boundaries (e.g. -inf / inf);
    ``labels`` must have len(bins) - 1.
    """
    if len(labels) != len(bins) - 1:
        raise ValueError("labels must have one fewer entry than bins")
    cats = pd.cut(values, bins=list(bins), labels=list(labels))
    return cats.astype(int)


def amount_band(values: Sequence[float]) -> pd.Series:
    """Standard 1-6 band suitable for transaction-amount-like features."""
    return to_band(
        values,
        bins=[-np.inf, 50, 100, 500, 1000, 5000, np.inf],
        labels=[1, 2, 3, 4, 5, 6],
    )


def distance_band(values: Sequence[float]) -> pd.Series:
    return to_band(
        values,
        bins=[-np.inf, 0, 10, 50, 100, np.inf],
        labels=[1, 2, 3, 4, 5],
    )


def velocity_band_1hr(values: Sequence[float]) -> pd.Series:
    return to_band(
        values,
        bins=[-np.inf, 0, 2, 5, 10, np.inf],
        labels=[0, 1, 2, 3, 4],
    )


def velocity_band_24hr(values: Sequence[float]) -> pd.Series:
    return to_band(
        values,
        bins=[-np.inf, 2, 5, 10, 20, np.inf],
        labels=[0, 1, 2, 3, 4],
    )


def account_age_band(values: Sequence[float]) -> pd.Series:
    return to_band(
        values,
        bins=[-np.inf, 30, 90, 365, 730, np.inf],
        labels=[1, 2, 3, 4, 5],
    )


def risk_band(values: Sequence[float]) -> pd.Series:
    return to_band(
        values,
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
