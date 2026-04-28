"""Synthetic SHAP value generator.

Generates ``<feature>_shap`` columns drawn from N(0, sigma) where sigma is
larger for important features, so the Arize feature-importance UI tells a
coherent story without needing a real model.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def add_synthetic_shap(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    important_features: dict[str, float],
    rng: np.random.Generator,
    *,
    default_sigma: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Append a ``<feature>_shap`` column for each feature.

    Returns the modified dataframe plus a ``feature -> shap_col`` mapping
    suitable for Arize's ``shap_values_column_names`` schema field.
    """
    out = df.copy()
    n = len(out)
    mapping: dict[str, str] = {}
    for feat in feature_columns:
        col = f"{feat}_shap"
        sigma = important_features.get(feat, default_sigma)
        out[col] = rng.normal(loc=0.0, scale=sigma, size=n)
        mapping[feat] = col
    return out, mapping
