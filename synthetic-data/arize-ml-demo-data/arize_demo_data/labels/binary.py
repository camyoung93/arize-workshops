"""Binary classification labels with a target AUC.

Strategy
--------
1. Build a per-row "risk score" by adding feature-driven multipliers to a
   small base rate, plus a Gaussian noise term so AUC is not artificially
   high.
2. ``prediction_score`` is the clipped result; ``prediction_label`` is the
   thresholded result.
3. ``actual_label`` is sampled from a Bernoulli that mixes 70% of the
   prediction score with 30% of (score + extra noise). This mix-rate is
   chosen to land near a target AUC of ~0.80 in practice; tweak via
   ``actual_signal_weight`` if you need a different target.

For training environments we use a slightly tighter signal blend to keep
training AUC higher than production — i.e. the demo tells a "model degraded
when it hit prod" story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BinaryLabelSpec:
    """Configuration for a binary-classification label generation pass."""

    risk_features: dict[str, float] = field(default_factory=dict)
    """Map of feature column name -> linear multiplier on (value - min)."""

    base_rate: float = 0.002
    """Baseline positive-class rate when all multipliers are zero."""

    pred_noise_sigma: float = 0.20
    """Std-dev of Gaussian noise added to the risk score before clipping."""

    actual_noise_sigma: float = 0.10
    """Std-dev of additional noise blended into the actuals."""

    actual_signal_weight: float = 0.70
    """Fraction of pure signal vs. (signal + noise) used when sampling actuals.

    Lower => actuals decouple more from predictions => lower AUC.
    """

    label_strategy: str = "bernoulli"
    """How to derive ``prediction_label`` from ``prediction_score``.

    - ``bernoulli`` (default): sample from Bernoulli(score). Matches typical
      low-base-rate fraud demos where almost no row clears 0.5 but actuals
      and predictions both land near the prevalence.
    - ``threshold``: ``prediction_label = int(score >= threshold)``.
    """

    threshold: float = 0.5
    """Score threshold (only used when ``label_strategy='threshold'``)."""


def generate_binary_labels(
    df: pd.DataFrame,
    spec: BinaryLabelSpec,
    rng: np.random.Generator,
    *,
    pred_score_col: str = "prediction_score",
    pred_label_col: str = "prediction_label",
    actual_label_col: str = "actual_label",
) -> pd.DataFrame:
    """Append prediction_score, prediction_label, and actual_label columns.

    The function mutates a copy of ``df`` and returns it; risk features are
    expected to already exist as numeric columns. Categorical features can
    be made numeric in the flavor's feature builders before being passed in.
    """
    n = len(df)
    if n == 0:
        return df.copy()

    out = df.copy()

    risk = np.full(n, fill_value=1.0, dtype=float)
    for col, mult in spec.risk_features.items():
        if col not in out.columns:
            raise KeyError(f"risk feature column not in dataframe: {col}")
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            continue
        col_min = np.nanmin(values)
        adj = np.where(np.isnan(values), 0.0, values - col_min)
        risk += mult * adj

    risk += rng.normal(loc=0.0, scale=spec.pred_noise_sigma, size=n)

    score = np.clip(spec.base_rate * risk, 0.0, 1.0)
    out[pred_score_col] = score
    if spec.label_strategy == "bernoulli":
        out[pred_label_col] = rng.binomial(n=1, p=score).astype(int)
    elif spec.label_strategy == "threshold":
        out[pred_label_col] = (score >= spec.threshold).astype(int)
    else:
        raise ValueError(f"Unknown label_strategy: {spec.label_strategy}")

    actual_noise = rng.normal(loc=0.0, scale=spec.actual_noise_sigma, size=n)
    blended = (
        spec.actual_signal_weight * score
        + (1.0 - spec.actual_signal_weight) * np.clip(score + actual_noise, 0.0, 1.0)
    )
    blended = np.clip(blended, 0.0, 1.0)
    out[actual_label_col] = rng.binomial(n=1, p=blended).astype(int)

    return out
