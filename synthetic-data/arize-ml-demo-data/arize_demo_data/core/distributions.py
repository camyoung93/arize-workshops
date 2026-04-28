"""Numpy-backed distribution helpers used by feature builders.

All functions take an explicit ``rng: np.random.Generator`` so the entire
pipeline is reproducible from a single seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_choice(
    rng: np.random.Generator,
    choices: Sequence,
    weights: Sequence[float],
    n: int,
) -> np.ndarray:
    """Sample n items from choices using normalized weights."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("weights must have positive sum")
    p = w / w.sum()
    return rng.choice(np.asarray(choices, dtype=object), size=n, p=p)


def lognormal(rng: np.random.Generator, mean: float, sigma: float, n: int) -> np.ndarray:
    return rng.lognormal(mean=mean, sigma=sigma, size=n)


def exponential(rng: np.random.Generator, scale: float, n: int) -> np.ndarray:
    return rng.exponential(scale=scale, size=n)


def beta(rng: np.random.Generator, a: float, b: float, n: int) -> np.ndarray:
    return rng.beta(a=a, b=b, size=n)


def gamma(rng: np.random.Generator, shape: float, scale: float, n: int) -> np.ndarray:
    return rng.gamma(shape=shape, scale=scale, size=n)


def normal(rng: np.random.Generator, loc: float, scale: float, n: int) -> np.ndarray:
    return rng.normal(loc=loc, scale=scale, size=n)
