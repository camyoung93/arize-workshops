"""ID generation helpers."""

from __future__ import annotations

import uuid

import numpy as np


def transaction_ids(n: int, prefix: str = "TXN") -> list[str]:
    return [f"{prefix}_{i + 1:08d}" for i in range(n)]


def customer_ids(n_rows: int, n_unique: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n_rows customer IDs from a pool of n_unique ids."""
    if n_unique < 1:
        raise ValueError("n_unique must be at least 1")
    pool = np.array([f"CUST_{i + 1:06d}" for i in range(n_unique)], dtype=object)
    idx = rng.integers(0, n_unique, size=n_rows)
    return pool[idx]


def prediction_ids(n: int) -> list[str]:
    return [str(uuid.uuid4()) for _ in range(n)]
