"""Registry of available flavors."""

from __future__ import annotations

from arize_demo_data.flavors.base import Flavor
from arize_demo_data.flavors.payments_fraud import build_payments_fraud_flavor

FLAVORS: dict[str, Flavor] = {
    "payments_fraud": build_payments_fraud_flavor(),
}


def get_flavor(key: str) -> Flavor:
    if key not in FLAVORS:
        available = ", ".join(sorted(FLAVORS))
        raise KeyError(f"Unknown flavor '{key}'. Available: {available}")
    return FLAVORS[key]
