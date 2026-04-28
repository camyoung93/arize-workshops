"""Synthetic payments-fraud flavor (binary classification).

Anonymized industry analogue, not modeled on any real customer. Generates a
transaction dataset with merchant, customer, location, velocity, and entry-
method features, plus a binary fraud target.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from arize_demo_data.core import bands, distributions, ids, timestamps
from arize_demo_data.flavors.base import Flavor
from arize_demo_data.labels.binary import BinaryLabelSpec

PROVINCES = ["ON", "BC", "AB", "QC", "NS", "MB", "SK", "NB", "NL", "PE"]
PROVINCE_WEIGHTS = [0.40, 0.15, 0.12, 0.20, 0.03, 0.03, 0.03, 0.02, 0.01, 0.01]

URBAN_RURAL = ["urban", "rural"]
URBAN_RURAL_WEIGHTS = [0.7, 0.3]

MCC_CATEGORIES = {
    "grocery": ["5411", "5422", "5451"],
    "dining": ["5812", "5813", "5814"],
    "retail": ["5311", "5399", "5611", "5621"],
    "travel": ["4511", "4112", "7011"],
    "utilities": ["4900", "4814", "4899"],
    "gas": ["5541", "5542"],
    "online_retail": ["5964", "5965", "5969"],
}

TRANSACTION_TYPES = ["retail", "withdrawal", "e_transfer", "bill_payment", "deposit"]
TRANSACTION_TYPE_WEIGHTS = [0.60, 0.15, 0.15, 0.05, 0.05]
ENTRY_METHODS = ["chip", "tap", "manual", "online"]


FEATURE_COLUMNS = [
    "transaction_amount",
    "amount_band",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "mcc_code",
    "merchant_risk_band",
    "is_new_merchant",
    "account_age_days",
    "account_age_band",
    "customer_risk_score",
    "customer_risk_band",
    "txn_velocity_1hr",
    "txn_velocity_24hr",
    "txn_velocity_1hr_band",
    "txn_velocity_24hr_band",
    "province",
    "urban_rural",
    "distance_from_home",
    "distance_band",
    "transaction_type",
    "entry_method",
]

TAG_COLUMNS = ["transaction_id", "customer_id"]


def _flatten_mcc_choices(rng: np.random.Generator, n: int) -> np.ndarray:
    all_mccs: list[str] = []
    weights: list[float] = []
    common = {"grocery", "dining", "retail"}
    for category, mccs in MCC_CATEGORIES.items():
        per = 0.10 if category in common else 0.05
        all_mccs.extend(mccs)
        weights.extend([per] * len(mccs))
    return distributions.weighted_choice(rng, all_mccs, weights, n)


def _entry_method_for_type(rng: np.random.Generator, ttype: str) -> str:
    if ttype == "e_transfer":
        return "online"
    if ttype == "withdrawal":
        return str(distributions.weighted_choice(rng, ["chip", "manual"], [0.9, 0.1], 1)[0])
    return str(distributions.weighted_choice(rng, ENTRY_METHODS, [0.4, 0.4, 0.1, 0.1], 1)[0])


def build_payments_dataframe(
    n_rows: int,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a fully-featured payments dataframe (no labels yet)."""
    if n_rows <= 0:
        return pd.DataFrame()

    ts = timestamps.random_timestamps(n_rows, start, end, rng)

    amount = np.round(distributions.lognormal(rng, mean=4.0, sigma=1.0, n=n_rows), 2)
    distance = np.round(distributions.exponential(rng, scale=20.0, n=n_rows), 2)
    customer_risk = distributions.beta(rng, a=2.0, b=5.0, n=n_rows)
    account_age = distributions.gamma(rng, shape=2.0, scale=500.0, n=n_rows).astype(int)

    velocity_1hr = rng.integers(0, 5, size=n_rows)
    velocity_24hr = rng.integers(0, 25, size=n_rows)

    df = pd.DataFrame(
        {
            "transaction_id": ids.transaction_ids(n_rows),
            "timestamp": ts,
            "transaction_amount": amount,
            "hour_of_day": ts.dt.hour.astype(int),
            "day_of_week": ts.dt.weekday.astype(int),
            "is_weekend": (ts.dt.weekday >= 5).astype(int),
            "mcc_code": _flatten_mcc_choices(rng, n_rows),
            "merchant_risk_band": distributions.weighted_choice(
                rng,
                [1, 2, 3, 4, 5],
                [0.40, 0.30, 0.15, 0.10, 0.05],
                n_rows,
            ).astype(int),
            "is_new_merchant": distributions.weighted_choice(
                rng, [False, True], [0.9, 0.1], n_rows
            ).astype(bool),
            "customer_id": ids.customer_ids(n_rows, max(1, n_rows // 10), rng),
            "account_age_days": account_age,
            "customer_risk_score": np.round(customer_risk, 4),
            "txn_velocity_1hr": velocity_1hr,
            "txn_velocity_24hr": velocity_24hr,
            "province": distributions.weighted_choice(rng, PROVINCES, PROVINCE_WEIGHTS, n_rows),
            "urban_rural": distributions.weighted_choice(
                rng, URBAN_RURAL, URBAN_RURAL_WEIGHTS, n_rows
            ),
            "distance_from_home": distance,
        }
    )

    df["amount_band"] = bands.amount_band(df["transaction_amount"])
    df["distance_band"] = bands.distance_band(df["distance_from_home"])
    df["customer_risk_band"] = bands.risk_band(df["customer_risk_score"])
    df["account_age_band"] = bands.account_age_band(df["account_age_days"])
    df["txn_velocity_1hr_band"] = bands.velocity_band_1hr(df["txn_velocity_1hr"])
    df["txn_velocity_24hr_band"] = bands.velocity_band_24hr(df["txn_velocity_24hr"])

    types = distributions.weighted_choice(
        rng, TRANSACTION_TYPES, TRANSACTION_TYPE_WEIGHTS, n_rows
    )
    df["transaction_type"] = types
    df["entry_method"] = [_entry_method_for_type(rng, t) for t in types]

    return df.sort_values("timestamp").reset_index(drop=True)


def _label_spec() -> BinaryLabelSpec:
    return BinaryLabelSpec(
        risk_features={
            "amount_band": 0.30,
            "merchant_risk_band": 0.30,
            "customer_risk_band": 0.30,
            "txn_velocity_1hr_band": 0.40,
            "distance_band": 0.20,
            "is_new_merchant": 0.40,
        },
        base_rate=0.002,
        pred_noise_sigma=0.20,
        actual_noise_sigma=0.10,
        actual_signal_weight=0.70,
        threshold=0.5,
    )


def build_payments_fraud_flavor() -> Flavor:
    return Flavor(
        key="payments_fraud",
        industry="Financial / payments",
        default_model_type="binary_classification",
        feature_columns=list(FEATURE_COLUMNS),
        tag_columns=list(TAG_COLUMNS),
        default_important_features={
            "amount_band": 5.0,
            "transaction_amount": 4.0,
            "entry_method": 3.0,
        },
        default_spikes=[
            "feature_drift",
            "missing_values",
            "schema_regression",
        ],
        description=(
            "Anonymized payments fraud detection demo: "
            "transaction features, merchant + customer risk bands, "
            "velocity, location (Canadian provinces), entry method, "
            "binary fraud target with target AUC ~0.80."
        ),
        builder=build_payments_dataframe,
        label_spec_factory=_label_spec,
    )
