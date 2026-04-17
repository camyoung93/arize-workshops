#!/usr/bin/env python3
"""
StreamFlix Feature Store Monitoring — synthetic data ingestion for Arize (SDK v7).

30 days of paired daily snapshots: TRAINING (offline feature store materialization)
and PRODUCTION (online serving logs). Same model, model_version = date string for
day-over-day alignment in the UI.

Intentional skew:
- genre_affinity_score: offline/online distributions diverge from day 15 (mean shift ~0.1–0.15).
- device_type: mobile share increases in online logs vs offline baseline over the window.
- Other features remain stable.

Usage (use a venv with arize>=7,<8):
  export ARIZE_SPACE_ID=... ARIZE_API_KEY=...
  pip install -r requirements-v7.txt
  python streamflix_feature_store_ingest_v7.py
"""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Arize SDK v7
from arize.pandas.logger import Client as ArizeClient
from arize.utils.types import Schema, Environments, ModelTypes

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_NAME = "streamflix_feature_store_monitor-v7-1"
ROWS_PER_DAY = 10_000
NUM_DAYS = 30
RNG_SEED = 42

DEVICE_TYPES = ["mobile", "tv", "web", "tablet"]
LANGUAGES = ["en", "es", "fr", "ko", "de", "ja", "pt"]

# Offline (TRAINING) baseline: device_type distribution (mobile, tv, web, tablet)
DEVICE_PROBS_OFFLINE = [0.35, 0.30, 0.25, 0.10]
# Online (PRODUCTION) will shift: mobile increases over the 30-day window
MOBILE_START_PROB = 0.35
MOBILE_END_PROB = 0.55  # drift: mobile share grows in production

# genre_affinity_score: baseline mean for both envs until day 15, then production drifts
GENRE_BASELINE_MEAN = 0.45
GENRE_BASELINE_STD = 0.18
GENRE_DRIFT_START_DAY = 15
GENRE_DRIFT_MEAN_SHIFT = 0.12  # production mean increases gradually (up to ~0.12 by day 30)


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _hash_user(i: int, day: int) -> str:
    return hashlib.sha256(f"user_{i}_{day}".encode()).hexdigest()[:24]


def generate_day(
    rng: np.random.Generator,
    date: datetime,
    environment: str,
    day_index: int,
) -> pd.DataFrame:
    """Generate one day of synthetic rows for TRAINING or PRODUCTION."""
    n = ROWS_PER_DAY
    ts = int(date.replace(tzinfo=timezone.utc).timestamp())

    user_ids = [_hash_user(i, day_index) for i in range(n)]

    # recency_score: stable 0–1
    recency_score = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # watch_count_7d: stable
    watch_count_7d = rng.integers(0, 50, size=n)

    # content_language_pref: stable categorical
    content_language_pref = rng.choice(LANGUAGES, size=n).tolist()

    # genre_affinity_score: drift in PRODUCTION from day 15
    if environment == "TRAINING":
        genre_mean = GENRE_BASELINE_MEAN
        genre_std = GENRE_BASELINE_STD
    else:
        # Production: gradual mean shift from day GENRE_DRIFT_START_DAY
        if day_index < GENRE_DRIFT_START_DAY:
            genre_mean = GENRE_BASELINE_MEAN
            genre_std = GENRE_BASELINE_STD
        else:
            # Linear ramp from 0 to GENRE_DRIFT_MEAN_SHIFT over remaining days
            progress = (day_index - GENRE_DRIFT_START_DAY) / max(1, NUM_DAYS - GENRE_DRIFT_START_DAY)
            genre_mean = GENRE_BASELINE_MEAN + progress * GENRE_DRIFT_MEAN_SHIFT
            genre_std = GENRE_BASELINE_STD
        genre_mean = float(np.clip(genre_mean, 0.0, 1.0))
    raw = rng.normal(genre_mean, genre_std, size=n)
    genre_affinity_score = np.clip(raw, 0.0, 1.0).astype(np.float32)

    # device_type: stable for TRAINING; for PRODUCTION, mobile share increases over time
    if environment == "TRAINING":
        device_type = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_PROBS_OFFLINE).tolist()
    else:
        # Production: mobile probability increases linearly over 30 days
        t = day_index / max(1, NUM_DAYS - 1)
        mobile_p = MOBILE_START_PROB + t * (MOBILE_END_PROB - MOBILE_START_PROB)
        mobile_p = float(np.clip(mobile_p, 0.0, 1.0))
        remaining = 1.0 - mobile_p
        # Distribute remaining among tv, web, tablet (same relative proportions as offline)
        p_rest = [0.30, 0.25, 0.10]  # tv, web, tablet
        s = sum(p_rest)
        probs = [mobile_p] + [p * remaining / s for p in p_rest]
        device_type = rng.choice(DEVICE_TYPES, size=n, p=probs).tolist()

    # Stand-in prediction (regression target); use same as actual for pre-production requirement
    standin_prediction_value = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # Timestamp: same date for all rows (snapshot date)
    timestamp = np.full(n, ts, dtype=np.int64)

    # Prediction ID: unique per row for matching
    prediction_id = [f"{uid}_{ts}_{i}" for i, uid in enumerate(user_ids)]

    return pd.DataFrame({
        "prediction_id": prediction_id,
        "user_id": user_ids,
        "genre_affinity_score": genre_affinity_score,
        "recency_score": recency_score,
        "watch_count_7d": watch_count_7d,
        "device_type": device_type,
        "content_language_pref": content_language_pref,
        "standin_prediction_value": standin_prediction_value,
        "standin_actual_value": standin_prediction_value,  # same as prediction for schema requirement
        "timestamp": timestamp,
    })


def main() -> None:
    space_id = os.environ.get("ARIZE_SPACE_ID")
    api_key = os.environ.get("ARIZE_API_KEY")
    if not space_id or not api_key:
        raise SystemExit("Set ARIZE_SPACE_ID and ARIZE_API_KEY in the environment.")

    # v7: Client takes space_id and api_key at construction
    client = ArizeClient(space_id=space_id, api_key=api_key)

    schema = Schema(
        prediction_id_column_name="prediction_id",
        feature_column_names=[
            "user_id",
            "genre_affinity_score",
            "recency_score",
            "watch_count_7d",
            "device_type",
            "content_language_pref",
        ],
        timestamp_column_name="timestamp",
        prediction_score_column_name="standin_prediction_value",
        actual_score_column_name="standin_actual_value",
    )

    base_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    rng = _make_rng(RNG_SEED)

    for day_index in range(NUM_DAYS):
        date = base_date + timedelta(days=day_index)
        date_str = date.strftime("%Y-%m-%d")

        for env_name, environment in [
            ("TRAINING", Environments.TRAINING),
            ("PRODUCTION", Environments.PRODUCTION),
        ]:
            df = generate_day(rng, date, env_name, day_index)
            # v7: client.log() with model_id (no space_id in log call)
            resp = client.log(
                dataframe=df,
                schema=schema,
                environment=environment,
                model_id=MODEL_NAME,
                model_type=ModelTypes.REGRESSION,
                model_version=date_str,
                validate=True,
            )
            status = "ok" if resp.status_code == 200 else f"err={resp.status_code}"
            print(f"  {date_str} {env_name}: {len(df)} rows -> {status}")

    print("Done. 60 ingestion calls (30 days × 2 environments).")


if __name__ == "__main__":
    main()
