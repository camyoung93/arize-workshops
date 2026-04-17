# StreamFlix synthetic data

Synthetic data demos for StreamFlix prospect spaces.

## Contents

| Artifact | Description |
|----------|-------------|
| `synthetic_spans_streamflix_search_augment.ipynb` | Search augment traces (catalog, reranking, personalized response) — OpenInference spans |
| `streamflix_feature_store_ingest.py` | **Feature Store Monitoring (SDK v8)** — 30 days of offline/online feature snapshots; requires Python 3.10+ |
| `streamflix_feature_store_ingest_v7.py` | **Feature Store Monitoring (SDK v7)** — same as above for arize 7.x; use `requirements-v7.txt` and any Python 3.8+ |

---

## StreamFlix Feature Store Monitoring

### Overview

30 days of paired daily snapshots ingested via Arize SDK as a single model:

- **TRAINING** = offline (feature store materialization)
- **PRODUCTION** = online (serving logs)

`model_version` is the date string for both environments, enabling day-over-day alignment in the UI.

### Schema (both environments)

| Column | Type | Notes |
|--------|------|--------|
| `user_id` | string | Hashed, entity ID |
| `genre_affinity_score` | float | 0–1, primary drift feature |
| `recency_score` | float | 0–1, days-since-watch decay |
| `watch_count_7d` | int | Rolling 7-day view count |
| `device_type` | string | mobile / tv / web / tablet |
| `content_language_pref` | string | en, es, fr, ko, etc. |
| `standin_prediction_value` | float | Required prediction column (regression) |
| `timestamp` | datetime | Date of snapshot/serving log |

### Intentional skew

- **genre_affinity_score**: Offline/online distributions diverge starting day 15 (mean shift ~0.1–0.15).
- **device_type**: Mobile share increases in online logs vs offline baseline over the 30-day window.
- All other features remain stable.

### Ingestion

- **Model name:** `streamflix_feature_store_monitor`
- **Offline:** `Environments.TRAINING`, `model_version` = date string
- **Online:** `Environments.PRODUCTION`, `model_version` = date string
- 10k rows per day per environment → 60 ingestion calls total
- **Prediction type:** regression (`standin_prediction_value`)

### What this shows in the UI

- Drift monitor firing on `genre_affinity_score` around day 15
- `device_type` skew widening between offline and online over the 30-day window
- Temporal range for “how did this evolve” narrative
- Optional data quality checks on null rates or range guards per snapshot

### How to run

**v8 (default):** Requires Python 3.10+. From `arize-workshops/synthetic-data/`:

```bash
export ARIZE_SPACE_ID=... ARIZE_API_KEY=...
python streaming/streamflix_feature_store_ingest.py
```

**v7:** Use a separate venv with arize 7.x (e.g. `python3 -m venv .venv-v7 && source .venv-v7/bin/activate`, then `pip install -r requirements-v7.txt` from `arize-workshops/synthetic-data/`), then:

```bash
export ARIZE_SPACE_ID=... ARIZE_API_KEY=...
python streaming/streamflix_feature_store_ingest_v7.py
```

Single Python script; no notebook. Data is generated in-memory with numpy/pandas, RNG seeded for reproducibility. Drift is gradual (linear ramp), not a step function.
