# StreamFlix synthetic data

Synthetic data demos for StreamFlix prospect spaces. Two independent tracks:

1. **Search Augment (GenAI)** — OpenInference traces for a search-augment
   recommendation pipeline, plus registered LLM-as-judge evaluators and
   continuous eval tasks.
2. **Feature Store Monitoring (classic ML)** — 30 days of offline/online
   feature snapshots with seeded drift.

## Contents

| Artifact | Description |
|----------|-------------|
| `generate_traces.py` | Search-augment OpenInference traces (catalog, reranking, personalized response), grouped into sessions |
| `create_evaluators.py` | Registers 4 LLM-as-judge evaluators (span / trace / session) |
| `create_tasks.py` | Creates 3 continuous eval tasks wiring the evaluators to spans |
| `run_evals.py` | Triggers a task over a time window and waits for results |
| `templates/` | Synthetic world data (queries, catalog, scenarios, profiles) + LLM prompt template |
| `EVALUATORS.md` | Evaluator/task reference: filters, mappings, caveats |
| `streamflix_feature_store_ingest.py` | **Feature Store Monitoring (SDK v8)** — 30 days of offline/online feature snapshots; requires Python 3.10+ |
| `streamflix_feature_store_ingest_v7.py` | **Feature Store Monitoring (SDK v7)** — same as above for arize 7.x; use `requirements-v7.txt` and any Python 3.8+ |

---

## StreamFlix Search Augment

### Overview

Each trace is one user query through the recommendation pipeline (all content
synthetic — no retrieval system, reranker, or LLM is invoked; OpenInference
attributes are written directly):

```
StreamFlix Search Augment Pipeline        [CHAIN, root]
  ├── candidate_retrieval                  [RETRIEVER]  catalog first-pull with scores
  ├── rerank_candidates                    [RERANKER]   top picks passed to the LLM
  └── search_augment_agent                 [AGENT]
        ├── fetch_user_history             [RETRIEVER]  user profile / watch history
        └── personalize_recommendations    [LLM]        personalized picks
```

Traces are grouped into sessions of 2-5 (`session.id` on the CHAIN/AGENT/LLM
spans). Default project: `streamflix_search_augment`.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ARIZE_SPACE_ID / ARIZE_API_KEY
```

`create_evaluators.py` additionally needs `ARIZE_AI_INTEGRATION_NAME` — the
display name of an AI integration in your space (app.arize.com → Space
settings → AI Integrations).

### Run order

```bash
# 1. Traces — smoke-test one, then the full batch
python generate_traces.py --test
python generate_traces.py                # 500 traces

# 2. Register the LLM-as-judge evaluators
python create_evaluators.py --dry-run
python create_evaluators.py

# 3. Create the continuous eval tasks (span / trace / session)
python create_tasks.py

# 4. Backfill evals over existing traces (continuous tasks only score new ones)
python run_evals.py --task "StreamFlix - LLM Evals" --days 1
python run_evals.py --task "StreamFlix - Trace Evals" --days 1
python run_evals.py --task "StreamFlix - Session Evals" --days 1
```

See `EVALUATORS.md` for the evaluator/task reference and caveats (exact-match
span filters, session `{conversation}` mapping, backfill behavior).

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
