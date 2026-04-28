# arize-ml-demo-data

Generate realistic synthetic ML datasets for Arize POCs and demos. Builds a
clean baseline of model predictions + ground truth, optionally injects a
configurable mix of "spikes" (drift, data-quality regressions, performance
degradations) over a recent time window, and logs everything to Arize as a
classical ML model — production and/or training environments.

This is the pattern many Arize SEs use by hand for POC kickoffs, packaged as a
runnable Python CLI with industry-flavored use cases. Everything is
synthetic and anonymized; no real PII or customer data.

## What it generates

A typical run produces:

- **Base period** (1–3 months of clean traffic) — feature distributions,
  predictions, and ground truth labels with a realistic target metric
  (e.g. AUC ≈ 0.80 for binary classification).
- **Spike period** (last ~2 weeks) — a subset of rows in which one or more
  configurable issues are injected. These manifest in Arize as drift, data
  quality degradation, or performance regressions you can drill into.
- **Tabular embeddings** (optional) — generated via Arize's
  `EmbeddingGenerator` so the embeddings tab in the UI is populated.
- **Synthetic SHAP values** — biased toward a configurable set of "important"
  features so the feature-importance tab tells a coherent story.

Logged to Arize as a classical ML model with `Schema`, `ModelTypes`, and
`Environments.PRODUCTION` (and optionally `TRAINING`).

## Quickstart

```bash
cd arize-workshops/synthetic-data/arize-ml-demo-data

# Set up a virtualenv (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate

# Install (use the [embeddings] extra to enable the tabular embeddings step)
pip install -e ".[embeddings]"

# Configure Arize credentials
cp .env.example .env
# then edit .env with your ARIZE_SPACE_ID and ARIZE_API_KEY

# Run a payments-fraud POC: 100k clean rows over 90 days + 25k spike rows in
# the last 14 days, recommended spike pack, log to PRODUCTION.
python -m arize_demo_data --config configs/payments_fraud_binary.yaml
```

For a fully offline test (no Arize calls), use the smoke example:

```bash
python examples/smoke.py
```

## CLI shape

```bash
python -m arize_demo_data \
  --flavor payments_fraud \
  --model-type binary_classification \
  --base-rows 100000 --base-window-days 90 \
  --spike-rows 25000 --spike-window-days 14 \
  --spikes feature_drift,missing_values,schema_regression \
  --target-auc 0.80 \
  --embeddings tabular \
  --shap synthetic \
  --environments production \
  --model-id demo-payments-fraud-v1 \
  --model-version 1.0 \
  --seed 42
```

Or pass `--config <path-to-yaml>` and override individual fields with flags.

## Available flavors

| Flavor key                 | Industry             | Default model type        |
|----------------------------|----------------------|---------------------------|
| `payments_fraud`           | Financial / payments | binary classification     |

(More flavors land in subsequent milestones — see `MILESTONES.md`.)

## Available spikes

Mix and match with `--spikes a,b,c`.

| Spike key                | Surfaces in Arize as          |
|--------------------------|--------------------------------|
| `feature_drift`          | Drift tab                      |
| `missing_values`         | Data Quality tab               |
| `schema_regression`      | DQ + cardinality changes       |

## Environment variables

| Variable             | Required        | Description                       |
|----------------------|-----------------|-----------------------------------|
| `ARIZE_SPACE_ID`     | yes (for upload)| Arize space ID                    |
| `ARIZE_API_KEY`      | yes (for upload)| Arize API key                     |
| `ARIZE_ENABLE_LOG`   | no              | Set to `false` to skip the upload |

If `ARIZE_SPACE_ID` / `ARIZE_API_KEY` are absent, the package will generate
the dataset and write a parquet file to the configured output directory but
will not attempt to upload.

## License

Apache-2.0. See `LICENSE`.
