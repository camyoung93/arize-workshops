"""Offline smoke test: generates a tiny dataset without uploading to Arize.

Run with:

    python examples/smoke.py

Verifies the full pipeline (features + labels + spikes + SHAP) without
network calls. Skips embeddings since those require the heavy extras.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arize_demo_data.config import GenerationConfig  # noqa: E402
from arize_demo_data.pipeline import run  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    cfg = GenerationConfig(
        flavor="payments_fraud",
        model_type="binary_classification",
        model_id="smoke-payments-fraud",
        model_version="0.0.1",
        base_rows=2_000,
        base_window_days=30,
        spike_rows=500,
        spike_window_days=7,
        spikes=["feature_drift", "missing_values", "schema_regression"],
        embeddings="none",
        shap="synthetic",
        environments=["production"],
        seed=7,
        output_dir=str(REPO_ROOT / ".scratch" / "smoke"),
        log_to_arize=False,
    )
    written = run(cfg)
    print("\nGenerated:")
    for env, path in written.items():
        print(f"  {env}: {path}")
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
