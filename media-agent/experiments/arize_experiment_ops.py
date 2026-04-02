#!/usr/bin/env python3
"""
Reusable: create an Arize experiment from collected run records.

Reads a runs JSON file (output of run_experiment_batch.py) and uploads it as
an experiment linked to the experiment dataset.

Supports both Python SDK and ax CLI paths.

Usage (SDK path):
    python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_20260330_120000.json

Usage (ax CLI path — preferred for demos):
    ax experiments create \
      --name "media-experiment-v1-run1" \
      --dataset-id <DATASET_ID> \
      --file .scratch/experiment_runs_20260330_120000.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_APP_DIR / ".env")

DATASET_NAME = os.environ.get(
    "EXPERIMENT_DATASET_NAME", "media-agent-experiment"
)


def _remap_example_ids(runs: list[dict], client, dataset_name: str, space_id: str) -> list[dict]:
    """Map local example_ids (eval_001, etc.) to Arize platform UUIDs.

    The platform assigns its own IDs when the dataset is created. Our run
    records reference the local IDs, so we need to translate them by fetching
    the dataset examples and matching on the 'example_id' user field.
    """
    try:
        resp = client.datasets.list_examples(dataset=dataset_name, space=space_id, all=True)
        examples = resp.examples or []
    except Exception:
        resp = client.datasets.list_examples(dataset=dataset_name, space=space_id)
        examples = resp.examples or []

    id_map = {}
    for ex in examples:
        props = getattr(ex, "additional_properties", {}) or {}
        local_id = props.get("example_id", "")
        if local_id and hasattr(ex, "id"):
            id_map[local_id] = ex.id

    if not id_map:
        print("  WARNING: could not build ID mapping — uploading with local IDs.")
        return runs

    mapped = 0
    for run in runs:
        old_id = run["example_id"]
        if old_id in id_map:
            run["example_id"] = id_map[old_id]
            mapped += 1

    print(f"  Mapped {mapped}/{len(runs)} example_ids to platform UUIDs.")
    return runs


def create_experiment_sdk(
    runs_path: str,
    experiment_name: str,
    dataset_name: str | None = None,
):
    """Create an Arize experiment from a runs JSON file using the SDK."""
    try:
        from arize import ArizeClient
        from arize.experiments.types import ExperimentTaskFieldNames
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install arize\n"
            "Or use the ax CLI path instead."
        )
        sys.exit(1)

    api_key = os.environ.get("ARIZE_API_KEY")
    space_id = os.environ.get("ARIZE_SPACE_ID")
    if not api_key or not space_id:
        print("Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env or environment.")
        sys.exit(1)

    with open(runs_path) as f:
        runs = json.load(f)

    ds_name = dataset_name or DATASET_NAME
    client = ArizeClient(api_key=api_key)

    # Remap local IDs to platform UUIDs if needed
    first_id = runs[0].get("example_id", "") if runs else ""
    if first_id and not _is_uuid(first_id):
        print("  Local example_ids detected — remapping to platform UUIDs...")
        runs = _remap_example_ids(runs, client, ds_name, space_id)

    # Strip null values (API rejects them)
    runs = [{k: v for k, v in run.items() if v is not None} for run in runs]

    experiment, results_df = client.experiments.create(
        name=experiment_name,
        dataset=ds_name,
        space=space_id,
        experiment_runs=runs,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id",
            output="output",
        ),
    )
    print(f"Experiment created: {experiment.id}")
    print(f"  Name: {experiment_name}")
    print(f"  Dataset: {ds_name}")
    print(f"  Runs: {len(runs)}")
    return experiment


def _is_uuid(s: str) -> bool:
    """Quick check if a string looks like a UUID."""
    import re
    return bool(re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", s))


def main():
    parser = argparse.ArgumentParser(
        description="Create an Arize experiment from run records."
    )
    parser.add_argument(
        "--runs", required=True,
        help="Path to experiment_runs_*.json file.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Experiment name (default: auto-generated).",
    )
    parser.add_argument(
        "--dataset", default=None,
        help=f"Dataset name or ID (default: {DATASET_NAME}).",
    )
    args = parser.parse_args()

    experiment_name = args.name
    if not experiment_name:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"media-experiment-{ts}"

    print(f"Creating experiment '{experiment_name}' from {args.runs}")
    create_experiment_sdk(
        runs_path=args.runs,
        experiment_name=experiment_name,
        dataset_name=args.dataset,
    )

    print(
        "\nAlternative (ax CLI):\n"
        f"  ax experiments create --name \"{experiment_name}\" "
        f"--dataset-id <DATASET_ID> --file {args.runs}"
    )
    print(
        "\nNext: use AX skills to create evaluators and run tasks:\n"
        "  1. Create evaluators via arize-evaluator skill\n"
        "  2. Create task: ax tasks create --dataset-id <ID> "
        "--experiment-ids <EXP_ID> ...\n"
        "  3. Trigger run: ax tasks trigger-run <TASK_ID> "
        "--experiment-ids <EXP_ID> --wait"
    )


if __name__ == "__main__":
    main()
