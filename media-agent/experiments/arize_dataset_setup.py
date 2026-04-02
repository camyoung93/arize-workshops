#!/usr/bin/env python3
"""
One-time: create or update the Arize dataset from the experiment dataset JSON.

Reads experiments/data/experiment_dataset.json and uploads it as an Arize dataset
named 'media-agent-experiment'. Supports both the ax CLI path and the
Python SDK.

Usage (SDK path):
    python -m experiments.arize_dataset_setup

Usage (ax CLI path — preferred for demos):
    ax datasets create \
      --name "media-agent-experiment" \
      --space-id $ARIZE_SPACE_ID \
      --file experiments/data/experiment_dataset.json
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_APP_DIR / ".env")

DATASET_PATH = Path(__file__).parent / "data" / "experiment_dataset.json"
DATASET_NAME = os.environ.get(
    "EXPERIMENT_DATASET_NAME", "media-agent-experiment"
)


def create_dataset_sdk():
    """Create the Arize dataset using the Python SDK (v8+)."""
    try:
        from arize import ArizeClient
    except ImportError:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install arize\n"
            "Or use the ax CLI path instead (see --help)."
        )
        sys.exit(1)

    api_key = os.environ.get("ARIZE_API_KEY")
    space_id = os.environ.get("ARIZE_SPACE_ID")
    if not api_key or not space_id:
        print("Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env or environment.")
        sys.exit(1)

    with open(DATASET_PATH) as f:
        examples = json.load(f)

    for ex in examples:
        # Arize reserves "id" as a platform-managed column — move ours to "example_id"
        if "id" in ex:
            ex["example_id"] = ex.pop("id")
        for key in ("must_include", "must_not_include"):
            if isinstance(ex.get(key), list):
                ex[key] = json.dumps(ex[key])

    client = ArizeClient(api_key=api_key)
    dataset = client.datasets.create(
        name=DATASET_NAME,
        space=space_id,
        examples=examples,
    )
    print(f"Dataset created: {dataset.id}")
    print(f"  Name: {DATASET_NAME}")
    print(f"  Examples: {len(examples)}")
    return dataset


def main():
    print(f"Creating Arize dataset '{DATASET_NAME}' from {DATASET_PATH}")
    create_dataset_sdk()
    print(
        "\nAlternative (ax CLI):\n"
        f"  ax datasets create --name \"{DATASET_NAME}\" "
        f"--space-id $ARIZE_SPACE_ID --file experiments/data/experiment_dataset.json"
    )


if __name__ == "__main__":
    main()
