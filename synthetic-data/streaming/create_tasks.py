#!/usr/bin/env python3
"""
Create the evaluation tasks that apply each registered evaluator to the right
spans / traces / sessions in the streamflix_search_augment project. Idempotent
and space-portable: evaluators are resolved by NAME to id at runtime, and tasks
that already exist are skipped.

Tasks:
  - LLM Evals (span)        Recommendation Relevance + Recommendation
                            Hallucination -> personalize_recommendations
  - Trace Evals (trace)     Pipeline Trajectory -> root CHAIN span
  - Session Evals (session) Session Quality -> root CHAIN span

Usage:
  python create_tasks.py                 # continuous tasks, sampling 1.0
  python create_tasks.py --no-continuous # backfill-only tasks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from arize import ArizeClient

# Span selectors (exact-match only — LIKE silently matches zero rows in the
# eval task backend). The root of this pipeline is the CHAIN span, so trace-
# and session-level tasks target it by its exact name.
_LLM = "name = 'personalize_recommendations'"
_ROOT = "name = 'StreamFlix Search Augment Pipeline'"

_IO = {"input": "attributes.input.value", "output": "attributes.output.value"}

# (task_name, task_query_filter, [(evaluator_name, column_mappings), ...])
PROJECT_TASKS = [
    ("StreamFlix - LLM Evals", _LLM, [
        ("StreamFlix - Recommendation Relevance", dict(_IO)),
        ("StreamFlix - Recommendation Hallucination", dict(_IO)),
    ]),
    ("StreamFlix - Trace Evals", _ROOT, [
        ("StreamFlix - Pipeline Trajectory", dict(_IO)),
    ]),
    ("StreamFlix - Session Evals", _ROOT, [
        # Session evaluators reference {conversation}; Arize supplies the turns.
        ("StreamFlix - Session Quality", {}),
    ]),
]


def _drift_ok(exc: Exception) -> bool:
    """The task is created server-side but the response body carries newer fields
    the pinned generated model can't deserialize (same SDK<->server drift as the
    evaluators)."""
    s = str(exc)
    return "No match found when deserializing" in s or "additional fields" in s


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evaluation tasks for the registered evaluators")
    parser.add_argument("--no-continuous", action="store_true",
                        help="Create backfill-only tasks (default: continuous, sampling 1.0)")
    args = parser.parse_args()

    api_key = os.environ.get("ARIZE_API_KEY")
    space = os.environ.get("ARIZE_SPACE_ID")
    project = os.environ.get("ARIZE_PROJECT_NAME", "streamflix_search_augment")
    if not api_key or not space:
        print("ERROR: Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env.")
        sys.exit(1)

    client = ArizeClient(api_key=api_key)

    # Resolve evaluator name -> id.
    resp = client.evaluators.list(space=space, limit=100)
    evs = getattr(resp, "evaluators", resp)
    by_name = {e.name: e.id for e in evs}
    print(f"Resolved {len(by_name)} evaluators in the space.\n")

    # Existing task names (skip duplicates).
    try:
        tresp = client.tasks.list(project=project, space=space, limit=100)
        existing = {t.name for t in getattr(tresp, "tasks", tresp)}
    except Exception:
        existing = set()

    print(f"Tasks in '{project}':")
    for name, task_filter, evspec in PROJECT_TASKS:
        if name in existing:
            print(f"  exists, skipping: {name}")
            continue
        evaluators, missing = [], []
        for ev_name, mappings in evspec:
            eid = by_name.get(ev_name)
            if not eid:
                missing.append(ev_name)
                continue
            evaluators.append({"evaluator_id": eid, "column_mappings": mappings})
        if missing:
            print(f"  SKIP {name}: missing evaluators {missing} (run create_evaluators.py first)")
            continue
        try:
            # task_type enum is uppercase per arize >= 8.43 (older 8.x used lowercase).
            client.tasks.create_evaluation_task(
                name=name, task_type="TEMPLATE_EVALUATION", project=project, space=space,
                evaluators=evaluators, query_filter=task_filter,
                is_continuous=not args.no_continuous,
                sampling_rate=None if args.no_continuous else 1.0,
            )
            mode = "backfill" if args.no_continuous else "continuous"
            print(f"  created [{mode}]: {name}  ({len(evaluators)} evaluators)")
        except Exception as exc:
            if _drift_ok(exc):
                print(f"  created (server ok; response parse skipped): {name}")
            else:
                print(f"  FAILED {name}: {exc.__class__.__name__}: {str(exc)[:180]}")

    print("\nDone. Continuous tasks score new traces automatically; backfill older "
          "spans with run_evals.py.")


if __name__ == "__main__":
    main()
