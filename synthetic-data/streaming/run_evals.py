#!/usr/bin/env python3
"""
Trigger an Arize evaluation task over a time window of project traces and wait
for results — via the SDK.

The SDK is used directly because the `ax tasks trigger-run` CLI has sent naive
(timezone-less) datetimes that the API rejects; the SDK takes tz-aware datetimes.

Usage:
  python run_evals.py                                          # LLM evals, last day
  python run_evals.py --task "StreamFlix - Trace Evals" --days 7
  python run_evals.py --task "StreamFlix - Session Evals" --override
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from arize import ArizeClient


def _wait(client, run_id, timeout):
    final = client.tasks.wait_for_run(run_id=run_id, timeout=timeout)
    d = final.model_dump() if hasattr(final, "model_dump") else vars(final)
    print(f"\nstatus:    {d.get('status')}")
    print(f"successes: {d.get('num_successes')}")
    print(f"errors:    {d.get('num_errors')}")
    print(f"skipped:   {d.get('num_skipped')}")
    if d.get("num_errors"):
        print("\nSome evaluations errored — open the task in the Arize UI for details.")
    elif d.get("num_successes"):
        print("\nScores written. They appear on the spans (eval index may lag a few minutes).")


def main() -> None:
    p = argparse.ArgumentParser(description="Trigger an Arize eval task (SDK, tz-aware)")
    p.add_argument("--task", default="StreamFlix - LLM Evals", help="task name or base64 id")
    p.add_argument("--days", type=int, default=1, help="lookback window")
    p.add_argument("--max-spans", type=int, default=3000)
    p.add_argument("--override", action="store_true", help="re-score items that already have labels")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    api_key = os.environ.get("ARIZE_API_KEY")
    space = os.environ.get("ARIZE_SPACE_ID")
    if not api_key or not space:
        print("ERROR: Set ARIZE_API_KEY and ARIZE_SPACE_ID in .env.")
        sys.exit(1)

    client = ArizeClient(api_key=api_key)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"Triggering task '{args.task}' over {start.isoformat()} .. {end.isoformat()}")
    run = client.tasks.trigger_run(
        task=args.task, space=space,
        data_start_time=start, data_end_time=end,
        max_spans=args.max_spans, override_evaluations=args.override or None,
    )

    run_id = getattr(run, "id", None)
    print(f"run id: {run_id}")
    if run_id:
        _wait(client, run_id, args.timeout)


if __name__ == "__main__":
    main()
