#!/usr/bin/env python3
"""
Run the Media Agent pipeline on the experiment dataset and capture
run records for Arize experiment upload.

Reusable: run this each time you want to create a new experiment.

Outputs:
  .scratch/experiment_runs_<timestamp>.json   — run records for Arize experiment
  Traces are automatically sent to Arize via the existing instrumentation.

Usage:
    python -m experiments.run_experiment_batch
    python -m experiments.run_experiment_batch --count 10          # first 10 only (dry-run)
    python -m experiments.run_experiment_batch --prompt-version v1 # force v1 synth prompt
    python -m experiments.run_experiment_batch --output runs.json  # custom output path
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Ensure the media-agent directory is on sys.path so we can import
# the app modules (instrumentation, agents, demo) regardless of how the script
# is invoked.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Instrumentation must run before importing agents.
import instrumentation  # noqa: F401, E402

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents import (
    _current_session_id,
    _prompt_version,
    _user_role,
    pipeline_agent,
)

DATASET_PATH = Path(__file__).parent / "data" / "experiment_dataset.json"
SCRATCH_DIR = _APP_DIR.parent.parent / ".scratch"
APP_NAME = os.environ.get("ADK_APP_NAME", "media-agent-experiment")
USER_ID = os.environ.get("ADK_USER_ID", "experiment-runner")

_session_service = InMemorySessionService()


async def run_single(
    question: str,
    query_index: int,
    prompt_version: str,
    role: str,
) -> tuple[str, float]:
    """Run one question through the pipeline. Returns (answer, latency_ms)."""
    session = await _session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=f"eval-{query_index:03d}-{uuid4().hex[:6]}",
    )

    _current_session_id.set(session.id)
    _prompt_version.set(prompt_version)
    _user_role.set(role)

    runner = Runner(
        app_name=APP_NAME,
        agent=pipeline_agent,
        session_service=_session_service,
    )
    content = types.Content(role="user", parts=[types.Part(text=question)])

    t0 = time.perf_counter()
    final_answer = ""
    try:
        events_async = runner.run_async(
            session_id=session.id,
            user_id=USER_ID,
            new_message=content,
        )
        async for event in events_async:
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_answer = part.text.strip()
                        break
    except Exception as e:
        final_answer = f"(Error: {e})"
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract planner SQL from session state
    plan_result = session.state.get("plan_result", {})
    sql_generated = plan_result.get("sql", "")
    sql_plan = plan_result.get("plan", "")
    tables = json.dumps(plan_result.get("tables", []))

    return final_answer or "(no response)", latency_ms, sql_generated, sql_plan, tables


async def run_batch(
    dataset: list[dict],
    prompt_version: str,
    count: int | None = None,
) -> list[dict]:
    """Run the full dataset (or first `count` rows) and return run records."""
    examples = dataset[:count] if count else dataset
    runs = []

    for i, ex in enumerate(examples, start=1):
        question = ex["question"]
        role = ex.get("role", "finance")
        pv = prompt_version

        print(f"[{i:03d}/{len(examples):03d}] {ex['category']:>22s} | {question[:70]}...")

        answer, latency_ms, sql, plan, tables = await run_single(
            question=question,
            query_index=i,
            prompt_version=pv,
            role=role,
        )

        runs.append({
            "example_id": ex["id"],
            "output": answer,
            "sql_generated": sql,
            "sql_plan": plan,
            "tables_referenced": tables,
            "prompt_version": pv,
            "latency_ms": round(latency_ms, 1),
            "category": ex["category"],
            "difficulty": ex["difficulty"],
            "error": None if not answer.startswith("(Error") else answer,
        })

        print(f"       {latency_ms:>7.0f}ms | {answer[:80]}{'...' if len(answer) > 80 else ''}")

        if i < len(examples):
            await asyncio.sleep(0.5)

    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run the Media Agent experiment dataset batch."
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Only run the first N examples (default: all).",
    )
    parser.add_argument(
        "--prompt-version", choices=["v1", "v2", "mixed"], default="v1",
        help="Synthesizer prompt version (default: v1).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for run records JSON (default: .scratch/experiment_runs_<ts>.json).",
    )
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples from {DATASET_PATH}")
    if args.count:
        print(f"Running first {args.count} only.")

    runs = asyncio.run(run_batch(dataset, args.prompt_version, args.count))

    if args.output:
        out_path = Path(args.output)
    else:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SCRATCH_DIR / f"experiment_runs_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(runs, f, indent=2)

    errors = sum(1 for r in runs if r.get("error"))
    print(f"\nDone. {len(runs)} runs written to {out_path}")
    if errors:
        print(f"  {errors} runs had errors.")


if __name__ == "__main__":
    main()
