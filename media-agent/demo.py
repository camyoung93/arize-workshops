#!/usr/bin/env python3
"""
Media Agent Demo
================
Multi-agent pipeline: NL question → SQL → SQLite → synthesized answer.
Instrumented with Arize AX for full trace observability.

Usage:
  python demo.py "What was total revenue in Q2 2024?"
  python demo.py "What was total revenue in Q2 2024?" --role analyst   # denied
  python demo.py --demo --prompt-version v1
  python demo.py --demo --prompt-version mixed --role finance
  python demo.py --demo --prompt-version mixed --count 30
"""

import argparse
import asyncio
import json
import os
import random
import time
from uuid import uuid4

# Instrumentation must run before importing agents (registers Arize, ADK + Gen AI instrumentors).
import instrumentation  # noqa: F401, E402

# ── ADK session management and agent ───────────────────────────────────────────
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents import (
    _current_session_id,
    _prompt_version,
    _user_role,
    ROLE_PERMISSIONS,
    pipeline_agent,
)

_session_service = InMemorySessionService()
APP_NAME = os.environ.get("ADK_APP_NAME", "media-agent-demo")
USER_ID  = os.environ.get("ADK_USER_ID", "demo-user")
DEFAULT_SESSION_ID = "default"
DEMO_BATCH_SESSION_ID = "demo-session"
DEMO_TRACES_PER_SESSION = 4  # Every N traces share one session_id in demo mode

# Root agent: SequentialAgent with five stages (classify → plan → guardrail → retrieve → synthesize).
root_agent = pipeline_agent

# ── Demo query bank ───────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # Summarization stress tests
    {
        "question": (
            "Which authors in the Technology beat wrote the most articles in "
            "quarters where digital ad revenue exceeded $2M?"
        ),
    },
    {
        "question": "Compare the trajectory of print ad revenue vs newsletter-driven traffic in 2024",
    },
    {
        "question": "Break down Q4 2024 revenue by segment and rank by growth rate vs Q4 2023",
    },
    {
        "question": (
            "What's the correlation between article output in the Markets section "
            "and organic traffic over 2024?"
        ),
    },
    {
        "question": (
            "Summarize the top 3 revenue segments by total 2024 earnings "
            "and their quarter-over-quarter trends"
        ),
    },
    # Instruction-following stress tests
    {
        "question": (
            "In exactly 2 sentences, explain the relationship between "
            "article output and page views by section"
        ),
    },
    {
        "question": "Summarize 2024 revenue trends but exclude any mention of the events segment",
    },
    {
        "question": (
            "Explain Q3 traffic patterns as if briefing a non-technical CMO — "
            "no jargon, no raw numbers over 1 million"
        ),
    },
    {
        "question": (
            "List the top 5 authors by article count, then separately give a "
            "1-sentence summary of overall content production trends. "
            "Do not combine these into one paragraph."
        ),
    },
    {
        "question": (
            "Give me a 3-bullet summary of subscription revenue trends. "
            "Each bullet must start with the quarter name."
        ),
    },
    # Simple baselines
    {"question": "How many articles were published in 2024?"},
    {"question": "What was total revenue in Q2 2024?"},
    {"question": "Which author has the most articles?"},
    {"question": "What's the average daily page views from organic traffic?"},
    {"question": "List all revenue segments"},
    # Guardrail tests (revenue queries that get denied for non-finance roles)
    {
        "question": "Show me the quarterly revenue breakdown for digital ads vs subscriptions",
        "role": "analyst",
    },
    {
        "question": "What was the total licensing revenue across all quarters?",
        "role": "restricted",
    },
    {
        "question": "List every author and how many articles they wrote",
        "role": "restricted",
    },
    # Edge cases
    {
        "question": "How did the company's market share compare to Reuters in Q4 2024?",
        "extra_tags": ["edge-case", "missing-data"],
    },
    {
        "question": "Summarize article performance for January 2022",
        "extra_tags": ["edge-case", "out-of-range"],
    },
    {
        "question": (
            "What is the average word count per article broken down by author seniority level?"
        ),
        "extra_tags": ["edge-case", "schema-mismatch"],
    },
]


# ── Pipeline orchestration (ADK Runner + Agent, instrumented by GoogleADKInstrumentor) ─

async def run_pipeline(
    question: str,
    query_index: int,
    prompt_version: str,
    extra_tags: list[str] | None = None,
    session_id: str | None = None,
    role: str = "finance",
) -> str:
    """Run the five-stage pipeline via ADK Runner.

    Spans are created by OpenInference GoogleADKInstrumentor (AGENT + TOOL/CHAIN).
    Tool spans are augmented with CHAIN kind and custom attributes in agents.py.
    The guardrail stage creates GUARDRAIL + child TOOL/CHAIN spans for access control.
    LLM calls use using_prompt_template in agents.py and tools.py.

    If session_id is provided, that session is reused (caller must have created it).
    If session_id is None, a new session is created with a unique ID.
    """
    if session_id is None:
        session = await _session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=f"query-{query_index:03d}-{uuid4().hex[:6]}",
        )
        session_id = session.id

    _current_session_id.set(session_id)
    _prompt_version.set(prompt_version)
    _user_role.set(role)

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=_session_service,
    )
    content = types.Content(role="user", parts=[types.Part(text=question)])
    try:
        events_async = runner.run_async(
            session_id=session_id,
            user_id=USER_ID,
            new_message=content,
        )
        final_answer = ""
        async for event in events_async:
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_answer = part.text.strip()
                        break
        return final_answer or "(no response)"
    except Exception as e:
        return f"(Error: {e})"


def _detect_constraints(question: str) -> bool:
    """Return True if the question contains explicit formatting/exclusion instructions."""
    markers = [
        "in exactly", "do not", "exclude", "as if", "no jargon",
        "no raw numbers", "each bullet", "separately", "1-sentence", "2 sentences",
    ]
    q_lower = question.lower()
    return any(m in q_lower for m in markers)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Media multi-agent demo with Arize AX tracing."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural language question to answer (omit when using --demo).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the full built-in demo query bank.",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2", "mixed"],
        default="mixed",
        help=(
            "Synthesizer prompt to use. "
            "v1 = strict (default), v2 = intentionally weaker, "
            "mixed = random 50/50 (best for showing Arize comparisons)."
        ),
    )
    parser.add_argument(
        "--role",
        choices=sorted(ROLE_PERMISSIONS.keys()),
        default="finance",
        help=(
            "User role for the access-control guardrail. "
            "finance = full access (default), analyst/editor = no revenue, "
            "restricted = articles only. In --demo mode this is the default; "
            "some demo queries override it to demonstrate denials."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Cap the number of demo queries to run (default: all).",
    )
    return parser.parse_args()


async def _run_single(question: str, prompt_version: str, role: str) -> None:
    pv = prompt_version if prompt_version != "mixed" else random.choice(["v1", "v2"])
    print(f"\n{'─' * 70}")
    print(f"Question : {question}")
    print(f"Prompt   : {pv}")
    print(f"Role     : {role}")
    default_session = await _session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=DEFAULT_SESSION_ID,
    )
    answer = await run_pipeline(
        question=question,
        query_index=1,
        prompt_version=pv,
        session_id=default_session.id,
        role=role,
    )
    print(f"\nAnswer:\n{answer}")


async def _run_demo(prompt_version: str, count: int | None, default_role: str) -> None:
    queries = DEMO_QUERIES
    if count:
        queries = queries[:count]

    print(f"\nRunning {len(queries)} demo queries  (prompt-version={prompt_version}, default-role={default_role})\n")

    current_session = None
    for i, entry in enumerate(queries, start=1):
        if (i - 1) % DEMO_TRACES_PER_SESSION == 0:
            current_session = await _session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=f"{DEMO_BATCH_SESSION_ID}-{uuid4().hex}",
            )

        question   = entry["question"]
        extra_tags = entry.get("extra_tags")
        role       = entry.get("role", default_role)

        pv = prompt_version
        if pv == "mixed":
            pv = "v1" if i % 2 == 1 else "v2"

        print(f"\n[{i:02d}/{len(queries):02d}] ── {pv.upper()} / {role.upper()} ────────────────────")
        print(f"Q: {question}")

        answer = await run_pipeline(
            question=question,
            query_index=i,
            prompt_version=pv,
            extra_tags=extra_tags,
            session_id=current_session.id,
            role=role,
        )
        print(f"A: {answer[:400]}{'...' if len(answer) > 400 else ''}")

        if i < len(queries):
            delay = random.uniform(2, 4)
            time.sleep(delay)

    print(
        "\n✓ Demo complete.\n"
        "View traces in Arize AX → project 'media-agent-demo'\n\n"
        "What to explore:\n"
        "  • Filter synthesizer.prompt_version = 'v1.0' vs 'v2.0' to compare quality\n"
        "  • Sort by brand_voice.score to find lowest-scoring answers\n"
        "  • Filter retriever.empty_result = True to see edge cases\n"
        "  • Filter planner.retries > 0 to find SQL validation failures\n"
        "  • Group by query.complexity to compare latency across difficulty levels\n"
        "  • Filter guardrail.result = 'denied' to see access-control blocks\n"
        "  • Filter guardrail.role to compare behavior across roles\n"
        f"  • Session view: every {DEMO_TRACES_PER_SESSION} traces share one session (unique UUID per batch)\n"
    )


def main() -> None:
    args = _parse_args()

    if args.demo:
        asyncio.run(_run_demo(args.prompt_version, args.count, args.role))
    elif args.question:
        asyncio.run(_run_single(args.question, args.prompt_version, args.role))
    else:
        print("Provide a question or use --demo. Run with -h for help.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
