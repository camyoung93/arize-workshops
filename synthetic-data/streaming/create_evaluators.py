#!/usr/bin/env python3
"""
Register the StreamFlix search-augment LLM-as-judge evaluators via the Python
SDK (v8). This REGISTERS the judge templates in the space only — it does not
run them and does not attach scores to spans. Wire them to spans with
create_tasks.py, then trigger runs with run_evals.py (or let the continuous
tasks score new traces automatically).

Prerequisites:
  - arize >= 8.0.0
  - ARIZE_API_KEY, ARIZE_SPACE_ID, ARIZE_AI_INTEGRATION_NAME set in .env
  - An AI integration configured at app.arize.com → Space settings → AI Integrations

Usage:
  python create_evaluators.py
  python create_evaluators.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from arize import ArizeClient
from arize.evaluators.types import TemplateConfig, EvaluatorLlmConfig
from arize._generated.api_client.models import InvocationParams, ProviderParams
from arize._generated.api_client.exceptions import ConflictException


# data_granularity: "SPAN"/"TRACE"/"SESSION" -> that level (uppercase enum
# values per arize >= 8.43; older 8.x used lowercase).
EVALUATORS: list[dict] = [
    # ── Span-level (personalize_recommendations LLM span) ────────────────────
    {
        "name": "StreamFlix - Recommendation Relevance",
        "template_name": "recommendation_relevance",
        "commit_message": "Personalized picks must address the user's request from the candidate list",
        "classification_choices": {"relevant": 1, "irrelevant": 0},
        "data_granularity": "SPAN",
        "template": """You evaluate whether a streaming-service recommendation response is relevant to the user's request.

The LLM input contains the user's request, their profile/watch history, and the reranked candidate list:
{input}

The response:
{output}

RELEVANT: the response recommends 2-3 titles drawn from the reranked candidate list, and those picks address the user's stated request (genre, mood, occasion).
IRRELEVANT: the response ignores the request, recommends titles that are not in the candidate list, or fails to make concrete picks.

Respond with exactly one of these labels: relevant, irrelevant""",
    },
    {
        "name": "StreamFlix - Recommendation Hallucination",
        "template_name": "recommendation_hallucination",
        "commit_message": "Response must not fabricate titles or user watch history",
        "classification_choices": {"grounded": 1, "hallucinated": 0},
        "data_granularity": "SPAN",
        "template": """You perform a grounding check on a streaming-service recommendation response.

The LLM input contains the user's request, their profile/watch history, and the reranked candidate list:
{input}

The response:
{output}

GROUNDED: every recommended title appears in the reranked candidate list, and every claim about the user (recent watches, preferences) appears in the provided profile/watch history.
HALLUCINATED: the response invents a title not in the candidate list, or fabricates details of the user's watch history or preferences not present in the input.

Respond with exactly one of these labels: grounded, hallucinated""",
    },
    # ── Trace-level (root CHAIN span) ─────────────────────────────────────────
    {
        "name": "StreamFlix - Pipeline Trajectory",
        "template_name": "pipeline_trajectory",
        "commit_message": "End-to-end: the search-augment pipeline resolves the user's request",
        "classification_choices": {"resolved": 1, "unresolved": 0},
        "data_granularity": "TRACE",
        "template": """You judge whether the StreamFlix search-augment pipeline resolved the user's request end to end.

User request: {input}
Final response: {output}

RESOLVED: the user gets a concrete, on-request set of 2-3 recommendations they could act on immediately.
UNRESOLVED: the response is off-request, vague, empty, or gives no actionable picks.

Respond with exactly one of these labels: resolved, unresolved""",
    },
    # ── Session-level ─────────────────────────────────────────────────────────
    {
        "name": "StreamFlix - Session Quality",
        "template_name": "session_quality",
        "commit_message": "Across a session, recommendations stay coherent and non-contradictory",
        "classification_choices": {"coherent": 1, "incoherent": 0},
        "data_granularity": "SESSION",
        "template": """You evaluate the quality of a multi-turn streaming-recommendation session.

Within a session, each turn's picks should address that turn's request, and turns should not contradict each other (e.g. praising a title in one turn and dismissing it in another, or describing the same user's preferences inconsistently).

Conversation (all turns in the session): {conversation}

COHERENT: every turn's recommendations address that turn's request and the session is mutually consistent.
INCOHERENT: a turn's picks do not match its request, or turns contradict each other.

Respond with exactly one of these labels: coherent, incoherent""",
    },
]


def resolve_ai_integration_id(client: ArizeClient, space: str, name: str) -> str:
    resp = client.ai_integrations.list(space=space, name=name)
    matches = [i for i in resp.ai_integrations if i.name == name]
    if not matches:
        available = ", ".join(sorted({i.name for i in resp.ai_integrations})) or "<none>"
        raise SystemExit(
            f"ERROR: No AI integration named '{name}' in space {space}.\n"
            f"  Available: {available}\n"
            f"  Set one up at app.arize.com → Space settings → AI Integrations,\n"
            f"  then update ARIZE_AI_INTEGRATION_NAME in .env."
        )
    return matches[0].id


def main():
    parser = argparse.ArgumentParser(description="Register the StreamFlix search-augment evaluators")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without creating")
    args = parser.parse_args()

    api_key = os.environ.get("ARIZE_API_KEY")
    space = os.environ.get("ARIZE_SPACE_ID")
    integ_name = os.environ.get("ARIZE_AI_INTEGRATION_NAME")
    model = os.environ.get("ARIZE_EVALUATOR_MODEL", "gpt-4o")

    if not api_key or not space or not integ_name:
        print("ERROR: Set ARIZE_API_KEY, ARIZE_SPACE_ID, and ARIZE_AI_INTEGRATION_NAME in .env.")
        sys.exit(1)

    if args.dry_run:
        print(f"DRY RUN — would create {len(EVALUATORS)} evaluators in space {space}")
        for cfg in EVALUATORS:
            print(f"  [{cfg['data_granularity']:>7}] {cfg['name']} ({cfg['classification_choices']})")
        return

    client = ArizeClient(api_key=api_key)
    integration_id = resolve_ai_integration_id(client, space, integ_name)

    print(f"Creating {len(EVALUATORS)} StreamFlix evaluators...")
    print(f"  Space:       {space}")
    print(f"  Integration: {integ_name} ({integration_id})")
    print(f"  Model:       {model}\n")

    created = skipped = 0
    for idx, cfg in enumerate(EVALUATORS, start=1):
        print(f"[{idx}/{len(EVALUATORS)}] {cfg['name']}", end=" ")
        try:
            client.evaluators.create_template_evaluator(
                name=cfg["name"],
                space=space,
                commit_message=cfg["commit_message"],
                template_config=TemplateConfig(
                    name=cfg["template_name"],
                    template=cfg["template"],
                    include_explanations=True,
                    use_function_calling_if_available=True,
                    classification_choices=cfg["classification_choices"],
                    direction="MAXIMIZE",
                    data_granularity=cfg["data_granularity"],
                    llm_config=EvaluatorLlmConfig(
                        ai_integration_id=integration_id,
                        model_name=model,
                        invocation_parameters=InvocationParams(temperature=0),
                        provider_parameters=ProviderParams(),
                    ),
                ),
            )
            print("created")
            created += 1
        except ConflictException:
            print("already exists, skipping")
            skipped += 1
        except ValueError as e:
            # Known arize SDK <-> server drift: the evaluator IS created server-side,
            # but the success response carries newer fields the pinned generated
            # model can't deserialize, raising on response parse. Treat as created.
            if "EvaluatorVersion" in str(e) or "No match found when deserializing" in str(e):
                print("created (server ok; response parse skipped)")
                created += 1
            else:
                raise

    print(f"\n=== Done: {created} created, {skipped} skipped ({len(EVALUATORS)} total) ===")
    print("Wire them to spans with create_tasks.py — nothing is auto-scored here.")


if __name__ == "__main__":
    main()
