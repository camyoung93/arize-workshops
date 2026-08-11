#!/usr/bin/env python3
"""
StreamFlix Search Augment — Synthetic Trace Generator

Emits OpenInference traces for a StreamFlix search-augment recommendation
pipeline. All content is synthetic — no retrieval system, reranker, or LLM is
ever invoked; the OpenInference attributes are written directly so what lands
in Arize reads like a live app.

Topology (one trace per user query):

  StreamFlix Search Augment Pipeline        [CHAIN, root]
    ├── candidate_retrieval                  [RETRIEVER]  catalog first-pull with scores
    ├── rerank_candidates                    [RERANKER]   top picks passed to the LLM
    └── search_augment_agent                 [AGENT]
          ├── fetch_user_history             [RETRIEVER]  user profile / watch history
          └── personalize_recommendations    [LLM]        personalized picks

Traces are grouped into sessions of 2-5 (session.id on the CHAIN/AGENT/LLM
spans) so trace- and session-level evaluators have something to chew on.
Evaluations are NOT attached here — register LLM-as-judge evaluators with
create_evaluators.py and wire them to spans with create_tasks.py.

Usage:
  python generate_traces.py --test                 # one trace, session test_session_001
  python generate_traces.py                         # 500 traces in random sessions
  python generate_traces.py --count 20              # smaller batch
  python generate_traces.py --project my-project    # custom project
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.status import Status, StatusCode

from templates.world import BEDROCK_MODEL_ID, SEARCH_AUGMENT_SCENARIOS, get_search_augment_data
from templates.prompts import format_llm_input

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID)

SESSION_SIZE_MIN, SESSION_SIZE_MAX = 2, 5


def create_search_augment_trace(tracer, query_id=None, session_id=None):
    """
    Create a single StreamFlix search-augment trace:
    CHAIN -> RETRIEVER (candidate_retrieval) -> RERANKER (rerank_candidates)
    -> AGENT (search_augment_agent) -> RETRIEVER (fetch_user_history) + LLM
    (personalize_recommendations).
    """
    data = get_search_augment_data(query_id)
    query = data["query"]
    retrieved_docs = data["retrieved_docs"]
    reranked_docs = data["reranked_docs"]
    response = data["response"]
    user_history = data.get("user_history", "")
    user_profile = data.get("user_profile", {})
    llm_input = format_llm_input(query, reranked_docs, user_history)

    retrieval_latency_ms = random.uniform(80, 300)
    reranker_latency_ms = random.uniform(50, 200)
    agent_latency_ms = random.uniform(30, 150)
    user_history_latency_ms = random.uniform(30, 120)
    llm_latency_ms = random.uniform(1200, 3500)
    chain_overhead_ms = random.uniform(20, 80)
    chain_latency_ms = (retrieval_latency_ms + reranker_latency_ms + agent_latency_ms
                        + user_history_latency_ms + llm_latency_ms + chain_overhead_ms)

    prompt_tokens, completion_tokens = 450, 120

    with tracer.start_as_current_span("StreamFlix Search Augment Pipeline") as chain_span:
        chain_span.set_attribute("openinference.span.kind", "CHAIN")
        chain_span.set_attribute("input.value", query)
        chain_span.set_attribute("input.mime_type", "text/plain")
        if session_id:
            chain_span.set_attribute("session.id", session_id)
        if user_profile.get("user_id"):
            chain_span.set_attribute("user.id", user_profile["user_id"])
        time.sleep(0.003)

        with tracer.start_as_current_span("candidate_retrieval") as ret_span:
            ret_span.set_attribute("openinference.span.kind", "RETRIEVER")
            ret_span.set_attribute("input.value", json.dumps({"query": query}))
            ret_span.set_attribute("input.mime_type", "application/json")
            for idx, doc in enumerate(retrieved_docs):
                ret_span.set_attribute(f"retrieval.documents.{idx}.document.id", doc["id"])
                ret_span.set_attribute(f"retrieval.documents.{idx}.document.content", doc["content"])
                ret_span.set_attribute(f"retrieval.documents.{idx}.document.score", doc["score"])
            ret_span.set_attribute("output.value", json.dumps([{"id": d["id"], "score": d["score"]} for d in retrieved_docs]))
            ret_span.set_attribute("output.mime_type", "application/json")
            ret_span.set_attribute("latency_ms", round(retrieval_latency_ms, 2))
            ret_span.set_status(Status(StatusCode.OK))
            time.sleep(0.002)

        # Human-readable reranker input/output so Arize renders them (it mainly
        # shows input.value / output.value).
        def _doc_label(d):
            c = d.get("content", "")
            return c.split(":")[0].strip() if ":" in c else (c[:50] + "..." if len(c) > 50 else c)
        reranker_input_value = "\n".join(
            [f"User query: {query}", "", "Candidates from retrieval (id, title, score):"]
            + [f"  • {d['id']}: {_doc_label(d)} — {d['score']}" for d in retrieved_docs]
        )
        reranker_output_value = "\n".join(
            ["Reranked top picks (passed to LLM for personalization):"]
            + [f"  {i+1}. {_doc_label(d)} — score {d['score']}" for i, d in enumerate(reranked_docs)]
        )

        with tracer.start_as_current_span("rerank_candidates") as rerank_span:
            rerank_span.set_attribute("openinference.span.kind", "RERANKER")
            rerank_span.set_attribute("input.value", reranker_input_value)
            rerank_span.set_attribute("input.mime_type", "text/plain")
            rerank_span.set_attribute("output.value", reranker_output_value)
            rerank_span.set_attribute("output.mime_type", "text/plain")
            rerank_span.set_attribute("reranker.model_name", "synthetic-reranker-v1")
            rerank_span.set_attribute("latency_ms", round(reranker_latency_ms, 2))
            rerank_span.set_status(Status(StatusCode.OK))
            time.sleep(0.002)

        with tracer.start_as_current_span("search_augment_agent") as agent_span:
            agent_span.set_attribute("openinference.span.kind", "AGENT")
            agent_span.set_attribute("agent.name", "search_augment_agent")
            agent_span.set_attribute("input.value", json.dumps({
                "request": "personalize_recommendations", "query": query,
                "task": "retrieve -> rerank -> personalize",
            }))
            agent_span.set_attribute("input.mime_type", "application/json")
            if session_id:
                agent_span.set_attribute("session.id", session_id)
            agent_span.set_attribute("latency_ms", round(agent_latency_ms, 2))
            time.sleep(0.002)

            with tracer.start_as_current_span("fetch_user_history") as user_hist_span:
                user_hist_span.set_attribute("openinference.span.kind", "RETRIEVER")
                user_hist_span.set_attribute("input.value", json.dumps({
                    "session_id": session_id, "user_id": user_profile.get("user_id", "unknown"),
                }))
                user_hist_span.set_attribute("input.mime_type", "application/json")
                user_hist_span.set_attribute("output.value", user_history)
                user_hist_span.set_attribute("output.mime_type", "text/plain")
                user_hist_span.set_attribute("latency_ms", round(user_history_latency_ms, 2))
                user_hist_span.set_status(Status(StatusCode.OK))
                time.sleep(0.001)

            with tracer.start_as_current_span("personalize_recommendations") as llm_span:
                if session_id:
                    llm_span.set_attribute("session.id", session_id)
                llm_span.set_attribute("openinference.span.kind", "LLM")
                llm_span.set_attribute("llm.model_name", MODEL_ID)
                llm_span.set_attribute("llm.provider", "aws")
                llm_span.set_attribute("llm.system", "anthropic")
                llm_span.set_attribute("llm.input_messages.0.message.role", "user")
                llm_span.set_attribute("llm.input_messages.0.message.content", llm_input)
                llm_span.set_attribute("llm.output_messages.0.message.role", "assistant")
                llm_span.set_attribute("llm.output_messages.0.message.content", response)
                llm_span.set_attribute("llm.token_count.prompt", prompt_tokens)
                llm_span.set_attribute("llm.token_count.completion", completion_tokens)
                llm_span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
                # Untruncated: the Hallucination judge grounds the answer against
                # the user-history block embedded in this input.
                llm_span.set_attribute("input.value", llm_input)
                llm_span.set_attribute("output.value", response)
                llm_span.set_attribute("latency_ms", round(llm_latency_ms, 2))
                llm_span.set_status(Status(StatusCode.OK))

            agent_span.set_attribute("output.value", response)
            agent_span.set_attribute("output.mime_type", "text/plain")

        chain_span.set_attribute("output.value", response)
        chain_span.set_attribute("output.mime_type", "text/plain")
        chain_span.set_attribute("latency_ms", round(chain_latency_ms, 2))
        chain_span.set_status(Status(StatusCode.OK))

    return {"query": query, "response": response, "query_id": data["query_id"],
            "session_id": session_id, "user_id": user_profile.get("user_id")}


def setup_tracing(project_name: str) -> str:
    space = os.environ.get("ARIZE_SPACE_ID")
    key = os.environ.get("ARIZE_API_KEY")
    if not (space and key):
        raise SystemExit(
            "ERROR: No tracing destination configured. Set ARIZE_SPACE_ID and "
            "ARIZE_API_KEY in .env."
        )
    resource = Resource.create({
        "model_id": project_name,
        "openinference.project.name": project_name,
        "service.name": project_name,
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(GrpcOTLPExporter(
        endpoint="otlp.arize.com:443",
        headers={"space_id": space, "api_key": key},
    )))
    trace.set_tracer_provider(provider)
    return f"Arize (space {space[:8]}...)"


def main():
    parser = argparse.ArgumentParser(description="Generate StreamFlix search-augment synthetic traces")
    parser.add_argument("--count", type=int, default=500,
                        help="Number of traces (default: 500)")
    parser.add_argument("--test", action="store_true", help="Single trace test mode")
    parser.add_argument("--project", default=os.environ.get("ARIZE_PROJECT_NAME", "streamflix_search_augment"),
                        help="Arize project name (env: ARIZE_PROJECT_NAME)")
    args = parser.parse_args()

    destination = setup_tracing(args.project)
    tracer = trace.get_tracer("streamflix_search_augment_synthetic", "1.0.0")

    print(f"Tracing -> {destination}")
    print(f"Project: {args.project}")
    print(f"Model on spans: {MODEL_ID}")

    if args.test:
        print("Generating 1 test trace (session test_session_001)...")
        sd = create_search_augment_trace(tracer, query_id=0, session_id="test_session_001")
        print(f"  query:    {sd['query']}")
        print(f"  response: {sd['response'][:100]}...")
        count = 1
    else:
        count = args.count
        print(f"Generating {count:,} traces (CHAIN -> RETRIEVER -> RERANKER -> AGENT -> LLM), "
              f"sessions of {SESSION_SIZE_MIN}-{SESSION_SIZE_MAX}...")
        current_session_id = None
        traces_in_session = 0
        session_size = 0
        for i in range(count):
            if current_session_id is None or traces_in_session >= session_size:
                current_session_id = f"session_{random.randint(100000, 999999)}"
                session_size = random.randint(SESSION_SIZE_MIN, SESSION_SIZE_MAX)
                traces_in_session = 0
            create_search_augment_trace(tracer, query_id=random.randint(0, len(SEARCH_AUGMENT_SCENARIOS) - 1),
                                        session_id=current_session_id)
            traces_in_session += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{count}]")

    print("Flushing spans...")
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=30000)
    print(f"Done. {count} trace(s) sent to project '{args.project}'.")


if __name__ == "__main__":
    main()
