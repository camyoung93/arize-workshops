"""
FastAPI backend for the Claude App Crossing demo.

Cross-app tracing flow:
  1. Browser generates a W3C traceparent (trace_id + span_id) on click
  2. Browser POSTs to /ask with `traceparent` in the request headers + event metadata in body
  3. /ask manually extracts the traceparent with OTel's propagate.extract() and attaches
     the browser's trace context — this is the cross-app link
  4. llm.ask span and the auto-instrumented OpenAI spans become children of the browser trace
  5. Browser POSTs its root span to /telemetry so it appears as the trace root in Arize

NOTE: FastAPIInstrumentor is intentionally NOT used. Its ASGI middleware spans are rejected
by Arize's ingestion pipeline, causing the entire export batch to fail. Manual extraction
(propagate.extract + context.attach) achieves the same cross-app linking without the issue.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract as extract_context
from pydantic import BaseModel

from tracing import create_browser_span, setup_tracing

from dotenv import load_dotenv

load_dotenv(override=True)
# override=True: ensures .env values take precedence over OS env vars (e.g. stale
# arize-otel defaults cached in shell environment can cause auth failures).

# Must call setup_tracing() BEFORE trace.get_tracer() so the real TracerProvider
# is set as the global provider before any tracer is acquired.
setup_tracing()

app = FastAPI(title="Claude App Crossing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracer = trace.get_tracer(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    message: str
    event_type: str = "click"
    element_id: str = "ask-btn"
    timestamp: float = 0.0


class AskResponse(BaseModel):
    response: str


class FrontendSpan(BaseModel):
    trace_id: str
    span_id: str
    name: str
    start_time_ms: float
    end_time_ms: float
    http_status_code: int = 200


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, raw_request: Request):
    """
    Process a user question that originated from a browser click event.

    Manually extracts the W3C traceparent header so that llm.ask and all
    OpenAI child spans become part of the browser-initiated trace.
    """
    # Cross-app link: extract the browser's traceparent and attach it as the
    # active OTel context so all child spans share the browser's trace_id.
    ctx = extract_context(carrier=dict(raw_request.headers))
    token = otel_context.attach(ctx)

    try:
        with tracer.start_as_current_span("llm.ask") as span:
            # Attach frontend event metadata so it's visible in Arize alongside the LLM spans
            span.set_attribute("frontend.event_type", body.event_type)
            span.set_attribute("frontend.element_id", body.element_id)
            span.set_attribute("frontend.timestamp_ms", body.timestamp)

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a helpful assistant. The user triggered a "
                            f"'{body.event_type}' event on element '{body.element_id}'. "
                            "Answer their question concisely (2-3 sentences max)."
                        ),
                    },
                    {"role": "user", "content": body.message},
                ],
            )

            return AskResponse(response=completion.choices[0].message.content or "")
    finally:
        otel_context.detach(token)


@app.post("/telemetry")
async def receive_telemetry(span_data: FrontendSpan):
    """
    Proxy the browser's root span into Arize.

    The browser can't export directly to Arize, so it sends its span metadata
    here and we recreate it with the original trace_id/span_id so it appears
    as the root of the distributed trace.
    """
    ok = create_browser_span(
        trace_id_hex=span_data.trace_id,
        span_id_hex=span_data.span_id,
        name=span_data.name,
        start_time_ms=span_data.start_time_ms,
        end_time_ms=span_data.end_time_ms,
        http_status_code=span_data.http_status_code,
    )
    return {"ok": ok}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
