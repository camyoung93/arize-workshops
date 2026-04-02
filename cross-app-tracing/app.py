"""
Cross-App Tracing Demo — minimal example.

INSTRUMENTATION TYPE 1 (Backend):
  - OpenAIInstrumentor captures LLM call spans
  - Manual traceparent extraction links them to the browser's trace

INSTRUMENTATION TYPE 2 (Frontend):
  - Vanilla JS generates trace/span IDs and injects traceparent header
  - POSTs span data to /telemetry so the browser span appears in Arize

Run: python app.py → open http://localhost:8000
"""

import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from opentelemetry.propagate import extract as extract_context
from opentelemetry import context as otel_context, trace as trace_api
from opentelemetry.trace import SpanContext, TraceFlags, SpanKind, Status, StatusCode
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope


# ---------------------------------------------------------------------------
# Tracing setup — runs at module level (outside async context) so the gRPC
# channel is created in the main thread. batch=False means each span exports
# individually, so no init-time span can poison a batch.
# ---------------------------------------------------------------------------

def _init_tracing():
    from arize.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    project = os.getenv("ARIZE_PROJECT_NAME", "cross-app-tracing-workshop")

    if not space_id or not api_key:
        print("Warning: ARIZE_SPACE_ID / ARIZE_API_KEY not set — tracing disabled")
        return

    register(space_id=space_id, api_key=api_key, project_name=project, batch=False)
    OpenAIInstrumentor().instrument()
    print(f"Tracing initialized → project: {project}")

_init_tracing()


# ---------------------------------------------------------------------------
# Telemetry proxy — recreates the browser span so it appears in Arize
# ---------------------------------------------------------------------------

def create_browser_span(trace_id_hex, span_id_hex, name, start_ms, end_ms, status_code=200):
    provider = trace_api.get_tracer_provider()
    real = getattr(provider, "_real_provider", provider)
    processor = getattr(real, "_active_span_processor", None)
    resource = getattr(real, "resource", None)
    if not processor or not resource:
        return False

    ctx = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    status = Status(StatusCode.OK) if status_code < 400 else Status(StatusCode.ERROR)
    span = ReadableSpan(
        name=name,
        context=ctx,
        parent=None,
        kind=SpanKind.CLIENT,
        resource=resource,
        attributes={"service.origin": "browser", "openinference.span.kind": "CHAIN"},
        start_time=int(start_ms * 1_000_000),
        end_time=int(end_ms * 1_000_000),
        status=status,
        instrumentation_scope=InstrumentationScope("frontend-browser"),
    )
    processor.on_end(span)
    return True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI()


class ChatRequest(BaseModel):
    message: str

class TelemetryData(BaseModel):
    trace_id: str
    span_id: str
    name: str
    start_time_ms: float
    end_time_ms: float
    http_status_code: int = 200


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(req: ChatRequest, raw: Request):
    # Extract traceparent so OpenAI span joins the browser's trace
    ctx = extract_context(carrier=dict(raw.headers))
    token = otel_context.attach(ctx)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
                {"role": "user", "content": req.message},
            ],
        )
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        otel_context.detach(token)
        trace_api.get_tracer_provider().force_flush()


@app.post("/telemetry")
async def telemetry(data: TelemetryData):
    ok = create_browser_span(
        data.trace_id, data.span_id, data.name,
        data.start_time_ms, data.end_time_ms, data.http_status_code,
    )
    trace_api.get_tracer_provider().force_flush()
    return {"ok": ok}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
