"""
Tracing setup — two instrumentation layers.

LAYER 1: Cross-app context propagation (manual, in app.py)
  - The /ask endpoint manually extracts the W3C traceparent header and attaches
    the browser's trace context, so all child spans share the browser's trace_id.
  - NOTE: FastAPIInstrumentor is intentionally NOT used here — its ASGI middleware
    spans are rejected by Arize's ingestion pipeline (see commit f6e0106 in the
    cross-app-tracing-demo for the same root cause fix).

LAYER 2: LLM calls (OpenAIInstrumentor)
  - Auto-instruments all OpenAI client calls → LLM spans with prompt/completion.

LAYER 3: Browser span proxy (/telemetry endpoint)
  - Browser POSTs its root span data here.
  - We recreate it in the backend TracerProvider so it appears as the trace root
    in Arize alongside the backend LLM spans.

Exports to Arize via arize.otel.register() (wraps OTLPSpanExporter + BatchSpanProcessor).
"""

import os
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.trace import SpanContext, TraceFlags, SpanKind, Status, StatusCode
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope


def setup_tracing(project_name: str = "claude-cross-app") -> None:
    """
    Initialize Arize tracing.

    Args:
        project_name: Arize project name
    """
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("Warning: ARIZE_SPACE_ID and ARIZE_API_KEY not set. Tracing disabled.")
        return

    from arize.otel import Transport

    # Sets up TracerProvider with BatchSpanProcessor + OTLPSpanExporter → Arize.
    # Resource attributes (service.name, arize.space_id, arize.project.name) are
    # added automatically by arize.otel.register().
    # Using HTTP transport + explicit full path avoids two issues:
    #   1. gRPC sync channel causes StatusCode.INTERNAL inside uvicorn's asyncio loop
    #   2. arize-otel default endpoint ("https://otlp.arize.com/v1") silently discards
    #      spans — Arize returns 200 OK but only ingests at "/v1/traces"
    register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
        transport=Transport.HTTP,
        endpoint="https://otlp.arize.com/v1/traces",
    )

    # Instrument all OpenAI client calls → LLM spans with prompt/completion content
    OpenAIInstrumentor().instrument()

    print(f"Arize tracing initialized for project: {project_name}")


# ---------------------------------------------------------------------------
# Browser span proxy — recreates the frontend root span in the backend
# TracerProvider so it appears in Arize as the trace root.
# ---------------------------------------------------------------------------

def _get_sdk_internals():
    """Access the active span processor and resource from the TracerProvider."""
    provider = trace_api.get_tracer_provider()
    for candidate in [provider, getattr(provider, "_real_provider", None)]:
        if candidate and hasattr(candidate, "_active_span_processor"):
            return candidate._active_span_processor, candidate.resource
    return None, None


def create_browser_span(
    trace_id_hex: str,
    span_id_hex: str,
    name: str,
    start_time_ms: float,
    end_time_ms: float,
    http_status_code: int = 200,
) -> bool:
    """
    Recreate a browser span with its original trace_id/span_id so it appears
    as the root of the trace in Arize.

    Returns True if the span was successfully queued for export.
    """
    processor, resource = _get_sdk_internals()
    if processor is None or resource is None:
        print("Warning: TracerProvider not ready — browser span dropped.")
        return False

    ctx = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )

    status = (
        Status(StatusCode.OK)
        if http_status_code < 400
        else Status(StatusCode.ERROR, f"HTTP {http_status_code}")
    )

    span = ReadableSpan(
        name=name,
        context=ctx,
        parent=None,
        kind=SpanKind.CLIENT,
        resource=resource,
        attributes={
            "service.origin": "browser",
            "http.status_code": http_status_code,
            "openinference.span.kind": "CHAIN",
        },
        start_time=int(start_time_ms * 1_000_000),
        end_time=int(end_time_ms * 1_000_000),
        status=status,
        instrumentation_scope=InstrumentationScope("frontend-browser"),
    )

    processor.on_end(span)
    return True
