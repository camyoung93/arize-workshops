# Cross-App Tracing Workshop

Minimal demo of **frontend-to-backend OpenTelemetry tracing** with
[Arize AX](https://arize.com/docs/ax). When a user sends a chat
message, a single trace flows from the browser through the backend
and into an OpenAI LLM call — all visible in Arize as one connected
trace.

Two files. No Docker. No npm. No framework.

## Quick Start

```bash
cd cross-app-tracing
python -m venv .venv && source .venv/bin/activate
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

## What Happens When You Send a Message

```
 Browser (index.html)                    Server (app.py)
 ────────────────────                    ───────────────
 1. User clicks "Send"

 2. JS generates a unique
    trace ID + span ID
    (per W3C Trace Context spec)

 3. JS calls fetch("/chat") with        4. FastAPI receives the request
    header: traceparent: 00-{traceId}
            -{spanId}-01                 5. extract(headers) reads the
                                            traceparent and sets it as
                                            the active OTel context

                                         6. OpenAI client.chat.completions
                                            .create() runs — the OpenAI
                                            Instrumentor creates a
                                            "ChatCompletion" span that
                                            automatically inherits the
                                            browser's traceId

                                         7. Response sent back
 8. JS receives the response
                                         
 9. JS fire-and-forgets a POST
    to /telemetry with the span     →   10. /telemetry recreates the
    metadata (traceId, spanId,              browser span with the exact
    name, timing)                           same IDs and exports it to
                                            Arize via the same
                                            TracerProvider
```

The end result in Arize: both spans share the same trace ID, and the
browser span is the root with the LLM span nested underneath.

## Trace Structure in Arize

```
Frontend: POST /chat     ← root (browser span)
  └─ ChatCompletion      ← OpenAI LLM call
```

## How the Instrumentation Works

### Frontend — the initiator (`index.html`)

The browser **starts** the trace. It runs first.

- Vanilla JS generates a unique 16-byte trace ID and 8-byte span ID
  using `crypto.getRandomValues()` (the same way OTel SDKs do it)
- Formats them into a
  [W3C `traceparent`](https://www.w3.org/TR/trace-context/) header:
  `00-{traceId}-{spanId}-01`
- Attaches the header to the `fetch("/chat")` call
- After the response arrives, POSTs the span data (IDs + timing) to
  `/telemetry` so the browser span appears in Arize as the trace root

### Backend — the continuer (`app.py`)

The backend **continues** the trace that the browser started.

- `arize.otel.register()` sets up the TracerProvider that exports
  spans to Arize
- `OpenAIInstrumentor` automatically wraps every OpenAI API call in
  a span
- The `/chat` endpoint calls `opentelemetry.propagate.extract()` to
  read the `traceparent` header and set it as the active context —
  this is the critical line that links the OpenAI span to the
  browser's trace
- The `/telemetry` endpoint receives the browser's span data and
  recreates it with the exact trace/span IDs using the same
  TracerProvider, so it appears in Arize alongside the backend spans

### Why the backend proxies the browser span

Browsers can't send spans directly to Arize (no gRPC from the
browser, and API keys shouldn't be exposed client-side). So the
browser sends its span metadata to the backend's `/telemetry`
endpoint, which recreates the span and exports it through the same
pipeline that handles the OpenAI spans.

## Key Code Locations

| What | Where | Lines |
|---|---|---|
| Trace ID + span ID generation | `index.html` | `randomHex()` function |
| `traceparent` injection on fetch | `index.html` | `tracedFetch()` function |
| Span reporting to `/telemetry` | `index.html` | end of `tracedFetch()` |
| `traceparent` extraction | `app.py` | `extract_context(carrier=dict(raw.headers))` in `/chat` |
| OpenAI call (auto-instrumented) | `app.py` | `client.chat.completions.create()` in `/chat` |
| Browser span proxy | `app.py` | `create_browser_span()` called from `/telemetry` |
| Tracing init (deferred to startup) | `app.py` | `init_tracing()` in lifespan |

## References

- [Arize: Advanced Tracing — Manual Context Propagation](https://arize.com/docs/ax/observe/tracing/configure/advanced-tracing-otel-examples) (Section 1)
- [OpenTelemetry: Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
