# Cross-App Tracing Workshop

Minimal demo of **frontend + backend OpenTelemetry tracing** with
[Arize AX](https://arize.com/docs/ax). A single trace flows from a
browser click through a FastAPI backend into an OpenAI LLM call.

Two files. No Docker. No npm.

## Quick Start

```bash
cd cross-app-tracing
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

## Two Types of Instrumentation

### Type 1: Backend (`app.py`)

- `arize.otel.register()` + `OpenAIInstrumentor` capture the LLM call
- Manual `traceparent` extraction via `opentelemetry.propagate.extract()`
  links every LLM span to the browser's trace
- `/telemetry` endpoint receives the browser span and recreates it in
  the backend's TracerProvider so it appears in Arize as the trace root

### Type 2: Frontend (`index.html`)

- Vanilla JS generates a random trace ID + span ID
- Builds a W3C `traceparent` header and attaches it to every fetch
- After the response, POSTs the span data to `/telemetry`

### How they connect

This follows the standard
[OpenTelemetry Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
pattern — the same pattern used between any two microservices:

1. **Inject** — Browser builds `traceparent` and sends it on the fetch
2. **Extract** — Backend reads `traceparent` and sets it as the active
   context so the OpenAI span becomes a child of the browser's trace
3. **Proxy** — Browser POSTs span metadata to `/telemetry`; backend
   recreates the span with the exact IDs and exports it to Arize

## Trace structure in Arize

```
Frontend: POST /chat     ← root (browser span, proxied via /telemetry)
  └─ ChatCompletion      ← OpenAI LLM call
```

## References

- [Arize: Advanced Tracing — Manual Context Propagation](https://arize.com/docs/ax/observe/tracing/configure/advanced-tracing-otel-examples)
- [OpenTelemetry: Context Propagation](https://opentelemetry.io/docs/concepts/context-propagation/)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
