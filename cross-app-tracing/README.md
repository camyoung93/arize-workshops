# Claude App Crossing — Cross-App Tracing Demo

A minimal demo of W3C distributed tracing across a browser frontend and a FastAPI backend, with all spans exported to Arize under the project **`claude-cross-app`**.

```
Browser (index.html)                     Backend (FastAPI)                  Arize
────────────────────────────────────────────────────────────────────────────────
Click "Ask AI"
  │
  ├─ generate traceparent                                             trace root
  │  (trace_id + span_id)                                                 │
  │                                                                        │
  ├─ POST /ask  ──────────────────────► FastAPIInstrumentor          server span
  │   headers: traceparent                extracts traceparent             │
  │   body: { message, event_type,                                         │
  │           element_id, timestamp }     llm.ask span ──────────────  llm span
  │                                         │                              │
  │                                       OpenAI call ──────────────── LLM span
  │                                         │
  │◄──────────────────────────────────── response
  │
  └─ POST /telemetry ──────────────────► recreate browser span ──── trace root
     (trace_id, span_id, timing)
```

## Setup

```bash
cd cross-app-tracing

# 1. Copy and fill in credentials
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the backend
cd backend
uvicorn app:app --reload --port 8000
```

## Run the frontend

Open `frontend/index.html` directly in a browser (no build step needed):

```bash
open frontend/index.html
```

Type a question and click **Ask AI**. The traceparent generated for each click is shown below the response.

## What to look for in Arize

Go to **app.arize.com → project `claude-cross-app`** and open **Traces**:

1. **Trace root**: `browser.click → ask-btn` — this span was created in the browser and proxied via `/telemetry`. Its `trace_id` matches all backend spans.
2. **HTTP server span**: `POST /ask` — created by FastAPIInstrumentor, child of the browser span.
3. **`llm.ask` span**: custom span with `frontend.event_type`, `frontend.element_id`, `frontend.timestamp_ms` attributes.
4. **OpenAI LLM span**: created by `OpenAIInstrumentor` — includes model name, prompt, completion, token counts.

All four spans share the same `trace_id`, forming a single distributed trace that starts in the browser and ends at the LLM.

## Project structure

```
cross-app-tracing/
├── backend/
│   ├── app.py          # FastAPI app — /ask, /telemetry, /health
│   └── tracing.py      # arize.otel.register + OpenAIInstrumentor + browser span proxy
├── frontend/
│   └── index.html      # Single-file UI — traceparent generation + fetch
├── .env.example
├── requirements.txt
└── README.md
```
