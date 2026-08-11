# StreamFlix Search Augment — Evaluators & Tasks

Four LLM-as-judge evaluators registered by `create_evaluators.py`, wired to
spans by `create_tasks.py`, and triggered by `run_evals.py`. All evaluators use
`direction="maximize"` with 1 = good.

## Span-level (the `personalize_recommendations` LLM span)

| Evaluator | Checks | Choices | Variables |
|---|---|---|---|
| StreamFlix - Recommendation Relevance | Picks come from the reranked candidate list and address the user's request | relevant=1 / irrelevant=0 | `{input}`, `{output}` |
| StreamFlix - Recommendation Hallucination | No invented titles; no fabricated watch history or preferences | grounded=1 / hallucinated=0 | `{input}`, `{output}` |

Span filter: `name = 'personalize_recommendations'`. Both judges read the full
LLM `input.value`, which embeds the user request, the profile/watch-history
block, and the reranked candidate list — no extra column mappings needed.

## Trace-level (root CHAIN span)

| Evaluator | Checks | Choices | Variables |
|---|---|---|---|
| StreamFlix - Pipeline Trajectory | End-to-end: the request produced actionable recommendations | resolved=1 / unresolved=0 | `{input}`, `{output}` |

Span filter: `name = 'StreamFlix Search Augment Pipeline'`. The judge sees the
root span's I/O only (query in, final response out). Note the root of this
pipeline is a **CHAIN** span, not AGENT — `search_augment_agent` is mid-tree,
so filtering on `openinference.span.kind = 'AGENT'` would target the wrong span.

## Session-level (grouped by `session.id`)

| Evaluator | Checks | Choices | Variables |
|---|---|---|---|
| StreamFlix - Session Quality | Turns are coherent and mutually consistent across the session | coherent=1 / incoherent=0 | `{conversation}` |

Same root-span filter as trace-level, with **empty column mappings** — Arize
supplies `{conversation}` from the session's turns. A session evaluator template
must reference `{conversation}` (a plain `{output}` mapping is rejected at
session granularity).

## Caveats

- **Task filters are exact-match only.** `LIKE` silently matches zero rows in
  the eval-task backend. Filters must byte-match the span names emitted by
  `generate_traces.py`.
- **Continuous tasks do not backfill.** Traces emitted before `create_tasks.py`
  ran need a manual trigger via `run_evals.py`.
- `create_evaluators.py` requires an AI integration in the space matching
  `ARIZE_AI_INTEGRATION_NAME` (app.arize.com → Space settings → AI Integrations).
- **arize >= 8.43 required.** The SDK's enum casing changed in 8.43
  (`data_granularity="SPAN"`, `direction="MAXIMIZE"`,
  `task_type="TEMPLATE_EVALUATION"`); older 8.x expected lowercase and will
  reject these values.

## Commands

```bash
python create_evaluators.py --dry-run   # inspect what would be registered
python create_evaluators.py            # register the 4 evaluators
python create_tasks.py                 # create the 3 continuous tasks (sampling 1.0)
python create_tasks.py --no-continuous # backfill-only variant

# Backfill / manual trigger (defaults: LLM Evals, last 1 day, 3000 spans max)
python run_evals.py
python run_evals.py --task "StreamFlix - Trace Evals" --days 7
python run_evals.py --task "StreamFlix - Session Evals"
python run_evals.py --override         # re-score spans that already have labels
```
