# Media Agent Demo

A multi-agent pipeline that answers natural language questions about media
company data by generating SQL, executing it against a local SQLite database, and
synthesizing a final answer — all fully traced in **Arize AX**.

**Structure:** A **SequentialAgent** root with **five custom BaseAgent stages** that run in a fixed order. Data flows via **session state** (no LLM-driven delegation): classifier_stage → planner_stage → guardrail_stage → retriever_stage → synthesizer_stage. Each stage calls the corresponding pipeline function and writes its result into `ctx.session.state` for the next stage. The guardrail stage enforces role-based access control before any data is retrieved.

**Instrumentation:** All tracing is configured in **`instrumentation.py`** (imported before agents):

- **GoogleADKInstrumentor** — auto-instruments the ADK (agent and tool spans from the Runner).
- **GoogleGenAIInstrumentor** — auto-instruments the Google Gen AI SDK so every `generate_content` call gets an LLM span ([OpenInference Google GenAI](https://arize-ai.github.io/openinference/python/instrumentation/openinference-instrumentation-google-genai/)). Included in requirements; run `pip install -r requirements.txt` to ensure it is installed.
- **OpenInferenceSpanProcessor** — attached for optional normalization of span attributes.

We still use **`with_prompt_template`** around each LLM call so template, version, and variables are attached to the current (LLM) span for the Prompt Playground. Manual child spans for `validate_sql` and `review_brand_voice` remain so those steps appear in the trace.

```
NL question
    │
    ▼ media_pipeline (SequentialAgent — from GoogleADKInstrumentor)
    ├─ classifier_stage (BaseAgent)  → classify_query()            → state: classification
    ├─ planner_stage (BaseAgent)     → plan_query()                → state: plan_result
    │    └─ validate_sql (inside plan_query) → TOOL span per attempt
    ├─ guardrail_stage (BaseAgent)   → check_access_guardrail()    → state: guardrail
    │    ├─ resolve_user_role       → TOOL span (auth service lookup)
    │    └─ verify_table_permissions → CHAIN span (role vs. tables check)
    ├─ retriever_stage (BaseAgent)   → retrieve_data()             → state: data  (skipped if denied)
    └─ synthesizer_stage (BaseAgent) → synthesize_answer()         → final Event  (access-denied msg if denied)
         └─ review_brand_voice (inside synthesize_answer) → TOOL span per round
```

---

## Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI API enabled
- `gcloud auth application-default login` (or a service account key)
- An [Arize AX](https://app.arize.com) account

---

## Setup

```bash
# 1. Clone / navigate to this directory
cd media-agent

# 2. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your Arize credentials and GCP project ID.
# Set GOOGLE_GENAI_USE_VERTEXAI=true so the ADK agent uses Vertex AI (no Gemini API key needed).
# Optional: GEMINI_MODEL, ADK_APP_NAME, ADK_USER_ID, MEDIA_DB_PATH (see .env.example).

# 5. Seed the SQLite database (~1,800 rows across 4 tables)
python seed_db.py
```

---

## Running the demo

**Single question (default role: `finance` — full access):**

```bash
python demo.py "What was total revenue in Q2 2024?"
python demo.py "Which author has the most articles in the Technology section?"
```

**Guardrail scenarios — passing vs. denied requests:**

```bash
# finance role has access to all tables — guardrail passes
python demo.py "What was total revenue in Q2 2024?" --role finance

# analyst role cannot access the revenue table — guardrail denies
python demo.py "What was total revenue in Q2 2024?" --role analyst

# editor role can query articles + authors + traffic — guardrail passes
python demo.py "Which author has the most articles?" --role editor

# restricted role can only query articles — authors table denied
python demo.py "Which author has the most articles?" --role restricted

# restricted role on a multi-table join — both revenue and authors denied
python demo.py "Which Technology authors wrote the most in quarters where digital ads exceeded 2M?" --role restricted
```

**Full demo mode (21 built-in queries including 3 guardrail denial tests):**

```bash
python demo.py --demo
```

**Force a specific synthesizer prompt version:**

```bash
python demo.py --demo --prompt-version v1    # strict, instruction-following
python demo.py --demo --prompt-version v2    # intentionally weaker (for comparison)
python demo.py --demo --prompt-version mixed # 50/50 random (best for Arize comparison)
```

**Override the default role for the full demo run:**

```bash
python demo.py --demo --role analyst   # most queries denied (no revenue access)
python demo.py --demo --role finance   # all queries pass (default)
```

**Limit to N queries:**

```bash
python demo.py --demo --prompt-version mixed --count 10
```

---

## What to explore in Arize AX

Open **project: media-agent-demo** → Traces.

### Agent waterfall

Each trace shows the full span tree: media_pipeline (SequentialAgent) →
classifier_stage → planner_stage (with nested validate_sql) → retriever_stage →
synthesizer_stage (with nested review_brand_voice). Latency at each stage is visible at a glance.

### Compare v1 vs v2 synthesizer prompts

Filter: `synthesizer.prompt_version = "v1.0"` vs `"v2.0"`

v2 is intentionally weaker: no explicit instruction-following rules, uses
vague "comprehensive" framing. Run summarization or instruction-following
evals on synthesizer spans filtered by prompt version to quantify the gap.

### Brand voice quality

- Sort by `brand_voice.score` ascending → find the lowest-quality answers
- Filter `synthesizer.brand_voice_passed_first_draft = false` → traces that
needed a revision round
- Filter `synthesizer.revision_count > 0` → see how often v2 requires fixes

### SQL validation retries

- Filter `planner.retries > 0` → queries that required SQL regeneration
- The `validate_sql` child spans show the exact error on each attempt

### Edge cases

- Filter `retriever.empty_result = true` → competitor data, out-of-range dates
- Filter `tag.tags contains "schema-mismatch"` → seniority column query

### Complexity distribution

- Group by `query.complexity` → compare latency and brand voice scores across
simple / aggregation / multi_hop / constrained queries

### Session grouping

- **Demo runs** batch every 4 traces into one session; each batch gets a unique session ID (UUID), so each session groups 4 queries in the Session view.
- **Single-question runs** use the default session (`session.id = "default"`).
- Filter by `session.id` to see all spans for a given session; filter by `user.id = "demo-user"` to see every query in the demo run

---

## Suggested evaluations

Run in Arize AX → Evaluations after collecting traces:

1. **Instruction following**: On synthesizer spans where
  `query.has_formatting_constraints = true`, eval whether the answer
   respects the constraint. Compare pass rate by `synthesizer.prompt_version`.
2. **Summarization quality**: On `query.complexity = multi_hop` spans, eval
  whether the answer correctly synthesizes data across multiple tables.
3. **Empty-result handling**: On `retriever.empty_result = true` spans, eval
  whether the synthesizer acknowledges the data gap vs. hallucinating.
4. **Brand voice regression**: Track `brand_voice.score` over time to detect
  prompt drift or model behavior changes.

---

## Database schema


| Table    | Rows  | Key columns                                         |
| -------- | ----- | --------------------------------------------------- |
| articles | ~200  | title, section, publish_date, author_id, word_count |
| revenue  | 40    | quarter, year, segment, amount_usd                  |
| traffic  | ~1825 | date, page_views, unique_visitors, source, section  |
| authors  | 20    | name, beat, articles_published, join_date           |


Revenue trends baked in: digital_ads +8% QoQ, print_ads -5% QoQ,
subscriptions +1% QoQ (plateauing). Traffic shows Monday–Thursday peaks.

---

## Experiment pipeline

A full experiment harness lives in `experiments/`. It tests SQL generation and
final answer synthesis across 145 diverse questions.

### One-time setup

```bash
# 1. Generate the experiment dataset (145 questions with golden SQL + answers)
python -m experiments.build_experiment_dataset

# 2. Upload dataset to Arize (SDK path)
python -m experiments.arize_dataset_setup

# Or via ax CLI:
ax datasets create --name "media-agent-experiment" \
  --space-id $ARIZE_SPACE_ID --file experiments/data/experiment_dataset.json
```

### Run an experiment

```bash
# Full batch (all 145 questions → traced + run records saved)
python -m experiments.run_experiment_batch

# Dry-run (first 10 only)
python -m experiments.run_experiment_batch --count 10

# Force synthesizer prompt version
python -m experiments.run_experiment_batch --prompt-version v2
```

### Upload experiment to Arize

```bash
# SDK path
python -m experiments.arize_experiment_ops --runs .scratch/experiment_runs_*.json

# Or via ax CLI:
ax experiments create --name "media-experiment-v1" \
  --dataset-id <DATASET_ID> --file .scratch/experiment_runs_*.json
```

### Create evaluators and run scoring

Use the **arize-evaluator** AX skill or see `experiments/EVALUATORS.md` for
ready-to-use `ax evaluators create` commands covering:

1. **SQL Quality** — compares generated SQL vs golden SQL
2. **Response Quality** — scores final answer vs golden answer (rubric-based)
3. **Constraint Following** — checks formatting/exclusion compliance

See `experiments/EVALUATORS.md` for full column mappings and task setup.

---

## File structure

```
media-agent/
├── demo.py            # CLI, instrumentation setup, pipeline orchestration
├── agents.py          # Prompt templates + per-stage agent logic
├── tools.py           # schema_lookup, validate_sql, execute_sql, review_brand_voice
├── prompt_utils.py    # with_prompt_template helper for Arize prompt template/variables/version
├── seed_db.py         # Creates and populates data/media.db
├── requirements.txt
├── .env.example
├── data/
│   └── media.db   (created by seed_db.py)
└── experiments/
    ├── build_experiment_dataset.py  # One-time: generate 145-question dataset
    ├── arize_dataset_setup.py      # One-time: upload dataset to Arize
    ├── run_experiment_batch.py     # Reusable: run pipeline + capture outputs
    ├── arize_experiment_ops.py     # Reusable: create Arize experiments from runs
    ├── schemas.py                  # Typed row/run schemas
    ├── EVALUATORS.md               # Evaluator templates + column mapping reference
    └── data/
        └── experiment_dataset.json (created by build_experiment_dataset.py)
```

