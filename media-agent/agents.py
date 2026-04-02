"""
Media Agent — Agent Logic

Five pipeline stages, each as an async function. When run via ADK Runner,
the OpenInference GoogleADKInstrumentor creates the agent and tool spans;
we augment them with CHAIN kind and custom attributes and use
with_prompt_template (prompt_utils) for LLM prompt template/variables/version.

  classify_query           — Gemini call; returns complexity + reasoning
  plan_query               — Gemini + validate_sql retry loop (max 3 attempts)
  check_access_guardrail   — Role-based table access check (GUARDRAIL span)
  retrieve_data            — Pure Python; calls execute_sql, no LLM
  synthesize_answer        — Gemini + brand-voice revision loop (max 2 revisions)

All prompt templates are module-level constants — prospects can edit them
without touching any other code.
"""

import json
import os
import re
from contextvars import ContextVar
from prompt_utils import with_prompt_template
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry import trace

from google import genai

from google.adk.agents import Agent, BaseAgent, InvocationContext, SequentialAgent
from google.adk.events import Event
from google.genai import types

from tools import execute_sql, review_brand_voice, schema_lookup, validate_sql

# Context vars set by demo.py before Runner.run_async() for use in tools
_current_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_prompt_version: ContextVar[str] = ContextVar("prompt_version", default="v2")
_user_role: ContextVar[str] = ContextVar("user_role", default="finance")

# Role-based access control — maps role name to the set of tables the role may query.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "analyst":    {"articles", "authors", "traffic"},
    "finance":    {"articles", "authors", "traffic", "revenue"},
    "editor":     {"articles", "authors", "traffic"},
    "restricted": {"articles"},
}

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Gen AI client (Vertex AI backend); project/location from env (demo sets dotenv first).
_genai_client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
)


# ── Span attribute constants ──────────────────────────────────────────────────
_SPAN_KIND   = SpanAttributes.OPENINFERENCE_SPAN_KIND
_INPUT       = SpanAttributes.INPUT_VALUE
_OUTPUT      = SpanAttributes.OUTPUT_VALUE
_INPUT_MIME  = SpanAttributes.INPUT_MIME_TYPE
_OUTPUT_MIME = SpanAttributes.OUTPUT_MIME_TYPE
_JSON        = "application/json"
_TEXT        = "text/plain"


def _get_tracer():
    """Return the OTel tracer for creating child spans (e.g. validate_sql, review_brand_voice)."""
    return trace.get_tracer_provider().get_tracer(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────
# Edit these freely — they appear verbatim in Arize trace details.

QUERY_CLASSIFIER_TEMPLATE = """
Classify this question into exactly one category:
- simple: single table lookup, count, or filter
- aggregation: GROUP BY, ranking, YoY comparison
- multi_hop: requires joining 2+ tables or sequential reasoning
- constrained: has explicit formatting/exclusion/persona instructions

Question: {question}

Return JSON only (no markdown fences):
{{"complexity": "<simple|aggregation|multi_hop|constrained>", "reasoning": "<one sentence>"}}
"""
QUERY_CLASSIFIER_VERSION = "v1.0"

QUERY_PLANNER_TEMPLATE = """
Given the database schema:
{schema}

Query complexity classification: {complexity}

Generate SQL to answer: {question}

Return JSON only (no markdown fences) with these exact keys:
{{
  "sql": "<SQL query string>",
  "plan": "<1-2 sentences: what data you're retrieving and why>",
  "tables": ["<table1>", ...]
}}

Rules:
- Use standard SQLite syntax
- Always alias aggregated columns (e.g. SUM(amount_usd) AS total_revenue)
- Use explicit JOIN ... ON syntax, not implicit comma joins
- If the question references data that does not exist in the schema,
  generate the closest reasonable query and note the gap in the plan field
"""
QUERY_PLANNER_VERSION = "v1.0"

QUERY_PLANNER_RETRY_TEMPLATE = """
The SQL query you generated was invalid. Please fix it.

Original question: {question}

Previous SQL:
{sql}

Validation error:
{error}

Database schema for reference:
{schema}

Return JSON only (no markdown fences):
{{
  "sql": "<corrected SQL query>",
  "plan": "<updated 1-2 sentence plan>",
  "tables": ["<table1>", ...]
}}
"""
QUERY_PLANNER_RETRY_VERSION = "v1.0"

SYNTHESIZER_TEMPLATE_V1 = """
Original question: {question}

Retrieval plan: {plan}

Raw data:
{data}

Instructions:
1. Answer the question using ONLY the data provided above.
2. If the question includes formatting constraints (sentence count, exclusions,
   persona, structure), follow them exactly.
3. If asked to exclude a topic, do not mention it at all — not even to say
   you are excluding it.
4. Keep your answer concise. Prefer specific numbers over vague trends.
5. If the data is insufficient to fully answer, say what you can and note
   what is missing.

Formatting constraints from the question (if any): {constraints}
"""
SYNTHESIZER_V1_VERSION = "v1.0"

SYNTHESIZER_TEMPLATE_V2 = """
You are a helpful analyst. Given the following question and data, provide a
comprehensive answer.

Question: {question}
Plan: {plan}
Data: {data}

Additional context: {constraints}

Please be thorough in your response.
"""
SYNTHESIZER_V2_VERSION = "v2.0"

SYNTHESIZER_REVISION_TEMPLATE = """
Your previous answer did not fully meet editorial standards.

Original question: {question}

Your previous answer:
{draft}

Editorial review feedback:
{issues}

Specific revision guidance:
{revision_notes}

Raw data (for reference):
{data}

Please write an improved answer that addresses the feedback. Follow the same
formatting constraints as before: {constraints}
"""
SYNTHESIZER_REVISION_VERSION = "v1.0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = text.strip()
    if "```" in text:
        for part in text.split("```")[1::2]:
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise ValueError(f"Cannot parse JSON from LLM response: {text[:300]}")


def _extract_constraints(question: str) -> str:
    """Identify explicit formatting/exclusion constraints in the question."""
    patterns = [
        r"in exactly \d+ sentences?",
        r"do not mention",
        r"exclude[s]? (any mention of )?",
        r"as if briefing",
        r"no jargon",
        r"no raw numbers",
        r"each bullet must",
        r"separately",
        r"do not combine",
    ]
    found = [p.strip("()[]?\\") for p in patterns if re.search(p, question, re.IGNORECASE)]
    return "; ".join(found) if found else "none"


def _get_full_schema() -> str:
    """Return formatted CREATE TABLE statements for all four tables."""
    parts = []
    for table in ["articles", "revenue", "traffic", "authors"]:
        result = schema_lookup(table)
        if result.get("status") == "success":
            parts.append(result["schema_ddl"])
        else:
            parts.append(f"-- Table {table} unavailable: {result.get('error', '?')}")
    return "\n\n".join(parts)


# ── Stage 1: Query Classifier ─────────────────────────────────────────────────

async def classify_query(question: str) -> dict:
    """Classify query complexity via a single Gemini call.

    Returns:
        dict: complexity (str), reasoning (str)
    """
    prompt = QUERY_CLASSIFIER_TEMPLATE.format(question=question)

    result = None
    for attempt in range(3):
        with with_prompt_template(
            template=QUERY_CLASSIFIER_TEMPLATE,
            variables={"question": question},
            version=QUERY_CLASSIFIER_VERSION,
        ):
            response = await _genai_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
        if not (getattr(response, "text", None) or "").strip():
            continue
        try:
            result = _parse_json(response.text)
            break
        except (ValueError, json.JSONDecodeError):
            continue
    if not result:
        # Fallback so pipeline continues; user still gets an answer
        return {
            "complexity": "simple",
            "reasoning": "Model returned empty after 3 retries; defaulting to simple.",
        }
    return {
        "complexity": result.get("complexity", "simple"),
        "reasoning":  result.get("reasoning", ""),
    }


# ── Stage 2: Query Planner ────────────────────────────────────────────────────

async def plan_query(question: str, complexity: str, tracer=None, session_id: str | None = None) -> dict:
    """Generate SQL with up to 3 validation attempts.

    Child spans for each validate_sql call are nested under the caller's
    active span (ADK instrumentor tool span or manual parent).

    Returns:
        dict: sql, plan, tables, retries, validation_passed_first_try, validation
    """
    session_id = session_id or _current_session_id.get() or ""
    tracer = tracer or _get_tracer()
    schema = _get_full_schema()

    # Initial SQL generation (retry on empty or unparseable LLM response)
    prompt = QUERY_PLANNER_TEMPLATE.format(
        schema=schema, complexity=complexity, question=question
    )
    parsed = None
    for gen_attempt in range(3):
        with with_prompt_template(
            template=QUERY_PLANNER_TEMPLATE,
            variables={"schema": schema, "question": question, "complexity": complexity},
            version=QUERY_PLANNER_VERSION,
        ):
            response = await _genai_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
        if not (getattr(response, "text", None) or "").strip():
            continue
        try:
            parsed = _parse_json(response.text)
            break
        except (ValueError, json.JSONDecodeError):
            continue
    if not parsed:
        # Fallback so pipeline continues; retriever will return no rows, synthesizer can explain
        parsed = {
            "sql": "SELECT 1 LIMIT 0",
            "plan": "Could not generate query; model returned empty after 3 retries.",
            "tables": [],
        }
    sql    = parsed.get("sql", "")
    plan   = parsed.get("plan", "")
    tables = parsed.get("tables", [])

    retries    = 0
    validation = {}

    for attempt in range(3):
        with tracer.start_as_current_span("validate_sql") as vs_span:
            vs_span.set_attribute(_SPAN_KIND, "TOOL")
            vs_span.set_attribute("tool.name", "validate_sql")
            vs_span.set_attribute("session.id", session_id)
            vs_span.set_attribute(_INPUT, sql)

            validation = validate_sql(sql)

            vs_span.set_attribute(_OUTPUT, json.dumps(validation))
            vs_span.set_attribute(_OUTPUT_MIME, _JSON)
            vs_span.set_attribute("validation.valid",        validation["valid"])
            vs_span.set_attribute("validation.error",        validation.get("error") or "")
            vs_span.set_attribute("validation.tables_found", json.dumps(validation.get("tables_referenced", [])))

        if validation["valid"]:
            break

        retries += 1
        if attempt == 2:
            break  # Exhausted retries; proceed with best-effort SQL

        # Regenerate with the error context (retry on empty or unparseable)
        retry_prompt = QUERY_PLANNER_RETRY_TEMPLATE.format(
            question=question,
            sql=sql,
            error=validation.get("error", "unknown error"),
            schema=schema,
        )
        retry_parsed = None
        for gen_attempt in range(3):
            with with_prompt_template(
                template=QUERY_PLANNER_RETRY_TEMPLATE,
                variables={
                    "question": question,
                    "sql":      sql,
                    "error":    validation.get("error", ""),
                    "schema":   schema,
                },
                version=QUERY_PLANNER_RETRY_VERSION,
            ):
                response = await _genai_client.aio.models.generate_content(
                    model=GEMINI_MODEL, contents=retry_prompt
                )
            if not (getattr(response, "text", None) or "").strip():
                continue
            try:
                retry_parsed = _parse_json(response.text)
                break
            except (ValueError, json.JSONDecodeError):
                continue
        if retry_parsed:
            parsed = retry_parsed
            sql    = parsed.get("sql", sql)
            plan   = parsed.get("plan", plan)
            tables = parsed.get("tables", tables)
        else:
            break  # Model returned empty on retries; proceed with current sql

    return {
        "sql":                        sql,
        "plan":                       plan,
        "tables":                     tables,
        "retries":                    retries,
        "validation_passed_first_try": retries == 0,
        "validation":                 validation,
    }


# ── Access Control Guardrail ──────────────────────────────────────────────────

async def check_access_guardrail(question: str, tables: list[str], sql: str) -> dict:
    """Validate that the current user's role permits access to the requested tables.

    Creates a parent GUARDRAIL span with two children:
      - resolve_user_role  (TOOL)  — look up role from the identity service
      - verify_table_permissions (CHAIN) — compare requested vs allowed tables

    Returns:
        dict with keys: allowed (bool), role, denied_tables, requested_tables, reason
    """
    tracer = _get_tracer()
    role = _user_role.get()
    session_id = _current_session_id.get() or ""
    requested = {t.lower() for t in tables}

    with tracer.start_as_current_span("access_control_check") as guard_span:
        guard_span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.GUARDRAIL.value)
        guard_span.set_attribute(_INPUT, question)
        guard_span.set_attribute(_INPUT_MIME, _TEXT)
        if session_id:
            guard_span.set_attribute("session.id", session_id)
        guard_span.set_attribute("guardrail.requested_tables", json.dumps(sorted(requested)))

        # Child 1 — simulate calling an external auth / identity service
        allowed_tables: set[str] = set()
        with tracer.start_as_current_span("resolve_user_role") as role_span:
            role_span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
            role_span.set_attribute(_INPUT, f"user_id=demo-user")
            role_span.set_attribute(_INPUT_MIME, _TEXT)

            allowed_tables = ROLE_PERMISSIONS.get(role, set())
            role_result = {
                "role": role,
                "allowed_tables": sorted(allowed_tables),
            }
            role_span.set_attribute(_OUTPUT, json.dumps(role_result))
            role_span.set_attribute(_OUTPUT_MIME, _JSON)

        # Child 2 — compare requested tables against the role's permissions
        with tracer.start_as_current_span("verify_table_permissions") as verify_span:
            verify_span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
            verify_span.set_attribute(
                _INPUT,
                json.dumps({"role": role, "requested_tables": sorted(requested), "sql": sql}),
            )
            verify_span.set_attribute(_INPUT_MIME, _JSON)

            denied = sorted(requested - allowed_tables)
            allowed = len(denied) == 0

            if allowed:
                reason = f"Role '{role}' has access to all requested tables."
            else:
                reason = (
                    f"Role '{role}' does not have access to table(s): {', '.join(denied)}. "
                    f"Query denied."
                )

            verdict = {
                "allowed": allowed,
                "denied_tables": denied,
                "reason": reason,
            }
            verify_span.set_attribute(_OUTPUT, json.dumps(verdict))
            verify_span.set_attribute(_OUTPUT_MIME, _JSON)

        # Set summary attributes on the parent guardrail span
        result = {
            "allowed": allowed,
            "role": role,
            "denied_tables": denied,
            "requested_tables": sorted(requested),
            "reason": reason,
        }
        guard_span.set_attribute(_OUTPUT, json.dumps(result))
        guard_span.set_attribute(_OUTPUT_MIME, _JSON)
        guard_span.set_attribute("guardrail.result", "allowed" if allowed else "denied")
        guard_span.set_attribute("guardrail.role", role)
        guard_span.set_attribute("guardrail.denied_tables", json.dumps(denied))

    return result


# ── Stage 3: Data Retriever ───────────────────────────────────────────────────

async def retrieve_data(sql: str) -> dict:
    """Execute the SQL query. Pure tool call — no LLM involved.

    Creates a RETRIEVER child span so the SQL and result metadata appear in the trace.

    Returns:
        dict: rows, row_count, column_names, status
    """
    tracer = _get_tracer()
    session_id = _current_session_id.get() or ""

    with tracer.start_as_current_span("sql_execution") as span:
        span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        span.set_attribute(_INPUT, sql)
        span.set_attribute(_INPUT_MIME, _TEXT)
        if session_id:
            span.set_attribute("session.id", session_id)

        result = execute_sql(sql)

        span.set_attribute(_OUTPUT, json.dumps(result))
        span.set_attribute(_OUTPUT_MIME, _JSON)

    return result


# ── Stage 4: Synthesizer ──────────────────────────────────────────────────────

async def synthesize_answer(
    question: str,
    plan: str,
    data: dict,
    complexity: str,
    prompt_version: str | None = None,
    tracer=None,
    session_id: str | None = None,
) -> dict:
    """Generate a final answer with up to 2 brand-voice revision rounds.

    Child spans for each review_brand_voice call are nested under the caller's
    active span (ADK instrumentor tool span or manual parent).

    Args:
        question:       Original NL question.
        plan:           Retrieval plan from Query Planner.
        data:           Raw results dict from Data Retriever.
        complexity:     Classification from Query Classifier.
        prompt_version: "v1" (strict) or "v2" (intentionally weaker). Default from context.
        tracer:         OTel tracer for child spans (optional).
        session_id:     ADK session ID for span attribute (optional, from context).

    Returns:
        dict: answer, prompt_version_used, revision_count,
              brand_voice_score, brand_voice_passed_first_draft
    """
    prompt_version = prompt_version or _prompt_version.get()
    session_id = session_id or _current_session_id.get() or ""
    tracer = tracer or _get_tracer()
    constraints = _extract_constraints(question)
    data_str = (
        json.dumps(data.get("rows", []), indent=2)
        if data.get("rows")
        else "No data returned."
    )

    # Select prompt template based on CLI flag
    if prompt_version == "v2":
        template         = SYNTHESIZER_TEMPLATE_V2
        template_version = SYNTHESIZER_V2_VERSION
    else:
        template         = SYNTHESIZER_TEMPLATE_V1
        template_version = SYNTHESIZER_V1_VERSION

    draft_prompt = template.format(
        question=question, plan=plan, data=data_str, constraints=constraints
    )
    draft = ""
    for attempt in range(3):
        with with_prompt_template(
            template=template,
            variables={"question": question, "plan": plan, "data": data_str, "constraints": constraints},
            version=template_version,
        ):
            response = await _genai_client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=draft_prompt
            )
        draft = (getattr(response, "text", None) or "").strip()
        if draft:
            break
    if not draft:
        draft = (
            "I couldn't generate a full answer because the model returned empty after retries. "
            "Please try rephrasing your question or try again."
        )

    revision_count           = 0
    passed_first_draft       = False
    final_score              = 0.0

    for revision_round in range(1, 3):  # rounds 1 and 2
        with tracer.start_as_current_span("review_brand_voice") as bv_span:
            bv_span.set_attribute(_SPAN_KIND, "TOOL")
            bv_span.set_attribute("tool.name", "review_brand_voice")
            bv_span.set_attribute("session.id", session_id)
            bv_span.set_attribute("brand_voice.revision_round", revision_round)
            bv_span.set_attribute(_INPUT, draft)

            review = review_brand_voice(draft, question)

            bv_span.set_attribute(_OUTPUT,               json.dumps(review))
            bv_span.set_attribute(_OUTPUT_MIME,          _JSON)
            bv_span.set_attribute("brand_voice.score",   review.get("score", 0.0))
            bv_span.set_attribute("brand_voice.passes",  review.get("passes", False))
            bv_span.set_attribute("brand_voice.issues",  json.dumps(review.get("issues", [])))

        final_score = review.get("score", 0.0)

        if review.get("passes") or final_score >= 0.70:
            if revision_round == 1:
                passed_first_draft = True
            break

        if revision_round == 2:
            break  # Return best attempt regardless

        revision_count += 1
        issues_str     = "; ".join(review.get("issues", []))
        revision_notes = review.get("suggested_revision_notes", "")

        revision_prompt = SYNTHESIZER_REVISION_TEMPLATE.format(
            question=question,
            draft=draft,
            issues=issues_str,
            revision_notes=revision_notes,
            data=data_str,
            constraints=constraints,
        )
        revision_draft = ""
        for attempt in range(3):
            with with_prompt_template(
                template=SYNTHESIZER_REVISION_TEMPLATE,
                variables={
                    "question":       question,
                    "draft":          draft,
                    "issues":         issues_str,
                    "revision_notes": revision_notes,
                    "data":           data_str,
                    "constraints":    constraints,
                },
                version=SYNTHESIZER_REVISION_VERSION,
            ):
                response = await _genai_client.aio.models.generate_content(
                    model=GEMINI_MODEL, contents=revision_prompt
                )
            revision_draft = (getattr(response, "text", None) or "").strip()
            if revision_draft:
                break
        draft = revision_draft if revision_draft else draft

    return {
        "answer":                       draft,
        "prompt_version_used":          template_version,
        "revision_count":               revision_count,
        "brand_voice_score":            final_score,
        "brand_voice_passed_first_draft": passed_first_draft,
    }


# ── ADK tool wrappers (augment current span from GoogleADKInstrumentor) ───────

def _augment_span(attrs: dict) -> None:
    """Set attributes on the current span (created by ADK instrumentor)."""
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


async def classify_query_tool(question: str) -> dict:
    """ADK tool: classify query complexity. Augments current span with CHAIN + attributes."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(_INPUT, question)
        span.set_attribute(_INPUT_MIME, _TEXT)
        sid = _current_session_id.get()
        if sid:
            span.set_attribute("session.id", sid)
    classification = await classify_query(question)
    _augment_span({
        _OUTPUT: json.dumps(classification),
        _OUTPUT_MIME: _JSON,
        "classifier.result": classification.get("complexity", ""),
        "classifier.reasoning": classification.get("reasoning", ""),
    })
    return classification


async def plan_query_tool(question: str, complexity: str) -> dict:
    """ADK tool: plan SQL. Augments current span with CHAIN + attributes."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(_INPUT, f"question: {question}\ncomplexity: {complexity}")
        span.set_attribute(_INPUT_MIME, _TEXT)
        sid = _current_session_id.get()
        if sid:
            span.set_attribute("session.id", sid)
    plan_result = await plan_query(question, complexity)
    _augment_span({
        _OUTPUT: json.dumps({k: plan_result[k] for k in ("sql", "plan", "tables", "retries")}),
        _OUTPUT_MIME: _JSON,
        "planner.sql": plan_result.get("sql", ""),
        "planner.tables_referenced": json.dumps(plan_result.get("tables", [])),
        "planner.retries": plan_result.get("retries", 0),
        "planner.validation_passed_first_try": plan_result.get("validation_passed_first_try", False),
    })
    return plan_result


async def retrieve_data_tool(sql: str) -> dict:
    """ADK tool: execute SQL and return data."""
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
        span.set_attribute(_INPUT, sql)
        span.set_attribute(_INPUT_MIME, _TEXT)
        sid = _current_session_id.get()
        if sid:
            span.set_attribute("session.id", sid)
    data = await retrieve_data(sql)
    rows_preview = data.get("rows", [])[:5]
    _augment_span({
        _OUTPUT: json.dumps(rows_preview),
        _OUTPUT_MIME: _JSON,
        "retriever.row_count": data.get("row_count", 0),
        "retriever.column_names": json.dumps(data.get("column_names", [])),
        "retriever.empty_result": data.get("row_count", 0) == 0,
    })
    return data


async def synthesize_answer_tool(
    question: str,
    plan: str,
    data_json: str,
    complexity: str,
    prompt_version: str = "v2",
) -> dict:
    """ADK tool: synthesize answer. Augments current span with CHAIN + attributes."""
    try:
        data = json.loads(data_json)
    except (TypeError, json.JSONDecodeError):
        data = {"rows": [], "row_count": 0, "column_names": []}
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute(_INPUT, f"question: {question}\nplan: {plan}\nrows: {data.get('row_count', 0)}")
        span.set_attribute(_INPUT_MIME, _TEXT)
        sid = _current_session_id.get()
        if sid:
            span.set_attribute("session.id", sid)
    synthesis = await synthesize_answer(question=question, plan=plan, data=data, complexity=complexity, prompt_version=prompt_version)
    _augment_span({
        _OUTPUT: synthesis.get("answer", ""),
        _OUTPUT_MIME: _TEXT,
        "synthesizer.prompt_version": synthesis.get("prompt_version_used", ""),
        "synthesizer.revision_count": synthesis.get("revision_count", 0),
        "synthesizer.brand_voice_score": synthesis.get("brand_voice_score", 0.0),
        "synthesizer.brand_voice_passed_first_draft": synthesis.get("brand_voice_passed_first_draft", False),
        "synthesizer.answer_length_chars": len(synthesis.get("answer", "")),
    })
    return synthesis


# ── Sub-agents (each runs one pipeline stage; root delegates in order) ─────────

query_classifier_agent = Agent(
    model=GEMINI_MODEL,
    name="query_classifier",
    description="Classifies the user question into complexity (simple, aggregation, multi_hop, constrained).",
    instruction="When you receive a question, call the classify_query tool with that question and return the classification result.",
    tools=[classify_query_tool],
)

query_planner_agent = Agent(
    model=GEMINI_MODEL,
    name="query_planner",
    description="Generates and validates SQL for the question given the complexity from the classifier.",
    instruction="When you receive a question and complexity, call the plan_query tool with them and return the result (sql, plan, tables).",
    tools=[plan_query_tool],
)

data_retriever_agent = Agent(
    model=GEMINI_MODEL,
    name="data_retriever",
    description="Executes SQL against the media database and returns the result.",
    instruction="When you receive a SQL string, call the retrieve_data tool with it and return the result.",
    tools=[retrieve_data_tool],
)

synthesizer_agent = Agent(
    model=GEMINI_MODEL,
    name="synthesizer",
    description="Synthesizes the final answer from the question, plan, data, and complexity; uses brand-voice review.",
    instruction="When you receive question, plan, data_json, complexity, and prompt_version, call the synthesize_answer tool and return the result.",
    tools=[synthesize_answer_tool],
)


# ── Sequential pipeline stages (BaseAgent; state-passing, no LLM delegation) ───

def _question_from_user_content(user_content) -> str:
    """Extract question text from InvocationContext.user_content (Content | None)."""
    if not user_content or not getattr(user_content, "parts", None):
        return ""
    for part in user_content.parts:
        if getattr(part, "text", None):
            return (part.text or "").strip()
    return ""


class ClassifierStageAgent(BaseAgent):
    """Stage 1: classify query, write question + classification to session state."""

    async def _run_async_impl(self, ctx: InvocationContext):
        question = ctx.session.state.get("question") or _question_from_user_content(ctx.user_content)
        if not question:
            question = ""
        ctx.session.state["question"] = question
        try:
            classification = await classify_query(question)
        except Exception as e:
            classification = {"complexity": "simple", "reasoning": f"Fallback after error: {e}"}
        ctx.session.state["classification"] = classification
        yield Event(author=self.name, content=None, turn_complete=False)


class PlannerStageAgent(BaseAgent):
    """Stage 2: plan query from question + classification, write plan_result to state."""

    async def _run_async_impl(self, ctx: InvocationContext):
        question = ctx.session.state.get("question", "")
        classification = ctx.session.state.get("classification") or {}
        complexity = classification.get("complexity", "simple")
        try:
            plan_result = await plan_query(question, complexity)
        except Exception as e:
            plan_result = {
                "sql": "SELECT 1 LIMIT 0",
                "plan": f"Planning failed: {e}",
                "tables": [],
            }
        ctx.session.state["plan_result"] = plan_result
        yield Event(author=self.name, content=None, turn_complete=False)


class GuardrailStageAgent(BaseAgent):
    """Stage 3: role-based access control — verify the user may query the planned tables."""

    async def _run_async_impl(self, ctx: InvocationContext):
        question = ctx.session.state.get("question", "")
        plan_result = ctx.session.state.get("plan_result") or {}
        tables = plan_result.get("tables", [])
        sql = plan_result.get("sql", "")

        guardrail = await check_access_guardrail(question, tables, sql)
        ctx.session.state["guardrail"] = guardrail

        if not guardrail.get("allowed", True):
            denied = ", ".join(guardrail.get("denied_tables", []))
            role = guardrail.get("role", "unknown")
            ctx.session.state["answer"] = (
                f"Access denied. Your role ('{role}') does not have permission to "
                f"query the following table(s): {denied}. "
                f"Please contact your administrator for elevated access."
            )

        yield Event(author=self.name, content=None, turn_complete=False)


class RetrieverStageAgent(BaseAgent):
    """Stage 4: run SQL from plan_result, write data to state. Skipped on guardrail denial."""

    async def _run_async_impl(self, ctx: InvocationContext):
        if not ctx.session.state.get("guardrail", {}).get("allowed", True):
            yield Event(author=self.name, content=None, turn_complete=False)
            return

        plan_result = ctx.session.state.get("plan_result") or {}
        sql = plan_result.get("sql", "SELECT 1 LIMIT 0")
        try:
            data = await retrieve_data(sql)
        except Exception as e:
            data = {"rows": [], "row_count": 0, "column_names": [], "status": "error", "error": str(e)}
        ctx.session.state["data"] = data
        yield Event(author=self.name, content=None, turn_complete=False)


class SynthesizerStageAgent(BaseAgent):
    """Stage 5: synthesize answer from state; yield final Event with turn_complete=True."""

    async def _run_async_impl(self, ctx: InvocationContext):
        if not ctx.session.state.get("guardrail", {}).get("allowed", True):
            answer = ctx.session.state.get("answer", "Access denied.")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=answer)]),
                turn_complete=True,
            )
            return

        question = ctx.session.state.get("question", "")
        plan_result = ctx.session.state.get("plan_result") or {}
        data = ctx.session.state.get("data") or {"rows": [], "row_count": 0, "column_names": []}
        classification = ctx.session.state.get("classification") or {}
        complexity = classification.get("complexity", "simple")
        plan = plan_result.get("plan", "")
        prompt_version = _prompt_version.get()
        try:
            synthesis = await synthesize_answer(
                question=question,
                plan=plan,
                data=data,
                complexity=complexity,
                prompt_version=prompt_version,
            )
            answer = synthesis.get("answer", "(no answer generated)")
        except Exception as e:
            answer = f"I encountered an error while generating the answer: {e}"
            ctx.session.state["answer"] = answer
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=answer)]),
                turn_complete=True,
            )
            return
        ctx.session.state["answer"] = answer
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=answer)]),
            turn_complete=True,
        )


classifier_stage_agent = ClassifierStageAgent(name="classifier_stage")
planner_stage_agent = PlannerStageAgent(name="planner_stage")
guardrail_stage_agent = GuardrailStageAgent(name="guardrail_stage")
retriever_stage_agent = RetrieverStageAgent(name="retriever_stage")
synthesizer_stage_agent = SynthesizerStageAgent(name="synthesizer_stage")

pipeline_agent = SequentialAgent(
    name="media_pipeline",
    description="Fixed-order pipeline: classify → plan → guardrail → retrieve → synthesize.",
    sub_agents=[
        classifier_stage_agent,
        planner_stage_agent,
        guardrail_stage_agent,
        retriever_stage_agent,
        synthesizer_stage_agent,
    ],
)
