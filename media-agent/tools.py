"""
Media Agent — Tools

All four tool functions used by the agent pipeline. Each is a plain Python function
with type hints and a docstring so Google ADK can auto-infer the schema if needed.

Tool functions:
  - schema_lookup      : Return DDL + sample rows for a table
  - validate_sql       : EXPLAIN + dry-run a query against SQLite
  - execute_sql        : Run a query and return JSON results
  - review_brand_voice : LLM-based editorial review (calls Gemini directly)
"""

import json
import os
import re
import sqlite3
from pathlib import Path

from prompt_utils import with_prompt_template
from google import genai


_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "media.db"
DB_PATH = Path(os.environ.get("MEDIA_DB_PATH", str(_DEFAULT_DB_PATH)))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Gen AI client (Vertex AI backend); project/location from env.
_genai_client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
)

# ── Prompt template ───────────────────────────────────────────────────────────
# Keep here so prospects can easily edit the editorial rubric.

BRAND_VOICE_RUBRIC = """
You are an editorial reviewer. Score the following text on these
dimensions (0.0–1.0 each), then compute an overall score as the average:

1. DATA_PRECISION  — Does it use specific numbers from the data?
   Score 0.0 if it uses vague language like "significant growth" when exact
   figures are available.

2. ATTRIBUTION     — Does it reference the data source or timeframe?
   Score 0.0 if claims float without temporal or source context.

3. NO_HEDGING      — Is the language assertive and direct?
   Score 0.0 for each instance of "might", "perhaps", "seems", "arguably",
   "it appears". Deduct 0.2 per occurrence (floor 0.0).

4. CONCISENESS     — Is every sentence earning its place?
   Score 0.0 for throat-clearing, preamble, or restating the question.

Text to review:
{draft_answer}

Original question for context:
{question}

Return JSON only (no markdown fences):
{{
  "passes": <true if overall score >= 0.70>,
  "score": <float 0.0–1.0, average of four dimensions>,
  "dimension_scores": {{
    "data_precision": <float>,
    "attribution": <float>,
    "no_hedging": <float>,
    "conciseness": <float>
  }},
  "issues": [<short description of each failing dimension>],
  "suggested_revision_notes": "<one sentence of specific guidance>"
}}
"""
BRAND_VOICE_RUBRIC_VERSION = "v1.0"

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
        raise ValueError(f"Cannot parse JSON from response: {text[:300]}")


def _get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python seed_db.py"
        )
    return sqlite3.connect(DB_PATH)


# ── Tool: schema_lookup ───────────────────────────────────────────────────────

def schema_lookup(table_name: str) -> dict:
    """Return the schema and sample rows for a table in the media database.

    Args:
        table_name: Name of the table. Valid values: articles, revenue, traffic, authors.

    Returns:
        dict with keys: table_name, schema_ddl (str), columns (list of dicts),
        sample_rows (list of dicts, up to 3), row_count (int), status (str).
    """
    try:
        con = _get_connection()
        cur = con.cursor()

        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = cur.fetchone()
        if not row:
            con.close()
            return {
                "status": "error",
                "error": f"Table '{table_name}' not found. "
                         "Available tables: articles, revenue, traffic, authors",
            }

        ddl = row[0]
        cur.execute(f"PRAGMA table_info({table_name})")
        pragma_rows = cur.fetchall()
        columns = [
            {
                "name": r[1],
                "type": r[2],
                "not_null": bool(r[3]),
                "primary_key": bool(r[5]),
            }
            for r in pragma_rows
        ]

        col_names = [c["name"] for c in columns]
        cur.execute(f"SELECT * FROM {table_name} LIMIT 3")  # noqa: S608
        sample_rows = [dict(zip(col_names, r)) for r in cur.fetchall()]

        cur.execute(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608
        row_count = cur.fetchone()[0]

        con.close()
        return {
            "status": "success",
            "table_name": table_name,
            "schema_ddl": ddl,
            "columns": columns,
            "sample_rows": sample_rows,
            "row_count": row_count,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ── Tool: validate_sql ────────────────────────────────────────────────────────

def validate_sql(sql: str) -> dict:
    """Validate a SQL query using EXPLAIN and a LIMIT 0 dry-run against SQLite.

    Args:
        sql: The SQL query string to validate.

    Returns:
        dict with keys: valid (bool), error (str or None),
        tables_referenced (list[str]), estimated_joins (int), status (str).
    """
    try:
        con = _get_connection()
        cur = con.cursor()

        # EXPLAIN catches syntax errors and references to non-existent tables
        cur.execute(f"EXPLAIN {sql}")  # noqa: S608

        # Dry-run: wrap in subquery and apply LIMIT 0 to avoid reading data
        dry_sql = f"SELECT * FROM ({sql}) AS _dry_run LIMIT 0"  # noqa: S608
        try:
            cur.execute(dry_sql)
        except sqlite3.OperationalError:
            # Some queries (e.g. those with ORDER BY + LIMIT) don't wrap cleanly;
            # EXPLAIN already confirmed syntax validity above.
            pass

        con.close()

        tables = re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
        tables = [t for t in tables if t.upper() not in ("SELECT", "WITH")]
        estimated_joins = len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))

        return {
            "status": "success",
            "valid": True,
            "error": None,
            "tables_referenced": list(dict.fromkeys(tables)),  # deduplicated, order-preserved
            "estimated_joins": estimated_joins,
        }
    except sqlite3.OperationalError as exc:
        return {
            "status": "error",
            "valid": False,
            "error": str(exc),
            "tables_referenced": [],
            "estimated_joins": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "valid": False,
            "error": str(exc),
            "tables_referenced": [],
            "estimated_joins": 0,
        }
    finally:
        try:
            con.close()
        except Exception:
            pass


# ── Tool: execute_sql ─────────────────────────────────────────────────────────

def execute_sql(sql: str) -> dict:
    """Execute a SQL query against the media SQLite database.

    Args:
        sql: The SQL query to execute. Must be a SELECT statement.

    Returns:
        dict with keys: rows (list of dicts), row_count (int),
        column_names (list[str]), status (str).
    """
    try:
        con = _get_connection()
        cur = con.cursor()
        cur.execute(sql)  # noqa: S608
        col_names = [d[0] for d in (cur.description or [])]
        rows = [dict(zip(col_names, r)) for r in cur.fetchall()]
        con.close()
        return {
            "status": "success",
            "rows": rows,
            "row_count": len(rows),
            "column_names": col_names,
        }
    except Exception as exc:
        return {
            "status": "error",
            "rows": [],
            "row_count": 0,
            "column_names": [],
            "error": str(exc),
        }
    finally:
        try:
            con.close()
        except Exception:
            pass


# ── Tool: review_brand_voice ──────────────────────────────────────────────────

def review_brand_voice(draft_answer: str, question: str) -> dict:
    """Score a draft answer against brand voice guidelines using Gemini.

    Evaluates four dimensions: DATA_PRECISION, ATTRIBUTION, NO_HEDGING, CONCISENESS.
    Returns a score (0.0–1.0) and actionable revision notes.

    Args:
        draft_answer: The draft text to evaluate.
        question: The original question, used as context for the review.

    Returns:
        dict with keys: passes (bool), score (float), dimension_scores (dict),
        issues (list[str]), suggested_revision_notes (str), status (str).
    """
    try:
        prompt = BRAND_VOICE_RUBRIC.format(
            draft_answer=draft_answer, question=question
        )
        with with_prompt_template(
            template=BRAND_VOICE_RUBRIC,
            variables={"draft_answer": draft_answer, "question": question},
            version=BRAND_VOICE_RUBRIC_VERSION,
        ):
            response = _genai_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
        result = _parse_json(response.text)
        result["status"] = "success"
        # Ensure required keys exist with safe defaults
        result.setdefault("passes", result.get("score", 0) >= 0.70)
        result.setdefault("score", 0.0)
        result.setdefault("issues", [])
        result.setdefault("suggested_revision_notes", "")
        return result
    except Exception as exc:
        return {
            "status": "error",
            "passes": False,
            "score": 0.0,
            "issues": [str(exc)],
            "suggested_revision_notes": "",
            "error": str(exc),
        }
