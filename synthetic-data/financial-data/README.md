# FinData Synthetic Traces

Generators for FinData-themed synthetic OpenInference traces.

## Scripts

| File | What it generates |
|------|-------------------|
| `synthetic_spans_findata_media_agent.py` | Data Analytics team NL-to-SQL agent traces (coordinator + planning/engineer/response agents) |
| `synthetic_spans_findata_briefing.ipynb` | Personalized briefing traces (wires, aggregation, personas) |
| `synthetic_spans_findata_research_augment.ipynb` | Research report augmentation traces (Vertex/Gemini metadata) |

---

## Data Analytics Agent Synthetic Traces

Generates traces matching the FinData Data Analytics team's NL-to-SQL agent system. The architecture follows their actual flowchart: a **Coordinator Agent** orchestrating three task agents that query BillingCo subscription data via BigQuery.

### Architecture

```
Coordinator Agent
├── Planning Agent ── get_catalog (MCP) → get_memory (MCP) → select tables (LLM) → get_details (MCP) × N
│                     Loops up to 3 iterations if context is insufficient
├── Engineer Agent ── Generates SQL from plan, returns SUCCESS or BLOCKING_AMBIGUITY
└── Response Agent ── Builds final response (success summary / failure explanation / clarification / direct answer)
```

**MCP tools** (findata-db server): `get_catalog`, `get_memory`, `get_details`, `execute_sql`
**MCP tools** (identity-service server): `resolve_user_role`
**User-defined tools**: `verify_table_permissions`

### Domain: FinData Subscription Data

Three BigQuery tables from the BillingCo billing system (`findata-analytics-dev.agent_eval_dataset`):

- **cb_subscriptions** -- BillingCo billing data: customer_id, plan_amount, MRR, currency, cancellation reasons, status
- **subscriptions_web** -- Web subscription lifecycle: rate plans, payments, discounts, cancellation details, billing periods, 50+ columns
- **cb_subscriptions_coupons** -- Coupon-subscription mappings: coupon_id, subscription_id, apply_till

Key business rules baked into the traces:
- Join via `cf_account_id`, never `customer_id`
- Use `mrr` only when user explicitly asks for MRR; otherwise `plan_amount`
- Never sum across currencies
- `cancelled_date` can be NULL (still active) or future-dated
- Use `status_reporting` for churn/active reporting

### Prompt Templates

Every LLM call carries a named prompt template with version tracking via `llm.prompt_template.*` span attributes. This enables filtering, comparison, and A/B testing across agent versions in Arize.

| Agent | Template Name | Version | Variables |
|-------|---------------|---------|-----------|
| Coordinator | `coordinator_routing` | `v1.0` (fixed) | `question`, `state` |
| Planning Agent | `planning_table_selection` | `v1.0` (fixed) | `question`, `available_tables` |
| Engineer Agent | `engineer_sql_generation` | `v1.0` or `v2.0` | `question`, `tables`, `context` |
| Response Agent | `response_synthesis` | `v1.0` or `v2.0` | `question`, `mode` |

**Version strategy**: Coordinator and Planning templates are fixed at v1.0 (infrastructure routing). Engineer and Response templates are randomly assigned v1.0 (70%) or v2.0 (30%) per trace to simulate A/B testing of prompt versions. Both agents within a trace share the same version, enabling clean comparison in Arize by filtering on `llm.prompt_template.version`.

### Scenario Types

| Scenario | ~% | What happens |
|----------|----|-------------|
| **Happy path** | 28 | Planning (1 iter) -> Engineer (SUCCESS) -> execute_sql -> Response summary. Includes planning deep loops (2-3 iter) and adversarial queries within this bucket. |
| **Guardrail denial** | 10 | Full pipeline runs but `access_control_check` blocks execution. `AccessControlViolation` and `GuardrailBlocked` exceptions on the guardrail spans. Non-finance roles denied access to `cb_subscriptions`. |
| **SQL retry** | 8 | Engineer generates bad SQL (syntax error, wrong column). Coordinator re-routes to Engineer for a corrected query. |
| **BLOCKING_AMBIGUITY (resolved)** | 8 | Engineer returns `BLOCKING_AMBIGUITY` -- query is ambiguous (e.g., "which table?", "which metric?"). Response Agent asks clarification question. `await_user_clarification` span shows user responding. Pipeline re-runs with clarified question. |
| **BLOCKING_AMBIGUITY (abandoned)** | 4 | Same as above but user never responds. `await_user_clarification` ends with `ClarificationTimeout` error. Trace shows exactly where the conversation stalled. |
| **Execution failure** | 8 | SQL executes but fails (timeout, too much data). `SQLExecutionError` exception on `execute_sql` span. Response Agent explains the failure. |
| **No SQL needed** | 6 | Coordinator skips Planning + Engineer entirely, routes directly to Response for a minimal answer (e.g., "What tables are available?"). Short trace. |
| **Schema mismatch** | 6 | Query references data that doesn't exist (e.g., customer LTV). Engineer generates best-effort SQL. Response notes gaps. |
| **Coordinator retry** | 8 | Full pipeline runs, results seem incomplete. Coordinator re-runs the whole pipeline with deeper Planning Agent iterations. Two full cycles visible in the trace. |

### Golden Queries

Three queries from the Data Analytics team's evaluation dataset are included verbatim:

1. **cb_subscriptions_q1**: Retrieve all bulk (self-serve) subscriptions with creation date, cancellation date, quantity, amount, currency, status
2. **subscriptions_web_q1**: Daily cancellation count for current year, ordered by cancelled_date desc
3. **cb_subscriptions_coupons_q1**: All coupons per subscription with apply_till converted to America/New_York

### Evaluations

When run with `--with-evals`, generates four evaluation types per trace:

| Evaluation | What it measures |
|------------|-----------------|
| `AgentTrajectoryAccuracy` | Did the coordinator route agents correctly? |
| `SQLQuality` | Was the SQL valid and did it follow usage rules? |
| `CoordinationQuality` | Did the multi-agent coordination succeed? |
| `TableSelectionPrecision` | Did the Planning Agent select the right tables? |

### Setup

1. Create a `.env` file in this directory:
   ```
   ARIZE_SPACE_ID="your-space-id"
   ARIZE_API_KEY="your-api-key"
   ARIZE_PROJECT_NAME="findata_da_agent_synthetic"
   ```

2. Install dependencies (from the `synthetic-data/` root):
   ```bash
   pip install -r requirements.txt
   ```

### Usage

```bash
# Test traces (one per scenario type, always runs evals)
python synthetic_spans_findata_media_agent.py --test

# Full batch (default 500 traces)
python synthetic_spans_findata_media_agent.py

# Custom count with reproducible seed
python synthetic_spans_findata_media_agent.py --count 200 --seed 42

# Include evaluations
python synthetic_spans_findata_media_agent.py --count 500 --with-evals

# Override project name
python synthetic_spans_findata_media_agent.py --project-name my_project
```

### What to explore in Arize

- **Trace tree**: Expand coordinator -> planning/engineer/response to see the full delegation chain
- **Prompt template versions**: Filter by `llm.prompt_template.version` to compare v1.0 vs v2.0 A/B performance on engineer and response agents
- **Template comparison**: Group by `llm.prompt_template.template` to see all LLM calls by template name across agents
- **Planning Agent iterations**: Filter by `planning.actual_iterations > 1` to find deep-loop traces
- **BLOCKING_AMBIGUITY**: Filter by `engineer.output_status = BLOCKING_AMBIGUITY` to find ambiguous queries
- **Abandoned clarifications**: Filter by `clarification.resolved = false` to find stalled conversations
- **Guardrail denials**: Filter by `guardrail.result = denied` with red error spans and exception details
- **MCP tools**: Filter by `mcp.server.name` to see `findata-db` and `identity-service` calls
- **Execution failures**: Filter by `execution.success = false` for SQL errors
- **Role-based access**: Group by `guardrail.role` to compare analyst, finance, marketing, restricted
- **Table selection**: Check `planning.selected_tables` to see which tables the Planning Agent chose
