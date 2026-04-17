#!/usr/bin/env python3
"""
FinData Data Analytics Agent — Synthetic Trace Generator

Generates OpenTelemetry traces matching the Data Analytics team's NL-to-SQL agent system:

  Coordinator Agent → Planning Agent / Engineer Agent / Response Agent

Architecture (from Data Analytics team flowchart):
  - Coordinator: routes flow, handles OUTPUT.status (SUCCESS / BLOCKING_AMBIGUITY),
    calls execute_sql tool, manages the "Requires SQL generation?" gate
  - Planning Agent: context retrieval with 3 named tools (get_catalog, get_memory,
    get_details), iterates up to N=3, returns {user_query, selected_tables, retrieved_context}
  - Engineer Agent: consumes plan, generates SQL, returns OUTPUT.status
  - Response Agent: builds final response (success summary, failure explanation,
    or minimal direct answer)

Domain: FinData subscription data (cb_subscriptions, subscriptions_web,
  cb_subscriptions_coupons) from BillingCo billing system.

Usage:
  python synthetic_spans_findata_media_agent.py                    # 500 traces
  python synthetic_spans_findata_media_agent.py --count 100        # 100 traces
  python synthetic_spans_findata_media_agent.py --test              # single trace test
  python synthetic_spans_findata_media_agent.py --with-evals        # include evaluations
"""

import argparse
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from arize.otel import register
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode


# ── OpenInference span attribute keys ────────────────────────────────────────

_KIND = "openinference.span.kind"
_INPUT = "input.value"
_OUTPUT = "output.value"
_INPUT_MIME = "input.mime_type"
_OUTPUT_MIME = "output.mime_type"
_JSON = "application/json"
_TEXT = "text/plain"

GEMINI_MODEL = "gemini-2.0-flash"
LLM_PROVIDER = "google"
LLM_SYSTEM = "vertexai"

BQ_DATASET = "findata-analytics-dev.agent_eval_dataset"

ROLE_PERMISSIONS = {
    "analyst": {"cb_subscriptions", "subscriptions_web", "cb_subscriptions_coupons"},
    "finance": {"cb_subscriptions", "subscriptions_web", "cb_subscriptions_coupons"},
    "marketing": {"subscriptions_web", "cb_subscriptions_coupons"},
    "restricted": {"subscriptions_web"},
}

ALL_TABLES = ["cb_subscriptions", "subscriptions_web", "cb_subscriptions_coupons"]

TABLE_DESCRIPTIONS = {
    "cb_subscriptions": "BillingCo raw subscription data: billing, plan amounts, MRR, cancellation reasons, currency. Join to cb_customers via cf_account_id.",
    "subscriptions_web": "Web subscription lifecycle: rate plans, payments, discounts, cancellation details, billing periods. Primary key is subscription. Only individual web subs (no group).",
    "cb_subscriptions_coupons": "Coupons applied to subscriptions: coupon_id, subscription_id, apply_till timestamp. Multiple rows per subscription possible.",
}

SCHEMA_DDL = f"""CREATE TABLE `{BQ_DATASET}.cb_subscriptions` (
  customer_id STRING, plan_id STRING, cancel_reason STRING,
  cancelled_at TIMESTAMP, channel STRING, created_at TIMESTAMP,
  current_term_end TIMESTAMP, current_term_start TIMESTAMP,
  id STRING, next_billing_at TIMESTAMP, plan_amount NUMERIC,
  plan_quantity INTEGER, status STRING, cf_account_id STRING,
  cf_is_bulk_subscription STRING, cf_channel STRING, currency STRING,
  next_billing_amount NUMERIC, mrr NUMERIC, cancel_reason_code STRING
);

CREATE TABLE `{BQ_DATASET}.subscriptions_web` (
  bconnect_id STRING, findata_customer_id STRING, email_domain STRING,
  account_signup_date DATE, previously_subscribed BOOLEAN,
  total_account_subscriptions INTEGER, country STRING, state STRING,
  postal_code STRING, subscription STRING, subscription_order INTEGER,
  status STRING, status_reporting STRING, platform STRING,
  created_month DATE, created_date DATE, cancelled_month DATE,
  cancellation_contact_date DATE, cancelled_date DATE,
  cancellation_type STRING, cancel_code STRING, product STRING,
  currency STRING, bill_cycle_day INTEGER, billing_period STRING,
  current_term_start DATE, current_term_end DATE, step_up_date DATE,
  total FLOAT, total_with_discount FLOAT, discount_amount FLOAT,
  discount_months INTEGER, quantity INTEGER, payments INTEGER,
  next_payment_amount NUMERIC, next_payment_date DATE,
  last_invoice_date DATE, last_payment_amount NUMERIC,
  last_payment_date DATE, last_transaction_status STRING,
  last_gateway_response STRING, gateway STRING, payment_method STRING,
  credit_card_brand STRING, rate_level STRING, current_plan STRING,
  current_coupon STRING, coupon_start_date DATE, coupon_end_date DATE,
  current_rate_plan_payments INTEGER, current_promotion_type STRING,
  current_rate_plan STRING, current_rate_plan_id STRING,
  current_version INTEGER, current_rate_plan_created_date DATE,
  current_rate_plan_start_date DATE, current_rate_plan_end_date DATE,
  original_promotion_type STRING, original_rate_plan STRING,
  original_rate_plan_id STRING, original_rate_plan_created_date DATE,
  original_rate_plan_start_date DATE, original_rate_plan_end_date DATE,
  previous_promotion_type STRING, previous_rate_plan STRING,
  previous_rate_plan_id STRING, previous_rate_plan_created_date DATE,
  previous_rate_plan_start_date DATE, previous_rate_plan_end_date DATE,
  future_promotion_type STRING, future_rate_plan STRING,
  future_rate_plan_id STRING, future_rate_plan_created_date DATE,
  future_rate_plan_start_date DATE
);

CREATE TABLE `{BQ_DATASET}.cb_subscriptions_coupons` (
  coupon_id STRING, coupon_code_id STRING, subscription_id STRING,
  apply_till TIMESTAMP
);"""


# ── Glossary and usage rules (from the Data Analytics team document) ─────────────────────

GLOSSARY_AND_RULES = """
Glossary:
- cancelled_date can be NULL (subscription still active) or contain future dates.
- Cancellation Rate = (cancelled during period) / (active at start of period).
- A rate plan is not a subscription; one subscription can have multiple rate plans over time.
- total_page_views includes page_view, campaign_screen_view, and custom_screen_view events.

Usage Rules:
- [join rule] Join cb_subscriptions to other tables via cf_account_id, NOT customer_id.
- [join rule] bconnect_id in subscriptions_web is NOT unique -- do not left join on it carelessly.
- [reporting rule] Use status_reporting for high-level churn/active reporting.
- [reporting rule] Use cancellation_type + cancel_code for churn analytics.
- [dates rule] cancelled_date and created_date can be in the future; filter with <= CURRENT_DATE() when needed.
- [financial rule] Aggregate amounts per currency; never sum across currencies.
- [financial rule] Use mrr ONLY when user explicitly asks for Monthly Recurring Revenue; otherwise use plan_amount.
- [segmentation rule] Web subscriptions: cf_channel = 'web'. Use LOWER(cf_is_bulk_subscription) for self-serve check.
- [segmentation rule] subscriptions_web excludes group/bulk subscriptions by design.
"""


# ── Prompt templates ─────────────────────────────────────────────────────────

COORDINATOR_SYSTEM = (
    "You are the Data Analytics agent coordinator. You orchestrate three task agents:\n"
    "1. planning_agent - Retrieves context: get_catalog, get_memory, get_details (loops up to N=3)\n"
    "2. engineer_agent - Generates SQL from the plan, returns SUCCESS or BLOCKING_AMBIGUITY\n"
    "3. response_agent - Builds the final user-facing response\n"
    "Route the user's question through these agents. If the question doesn't require SQL, route directly to response_agent."
)

COORDINATOR_ROUTING_TEMPLATE = (
    "User question: {question}\n\nCurrent state: {state}\n\n"
    "Decide the next step. Return JSON:\n"
    '  {{"action": "<route_to_agent|execute_sql|return_response>", "target": "<agent_name>", "reasoning": "<explanation>"}}'
)

PLANNING_SYSTEM = (
    "You are the Planning Agent. Your job is to retrieve context for the user's question.\n"
    "Steps: 1) get_catalog to fetch available tables, 2) get_memory for domain knowledge,\n"
    "3) select relevant tables, 4) get_details for each selected table.\n"
    "You may iterate up to 3 times if context is insufficient.\n\n"
    + GLOSSARY_AND_RULES
)

ENGINEER_SYSTEM = (
    "You are the Engineer Agent. You consume the Planning Agent's output and generate SQL.\n"
    "If the query is ambiguous and you cannot produce a reliable SQL query, return BLOCKING_AMBIGUITY.\n"
    "Otherwise return SUCCESS with the generated SQL.\n\n"
    "Rules:\n"
    "- Use BigQuery Standard SQL syntax\n"
    "- Always alias aggregated columns\n"
    "- Never sum across currencies\n"
    "- Use LOWER(cf_is_bulk_subscription) for self-serve filtering\n"
    "- Filter future dates with <= CURRENT_DATE() when counting current records"
)

RESPONSE_SYSTEM = (
    "You are the Response Agent. Build a clear, concise response for the user.\n"
    "Three modes:\n"
    "1. Success: summarize results with row count and key findings\n"
    "2. Execution failure: explain what went wrong\n"
    "3. Clarification: ask the user a specific clarifying question"
)

PLANNING_TABLE_SELECTION_TEMPLATE = (
    "Select relevant tables for: {question}\nAvailable: {available_tables}"
)

ENGINEER_SQL_TEMPLATE = (
    "Generate SQL for: {question}\nTables: {tables}\nContext: {context}"
)

RESPONSE_TEMPLATES = {
    "success": "Summarize results for: {question}\nRow count: {row_count}\nData: {data}",
    "failure": "Explain execution failure for: {question}\nError: {error}",
    "clarification": "Ask the user to clarify: {question}\nAmbiguity: {ambiguity_reason}",
    "direct": "Answer directly (no SQL): {question}",
}


# ── Query bank ───────────────────────────────────────────────────────────────

QUERY_BANK = [
    # Golden query 1: cb_subscriptions_q1
    {
        "question": "Generate a SQL query to retrieve all bulk (self-serve) subscriptions, including their creation date, cancellation date, seat quantity, plan amount, currency, and status.",
        "tables": ["cb_subscriptions"],
        "sql": f"SELECT cb.id, cb.created_at AS CreatedDate, cb.cancelled_at AS CancelledDate, cb.plan_quantity AS Quantity, cb.plan_amount AS Amount, cb.currency AS Currency, cb.status AS Status FROM `{BQ_DATASET}.cb_subscriptions` AS cb WHERE LOWER(cb.cf_is_bulk_subscription)='true' ORDER BY cb.created_at DESC",
        "result": {"status": "success", "rows": [{"id": "sub_HkZ9x3", "CreatedDate": "2025-02-15T10:30:00Z", "CancelledDate": None, "Quantity": 25, "Amount": 12500.00, "Currency": "USD", "Status": "active"}, {"id": "sub_Jm4Kp1", "CreatedDate": "2025-01-08T14:22:00Z", "CancelledDate": "2025-06-30T00:00:00Z", "Quantity": 10, "Amount": 5000.00, "Currency": "USD", "Status": "non_renewing"}], "row_count": 847, "column_names": ["id", "CreatedDate", "CancelledDate", "Quantity", "Amount", "Currency", "Status"]},
        "plan": "Retrieve all bulk/self-serve subscriptions using cf_is_bulk_subscription filter with key billing fields.",
        "answer": "Found 847 bulk (self-serve) subscriptions. The most recent was created on 2025-02-15 with 25 seats at $12,500 USD. Active subscriptions predominate, with some in non_renewing status.",
        "complexity": "simple",
    },
    # Golden query 2: subscriptions_web_q1
    {
        "question": "What are the number of cancellations of web subscriptions every day in the current year? In final result, order the table by cancelled_date, in reverse order.",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT cancelled_date, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE cancelled_date >= DATE_TRUNC(CURRENT_DATE(), YEAR) AND cancelled_date <= CURRENT_DATE() GROUP BY cancelled_date ORDER BY cancelled_date DESC",
        "result": {"status": "success", "rows": [{"cancelled_date": "2025-03-21", "ct": 42}, {"cancelled_date": "2025-03-20", "ct": 38}, {"cancelled_date": "2025-03-19", "ct": 45}, {"cancelled_date": "2025-03-18", "ct": 31}], "row_count": 80, "column_names": ["cancelled_date", "ct"]},
        "plan": "Count daily cancellations for the current year from subscriptions_web, filtering cancelled_date <= CURRENT_DATE() to exclude future-dated cancellations.",
        "answer": "Daily web subscription cancellations for 2025 so far: averaging ~39 cancellations per day. The highest day was March 19 with 45 cancellations. Data covers 80 days year-to-date.",
        "complexity": "aggregation",
    },
    # Golden query 3: cb_subscriptions_coupons_q1
    {
        "question": "Generate a SQL query to retrieve all coupons applied to each subscription, with the apply-till timestamp converted to America/New_York. The output should include the coupon ID, subscription ID, and apply-till timestamp.",
        "tables": ["cb_subscriptions_coupons"],
        "sql": f"SELECT coupon_id, subscription_id, DATETIME(apply_till, 'America/New_York') AS apply_till FROM `{BQ_DATASET}.cb_subscriptions_coupons`",
        "result": {"status": "success", "rows": [{"coupon_id": "intro_50_off_annual", "subscription_id": "sub_HkZ9x3", "apply_till": "2025-08-15T06:00:00"}, {"coupon_id": "winback_30_off", "subscription_id": "sub_Jm4Kp1", "apply_till": "2025-04-01T20:00:00"}], "row_count": 3250, "column_names": ["coupon_id", "subscription_id", "apply_till"]},
        "plan": "Retrieve all coupon-subscription mappings with apply_till converted to Eastern time.",
        "answer": "Retrieved 3,250 coupon applications across all subscriptions. Timestamps have been converted to America/New_York timezone. Most common coupons include intro_50_off_annual and winback_30_off.",
        "complexity": "simple",
    },
    # MRR by currency
    {
        "question": "What is the total Monthly Recurring Revenue (MRR) broken down by currency?",
        "tables": ["cb_subscriptions"],
        "sql": f"SELECT currency, SUM(mrr) AS total_mrr FROM `{BQ_DATASET}.cb_subscriptions` WHERE status = 'active' GROUP BY currency ORDER BY total_mrr DESC",
        "result": {"status": "success", "rows": [{"currency": "USD", "total_mrr": 2845000.00}, {"currency": "GBP", "total_mrr": 412000.00}, {"currency": "EUR", "total_mrr": 378000.00}], "row_count": 3, "column_names": ["currency", "total_mrr"]},
        "plan": "Aggregate MRR from cb_subscriptions grouped by currency. Using mrr column since user explicitly asked for MRR.",
        "answer": "Total MRR across active subscriptions: $2.85M USD, £412K GBP, €378K EUR. USD accounts for ~78% of recurring revenue.",
        "complexity": "aggregation",
    },
    # Active vs cancelled
    {
        "question": "How many web subscriptions are currently active versus cancelled?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT status_reporting, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE created_date <= CURRENT_DATE() GROUP BY status_reporting ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"status_reporting": "Active", "ct": 124500}, {"status_reporting": "Cancelled", "ct": 68200}], "row_count": 2, "column_names": ["status_reporting", "ct"]},
        "plan": "Count subscriptions by status_reporting (the reporting-friendly field) from subscriptions_web.",
        "answer": "Currently 124,500 active and 68,200 cancelled web subscriptions, giving an overall churn rate of approximately 35%.",
        "complexity": "simple",
    },
    # Rate plan distribution
    {
        "question": "What is the distribution of current rate plans across active web subscriptions?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT current_rate_plan, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY current_rate_plan ORDER BY ct DESC LIMIT 10",
        "result": {"status": "success", "rows": [{"current_rate_plan": "Digital Annual Full Price", "ct": 42100}, {"current_rate_plan": "All Access Monthly", "ct": 28300}, {"current_rate_plan": "Digital Monthly Intro Offer", "ct": 18700}, {"current_rate_plan": "Student Annual", "ct": 8400}], "row_count": 4, "column_names": ["current_rate_plan", "ct"]},
        "plan": "Count active subscriptions grouped by current_rate_plan from subscriptions_web.",
        "answer": "Top rate plans: Digital Annual Full Price (42,100), All Access Monthly (28,300), Digital Monthly Intro Offer (18,700), Student Annual (8,400).",
        "complexity": "aggregation",
    },
    # Payment method distribution
    {
        "question": "What payment methods are our active subscribers using?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT payment_method, COUNT(*) AS ct, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY payment_method ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"payment_method": "card", "ct": 89200, "pct": 71.7}, {"payment_method": "apple_pay", "ct": 18500, "pct": 14.9}, {"payment_method": "paypal_express_checkout", "ct": 10800, "pct": 8.7}, {"payment_method": "google_pay", "ct": 5100, "pct": 4.1}], "row_count": 4, "column_names": ["payment_method", "ct", "pct"]},
        "plan": "Count active subscriptions by payment_method with percentage.",
        "answer": "Card payments dominate at 71.7% (89,200 subscribers), followed by Apple Pay (14.9%), PayPal (8.7%), and Google Pay (4.1%).",
        "complexity": "aggregation",
    },
    # Discount vs full price
    {
        "question": "How many active subscribers are on a discount versus full price?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT rate_level, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY rate_level ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"rate_level": "full_price", "ct": 72400}, {"rate_level": "discount", "ct": 43800}, {"rate_level": "free_trial", "ct": 8300}], "row_count": 3, "column_names": ["rate_level", "ct"]},
        "plan": "Count active subscribers by rate_level from subscriptions_web.",
        "answer": "72,400 at full price (58%), 43,800 on discount (35%), and 8,300 on free trial (7%). Over a third of active subscribers are still on promotional pricing.",
        "complexity": "simple",
    },
    # Cancellation by type
    {
        "question": "Break down cancellations by cancellation type for Q1 2025.",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT cancellation_type, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE cancelled_date >= '2025-01-01' AND cancelled_date <= '2025-03-31' AND cancelled_date <= CURRENT_DATE() GROUP BY cancellation_type ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"cancellation_type": "elected", "ct": 4200}, {"cancellation_type": "payments", "ct": 2100}, {"cancellation_type": "internal", "ct": 850}, {"cancellation_type": "disputes", "ct": 320}, {"cancellation_type": "group", "ct": 180}], "row_count": 5, "column_names": ["cancellation_type", "ct"]},
        "plan": "Count Q1 2025 cancellations by cancellation_type, filtering <= CURRENT_DATE() for future-dated records.",
        "answer": "Q1 2025 cancellations by type: elected (4,200 — 55%), payments (2,100 — 27%), internal (850), disputes (320), group (180). Voluntary churn (elected) is the primary driver.",
        "complexity": "aggregation",
    },
    # Multi-table join
    {
        "question": "Show me the plan amount and coupon details for active web subscriptions that have coupons applied.",
        "tables": ["subscriptions_web", "cb_subscriptions", "cb_subscriptions_coupons"],
        "sql": f"SELECT sw.subscription, cb.plan_amount, cb.currency, c.coupon_id, DATETIME(c.apply_till, 'America/New_York') AS coupon_expires FROM `{BQ_DATASET}.subscriptions_web` sw JOIN `{BQ_DATASET}.cb_subscriptions` cb ON sw.bconnect_id = cb.cf_account_id JOIN `{BQ_DATASET}.cb_subscriptions_coupons` c ON cb.id = c.subscription_id WHERE sw.status_reporting = 'Active' ORDER BY cb.plan_amount DESC LIMIT 100",
        "result": {"status": "success", "rows": [{"subscription": "sub_HkZ9x3", "plan_amount": 449.00, "currency": "USD", "coupon_id": "intro_50_off_annual", "coupon_expires": "2025-08-15T06:00:00"}], "row_count": 12400, "column_names": ["subscription", "plan_amount", "currency", "coupon_id", "coupon_expires"]},
        "plan": "Three-table join: subscriptions_web -> cb_subscriptions via bconnect_id/cf_account_id -> cb_subscriptions_coupons via subscription_id. Filter active only.",
        "answer": "12,400 active web subscriptions have coupons applied. The highest plan amount is $449 USD with an intro_50_off_annual coupon expiring August 2025.",
        "complexity": "multi_hop",
    },
    # Step-up date analysis
    {
        "question": "How many active subscribers have a step-up date in the next 90 days?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT COUNT(*) AS upcoming_stepups FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' AND step_up_date BETWEEN CURRENT_DATE() AND DATE_ADD(CURRENT_DATE(), INTERVAL 90 DAY)",
        "result": {"status": "success", "rows": [{"upcoming_stepups": 8750}], "row_count": 1, "column_names": ["upcoming_stepups"]},
        "plan": "Count active subscribers with step_up_date within next 90 days — these are subscribers whose promotional pricing is about to expire.",
        "answer": "8,750 active subscribers will step up to full price within the next 90 days. This is a key churn risk cohort — promotional pricing expiring is a common cancellation trigger.",
        "complexity": "simple",
    },
    # Cancellation rate calculation
    {
        "question": "What is the 3-month cancellation rate for web subscriptions?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT ROUND(COUNTIF(cancelled_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND CURRENT_DATE()) * 100.0 / COUNTIF(created_date <= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND (cancelled_date IS NULL OR cancelled_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH))), 2) AS cancellation_rate_pct FROM `{BQ_DATASET}.subscriptions_web`",
        "result": {"status": "success", "rows": [{"cancellation_rate_pct": 8.42}], "row_count": 1, "column_names": ["cancellation_rate_pct"]},
        "plan": "Calculate 3-month cancellation rate: (cancelled in last 3 months) / (active at start of 3-month window). Using cancelled_date <= CURRENT_DATE() to exclude future-dated cancellations.",
        "answer": "The 3-month cancellation rate is 8.42%. This means roughly 1 in 12 subscribers who were active 3 months ago have since cancelled.",
        "complexity": "aggregation",
    },
    # Billing period breakdown
    {
        "question": "What is the breakdown of billing periods for active subscriptions?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT billing_period, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY billing_period ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"billing_period": "Annual", "ct": 68200}, {"billing_period": "Month", "ct": 45100}, {"billing_period": "Semi_Annual", "ct": 8200}, {"billing_period": "Two_Years", "ct": 2400}], "row_count": 4, "column_names": ["billing_period", "ct"]},
        "plan": "Count active subscriptions by billing_period.",
        "answer": "Annual leads at 68,200 (55%), monthly at 45,100 (36%), semi-annual at 8,200 (7%), and two-year at 2,400 (2%).",
        "complexity": "simple",
    },
    # Product distribution
    {
        "question": "How many subscribers are on each product type?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT product, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY product ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"product": "Digital", "ct": 78500}, {"product": "All Access", "ct": 35200}, {"product": "Student", "ct": 7800}, {"product": "Newsletter", "ct": 3000}], "row_count": 4, "column_names": ["product", "ct"]},
        "plan": "Count active subscribers by product type from subscriptions_web.",
        "answer": "Digital leads with 78,500 subscribers (63%), All Access at 35,200 (28%), Student at 7,800 (6%), and Newsletter at 3,000 (2%).",
        "complexity": "simple",
    },
    # Edge case: cross-currency (adversarial)
    {
        "question": "What is the total plan amount across all active subscriptions?",
        "tables": ["cb_subscriptions"],
        "sql": f"SELECT currency, SUM(plan_amount) AS total_amount FROM `{BQ_DATASET}.cb_subscriptions` WHERE status = 'active' GROUP BY currency ORDER BY total_amount DESC",
        "result": {"status": "success", "rows": [{"currency": "USD", "total_amount": 14250000.00}, {"currency": "GBP", "total_amount": 2180000.00}, {"currency": "EUR", "total_amount": 1920000.00}], "row_count": 3, "column_names": ["currency", "total_amount"]},
        "plan": "Aggregate plan_amount per currency (never cross-currency sums per usage rules).",
        "answer": "Total plan amounts by currency: $14.25M USD, £2.18M GBP, €1.92M EUR. Note: amounts are reported per currency and cannot be summed across currencies without conversion.",
        "complexity": "aggregation",
        "is_adversarial": True,
    },
    # Edge case: future-dated records
    {
        "question": "How many subscriptions were created this year?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE created_date >= DATE_TRUNC(CURRENT_DATE(), YEAR) AND created_date <= CURRENT_DATE()",
        "result": {"status": "success", "rows": [{"ct": 28400}], "row_count": 1, "column_names": ["ct"]},
        "plan": "Count subscriptions created in current year, filtering created_date <= CURRENT_DATE() to exclude future-dated records per usage rules.",
        "answer": "28,400 web subscriptions have been created so far in 2025. This excludes future-dated records.",
        "complexity": "simple",
        "is_adversarial": True,
    },
    # Edge case: schema mismatch
    {
        "question": "What is the average customer lifetime value?",
        "tables": ["subscriptions_web", "cb_subscriptions"],
        "sql": f"SELECT sw.product, AVG(sw.payments * cb.plan_amount) AS approx_ltv, sw.currency FROM `{BQ_DATASET}.subscriptions_web` sw JOIN `{BQ_DATASET}.cb_subscriptions` cb ON sw.bconnect_id = cb.cf_account_id GROUP BY sw.product, sw.currency ORDER BY approx_ltv DESC",
        "result": {"status": "success", "rows": [{"product": "All Access", "approx_ltv": 2840.00, "currency": "USD"}, {"product": "Digital", "approx_ltv": 1560.00, "currency": "USD"}], "row_count": 8, "column_names": ["product", "approx_ltv", "currency"]},
        "plan": "No direct LTV field exists. Approximating as payments * plan_amount per product and currency. This is a rough estimate.",
        "answer": "No explicit LTV field is available. Approximate LTV by product (USD): All Access ~$2,840, Digital ~$1,560. This is computed as (number of payments * plan amount) and should be treated as an approximation.",
        "complexity": "multi_hop",
    },
]

# Ambiguous queries that trigger BLOCKING_AMBIGUITY
AMBIGUOUS_QUERIES = [
    {
        "question": "How many subscriptions were cancelled?",
        "ambiguity_reason": "Multiple tables contain cancellation data (cb_subscriptions.cancelled_at vs subscriptions_web.cancelled_date). Also unclear what time period is intended.",
        "clarification_question": "Could you clarify: (1) Are you asking about BillingCo billing cancellations (cb_subscriptions) or web subscription lifecycle cancellations (subscriptions_web)? (2) What time period — all time, current year, or a specific range?",
        "clarified_question": "How many web subscriptions were cancelled in the current year?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE cancelled_date >= DATE_TRUNC(CURRENT_DATE(), YEAR) AND cancelled_date <= CURRENT_DATE()",
        "result": {"status": "success", "rows": [{"ct": 7650}], "row_count": 1, "column_names": ["ct"]},
        "plan": "Count web subscription cancellations for the current year.",
        "answer": "7,650 web subscriptions have been cancelled so far in 2025.",
        "complexity": "simple",
    },
    {
        "question": "What's the revenue?",
        "ambiguity_reason": "Unclear whether user wants MRR (monthly recurring), plan_amount (invoice amount), or total revenue. Also unclear on currency and time period.",
        "clarification_question": "Could you clarify what revenue metric you're looking for? Options: (1) Monthly Recurring Revenue (MRR) — ongoing subscription revenue, (2) Plan amounts — invoice amounts for current billing period. Also, for which currency and time period?",
        "clarified_question": "What is the total MRR by currency for active subscriptions?",
        "tables": ["cb_subscriptions"],
        "sql": f"SELECT currency, SUM(mrr) AS total_mrr FROM `{BQ_DATASET}.cb_subscriptions` WHERE status = 'active' GROUP BY currency",
        "result": {"status": "success", "rows": [{"currency": "USD", "total_mrr": 2845000.00}, {"currency": "GBP", "total_mrr": 412000.00}], "row_count": 2, "column_names": ["currency", "total_mrr"]},
        "plan": "Sum MRR by currency for active subscriptions. User explicitly asked for MRR so using mrr column.",
        "answer": "Total MRR: $2.85M USD, £412K GBP.",
        "complexity": "aggregation",
    },
    {
        "question": "Show me active subscribers",
        "ambiguity_reason": "status='active' in cb_subscriptions vs status_reporting='Active' in subscriptions_web return different counts. Also unclear what fields to show.",
        "clarification_question": "There are two ways to identify active subscribers: (1) BillingCo status (cb_subscriptions.status = 'active') or (2) Reporting status (subscriptions_web.status_reporting = 'Active', which includes non_renewing). Which would you prefer, and what subscriber details do you need?",
        "clarified_question": "Show me the count of active web subscriptions using the reporting status, broken down by product.",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT product, COUNT(*) AS ct FROM `{BQ_DATASET}.subscriptions_web` WHERE status_reporting = 'Active' GROUP BY product ORDER BY ct DESC",
        "result": {"status": "success", "rows": [{"product": "Digital", "ct": 78500}, {"product": "All Access", "ct": 35200}], "row_count": 4, "column_names": ["product", "ct"]},
        "plan": "Count active (status_reporting) subscribers by product.",
        "answer": "Active web subscribers by product: Digital (78,500), All Access (35,200), Student (7,800), Newsletter (3,000).",
        "complexity": "simple",
    },
    {
        "question": "What's the cancellation rate?",
        "ambiguity_reason": "Cancellation rate requires a defined period and denominator. Standard formula is (cancelled during period) / (active at start of period), but the period is unspecified.",
        "clarification_question": "To calculate the cancellation rate, I need to know: (1) What time period? (e.g., last 30 days, last 3 months, last year), (2) Which subscriber base — all web subscriptions or a specific product/segment?",
        "clarified_question": "What is the 3-month cancellation rate for all web subscriptions?",
        "tables": ["subscriptions_web"],
        "sql": f"SELECT ROUND(COUNTIF(cancelled_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND CURRENT_DATE()) * 100.0 / COUNTIF(created_date <= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND (cancelled_date IS NULL OR cancelled_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH))), 2) AS cancellation_rate_pct FROM `{BQ_DATASET}.subscriptions_web`",
        "result": {"status": "success", "rows": [{"cancellation_rate_pct": 8.42}], "row_count": 1, "column_names": ["cancellation_rate_pct"]},
        "plan": "Calculate 3-month cancellation rate using standard formula.",
        "answer": "The 3-month cancellation rate is 8.42%.",
        "complexity": "aggregation",
    },
]

# Queries that don't need SQL
NO_SQL_QUERIES = [
    {"question": "What tables are available?", "answer": "Three tables are available: cb_subscriptions (BillingCo billing data), subscriptions_web (web subscription lifecycle), and cb_subscriptions_coupons (coupon-subscription mappings)."},
    {"question": "What does MRR stand for?", "answer": "MRR stands for Monthly Recurring Revenue — the recurring revenue earned from active subscriptions on a monthly basis."},
    {"question": "How do I join subscriptions_web with cb_subscriptions?", "answer": "Join using subscriptions_web.bconnect_id = cb_subscriptions.cf_account_id. Do not use customer_id as that is BillingCo's internal ID and is less reliable."},
]

BAD_SQL_EXAMPLES = [
    {"sql": f"SELECT SUM(plan_amount) FROM `{BQ_DATASET}.cb_subscriptions` WHERE status = 'active'", "error": "Cross-currency aggregation: plan_amount summed without GROUP BY currency violates financial rules."},
    {"sql": f"SELECT * FROM `{BQ_DATASET}.subscriptions_web` WHERE customer_id = 'abc'", "error": "Column customer_id does not exist in subscriptions_web. Did you mean bconnect_id or findata_customer_id?"},
    {"sql": f"SELCT cancelled_date FROM `{BQ_DATASET}.subscriptions_web`", "error": "Syntax error: SELCT is not a valid keyword. Did you mean SELECT?"},
    {"sql": f"SELECT cancelled_date, COUNT(*) FROM `{BQ_DATASET}.subscriptions_web` GROUP BY", "error": "Syntax error near GROUP BY: missing column specification."},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def timer():
    start = time.time()
    yield
    print(f"Execution time: {time.time() - start:.2f}s")


def _span_id(span):
    return format(span.get_span_context().span_id, "016x")


def _set_llm_attrs(span, *, system_msg=None, user_msg, assistant_msg,
                   prompt_tokens=None, completion_tokens=None,
                   template=None, template_version=None, template_vars=None):
    span.set_attribute(_KIND, "LLM")
    span.set_attribute("llm.model_name", GEMINI_MODEL)
    span.set_attribute("llm.provider", LLM_PROVIDER)
    span.set_attribute("llm.system", LLM_SYSTEM)
    idx = 0
    if system_msg:
        span.set_attribute(f"llm.input_messages.{idx}.message.role", "system")
        span.set_attribute(f"llm.input_messages.{idx}.message.content", system_msg)
        idx += 1
    span.set_attribute(f"llm.input_messages.{idx}.message.role", "user")
    span.set_attribute(f"llm.input_messages.{idx}.message.content", user_msg)
    span.set_attribute("llm.output_messages.0.message.role", "assistant")
    span.set_attribute("llm.output_messages.0.message.content", assistant_msg)
    pt = prompt_tokens or random.randint(150, 600)
    ct = completion_tokens or random.randint(40, 300)
    span.set_attribute("llm.token_count.prompt", pt)
    span.set_attribute("llm.token_count.completion", ct)
    span.set_attribute("llm.token_count.total", pt + ct)
    if template:
        span.set_attribute("llm.prompt_template.template", template)
    if template_version:
        span.set_attribute("llm.prompt_template.version", template_version)
    if template_vars:
        span.set_attribute("llm.prompt_template.variables", json.dumps(template_vars))


def _set_tool_attrs(span, *, tool_name, input_val, output_val, is_mcp=False,
                    mcp_server=None, mcp_version="1.0"):
    span.set_attribute(_KIND, "TOOL")
    span.set_attribute("tool.name", tool_name)
    span.set_attribute(_INPUT, input_val if isinstance(input_val, str) else json.dumps(input_val))
    span.set_attribute(_INPUT_MIME, _JSON if not isinstance(input_val, str) else _TEXT)
    span.set_attribute(_OUTPUT, output_val if isinstance(output_val, str) else json.dumps(output_val))
    span.set_attribute(_OUTPUT_MIME, _JSON)
    if is_mcp and mcp_server:
        span.set_attribute("mcp.server.name", mcp_server)
        span.set_attribute("mcp.server.version", mcp_version)


# ── Span builders ────────────────────────────────────────────────────────────

def _emit_coordinator_routing(tracer, question, state_summary, decision, session_id):
    user_msg = COORDINATOR_ROUTING_TEMPLATE.format(question=question, state=state_summary)
    with tracer.start_as_current_span("AsyncGenerateContent") as span:
        _set_llm_attrs(
            span, system_msg=COORDINATOR_SYSTEM, user_msg=user_msg,
            assistant_msg=json.dumps(decision),
            prompt_tokens=random.randint(250, 450),
            completion_tokens=random.randint(40, 120),
            template=COORDINATOR_ROUTING_TEMPLATE,
            template_version="v1.0",
            template_vars={"question": question[:200], "state": state_summary[:200]},
        )
        if session_id:
            span.set_attribute("session.id", session_id)
        span.set_attribute("latency_ms", round(random.uniform(400, 1200), 2))
        span.set_status(Status(StatusCode.OK))
        time.sleep(0.002)
        return _span_id(span)


def _emit_planning_agent(tracer, scenario, session_id, *, iterations=1):
    question = scenario["question"]
    tables = scenario["tables"]

    with tracer.start_as_current_span("agent_run [planning_agent]") as agent:
        agent.set_attribute(_KIND, "AGENT")
        agent.set_attribute("agent.name", "planning_agent")
        agent.set_attribute(_INPUT, question)
        agent.set_attribute(_INPUT_MIME, _TEXT)
        agent.set_attribute("planning.max_iterations", 3)
        agent.set_attribute("planning.actual_iterations", iterations)
        if session_id:
            agent.set_attribute("session.id", session_id)
        time.sleep(0.002)

        for iteration in range(1, iterations + 1):
            iter_suffix = f" (iter {iteration})" if iterations > 1 else ""

            # Step 1: get_catalog
            with tracer.start_as_current_span(f"get_catalog{iter_suffix}") as cat:
                catalog_output = [{"table": t, "description": TABLE_DESCRIPTIONS[t]} for t in ALL_TABLES]
                _set_tool_attrs(cat, tool_name="get_catalog",
                    input_val={"scope": "da_agent_eval_dataset"},
                    output_val={"tables": catalog_output},
                    is_mcp=True, mcp_server="findata-db")
                if session_id:
                    cat.set_attribute("session.id", session_id)
                cat.set_attribute("latency_ms", round(random.uniform(50, 200), 2))
                cat.set_status(Status(StatusCode.OK))
                time.sleep(0.002)

            # Step 2: get_memory (always called)
            with tracer.start_as_current_span(f"get_memory{iter_suffix}") as mem:
                _set_tool_attrs(mem, tool_name="get_memory",
                    input_val={"user_query": question},
                    output_val={"glossary": GLOSSARY_AND_RULES[:500], "prior_queries": []},
                    is_mcp=True, mcp_server="findata-db")
                if session_id:
                    mem.set_attribute("session.id", session_id)
                mem.set_attribute("latency_ms", round(random.uniform(80, 300), 2))
                mem.set_status(Status(StatusCode.OK))
                time.sleep(0.002)

            # Select N related tables (LLM)
            ranked_tables = [{"table": t, "score": round(random.uniform(0.6, 0.99), 2)} for t in tables]
            if iterations > 1 and iteration < iterations:
                extra = [t for t in ALL_TABLES if t not in tables]
                if extra:
                    ranked_tables.append({"table": extra[0], "score": round(random.uniform(0.3, 0.5), 2)})
            ranked_tables.sort(key=lambda x: -x["score"])

            with tracer.start_as_current_span(f"AsyncGenerateContent{iter_suffix}") as sel:
                plan_user_msg = PLANNING_TABLE_SELECTION_TEMPLATE.format(
                    question=question, available_tables=ALL_TABLES)
                _set_llm_attrs(sel, system_msg=PLANNING_SYSTEM,
                    user_msg=plan_user_msg,
                    assistant_msg=json.dumps({"selected_tables": ranked_tables}),
                    prompt_tokens=random.randint(400, 700),
                    completion_tokens=random.randint(60, 150),
                    template=PLANNING_TABLE_SELECTION_TEMPLATE,
                    template_version="v1.0",
                    template_vars={"question": question, "available_tables": str(ALL_TABLES)})
                if session_id:
                    sel.set_attribute("session.id", session_id)
                sel.set_attribute("latency_ms", round(random.uniform(800, 2000), 2))
                sel.set_attribute("planning.selected_tables", json.dumps(ranked_tables))
                sel.set_status(Status(StatusCode.OK))
                time.sleep(0.002)

            # Step 3: get_details for each selected table
            for tbl in ranked_tables:
                with tracer.start_as_current_span(f"get_details{iter_suffix}") as det:
                    _set_tool_attrs(det, tool_name="get_details",
                        input_val={"table": tbl["table"]},
                        output_val={"table": tbl["table"], "schema": f"[columns for {tbl['table']}]", "sample_rows": 3},
                        is_mcp=True, mcp_server="findata-db")
                    if session_id:
                        det.set_attribute("session.id", session_id)
                    det.set_attribute("latency_ms", round(random.uniform(100, 400), 2))
                    det.set_status(Status(StatusCode.OK))
                    time.sleep(0.002)

        plan_output = {
            "user_query": question,
            "selected_tables": [t["table"] for t in ranked_tables],
            "retrieved_context": f"Schema and details for {len(ranked_tables)} tables retrieved.",
        }
        agent.set_attribute(_OUTPUT, json.dumps(plan_output))
        agent.set_attribute(_OUTPUT_MIME, _JSON)
        agent.set_attribute("latency_ms", round(random.uniform(2000, 8000), 2))
        agent.set_status(Status(StatusCode.OK))
        planning_span_id = _span_id(agent)

    return plan_output, planning_span_id


def _emit_engineer_agent(tracer, scenario, session_id, *, output_status="SUCCESS"):
    question = scenario["question"]

    with tracer.start_as_current_span("agent_run [engineer_agent]") as agent:
        agent.set_attribute(_KIND, "AGENT")
        agent.set_attribute("agent.name", "engineer_agent")
        agent.set_attribute(_INPUT, json.dumps({"user_query": question, "tables": scenario["tables"]}))
        agent.set_attribute(_INPUT_MIME, _JSON)
        if session_id:
            agent.set_attribute("session.id", session_id)
        time.sleep(0.002)

        if output_status == "SUCCESS":
            sql_output = {"status": "SUCCESS", "sql": scenario["sql"]}
            with tracer.start_as_current_span("AsyncGenerateContent") as llm:
                eng_version = scenario.get("prompt_version", "v1.0")
                eng_vars = {"question": question, "tables": str(scenario["tables"]), "context": scenario.get("plan", "")}
                eng_user_msg = ENGINEER_SQL_TEMPLATE.format(**eng_vars)
                _set_llm_attrs(llm, system_msg=ENGINEER_SYSTEM,
                    user_msg=eng_user_msg,
                    assistant_msg=json.dumps(sql_output),
                    prompt_tokens=random.randint(500, 900),
                    completion_tokens=random.randint(100, 400),
                    template=ENGINEER_SQL_TEMPLATE,
                    template_version=eng_version,
                    template_vars=eng_vars)
                if session_id:
                    llm.set_attribute("session.id", session_id)
                llm.set_attribute("latency_ms", round(random.uniform(1000, 3000), 2))
                llm.set_status(Status(StatusCode.OK))
                time.sleep(0.002)
        else:
            ambiguity = scenario.get("ambiguity_reason", "Query is ambiguous.")
            sql_output = {"status": "BLOCKING_AMBIGUITY", "reason": ambiguity}
            with tracer.start_as_current_span("AsyncGenerateContent") as llm:
                eng_version = scenario.get("prompt_version", "v1.0")
                eng_vars = {"question": question, "tables": str(scenario["tables"]), "context": ""}
                eng_user_msg = ENGINEER_SQL_TEMPLATE.format(**eng_vars)
                _set_llm_attrs(llm, system_msg=ENGINEER_SYSTEM,
                    user_msg=eng_user_msg,
                    assistant_msg=json.dumps(sql_output),
                    prompt_tokens=random.randint(500, 800),
                    completion_tokens=random.randint(80, 200),
                    template=ENGINEER_SQL_TEMPLATE,
                    template_version=eng_version,
                    template_vars=eng_vars)
                if session_id:
                    llm.set_attribute("session.id", session_id)
                llm.set_attribute("latency_ms", round(random.uniform(1000, 2500), 2))
                llm.set_status(Status(StatusCode.OK))
                time.sleep(0.002)

        agent.set_attribute(_OUTPUT, json.dumps(sql_output))
        agent.set_attribute(_OUTPUT_MIME, _JSON)
        agent.set_attribute("engineer.output_status", output_status)
        agent.set_attribute("latency_ms", round(random.uniform(1500, 4000), 2))
        agent.set_status(Status(StatusCode.OK))
        engineer_span_id = _span_id(agent)

    return sql_output, engineer_span_id


def _emit_execute_sql(tracer, scenario, session_id):
    with tracer.start_as_current_span("execute_sql") as span:
        _set_tool_attrs(span, tool_name="execute_sql",
            input_val=scenario["sql"],
            output_val=scenario["result"],
            is_mcp=True, mcp_server="findata-db")
        if session_id:
            span.set_attribute("session.id", session_id)
        span.set_attribute("latency_ms", round(random.uniform(200, 2000), 2))

        success = scenario["result"].get("status") == "success"
        span.set_attribute("execution.success", success)
        span.set_attribute("execution.row_count", scenario["result"].get("row_count", 0))
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            error_msg = scenario["result"].get("error", "Query execution failed")
            span.set_status(Status(StatusCode.ERROR, error_msg))
            span.add_event("exception", attributes={
                "exception.type": "SQLExecutionError",
                "exception.message": error_msg,
            })
        time.sleep(0.002)
    return success


def _emit_response_agent(tracer, scenario, session_id, *, mode="success"):
    with tracer.start_as_current_span("agent_run [response_agent]") as agent:
        agent.set_attribute(_KIND, "AGENT")
        agent.set_attribute("agent.name", "response_agent")
        agent.set_attribute("response.mode", mode)
        if session_id:
            agent.set_attribute("session.id", session_id)
        time.sleep(0.002)

        if mode == "success":
            answer = scenario["answer"]
            row_count = scenario["result"].get("row_count", 0)
            resp_vars = {"question": scenario["question"], "row_count": str(row_count),
                         "data": json.dumps(scenario["result"]["rows"][:3])}
        elif mode == "failure":
            answer = f"The query could not be executed successfully. {scenario['result'].get('error', 'An unexpected error occurred during SQL execution.')} Please try rephrasing your question."
            resp_vars = {"question": scenario["question"],
                         "error": scenario["result"].get("error", "unknown")}
        elif mode == "clarification":
            answer = scenario.get("clarification_question", "Could you please clarify your question?")
            resp_vars = {"question": scenario["question"],
                         "ambiguity_reason": scenario.get("ambiguity_reason", "")}
        elif mode == "direct":
            answer = scenario.get("answer", "")
            resp_vars = {"question": scenario["question"]}
        else:
            answer = scenario.get("answer", "(no response)")
            resp_vars = {"question": scenario["question"]}

        resp_template = RESPONSE_TEMPLATES.get(mode, RESPONSE_TEMPLATES["direct"])
        user_msg = resp_template.format(**resp_vars)
        agent.set_attribute(_INPUT, user_msg)
        agent.set_attribute(_INPUT_MIME, _TEXT)

        resp_version = scenario.get("prompt_version", "v1.0")
        with tracer.start_as_current_span("AsyncGenerateContent") as llm:
            _set_llm_attrs(llm, system_msg=RESPONSE_SYSTEM,
                user_msg=user_msg, assistant_msg=answer,
                prompt_tokens=random.randint(200, 500),
                completion_tokens=random.randint(60, 300),
                template=resp_template,
                template_version=resp_version,
                template_vars=resp_vars)
            if session_id:
                llm.set_attribute("session.id", session_id)
            llm.set_attribute("latency_ms", round(random.uniform(500, 2000), 2))
            llm.set_status(Status(StatusCode.OK))
            time.sleep(0.002)

        agent.set_attribute(_OUTPUT, answer)
        agent.set_attribute(_OUTPUT_MIME, _TEXT)
        agent.set_attribute("latency_ms", round(random.uniform(800, 3000), 2))
        agent.set_status(Status(StatusCode.OK))
        response_span_id = _span_id(agent)

    return answer, response_span_id


def _emit_await_user_clarification(tracer, scenario, session_id, *, resolved=True):
    with tracer.start_as_current_span("await_user_clarification") as span:
        span.set_attribute(_KIND, "CHAIN")
        span.set_attribute("clarification.question", scenario.get("clarification_question", ""))
        span.set_attribute("clarification.resolved", resolved)
        if session_id:
            span.set_attribute("session.id", session_id)

        if resolved:
            response = scenario.get("clarified_question", scenario["question"])
            span.set_attribute("clarification.user_response", response)
            span.set_attribute("clarification.wait_time_ms", round(random.uniform(5000, 45000), 2))
            span.set_attribute(_INPUT, scenario.get("clarification_question", ""))
            span.set_attribute(_OUTPUT, response)
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_attribute("clarification.wait_time_ms", round(random.uniform(60000, 300000), 2))
            span.set_attribute(_INPUT, scenario.get("clarification_question", ""))
            span.set_attribute(_OUTPUT, "")
            timeout_msg = "User did not respond to clarification request within the session timeout period."
            span.set_status(Status(StatusCode.ERROR, timeout_msg))
            span.add_event("exception", attributes={
                "exception.type": "ClarificationTimeout",
                "exception.message": timeout_msg,
            })
        time.sleep(0.002)
    return resolved


def _emit_guardrail_check(tracer, scenario, session_id):
    role = scenario.get("role", "finance")
    tables = scenario["tables"]
    requested = {t.lower() for t in tables}
    allowed = ROLE_PERMISSIONS.get(role, set())
    denied = sorted(requested - allowed)
    is_allowed = len(denied) == 0

    with tracer.start_as_current_span("access_control_check") as guard:
        guard.set_attribute(_KIND, "GUARDRAIL")
        guard.set_attribute(_INPUT, scenario["question"])
        guard.set_attribute("guardrail.requested_tables", json.dumps(sorted(requested)))
        guard.set_attribute("guardrail.role", role)
        if session_id:
            guard.set_attribute("session.id", session_id)
        time.sleep(0.002)

        with tracer.start_as_current_span("resolve_user_role") as role_span:
            _set_tool_attrs(role_span, tool_name="resolve_user_role",
                input_val="user_id=demo-user",
                output_val={"role": role, "allowed_tables": sorted(allowed)},
                is_mcp=True, mcp_server="identity-service")
            role_span.set_attribute("latency_ms", round(random.uniform(50, 200), 2))
            role_span.set_status(Status(StatusCode.OK))
            time.sleep(0.002)

        with tracer.start_as_current_span("verify_table_permissions") as perm:
            _set_tool_attrs(perm, tool_name="verify_table_permissions",
                input_val={"role": role, "requested_tables": sorted(requested)},
                output_val={"allowed": is_allowed, "denied_tables": denied})
            perm.set_attribute("latency_ms", round(random.uniform(2, 10), 2))
            if is_allowed:
                perm.set_status(Status(StatusCode.OK))
            else:
                err = f"AccessControlViolation: Role '{role}' denied access to: {', '.join(denied)}"
                perm.set_status(Status(StatusCode.ERROR, err))
                perm.add_event("exception", attributes={"exception.type": "AccessControlViolation", "exception.message": err})
            time.sleep(0.002)

        guard.set_attribute(_OUTPUT, json.dumps({"allowed": is_allowed, "denied_tables": denied}))
        guard.set_attribute(_OUTPUT_MIME, _JSON)
        guard.set_attribute("guardrail.result", "allowed" if is_allowed else "denied")
        guard.set_attribute("guardrail.denied_tables", json.dumps(denied))
        if is_allowed:
            guard.set_status(Status(StatusCode.OK))
        else:
            guard_err = f"GuardrailBlocked: Role '{role}' cannot access [{', '.join(denied)}]"
            guard.set_status(Status(StatusCode.ERROR, guard_err))
            guard.add_event("exception", attributes={"exception.type": "GuardrailBlocked", "exception.message": guard_err})

    return is_allowed, denied


# ── Main trace builder ───────────────────────────────────────────────────────

def create_trace(tracer, scenario, session_id=None):
    stype = scenario["type"]
    question = scenario["question"]
    span_ids = {"type": stype}

    with tracer.start_as_current_span("invocation [da-agent]") as root:
        root.set_attribute(_KIND, "CHAIN")
        root.set_attribute(_INPUT, question)
        root.set_attribute(_INPUT_MIME, _TEXT)
        if session_id:
            root.set_attribute("session.id", session_id)
        span_ids["root"] = _span_id(root)
        time.sleep(0.002)

        with tracer.start_as_current_span("agent_run [coordinator_agent]") as coord:
            coord.set_attribute(_KIND, "AGENT")
            coord.set_attribute("agent.name", "coordinator_agent")
            coord.set_attribute(_INPUT, question)
            coord.set_attribute(_INPUT_MIME, _TEXT)
            if session_id:
                coord.set_attribute("session.id", session_id)
            span_ids["coordinator"] = _span_id(coord)
            time.sleep(0.002)

            if stype == "no_sql":
                answer = _flow_no_sql(tracer, scenario, session_id, span_ids)
            elif stype == "ambiguity_resolved":
                answer = _flow_ambiguity(tracer, scenario, session_id, span_ids, resolved=True)
            elif stype == "ambiguity_abandoned":
                answer = _flow_ambiguity(tracer, scenario, session_id, span_ids, resolved=False)
            elif stype == "guardrail_denial":
                answer = _flow_guardrail_denial(tracer, scenario, session_id, span_ids)
            elif stype == "execution_failure":
                answer = _flow_execution_failure(tracer, scenario, session_id, span_ids)
            elif stype == "coordinator_retry":
                answer = _flow_coordinator_retry(tracer, scenario, session_id, span_ids)
            else:
                answer = _flow_standard(tracer, scenario, session_id, span_ids)

            span_ids["final_answer"] = answer
            coord.set_attribute(_OUTPUT, answer)
            coord.set_attribute(_OUTPUT_MIME, _TEXT)
            coord.set_attribute("latency_ms", round(random.uniform(5000, 20000), 2))
            coord.set_status(Status(StatusCode.OK))

        root.set_attribute(_OUTPUT, answer)
        root.set_attribute(_OUTPUT_MIME, _TEXT)
        root.set_status(Status(StatusCode.OK))

    return span_ids


def _flow_standard(tracer, scenario, session_id, span_ids):
    iterations = scenario.get("planning_iterations", 1)
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Query requires SQL generation."},
        session_id)

    _, plan_sid = _emit_planning_agent(tracer, scenario, session_id, iterations=iterations)
    span_ids["planning"] = plan_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete.",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Context retrieved. Generate SQL."},
        session_id)

    _, eng_sid = _emit_engineer_agent(tracer, scenario, session_id, output_status="SUCCESS")
    span_ids["engineer"] = eng_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Engineer returned SUCCESS.",
        {"action": "execute_sql", "reasoning": "SQL generated. Execute query."},
        session_id)

    _emit_execute_sql(tracer, scenario, session_id)

    _emit_coordinator_routing(tracer, scenario["question"], f"Execution succeeded. {scenario['result']['row_count']} rows.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Build response summary."},
        session_id)

    answer, resp_sid = _emit_response_agent(tracer, scenario, session_id, mode="success")
    span_ids["response"] = resp_sid
    return answer


def _flow_no_sql(tracer, scenario, session_id, span_ids):
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "No SQL needed. Answer directly."},
        session_id)
    answer, resp_sid = _emit_response_agent(tracer, scenario, session_id, mode="direct")
    span_ids["response"] = resp_sid
    return answer


def _flow_ambiguity(tracer, scenario, session_id, span_ids, *, resolved=True):
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Query requires SQL."},
        session_id)

    _, plan_sid = _emit_planning_agent(tracer, scenario, session_id, iterations=1)
    span_ids["planning"] = plan_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete.",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Generate SQL."},
        session_id)

    _, eng_sid = _emit_engineer_agent(tracer, scenario, session_id, output_status="BLOCKING_AMBIGUITY")
    span_ids["engineer"] = eng_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Engineer returned BLOCKING_AMBIGUITY.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Ask user for clarification."},
        session_id)

    _emit_response_agent(tracer, scenario, session_id, mode="clarification")

    user_resolved = _emit_await_user_clarification(tracer, scenario, session_id, resolved=resolved)

    if not user_resolved:
        return "(Conversation abandoned — user did not respond to clarification request.)"

    clarified_scenario = {**scenario, "question": scenario.get("clarified_question", scenario["question"])}

    _emit_coordinator_routing(tracer, clarified_scenario["question"], "User clarified. Re-running pipeline.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Re-run with clarified question."},
        session_id)

    _, plan_sid2 = _emit_planning_agent(tracer, clarified_scenario, session_id, iterations=1)
    span_ids["planning"] = plan_sid2

    _emit_coordinator_routing(tracer, clarified_scenario["question"], "Planning complete (clarified).",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Generate SQL for clarified question."},
        session_id)

    _, eng_sid2 = _emit_engineer_agent(tracer, clarified_scenario, session_id, output_status="SUCCESS")
    span_ids["engineer"] = eng_sid2

    _emit_coordinator_routing(tracer, clarified_scenario["question"], "Engineer SUCCESS.",
        {"action": "execute_sql", "reasoning": "Execute query."},
        session_id)

    _emit_execute_sql(tracer, clarified_scenario, session_id)

    _emit_coordinator_routing(tracer, clarified_scenario["question"], f"Execution succeeded. {clarified_scenario['result']['row_count']} rows.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Build final response."},
        session_id)

    answer, resp_sid = _emit_response_agent(tracer, clarified_scenario, session_id, mode="success")
    span_ids["response"] = resp_sid
    return answer


def _flow_guardrail_denial(tracer, scenario, session_id, span_ids):
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Query requires SQL."},
        session_id)

    _, plan_sid = _emit_planning_agent(tracer, scenario, session_id)
    span_ids["planning"] = plan_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete.",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Generate SQL."},
        session_id)

    _, eng_sid = _emit_engineer_agent(tracer, scenario, session_id, output_status="SUCCESS")
    span_ids["engineer"] = eng_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Engineer SUCCESS. Checking access.",
        {"action": "execute_sql", "reasoning": "Execute query after access check."},
        session_id)

    is_allowed, denied = _emit_guardrail_check(tracer, scenario, session_id)

    role = scenario.get("role", "restricted")
    denial = f"Access denied. Role '{role}' cannot query table(s): {', '.join(denied)}. Contact your administrator."

    _emit_coordinator_routing(tracer, scenario["question"], f"Guardrail DENIED: {denied}.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Inform user of access denial."},
        session_id)

    answer, resp_sid = _emit_response_agent(tracer, {**scenario, "answer": denial}, session_id, mode="direct")
    span_ids["response"] = resp_sid
    return answer


def _flow_execution_failure(tracer, scenario, session_id, span_ids):
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Query requires SQL."},
        session_id)

    _, plan_sid = _emit_planning_agent(tracer, scenario, session_id)
    span_ids["planning"] = plan_sid

    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete.",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Generate SQL."},
        session_id)

    _, eng_sid = _emit_engineer_agent(tracer, scenario, session_id, output_status="SUCCESS")
    span_ids["engineer"] = eng_sid

    fail_scenario = {**scenario, "result": {"status": "error", "rows": [], "row_count": 0, "column_names": [], "error": "Query timed out after 30s. The query may be scanning too much data. Consider adding date filters or LIMIT."}}

    _emit_coordinator_routing(tracer, scenario["question"], "Engineer SUCCESS. Executing.",
        {"action": "execute_sql", "reasoning": "Execute query."},
        session_id)

    _emit_execute_sql(tracer, fail_scenario, session_id)

    _emit_coordinator_routing(tracer, scenario["question"], "Execution FAILED.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Explain failure to user."},
        session_id)

    answer, resp_sid = _emit_response_agent(tracer, fail_scenario, session_id, mode="failure")
    span_ids["response"] = resp_sid
    return answer


def _flow_coordinator_retry(tracer, scenario, session_id, span_ids):
    _emit_coordinator_routing(tracer, scenario["question"], "No agents called.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Query requires SQL."},
        session_id)
    _emit_planning_agent(tracer, scenario, session_id, iterations=1)
    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete.",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Generate SQL."},
        session_id)
    _emit_engineer_agent(tracer, scenario, session_id, output_status="SUCCESS")
    _emit_coordinator_routing(tracer, scenario["question"], "Engineer SUCCESS.",
        {"action": "execute_sql", "reasoning": "Execute."},
        session_id)
    _emit_execute_sql(tracer, scenario, session_id)
    _emit_coordinator_routing(tracer, scenario["question"], "Execution OK but results seem incomplete.",
        {"action": "route_to_agent", "target": "planning_agent", "reasoning": "Re-run with deeper context."},
        session_id)
    _, plan_sid = _emit_planning_agent(tracer, scenario, session_id, iterations=2)
    span_ids["planning"] = plan_sid
    _emit_coordinator_routing(tracer, scenario["question"], "Planning complete (retry).",
        {"action": "route_to_agent", "target": "engineer_agent", "reasoning": "Re-generate SQL."},
        session_id)
    _, eng_sid = _emit_engineer_agent(tracer, scenario, session_id, output_status="SUCCESS")
    span_ids["engineer"] = eng_sid
    _emit_coordinator_routing(tracer, scenario["question"], "Engineer SUCCESS (retry).",
        {"action": "execute_sql", "reasoning": "Execute refined query."},
        session_id)
    _emit_execute_sql(tracer, scenario, session_id)
    _emit_coordinator_routing(tracer, scenario["question"], f"Execution succeeded. {scenario['result']['row_count']} rows.",
        {"action": "route_to_agent", "target": "response_agent", "reasoning": "Build final response."},
        session_id)
    answer, resp_sid = _emit_response_agent(tracer, scenario, session_id, mode="success")
    span_ids["response"] = resp_sid
    return answer


# ── Scenario construction ────────────────────────────────────────────────────

ADVERSARIAL_QUERIES = [q for q in QUERY_BANK if q.get("is_adversarial")]
STANDARD_QUERIES = [q for q in QUERY_BANK if not q.get("is_adversarial")]
CB_QUERIES = [q for q in QUERY_BANK if "cb_subscriptions" in q["tables"] and "subscriptions_web" not in q["tables"]]


def build_scenarios(count=500, seed=None):
    if seed is not None:
        random.seed(seed)

    scenarios = []
    for _ in range(count):
        r = random.random()
        if r < 0.28:
            stype = "happy_path"
        elif r < 0.38:
            stype = "guardrail_denial"
        elif r < 0.46:
            stype = "sql_retry"
        elif r < 0.54:
            stype = "ambiguity_resolved"
        elif r < 0.58:
            stype = "ambiguity_abandoned"
        elif r < 0.66:
            stype = "execution_failure"
        elif r < 0.74:
            stype = "planning_deep_loop"
        elif r < 0.80:
            stype = "no_sql"
        elif r < 0.88:
            stype = "coordinator_retry"
        elif r < 0.94:
            stype = "adversarial"
        else:
            stype = "schema_mismatch"

        if stype in ("ambiguity_resolved", "ambiguity_abandoned"):
            query = random.choice(AMBIGUOUS_QUERIES)
        elif stype == "no_sql":
            query = random.choice(NO_SQL_QUERIES)
            query = {**query, "tables": [], "sql": "", "result": {"status": "success", "rows": [], "row_count": 0, "column_names": []}, "plan": "", "complexity": "simple"}
        elif stype == "guardrail_denial":
            query = random.choice(CB_QUERIES) if CB_QUERIES else random.choice(STANDARD_QUERIES)
        elif stype == "adversarial":
            query = random.choice(ADVERSARIAL_QUERIES) if ADVERSARIAL_QUERIES else random.choice(STANDARD_QUERIES)
            stype = "happy_path"
        elif stype == "schema_mismatch":
            query = QUERY_BANK[-1]  # LTV approximation
        else:
            query = random.choice(STANDARD_QUERIES)

        role = "finance"
        if stype == "guardrail_denial":
            role = random.choice(["marketing", "restricted"])

        planning_iterations = 1
        if stype == "planning_deep_loop":
            planning_iterations = random.choice([2, 2, 3])
            stype = "happy_path"

        prompt_version = "v1.0" if random.random() < 0.70 else "v2.0"

        scenarios.append({
            **query,
            "type": stype,
            "role": role,
            "planning_iterations": planning_iterations,
            "prompt_version": prompt_version,
        })

    return scenarios


# ── Evaluation generators ────────────────────────────────────────────────────

def _eval_trajectory(scenario):
    stype = scenario["type"]
    question = scenario["question"][:120]
    tables = scenario.get("tables", [])
    rates = {"happy_path": 0.88, "guardrail_denial": 0.85, "sql_retry": 0.60,
             "ambiguity_resolved": 0.75, "ambiguity_abandoned": 0.30,
             "execution_failure": 0.50, "no_sql": 0.92, "coordinator_retry": 0.45,
             "schema_mismatch": 0.65}
    will_pass = random.random() < rates.get(stype, 0.70)

    if will_pass:
        explanations = {
            "happy_path": (
                f"I reviewed the coordinator's routing decisions for the query \"{question}\". "
                f"The coordinator correctly identified this as a SQL-required query and routed sequentially through "
                f"planning_agent (context retrieval for {tables}), engineer_agent (SQL generation), execute_sql, "
                f"and response_agent (summary). Each agent was invoked exactly once in the expected order with no "
                f"unnecessary steps. The tool trajectory matches the optimal path for a standard query."
            ),
            "guardrail_denial": (
                f"I examined the trajectory for \"{question}\" where access was denied. The coordinator correctly "
                f"routed through planning_agent and engineer_agent before reaching the guardrail check, which properly "
                f"blocked execution. After denial, the coordinator skipped execute_sql and routed directly to "
                f"response_agent to inform the user. This is the correct trajectory for an access-denied scenario — "
                f"no data was leaked and the user received an appropriate denial message."
            ),
            "ambiguity_resolved": (
                f"The trajectory for \"{question}\" shows proper handling of ambiguity. The coordinator routed to "
                f"planning_agent and engineer_agent, which correctly returned BLOCKING_AMBIGUITY. The coordinator then "
                f"engaged response_agent to ask a clarification question, waited for the user response via "
                f"await_user_clarification, and re-ran the full pipeline with the clarified input. The double-pass "
                f"through planning and engineer is expected for ambiguity resolution."
            ),
            "ambiguity_abandoned": (
                f"For \"{question}\", the engineer_agent returned BLOCKING_AMBIGUITY and the coordinator correctly "
                f"asked for clarification. The user did not respond (ClarificationTimeout). The coordinator appropriately "
                f"terminated the conversation at the await_user_clarification step without proceeding to execute any SQL. "
                f"This is the correct trajectory for an abandoned clarification."
            ),
            "no_sql": (
                f"The coordinator correctly identified \"{question}\" as not requiring SQL generation and routed "
                f"directly to response_agent, bypassing both planning_agent and engineer_agent. This is the optimal "
                f"short-circuit path — no unnecessary tool calls or context retrieval were performed."
            ),
            "coordinator_retry": (
                f"For \"{question}\", the coordinator ran the full pipeline (planning → engineer → execute_sql), "
                f"determined the results were insufficient, and correctly re-invoked planning_agent with deeper "
                f"iteration depth before re-running engineer and execute_sql. The retry trajectory shows appropriate "
                f"self-correction behavior, and the final response_agent call produced a better answer."
            ),
        }
        expl = explanations.get(stype,
            f"The coordinator's routing for \"{question}\" followed the expected trajectory for a {stype} scenario. "
            f"Agents were invoked in the correct order with appropriate tool calls at each step.")
    else:
        explanations = {
            "happy_path": (
                f"The trajectory for \"{question}\" deviated from the expected standard path. I observed an extra "
                f"routing decision between engineer_agent and execute_sql that introduced unnecessary latency. "
                f"The coordinator appeared to re-evaluate whether to proceed with execution despite a clean SUCCESS "
                f"status from the engineer. For a straightforward query against {tables}, this additional deliberation "
                f"step was wasteful and suggests the routing prompt may need tightening."
            ),
            "sql_retry": (
                f"While evaluating \"{question}\", the coordinator failed to detect the initial SQL generation error "
                f"promptly. The engineer_agent produced invalid SQL (cross-currency aggregation without GROUP BY), "
                f"but the coordinator routed to execute_sql before catching the issue. A more efficient trajectory "
                f"would validate the SQL structure before execution. The retry succeeded, but the wasted execution "
                f"attempt added ~2s of unnecessary latency."
            ),
            "coordinator_retry": (
                f"For \"{question}\", the coordinator's retry logic was triggered but the second pass through "
                f"planning_agent did not meaningfully improve context retrieval — the same tables were selected with "
                f"similar scores. The retry appears to have been unnecessary, suggesting the coordinator's quality "
                f"threshold for triggering a re-run may be too sensitive. The final answer was not substantially "
                f"different from the first pass."
            ),
            "execution_failure": (
                f"The trajectory for \"{question}\" shows the coordinator routed to execute_sql without first checking "
                f"whether the generated SQL was likely to timeout. The query scanned the full subscriptions_web table "
                f"without date filters, which is a known antipattern per the usage rules. The coordinator should have "
                f"detected the missing WHERE clause and routed back to engineer_agent for refinement before execution."
            ),
        }
        expl = explanations.get(stype,
            f"The coordinator's routing for \"{question}\" did not follow the optimal trajectory for a {stype} "
            f"scenario. I observed either missing steps, incorrect ordering, or unnecessary agent invocations that "
            f"indicate a routing decision error. The coordinator's reasoning at the deviation point did not adequately "
            f"account for the current pipeline state.")

    return {"label": "pass" if will_pass else "fail", "score": 0 if will_pass else 1, "explanation": expl}


def _eval_sql_quality(scenario):
    stype = scenario["type"]
    question = scenario["question"][:120]
    tables = scenario.get("tables", [])
    sql = scenario.get("sql", "")[:200]
    will_pass = random.random() < (0.92 if stype != "sql_retry" else 0.45)

    if will_pass:
        expl = (
            f"I analyzed the generated SQL for \"{question}\" targeting tables {tables}. "
            f"The query uses correct BigQuery Standard SQL syntax with proper table references "
            f"(`{BQ_DATASET}.*`). Column references align with the schema — no hallucinated columns detected. "
            f"The query respects the documented usage rules: aggregations are grouped by currency (no cross-currency "
            f"sums), date filters use <= CURRENT_DATE() to exclude future-dated records, and joins use "
            f"cf_account_id rather than customer_id. The WHERE clause is appropriately scoped and the "
            f"result columns match the user's request."
        )
    else:
        fail_reasons = [
            (
                f"The generated SQL for \"{question}\" contains a usage rule violation. The query aggregates "
                f"plan_amount across rows without a GROUP BY currency clause, which would produce misleading totals "
                f"by summing USD, EUR, and GBP amounts together. Per the financial rules, amounts must always be "
                f"aggregated per currency. Additionally, the query references `customer_id` for a join, but the "
                f"documented join key is `cf_account_id`. These issues would produce incorrect results."
            ),
            (
                f"Reviewing the SQL for \"{question}\": the query does not filter for cancelled_date <= CURRENT_DATE() "
                f"despite counting cancellations. Per the dates rule, cancelled_date can contain future dates, and "
                f"failing to filter these would overcount cancellations. The query also uses `status` instead of "
                f"`status_reporting` for the active/cancelled breakdown, which produces different counts (status "
                f"includes 'non_renewing' as a separate category while status_reporting groups it with 'Active')."
            ),
            (
                f"The SQL generated for \"{question}\" references tables {tables} but uses mrr for a revenue "
                f"calculation when the user did not explicitly request Monthly Recurring Revenue. Per the financial "
                f"rules, plan_amount should be used unless the user specifically asks for MRR. The column choice "
                f"would produce different (lower) figures since mrr is a monthly normalized value while plan_amount "
                f"reflects the full billing period amount."
            ),
        ]
        expl = random.choice(fail_reasons)

    return {"label": "pass" if will_pass else "fail", "score": 0 if will_pass else 1, "explanation": expl}


def _eval_coordination(scenario):
    stype = scenario["type"]
    question = scenario["question"][:120]
    rates = {"happy_path": 0.90, "guardrail_denial": 0.85, "ambiguity_resolved": 0.70,
             "ambiguity_abandoned": 0.35, "coordinator_retry": 0.45, "no_sql": 0.95}
    will_pass = random.random() < rates.get(stype, 0.70)

    if will_pass:
        expl = (
            f"End-to-end coordination for \"{question}\" completed successfully. I verified that: "
            f"(1) the coordinator correctly classified whether SQL generation was needed, "
            f"(2) data flowed cleanly between agents — the planning agent's selected tables were consumed by the "
            f"engineer agent, and the engineer's SQL output was passed to execute_sql, "
            f"(3) error states were handled appropriately (guardrail denials short-circuited execution, ambiguity "
            f"triggered clarification flow), and (4) the response agent received sufficient context to produce "
            f"a well-formed answer. No data was lost between agent handoffs and the session state was consistent "
            f"throughout the trace."
        )
    else:
        fail_reasons = [
            (
                f"Coordination for \"{question}\" showed a state consistency issue between agents. The planning "
                f"agent selected tables {scenario.get('tables', [])} and passed context to the engineer, but the "
                f"engineer's SQL referenced a column not present in the retrieved context (the get_details output "
                f"did not include this column's description). This suggests the planning agent's context retrieval "
                f"was insufficient, and the coordinator did not verify context completeness before routing to the "
                f"engineer. The end result may contain inaccuracies."
            ),
            (
                f"The multi-agent coordination for \"{question}\" failed at the handoff between engineer_agent and "
                f"execute_sql. The engineer returned SUCCESS with a valid SQL query, but the coordinator introduced "
                f"an unnecessary delay by re-evaluating the routing decision. In a {stype} scenario, I would expect "
                f"the coordinator to proceed directly to execution. The additional routing step consumed ~1.2s of "
                f"latency without changing the outcome, indicating the coordinator's decision confidence was low "
                f"despite clear signals from the engineer."
            ),
            (
                f"Reviewing the full trace for \"{question}\": the coordinator's state management lost track of "
                f"the planning agent's iteration count. The planning agent ran {scenario.get('planning_iterations', 1)} "
                f"iteration(s), but the coordinator's subsequent routing decision did not account for this depth "
                f"when deciding whether to proceed or retry. For queries requiring multi-table context, the "
                f"coordinator should factor in retrieval depth as a quality signal."
            ),
        ]
        expl = random.choice(fail_reasons)

    return {"label": "pass" if will_pass else "fail", "score": 0 if will_pass else 1, "explanation": expl}


def _eval_table_selection(scenario):
    question = scenario["question"][:120]
    expected_tables = scenario.get("tables", [])
    will_pass = random.random() < 0.82

    if will_pass:
        expl = (
            f"I evaluated the planning agent's table selection for \"{question}\". The agent selected "
            f"{expected_tables}, which matches the expected set of tables needed to answer this query. "
            f"The retrieval process followed the correct steps: get_catalog returned all available tables, "
            f"get_memory provided relevant glossary context and usage rules, and the LLM correctly ranked "
            f"the candidate tables by relevance. The subsequent get_details calls retrieved schema information "
            f"for each selected table, providing sufficient context for SQL generation. "
            f"Precision: 1.0, Recall: 1.0 — no extraneous tables were selected and no required tables were missed."
        )
    else:
        if len(expected_tables) == 1:
            extra = [t for t in ALL_TABLES if t not in expected_tables]
            expl = (
                f"The planning agent's table selection for \"{question}\" included unnecessary tables. The query "
                f"only requires {expected_tables}, but the agent also selected '{random.choice(extra)}' which is "
                f"not relevant to this question. This likely occurred because the get_memory step returned glossary "
                f"rules mentioning join relationships between tables, and the LLM over-generalized by including "
                f"related but unnecessary tables. The extra table adds noise to the engineer agent's context and "
                f"increases the risk of generating overly complex SQL with unnecessary JOINs. "
                f"Precision: {1/2:.1f}, Recall: 1.0."
            )
        else:
            missed = random.choice(expected_tables)
            expl = (
                f"The planning agent failed to select all required tables for \"{question}\". Expected tables "
                f"{expected_tables} but the agent missed '{missed}'. The get_catalog step returned all available "
                f"tables, and get_memory provided the correct join rules, but the LLM's table ranking assigned "
                f"'{missed}' a low relevance score (< 0.5) and it was filtered out during selection. This means "
                f"the engineer agent will lack schema context for '{missed}' and may generate SQL with missing "
                f"JOINs or incorrect column references. "
                f"Precision: 1.0, Recall: {(len(expected_tables)-1)/len(expected_tables):.2f}."
            )

    return {"label": "pass" if will_pass else "fail", "score": 0 if will_pass else 1, "explanation": expl}
def generate_evaluations(span_data_list):
    import pandas as pd
    span_rows = []
    trace_rows = []
    session_rows = []

    for sd in span_data_list:
        scenario = sd["scenario"]
        root_sid = sd.get("root_span_id", "")
        planning_sid = sd.get("planning_span_id")

        # ── trace_eval.* (trace-level, one row per trace with all evals) ──

        traj = _eval_trajectory(scenario)
        sql_q = _eval_sql_quality(scenario)
        coord = _eval_coordination(scenario)
        trace_rows.append({
            "context.span_id": root_sid,
            "trace_eval.AgentTrajectoryAccuracy.label": traj["label"],
            "trace_eval.AgentTrajectoryAccuracy.score": traj["score"],
            "trace_eval.AgentTrajectoryAccuracy.explanation": traj["explanation"],
            "trace_eval.SQLQuality.label": sql_q["label"],
            "trace_eval.SQLQuality.score": sql_q["score"],
            "trace_eval.SQLQuality.explanation": sql_q["explanation"],
            "trace_eval.CoordinationQuality.label": coord["label"],
            "trace_eval.CoordinationQuality.score": coord["score"],
            "trace_eval.CoordinationQuality.explanation": coord["explanation"],
        })

        # ── eval.* (span-level, visible when drilling into the agent) ──

        if planning_sid:
            ev = _eval_table_selection(scenario)
            span_rows.append({
                "context.span_id": planning_sid,
                "eval.TableSelectionPrecision.label": ev["label"],
                "eval.TableSelectionPrecision.score": ev["score"],
                "eval.TableSelectionPrecision.explanation": ev["explanation"],
            })

    # ── session_eval.* (session-level, visible in session view) ──

    session_groups = {}
    for sd in span_data_list:
        sid = sd.get("session_id", "")
        if not sid:
            continue
        session_groups.setdefault(sid, []).append(sd)

    for sid, traces in session_groups.items():
        first_root = traces[0].get("root_span_id", "")
        if not first_root:
            continue
        ev = _eval_session_resolution(traces)
        session_rows.append({
            "context.span_id": first_root,
            "session_eval.SessionResolution.label": ev["label"],
            "session_eval.SessionResolution.score": ev["score"],
            "session_eval.SessionResolution.explanation": ev["explanation"],
        })

    span_eval_df = pd.DataFrame(span_rows) if span_rows else pd.DataFrame()
    trace_eval_df = pd.DataFrame(trace_rows) if trace_rows else pd.DataFrame()
    session_eval_df = pd.DataFrame(session_rows) if session_rows else pd.DataFrame()
    return span_eval_df, trace_eval_df, session_eval_df


def _eval_session_resolution(traces):
    """Evaluate whether the session resolved all user questions successfully."""
    total = len(traces)
    scenarios = [t["scenario"] for t in traces]
    types = [s["type"] for s in scenarios]

    abandoned = sum(1 for t in types if t == "ambiguity_abandoned")
    failures = sum(1 for t in types if t in ("execution_failure", "ambiguity_abandoned"))
    guardrail_denials = sum(1 for t in types if t == "guardrail_denial")
    successful = total - failures

    resolved = abandoned == 0 and failures <= 1
    will_pass = resolved and random.random() < 0.85

    questions = [s["question"][:80] for s in scenarios]
    q_summary = "; ".join(f"({i+1}) \"{q}\"" for i, q in enumerate(questions[:4]))
    if total > 4:
        q_summary += f"; ... and {total - 4} more"

    if will_pass:
        expl = (
            f"I evaluated the full session ({total} turns) for completeness and resolution. "
            f"Questions asked: {q_summary}. "
            f"Of {total} queries, {successful} received substantive answers"
        )
        if guardrail_denials:
            expl += f" ({guardrail_denials} were correctly denied due to insufficient permissions, "
            expl += "which counts as a resolved interaction since the user received a clear explanation)"
        expl += (
            f". No conversations were abandoned mid-clarification and no critical failures went unaddressed. "
            f"The session demonstrates a productive user interaction where each question reached a terminal "
            f"state — either a data-backed answer, an appropriate access denial, or a clarified re-query "
            f"that succeeded on the second pass."
        )
    else:
        if abandoned > 0:
            expl = (
                f"Session evaluation ({total} turns) found {abandoned} abandoned clarification(s). "
                f"Questions: {q_summary}. "
                f"The user was asked to clarify an ambiguous question but did not respond, resulting in a "
                f"ClarificationTimeout. This leaves the user's original intent unresolved. From a session "
                f"perspective, the agent system failed to either (a) resolve the ambiguity autonomously by "
                f"making a reasonable assumption, or (b) retain the user's engagement through the clarification "
                f"flow. The session cannot be considered fully resolved when {abandoned} of {total} queries "
                f"terminated without an answer."
            )
        elif failures > 1:
            expl = (
                f"Session evaluation ({total} turns) found {failures} failed queries out of {total}. "
                f"Questions: {q_summary}. "
                f"Multiple execution failures within a single session suggest a systemic issue — possibly "
                f"the user is asking a category of questions that the agent consistently struggles with, "
                f"or the underlying data source was experiencing issues. A healthy session should have at "
                f"most one failure with subsequent queries succeeding. The {failures}/{total} failure rate "
                f"indicates the user likely left the session without getting the information they needed."
            )
        else:
            expl = (
                f"Session evaluation ({total} turns) showed mixed results. "
                f"Questions: {q_summary}. "
                f"While most queries resolved, the overall session quality was below expectations. "
                f"The agent required retries or produced responses that did not fully address the user's "
                f"questions. The session interaction pattern suggests the user may have needed to rephrase "
                f"or re-approach their questions to get useful answers, indicating friction in the "
                f"agent's ability to understand and resolve queries on the first attempt."
            )

    return {"label": "pass" if will_pass else "fail", "score": 0 if will_pass else 1, "explanation": expl}


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(tracer, count=500, with_evals=False, project_name=None):
    scenarios = build_scenarios(count)
    span_data_list = []
    session_id = None
    traces_in_session = 0
    session_size = 0

    print(f"Generating {count} synthetic traces...")
    from collections import Counter
    dist = Counter(s["type"] for s in scenarios)
    print("  Scenario distribution:")
    for stype, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {stype}: {cnt} ({cnt/count*100:.0f}%)")
    print()

    with timer():
        for i, scenario in enumerate(scenarios):
            if session_id is None or traces_in_session >= session_size:
                session_id = f"session_{random.randint(100000, 999999)}"
                session_size = random.randint(3, 6)
                traces_in_session = 0

            result = create_trace(tracer, scenario, session_id=session_id)
            traces_in_session += 1
            span_data_list.append({
                "root_span_id": result.get("root", ""),
                "coordinator_span_id": result.get("coordinator", ""),
                "planning_span_id": result.get("planning"),
                "engineer_span_id": result.get("engineer"),
                "response_span_id": result.get("response"),
                "scenario": scenario,
                "session_id": session_id,
            })

            if ((i + 1) % 100 == 0) or (i + 1 == count):
                print(f"  Created {i+1}/{count} traces...")

        print(f"\nFlushing spans...")
        flush_timeout = min(30000 + (count // 100) * 5000, 300000)
        ok = trace.get_tracer_provider().force_flush(timeout_millis=flush_timeout)
        print(f"  {'Flush successful' if ok else 'Flush timeout'}.")

    if with_evals and span_data_list:
        print(f"\nWaiting 10s for spans to be indexed before logging evaluations...")
        time.sleep(10)
        print(f"Generating evaluations for {len(span_data_list)} traces...")
        span_eval_df, trace_eval_df, session_eval_df = generate_evaluations(span_data_list)
        print(f"  Generated {len(span_eval_df)} span evals, {len(trace_eval_df)} trace evals, {len(session_eval_df)} session evals.")
        space_id = os.environ.get("ARIZE_SPACE_ID", "")
        api_key = os.environ.get("ARIZE_API_KEY", "")
        if space_id and api_key and project_name:
            try:
                from arize.pandas.logger import Client
                client = Client(space_id=space_id, api_key=api_key)
                if len(span_eval_df) > 0:
                    client.log_evaluations_sync(span_eval_df, project_name)
                    print(f"  Logged {len(span_eval_df)} span evaluations (eval.*).")
                if len(trace_eval_df) > 0:
                    client.log_evaluations_sync(trace_eval_df, project_name)
                    print(f"  Logged {len(trace_eval_df)} trace evaluations (trace_eval.*).")
                if len(session_eval_df) > 0:
                    client.log_evaluations_sync(session_eval_df, project_name)
                    print(f"  Logged {len(session_eval_df)} session evaluations (session_eval.*).")
                print(f"  All evaluations logged to Arize project '{project_name}'.")
            except Exception as e:
                print(f"  Error logging evaluations: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinData Data Analytics Agent synthetic trace generator.")
    parser.add_argument("--count", type=int, default=500, help="Number of traces (default: 500)")
    parser.add_argument("--test", action="store_true", help="Generate test traces")
    parser.add_argument("--with-evals", action="store_true", help="Generate and log evaluations")
    parser.add_argument("--project-name",
                        default=os.environ.get("ARIZE_PROJECT_NAME", "findata_da_agent_synthetic"),
                        help="Arize project name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    space_id = os.environ.get("ARIZE_SPACE_ID", "")
    api_key = os.environ.get("ARIZE_API_KEY", "")
    if not space_id or not api_key:
        print("Error: ARIZE_SPACE_ID and ARIZE_API_KEY must be set.")
        raise SystemExit(1)

    register(space_id=space_id, api_key=api_key, project_name=args.project_name)
    tracer = trace.get_tracer("findata_da_agent_synthetic")

    if args.seed is not None:
        random.seed(args.seed)

    if args.test:
        print("Running test traces...\n")
        tests = [
            ({**QUERY_BANK[0], "type": "happy_path", "role": "finance", "planning_iterations": 1, "prompt_version": "v1.0"}, "Happy path", "test_session_a"),
            ({**QUERY_BANK[1], "type": "happy_path", "role": "finance", "planning_iterations": 2, "prompt_version": "v1.0"}, "Planning deep loop (2 iter)", "test_session_a"),
            ({**AMBIGUOUS_QUERIES[0], "type": "ambiguity_resolved", "role": "finance", "planning_iterations": 1, "prompt_version": "v2.0"}, "Ambiguity resolved", "test_session_a"),
            ({**AMBIGUOUS_QUERIES[1], "type": "ambiguity_abandoned", "role": "finance", "planning_iterations": 1, "prompt_version": "v1.0"}, "Ambiguity abandoned", "test_session_b"),
            ({**QUERY_BANK[3], "type": "guardrail_denial", "role": "restricted", "planning_iterations": 1, "prompt_version": "v1.0"}, "Guardrail denial", "test_session_b"),
            ({**NO_SQL_QUERIES[0], "type": "no_sql", "role": "finance", "planning_iterations": 1, "prompt_version": "v1.0", "tables": [], "sql": "", "result": {"status": "success", "rows": [], "row_count": 0, "column_names": []}, "plan": "", "complexity": "simple"}, "No SQL needed", "test_session_b"),
        ]
        span_data_list = []
        for scenario, label, sess_id in tests:
            with timer():
                result = create_trace(tracer, scenario, session_id=sess_id)
                print(f"  {label} [{sess_id}]: root={result['root']}")
                span_data_list.append({
                    "root_span_id": result.get("root", ""),
                    "coordinator_span_id": result.get("coordinator", ""),
                    "planning_span_id": result.get("planning"),
                    "engineer_span_id": result.get("engineer"),
                    "response_span_id": result.get("response"),
                    "scenario": scenario,
                    "session_id": sess_id,
                })
                trace.get_tracer_provider().force_flush(timeout_millis=30000)
            print()

        print("Waiting 10s for spans to be indexed before logging evaluations...")
        time.sleep(10)
        print("Running evaluations on test traces...")
        span_eval_df, trace_eval_df, session_eval_df = generate_evaluations(span_data_list)
        print(f"  Generated {len(span_eval_df)} span evals, {len(trace_eval_df)} trace evals, {len(session_eval_df)} session evals.")
        try:
            from arize.pandas.logger import Client
            client = Client(space_id=space_id, api_key=api_key)
            if len(span_eval_df) > 0:
                client.log_evaluations_sync(span_eval_df, args.project_name)
                print(f"  Logged {len(span_eval_df)} span evaluations (eval.*).")
            if len(trace_eval_df) > 0:
                client.log_evaluations_sync(trace_eval_df, args.project_name)
                print(f"  Logged {len(trace_eval_df)} trace evaluations (trace_eval.*).")
            if len(session_eval_df) > 0:
                client.log_evaluations_sync(session_eval_df, args.project_name)
                print(f"  Logged {len(session_eval_df)} session evaluations (session_eval.*).")
            print(f"  All evaluations logged to Arize project '{args.project_name}'.")
        except Exception as e:
            print(f"  Error logging evaluations: {e}")
    else:
        run_batch(tracer, count=args.count, with_evals=args.with_evals, project_name=args.project_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
