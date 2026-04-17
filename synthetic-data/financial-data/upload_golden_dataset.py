"""
Upload FinData Data Analytics team golden SQL dataset to Arize AX.

Prerequisites:
    pip install arize
"""

import json
import os
from pathlib import Path

import pandas as pd
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# --- Config ---
ARIZE_SPACE_ID = os.environ["ARIZE_SPACE_ID"]
ARIZE_API_KEY = os.environ["ARIZE_API_KEY"]

client = ArizeDatasetsClient(api_key=ARIZE_API_KEY)

# --- Golden dataset examples ---
# Using native Python types so lists stay as lists in Arize
examples = [
    {
        "id": "cb_subscriptions_q1",
        "input": (
            "Generate a SQL query to retrieve all bulk (self-serve) subscriptions, "
            "including their creation date, cancellation date, seat quantity, "
            "plan amount, currency, and status."
        ),
        "expected_sql": (
            "SELECT cb.id, cb.created_at AS CreatedDate, cb.cancelled_at AS CancelledDate, "
            "cb.plan_quantity AS Quantity, cb.plan_amount AS Amount, cb.currency AS Currency, "
            "cb.status AS Status "
            "FROM `findata-analytics-dev.agent_eval_dataset.cb_subscriptions` AS cb "
            "WHERE LOWER(cb.cf_is_bulk_subscription)='true' "
            "ORDER BY cb.created_at DESC"
        ),
        "expected_cols": json.dumps(
            ["id", "CreatedDate", "CancelledDate", "Quantity", "Amount", "Currency", "Status"]
        ),
        "related_tables": json.dumps(["cb_subscriptions"]),
    },
    {
        "id": "subscriptions_web_q1",
        "input": (
            "What are the number of cancellations of web subscriptions every day "
            "in the current year? In final result, order the table by cancelled_date, "
            "in reverse order."
        ),
        "expected_sql": (
            "SELECT cancelled_date, COUNT(*) AS ct "
            "FROM `findata-analytics-dev.agent_eval_dataset.subscriptions_web` "
            "WHERE cancelled_date >= DATE_TRUNC(CURRENT_DATE(), YEAR) "
            "AND cancelled_date <= CURRENT_DATE() "
            "GROUP BY cancelled_date "
            "ORDER BY cancelled_date DESC"
        ),
        "expected_cols": json.dumps(["cancelled_date", "ct"]),
        "related_tables": json.dumps(["subscriptions_web"]),
    },
    {
        "id": "cb_subscriptions_coupons_q1",
        "input": (
            "Generate a SQL query to retrieve all coupons applied to each subscription, "
            "with the apply-till timestamp converted to America/New_York. The output "
            "should include the coupon ID, subscription ID, and apply-till timestamp."
        ),
        "expected_sql": (
            "SELECT coupon_id, subscription_id, "
            "DATETIME(apply_till, 'America/New_York') apply_till "
            "FROM `findata-analytics-dev.agent_eval_dataset.cb_subscriptions_coupons`"
        ),
        "expected_cols": json.dumps(["coupon_id", "subscription_id", "apply_till"]),
        "related_tables": json.dumps(["cb_subscriptions_coupons"]),
    },
]

df = pd.DataFrame(examples)

# --- Create dataset ---
dataset_id = client.create_dataset(
    space_id=ARIZE_SPACE_ID,
    dataset_name="findata_da_golden_sql",
    dataset_type=GENERATIVE,
    data=df,
)

print(f"Dataset created: {dataset_id}")
