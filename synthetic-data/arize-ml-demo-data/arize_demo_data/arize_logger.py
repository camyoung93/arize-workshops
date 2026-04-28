"""Thin wrapper around the Arize SDK v8 ML logger.

Centralizes Schema construction and ``client.ml.log()`` calls so the rest of
the pipeline doesn't need to know the v8 API surface.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


_MODEL_TYPE_MAP = {
    "binary_classification": "BINARY_CLASSIFICATION",
    "score_categorical": "SCORE_CATEGORICAL",
    "numeric": "NUMERIC",
    "ranking": "RANKING",
}

_ENV_MAP = {
    "production": "PRODUCTION",
    "training": "TRAINING",
    "validation": "VALIDATION",
}


@dataclass
class LoggerCredentials:
    space_id: str
    api_key: str

    @classmethod
    def from_env(cls) -> "LoggerCredentials | None":
        space_id = os.environ.get("ARIZE_SPACE_ID")
        api_key = os.environ.get("ARIZE_API_KEY")
        enabled = os.environ.get("ARIZE_ENABLE_LOG", "true").lower() != "false"
        if not enabled or not space_id or not api_key:
            return None
        return cls(space_id=space_id, api_key=api_key)


def _build_schema(
    *,
    feature_columns: Iterable[str],
    tag_columns: Iterable[str],
    pred_label_col: str,
    pred_score_col: str | None,
    actual_label_col: str,
    timestamp_col: str,
    embedding_vector_col: str | None,
    embedding_prompt_col: str | None,
    shap_mapping: dict[str, str] | None,
    prediction_id_col: str | None,
):
    from arize.ml.types import EmbeddingColumnNames, Schema

    embedding_features = None
    if embedding_vector_col and embedding_prompt_col:
        embedding_features = {
            "tabular embedding": EmbeddingColumnNames(
                vector_column_name=embedding_vector_col,
                data_column_name=embedding_prompt_col,
            )
        }

    return Schema(
        prediction_id_column_name=prediction_id_col,
        timestamp_column_name=timestamp_col,
        feature_column_names=list(feature_columns),
        embedding_feature_column_names=embedding_features,
        tag_column_names=list(tag_columns),
        prediction_label_column_name=pred_label_col,
        prediction_score_column_name=pred_score_col,
        actual_label_column_name=actual_label_col,
        shap_values_column_names=shap_mapping or None,
    )


def _resolve_model_type(key: str):
    from arize.ml.types import ModelTypes

    name = _MODEL_TYPE_MAP.get(key)
    if name is None:
        raise ValueError(f"Unsupported model_type '{key}'")
    return getattr(ModelTypes, name)


def _resolve_environment(key: str):
    from arize.ml.types import Environments

    name = _ENV_MAP.get(key)
    if name is None:
        raise ValueError(f"Unsupported environment '{key}'")
    return getattr(Environments, name)


def log_dataframe_to_arize(
    df: pd.DataFrame,
    *,
    credentials: LoggerCredentials,
    model_id: str,
    model_version: str,
    model_type: str,
    environment: str,
    feature_columns: Iterable[str],
    tag_columns: Iterable[str],
    pred_label_col: str = "prediction_label",
    pred_score_col: str | None = "prediction_score",
    actual_label_col: str = "actual_label",
    timestamp_col: str = "timestamp",
    embedding_vector_col: str | None = None,
    embedding_prompt_col: str | None = None,
    shap_mapping: dict[str, str] | None = None,
    prediction_id_col: str | None = None,
    batch_id: str | None = None,
):
    """Send a dataframe to Arize using the v8 unified client.

    Returns the SDK response on success; raises on hard SDK errors.
    """
    from arize import ArizeClient

    if df.empty:
        logger.warning("skipping Arize upload: dataframe is empty")
        return None

    schema = _build_schema(
        feature_columns=feature_columns,
        tag_columns=tag_columns,
        pred_label_col=pred_label_col,
        pred_score_col=pred_score_col,
        actual_label_col=actual_label_col,
        timestamp_col=timestamp_col,
        embedding_vector_col=embedding_vector_col,
        embedding_prompt_col=embedding_prompt_col,
        shap_mapping=shap_mapping,
        prediction_id_col=prediction_id_col,
    )

    client = ArizeClient(api_key=credentials.api_key)
    response = client.ml.log(
        space_id=credentials.space_id,
        model_name=model_id,
        model_version=model_version,
        model_type=_resolve_model_type(model_type),
        environment=_resolve_environment(environment),
        dataframe=df,
        schema=schema,
        batch_id=batch_id,
    )
    return response
