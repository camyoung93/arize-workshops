"""Tabular embeddings via the Arize SDK.

Wraps ``arize.embeddings.EmbeddingGenerator`` so the rest of the pipeline
doesn't need to care about whether the embedding extras are installed.
Imports are deferred so the package itself remains usable without torch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def add_tabular_embeddings(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    model_name: str = "distilbert-base-uncased",
    tokenizer_max_length: int = 512,
    batch_size: int = 100,
    vector_col: str = "tabular_embedding_vector",
    prompt_col: str = "tabular_embedding_prompt",
) -> pd.DataFrame:
    """Append a tabular embedding vector + prompt column to df.

    Returns the dataframe unchanged with a logged warning if the embeddings
    extra is not installed.
    """
    try:
        from arize.embeddings import EmbeddingGenerator, UseCases
    except ImportError:
        logger.warning(
            "arize[embeddings] extra not installed; skipping tabular embeddings. "
            "Install with: pip install 'arize[embeddings]>=8.0,<9'"
        )
        return df

    if df.empty:
        return df

    generator = EmbeddingGenerator.from_use_case(
        use_case=UseCases.STRUCTURED.TABULAR_EMBEDDINGS,
        model_name=model_name,
        tokenizer_max_length=tokenizer_max_length,
        batch_size=batch_size,
    )
    vectors, prompts = generator.generate_embeddings(
        df,
        selected_columns=list(feature_columns),
        return_prompt_col=True,
    )
    out = df.copy()
    out[vector_col] = list(vectors)
    out[prompt_col] = list(prompts)
    return out
