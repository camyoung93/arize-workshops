"""End-to-end orchestration: build base + spike data, label, embed, SHAP, log."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from arize_demo_data import embeddings as emb_mod
from arize_demo_data import shap_synth
from arize_demo_data.arize_logger import LoggerCredentials, log_dataframe_to_arize
from arize_demo_data.config import GenerationConfig
from arize_demo_data.core import timestamps as ts_mod
from arize_demo_data.flavors import get_flavor
from arize_demo_data.labels.binary import generate_binary_labels
from arize_demo_data.spikes import SpikeContext, apply_spikes

logger = logging.getLogger(__name__)


def _generate_environment(
    *,
    flavor,
    cfg: GenerationConfig,
    rng: np.random.Generator,
    base_rows: int,
    spike_rows: int,
    base_window_days: int,
    spike_window_days: int,
    label_signal_weight_override: float | None = None,
) -> pd.DataFrame:
    """Generate one environment's worth of data (base + optional spikes + labels)."""
    end = ts_mod.now_utc()
    base_start = end - timedelta(days=base_window_days)
    spike_start = end - timedelta(days=spike_window_days)
    base_end = spike_start

    if base_rows > 0 and (base_end - base_start).total_seconds() <= 0:
        base_end = end
        spike_rows = 0

    base_df = flavor.builder(base_rows, base_start, base_end, rng)
    spike_df = flavor.builder(spike_rows, spike_start, end, rng)

    if not spike_df.empty and cfg.spikes:
        ctx = SpikeContext(rng=rng, flavor_key=flavor.key)
        spike_df = apply_spikes(spike_df, cfg.spikes, ctx)
        spike_df["is_spike"] = 1
    else:
        spike_df["is_spike"] = 0
    if not base_df.empty:
        base_df["is_spike"] = 0

    combined = pd.concat([base_df, spike_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if cfg.model_type == "binary_classification":
        spec = flavor.label_spec_factory()
        if label_signal_weight_override is not None:
            spec.actual_signal_weight = label_signal_weight_override
        combined = generate_binary_labels(combined, spec, rng)
    else:
        raise NotImplementedError(
            f"model_type '{cfg.model_type}' not implemented yet (M1 supports binary_classification)"
        )

    return combined


def run(cfg: GenerationConfig) -> dict[str, Path]:
    """Run the full pipeline. Returns a dict of environment -> parquet path."""
    rng = np.random.default_rng(cfg.seed)
    flavor = get_flavor(cfg.flavor)

    if not cfg.spikes:
        cfg = cfg.merge_overrides({"spikes": list(flavor.default_spikes)})

    important = cfg.important_features or dict(flavor.default_important_features)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    credentials = LoggerCredentials.from_env() if cfg.log_to_arize else None
    if cfg.log_to_arize and credentials is None:
        logger.warning(
            "ARIZE_SPACE_ID/ARIZE_API_KEY missing or upload disabled — "
            "datasets will be written to disk only."
        )

    written: dict[str, Path] = {}

    for env in cfg.environments:
        if env == "training":
            df = _generate_environment(
                flavor=flavor,
                cfg=cfg,
                rng=rng,
                base_rows=cfg.base_rows,
                spike_rows=0,
                base_window_days=cfg.base_window_days * 2,
                spike_window_days=cfg.base_window_days,
                label_signal_weight_override=0.85,
            )
        elif env == "production":
            df = _generate_environment(
                flavor=flavor,
                cfg=cfg,
                rng=rng,
                base_rows=cfg.base_rows,
                spike_rows=cfg.spike_rows,
                base_window_days=cfg.base_window_days,
                spike_window_days=cfg.spike_window_days,
            )
        else:
            raise ValueError(f"Unsupported environment in M1: {env}")

        emb_vec_col: str | None = None
        emb_prompt_col: str | None = None
        if cfg.embeddings == "tabular":
            df = emb_mod.add_tabular_embeddings(
                df,
                feature_columns=flavor.feature_columns,
                model_name=cfg.embedding_model,
                batch_size=cfg.embedding_batch_size,
            )
            if "tabular_embedding_vector" in df.columns:
                emb_vec_col = "tabular_embedding_vector"
                emb_prompt_col = "tabular_embedding_prompt"

        shap_mapping: dict[str, str] | None = None
        if cfg.shap == "synthetic":
            df, shap_mapping = shap_synth.add_synthetic_shap(
                df,
                feature_columns=flavor.feature_columns,
                important_features=important,
                rng=rng,
            )

        out_path = out_dir / f"{cfg.flavor}_{cfg.model_id}_{env}.parquet"
        df.to_parquet(out_path, index=False)
        written[env] = out_path
        logger.info("wrote %s rows to %s", len(df), out_path)

        if credentials is not None:
            response = log_dataframe_to_arize(
                df,
                credentials=credentials,
                model_id=cfg.model_id,
                model_version=cfg.model_version,
                model_type=cfg.model_type,
                environment=env,
                feature_columns=flavor.feature_columns,
                tag_columns=flavor.tag_columns,
                embedding_vector_col=emb_vec_col,
                embedding_prompt_col=emb_prompt_col,
                shap_mapping=shap_mapping,
            )
            if response is not None:
                logger.info("logged %s rows to Arize (%s)", len(df), env)

    return written
