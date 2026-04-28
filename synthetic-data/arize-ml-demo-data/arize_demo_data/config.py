"""Configuration dataclasses + YAML/CLI loaders for the demo-data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GenerationConfig:
    """Top-level configuration for one demo-data run."""

    flavor: str = "payments_fraud"
    model_type: str = "binary_classification"
    model_id: str = "demo-model"
    model_version: str = "1.0"

    base_rows: int = 100_000
    base_window_days: int = 90

    spike_rows: int = 25_000
    spike_window_days: int = 14
    spikes: list[str] = field(default_factory=list)

    target_metric: float = 0.80

    embeddings: str = "none"
    embedding_model: str = "distilbert-base-uncased"
    embedding_batch_size: int = 100

    shap: str = "synthetic"
    important_features: dict[str, float] = field(default_factory=dict)

    environments: list[str] = field(default_factory=lambda: ["production"])

    seed: int = 42
    output_dir: str = ".scratch/arize-ml-demo-data"
    log_to_arize: bool = True

    def merge_overrides(self, overrides: dict[str, Any]) -> "GenerationConfig":
        """Return a new config with non-None overrides applied."""
        data = self.__dict__.copy()
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise ValueError(f"Unknown config key: {key}")
            data[key] = value
        return GenerationConfig(**data)


def load_config(path: str | Path | None) -> GenerationConfig:
    """Load a YAML config from disk, or return defaults if path is None."""
    if path is None:
        return GenerationConfig()
    p = Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {p} must be a YAML mapping, got {type(raw).__name__}")
    valid_keys = set(GenerationConfig().__dict__.keys())
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config keys in {p}: {sorted(unknown)}")
    return GenerationConfig(**raw)
