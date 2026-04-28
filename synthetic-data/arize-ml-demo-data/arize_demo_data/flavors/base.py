"""Base protocol for industry-flavored generators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd


class FlavorBuilder(Protocol):
    """Callable contract for flavors.

    A flavor builder produces a fully-featured dataframe with at least
    a ``timestamp`` column plus all flavor-specific feature columns. It does
    not produce predictions or actuals — those are added by the label
    generator the flavor selects.
    """

    def __call__(
        self,
        n_rows: int,
        start: datetime,
        end: datetime,
        rng: np.random.Generator,
    ) -> pd.DataFrame: ...


@dataclass
class Flavor:
    """A flavor bundles feature generation, label spec, and metadata."""

    key: str
    industry: str
    default_model_type: str
    feature_columns: list[str]
    tag_columns: list[str]
    default_important_features: dict[str, float]
    default_spikes: list[str]
    description: str
    builder: FlavorBuilder
    label_spec_factory: callable
    """Callable () -> a label spec object (e.g. BinaryLabelSpec)."""
