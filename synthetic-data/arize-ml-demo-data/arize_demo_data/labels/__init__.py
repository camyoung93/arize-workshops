"""Label generators that produce predictions + actuals at a target metric."""

from arize_demo_data.labels.binary import (
    BinaryLabelSpec,
    generate_binary_labels,
)

__all__ = [
    "BinaryLabelSpec",
    "generate_binary_labels",
]
