"""Argparse CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from arize_demo_data.config import GenerationConfig, load_config
from arize_demo_data.pipeline import run


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arize-ml-demo-data",
        description="Generate synthetic ML demo data for Arize POCs.",
    )
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    p.add_argument("--flavor", type=str, default=None)
    p.add_argument(
        "--model-type",
        dest="model_type",
        type=str,
        default=None,
        choices=["binary_classification"],
        help="(M1 supports binary_classification only)",
    )
    p.add_argument("--model-id", dest="model_id", type=str, default=None)
    p.add_argument("--model-version", dest="model_version", type=str, default=None)

    p.add_argument("--base-rows", dest="base_rows", type=int, default=None)
    p.add_argument("--base-window-days", dest="base_window_days", type=int, default=None)
    p.add_argument("--spike-rows", dest="spike_rows", type=int, default=None)
    p.add_argument("--spike-window-days", dest="spike_window_days", type=int, default=None)
    p.add_argument(
        "--spikes",
        type=str,
        default=None,
        help="Comma-separated spike keys (e.g. feature_drift,missing_values).",
    )

    p.add_argument(
        "--target-metric", dest="target_metric", type=float, default=None,
        help="Target metric for the label generator (e.g. AUC ~0.80 for binary).",
    )

    p.add_argument(
        "--embeddings",
        type=str,
        default=None,
        choices=["none", "tabular"],
        help="Embedding strategy.",
    )
    p.add_argument("--embedding-model", dest="embedding_model", type=str, default=None)
    p.add_argument(
        "--shap",
        type=str,
        default=None,
        choices=["none", "synthetic"],
        help="SHAP value strategy.",
    )
    p.add_argument(
        "--environments",
        type=str,
        default=None,
        help="Comma-separated env list (production,training).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    p.add_argument(
        "--no-arize",
        dest="log_to_arize",
        action="store_false",
        default=None,
        help="Skip Arize upload, only write parquet locally.",
    )

    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase log verbosity."
    )
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    base = load_config(args.config) if args.config else GenerationConfig()

    overrides: dict = {
        "flavor": args.flavor,
        "model_type": args.model_type,
        "model_id": args.model_id,
        "model_version": args.model_version,
        "base_rows": args.base_rows,
        "base_window_days": args.base_window_days,
        "spike_rows": args.spike_rows,
        "spike_window_days": args.spike_window_days,
        "spikes": _parse_csv(args.spikes),
        "target_metric": args.target_metric,
        "embeddings": args.embeddings,
        "embedding_model": args.embedding_model,
        "shap": args.shap,
        "environments": _parse_csv(args.environments),
        "seed": args.seed,
        "output_dir": args.output_dir,
        "log_to_arize": args.log_to_arize,
    }
    cfg = base.merge_overrides({k: v for k, v in overrides.items() if v is not None})

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    written = run(cfg)
    for env, path in written.items():
        print(f"{env}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
