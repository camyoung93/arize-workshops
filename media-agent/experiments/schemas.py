"""Typed schemas for experiment dataset rows and experiment run records."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentExample:
    """One row in the experiment dataset."""

    id: str
    question: str
    category: str  # simple | aggregation | multi_hop | constraint_following | edge_case
    difficulty: str  # easy | medium | hard
    expected_sql: str
    expected_answer: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    notes: str = ""
    role: str = "finance"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentRun:
    """One run record ready for Arize experiment upload."""

    example_id: str
    output: str  # final answer text
    sql_generated: str = ""
    sql_plan: str = ""
    tables_referenced: str = ""  # JSON list
    prompt_version: str = ""
    latency_ms: float = 0.0
    category: str = ""
    difficulty: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
