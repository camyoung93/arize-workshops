"""
Prompt template instrumentation for Arize AX.

Wraps OpenInference's using_prompt_template so every LLM call captures template,
version, and variables for the Arize Prompt Playground and evals.

Ref: https://arize.com/docs/ax/observe/tracing/configure/instrumenting-prompt-templates-and-prompt-variables
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import using_prompt_template


@contextmanager
def with_prompt_template(
    template: str,
    variables: dict[str, Any],
    version: str,
):
    """
    Capture prompt template, version, and variables on the current LLM span.

    OpenInference auto-instrumentors read this context and attach:
    - llm.prompt_template.template
    - llm.prompt_template.version
    - llm.prompt_template.variables

    so you can experiment with prompt changes in the Arize Prompt Playground.

    Args:
        template: The prompt template string with placeholders (e.g. {city}, {date}).
        variables: Dict of variable names to values used to fill the template.
        version: Version string for this template (e.g. "v1.0") for A/B comparison.

    Example:
        with with_prompt_template(
            template="Answer for {question}",
            variables={"question": "What is 2+2?"},
            version="v1.0",
        ):
            response = model.generate_content(prompt)
    """
    with using_prompt_template(
        template=template,
        variables=variables,
        version=version,
    ):
        yield
