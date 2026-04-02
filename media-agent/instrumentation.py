"""
Centralized OpenTelemetry / OpenInference instrumentation for Arize AX.

Import this module before any code that uses the ADK runner or the Gen AI client.
- Registers the Arize tracer provider.
- Auto-instruments Google ADK (agent/tool spans) and Google Gen AI (LLM spans).
  LLM spans: https://arize-ai.github.io/openinference/python/instrumentation/openinference-instrumentation-google-genai/
- Exposes OpenInference tracer for manual spans (e.g. @tracer.llm, .chain, .tool) if needed.

Vertex AI is initialized here so the google-genai client (used in agents/tools)
can use Vertex as the backend.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from arize.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

# Register Arize as the OTLP tracer provider (must run before instrumentors).
tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=os.environ.get("ARIZE_PROJECT_NAME", "media-agent-demo"),
)

# Auto-instrument: ADK (agent/tool spans), Gen AI (generate_content → LLM spans).
GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# OpenInference tracer for manual spans if needed.
tracer = tracer_provider.get_tracer(__name__)

# Vertex AI init so google-genai client (vertexai=True) in agents/tools can use it.
import vertexai

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
)
