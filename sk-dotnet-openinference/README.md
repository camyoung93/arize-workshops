# OpenInference span processor for Semantic Kernel .NET

POC OpenTelemetry `SpanProcessor` that rewrites Microsoft.SemanticKernel 1.54's
native OTel GenAI spans into [OpenInference](https://github.com/Arize-ai/openinference)
attributes so SK-instrumented .NET apps render correctly in Arize AX.

The processor sits on the `TracerProvider` ahead of the OTLP exporter and mutates
each `Microsoft.SemanticKernel.Diagnostics` activity in `OnEnd`. SK already emits
the OTel GenAI semantic convention; this is a translation layer, not an instrumentor.

## Layout

```
src/OpenInference.Instrumentation.SemanticKernel/   library (net8.0 + netstandard2.0)
samples/SkConsoleDemo/                              minimal chat completion sample
tests/...Tests/                                     one xUnit happy-path test
```

## Required diagnostics switches

SK 1.54 only emits the LLM activity (and the per-message events that carry the
prompt / completion content) when these switches are flipped to `true` **before**
SK loads:

```csharp
AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics", true);
AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnosticsSensitive", true);
```

Or via env vars:

```bash
export SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true
export SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE=true
```

Without `..._SENSITIVE`, the activity will appear in AX but `input.value` /
`output.value` / `llm.input_messages.*` will be empty. This is the single most
common failure mode for the demo.

## Mapping (verified against SK 1.54 source)

Source: [`dotnet/src/InternalUtilities/src/Diagnostics/ModelDiagnostics.cs`](https://github.com/microsoft/semantic-kernel/blob/dotnet-1.54.0/dotnet/src/InternalUtilities/src/Diagnostics/ModelDiagnostics.cs)

| SK / OTel GenAI                                   | OpenInference                                              |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `gen_ai.operation.name == "chat.completions"`     | `openinference.span.kind = "LLM"`                          |
| `gen_ai.operation.name == "text.completions"`     | `openinference.span.kind = "LLM"`                          |
| `gen_ai.operation.name == "invoke_agent"`         | `openinference.span.kind = "AGENT"`                        |
| `gen_ai.system == "openai"`                       | `llm.provider = "openai"`, `llm.system = "openai"`         |
| `gen_ai.system == "az.ai.openai"`                 | `llm.provider = "azure"`, `llm.system = "openai"`          |
| `gen_ai.system == <other>`                        | `llm.provider = <value>`, `llm.system = <value>`           |
| `gen_ai.request.model` (then `gen_ai.response.model`) | `llm.model_name`                                       |
| `gen_ai.usage.input_tokens`                       | `llm.token_count.prompt`                                   |
| `gen_ai.usage.output_tokens`                      | `llm.token_count.completion`                               |
| (prompt + completion if `total` absent)           | `llm.token_count.total`                                    |
| `gen_ai.request.{temperature, top_p, max_tokens, frequency_penalty, presence_penalty}` | packed into `llm.invocation_parameters` (JSON string) |
| Event `gen_ai.{system,user,assistant,tool}.message` with tag `gen_ai.event.content` (JSON `{role, content, ...}`) | `llm.input_messages.{i}.message.{role, content}` + `input.value` (JSON array) + `input.mime_type = "application/json"` |
| Event `gen_ai.choice` with tag `gen_ai.event.content` (JSON `{index, message:{role, content}, ...}`) | `llm.output_messages.{i}.message.{role, content}` + `output.value` + `output.mime_type` |

### Departures from the original handoff

The customer-facing handoff was based on the OpenLIT span shape. SK 1.54 differs:

- Operation values are `"chat.completions"` / `"text.completions"`, not `"chat"` / `"text_completion"`.
- SK does **not** emit `gen_ai.content.prompt` / `gen_ai.content.completion`
  with JSON-array bodies. It emits one event per message
  (`gen_ai.user.message`, `gen_ai.choice`, etc.), each carrying a
  `gen_ai.event.content` tag with the JSON body.
- SK only emits `temperature`, `top_p`, `max_tokens` invocation params.
  The processor still picks up `frequency_penalty` / `presence_penalty` if
  another emitter ever produces them, but SK won't.
- SK 1.54 sets `gen_ai.system = "openai"` for both the OpenAI and Azure OpenAI
  connectors. The `"az.ai.openai"` branch is dead code today but kept for
  forward compatibility.

## Usage

```csharp
using OpenInference.Instrumentation.SemanticKernel;
using OpenTelemetry;
using OpenTelemetry.Trace;

using var tracer = Sdk.CreateTracerProviderBuilder()
    .AddSource("Microsoft.SemanticKernel*")
    .AddProcessor(new OpenInferenceSpanProcessor())   // must come before exporters
    .AddOtlpExporter(opts =>
    {
        opts.Endpoint = new Uri("https://otlp.arize.com/v1/traces");
        opts.Protocol = OtlpExportProtocol.HttpProtobuf;
        opts.Headers = $"space_id={spaceId},api_key={apiKey}";
    })
    .Build();
```

## Run the sample

Two modes; both honour the diagnostics switches and the Arize OTLP exporter.

### Live LLM mode

Set Azure OpenAI **or** OpenAI direct creds, plus the Arize creds:

```bash
export AZURE_OPENAI_ENDPOINT="https://your.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
# OR
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"

export ARIZE_SPACE_ID="..."
export ARIZE_API_KEY="..."
export ARIZE_PROJECT_NAME="sk-dotnet-openinference-demo"
export ENABLE_OPENINFERENCE_PROCESSOR=true

dotnet run --project samples/SkConsoleDemo
```

### Synthetic mode (no LLM call)

Use this to verify the OTLP path and AX rendering without spending tokens:

```bash
export SYNTHETIC=true
export ENABLE_OPENINFERENCE_PROCESSOR=true
export ARIZE_SPACE_ID="..."
export ARIZE_API_KEY="..."
dotnet run --project samples/SkConsoleDemo
```

A hand-crafted SK-shaped activity is emitted on the
`Microsoft.SemanticKernel.Diagnostics` source, run through the processor, and
shipped to Arize.

### Inspect raw SK spans without the processor

Useful for confirming the SK 1.54 wire shape against future versions:

```bash
# leave ENABLE_OPENINFERENCE_PROCESSOR unset; console exporter prints raw spans
ENABLE_OPENINFERENCE_PROCESSOR=false dotnet run --project samples/SkConsoleDemo
```

## Test

```bash
dotnet test
```

One xUnit test builds a synthetic SK-shaped activity, runs it through
`OpenInferenceSpanProcessor.OnEnd`, and asserts every target OpenInference tag.

## Scope

POC. Optimised for "SK 1.54 + this processor + OTLP → working AX traces" and
nothing more. Known things explicitly out of scope:

- The third-party `OpenInference.NET` NuGet (constants are defined locally
  instead; swap in if Arize blesses that package).
- Package name `OpenInference.Instrumentation.SemanticKernel` is a placeholder
  pending Arize naming review.
- Tool/function call structure (POC stringifies the tool_calls JSON into
  `message.content` rather than emitting `TOOL`-kind spans).
- Kernel function spans (orchestration spans from
  `Microsoft.SemanticKernel`-prefixed sources other than `.Diagnostics`).
  The processor is a no-op on those; they flow through unmodified.
- DI registration helpers, retries, config systems, multiple processor instances.

## Verification

Demo trace (synthetic mode) sent to Arize space `LLM_test`, project
`sk-dotnet-openinference-demo`. AX rendered:

- Span kind `LLM`
- Model `gpt-4o-mini-2024-07-18` (response.model overrode request.model)
- Token counts (23 / 12 / 35), invocation parameters JSON
- Both input and output message panels populated
- Cost computed automatically by AX from model + token attribution
