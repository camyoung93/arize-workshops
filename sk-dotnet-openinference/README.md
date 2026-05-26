# OpenInference span processor for Semantic Kernel .NET

POC OpenTelemetry `SpanProcessor` that rewrites Microsoft.SemanticKernel 1.54's
native OTel GenAI spans into [OpenInference](https://github.com/Arize-ai/openinference)
attributes so SK-instrumented .NET apps render correctly in Arize AX.

The processor sits on the `TracerProvider` ahead of the OTLP exporter and mutates
SK activities from two ActivitySources in `OnEnd`:

- `Microsoft.SemanticKernel.Diagnostics` - LLM / agent model spans from `ModelDiagnostics`
- `Microsoft.SemanticKernel` - kernel function invocation spans from `KernelFunction.InvokeAsync`

SK already emits both; this is a translation layer, not an instrumentor.

## Layout

```
src/OpenInference.Instrumentation.SemanticKernel/   library (net8.0 + netstandard2.0)
samples/SkConsoleDemo/                              minimal chat completion sample
tests/...Tests/                                     xUnit happy-path + tool-call tests
```

## Required diagnostics switches

The LLM activity (and the per-message events that carry the prompt / completion
content) only flow when SK's diagnostics switches are flipped to `true`
**before** SK loads.

There are two switches with different scopes:

| Switch | Activity emitted? | Prompt/completion events emitted? |
|---|---|---|
| neither | no | no (default) |
| `EnableOTelDiagnostics` only | yes | no |
| `EnableOTelDiagnosticsSensitive` (alone or with the other) | yes | yes |

Setting just `Sensitive=true` is sufficient - it implies the activity too.
The sample sets both for belt-and-suspenders:

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

The `Microsoft.SemanticKernel` (kernel function) activities do **not** depend
on these switches; they emit unconditionally whenever a kernel function is
invoked and a TracerProvider is listening to the source.

## Mapping (verified against SK 1.54 source)

Sources:
[`ModelDiagnostics.cs`](https://github.com/microsoft/semantic-kernel/blob/dotnet-1.54.0/dotnet/src/InternalUtilities/src/Diagnostics/ModelDiagnostics.cs)
and
[`KernelFunction.cs`](https://github.com/microsoft/semantic-kernel/blob/dotnet-1.54.0/dotnet/src/SemanticKernel.Abstractions/Functions/KernelFunction.cs)

### LLM / agent spans (ActivitySource `Microsoft.SemanticKernel.Diagnostics`)

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
| Event `gen_ai.{system,user,assistant,tool}.message` with tag `gen_ai.event.content` (JSON `{role, content, tool_calls?, tool_call_id?}`) | `llm.input_messages.{i}.message.role` plus `.content` / `.tool_call_id` / `.tool_calls.{j}.tool_call.{id, function.name, function.arguments}` as present, plus `input.value` (OpenAI-shaped JSON array) + `input.mime_type = "application/json"` |
| Event `gen_ai.choice` with tag `gen_ai.event.content` (JSON `{index, message:{role, content, tool_calls?}, finish_reason}`) | `llm.output_messages.{i}.message.role` plus `.content` / `.tool_calls.{j}.tool_call.*` as present, plus `output.value` + `output.mime_type` |

Empty/null `content` (e.g. an assistant message that is only a tool call) is
omitted from both the indexed `message.content` tag and the `input.value` /
`output.value` JSON, rather than emitted as `""`.

### Kernel function spans (ActivitySource `Microsoft.SemanticKernel`)

| SK                                                | OpenInference                                              |
| ------------------------------------------------- | ---------------------------------------------------------- |
| Any activity                                      | `openinference.span.kind = "TOOL"`                         |
| `activity.DisplayName` (= kernel function name)   | `tool.name`                                                |

**SK 1.54 limitation.** `KernelFunction.InvokeAsync` calls
`s_activitySource.StartActivity(this.Name)` and adds **no tags** about the
arguments or return value. The translated `TOOL` span therefore renders with
only `tool.name` populated. The richer per-call payload lives inside the
parent LLM span's `gen_ai.tool.message` event body, which the processor
already exposes via `llm.input_messages.{i}.message.tool_call_id` + `.content`
on the LLM span. This is a Semantic Kernel limitation, not an Arize AX
limitation; see [Enriching kernel function spans](#enriching-kernel-function-spans-optional)
below for the recommended fix on the customer side.

### Enriching kernel function spans (optional)

To make a `TOOL` span carry the function's actual input and output, register
an `IFunctionInvocationFilter` in the customer's kernel that attaches them to
`Activity.Current` before the processor sees the span. The processor will
leave any pre-existing `input.value` / `output.value` tags alone:

```csharp
using System.Diagnostics;
using System.Text.Json;
using Microsoft.SemanticKernel;

internal sealed class TraceFunctionFilter : IFunctionInvocationFilter
{
    public async Task OnFunctionInvocationAsync(
        FunctionInvocationContext context,
        Func<FunctionInvocationContext, Task> next)
    {
        var activity = Activity.Current;
        activity?.SetTag("input.value", JsonSerializer.Serialize(context.Arguments));
        activity?.SetTag("input.mime_type", "application/json");

        await next(context);

        if (activity is not null && context.Result.GetValue<object>() is { } value)
        {
            activity.SetTag("output.value", JsonSerializer.Serialize(value));
            activity.SetTag("output.mime_type", "application/json");
        }
    }
}

// Wire into the kernel:
kernelBuilder.Services.AddSingleton<IFunctionInvocationFilter, TraceFunctionFilter>();
```

With this filter wired, the `TOOL` span in AX renders with the function name,
arguments, and return value.

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
- Kernel function spans are bare (display name only); see the SK 1.54
  limitation note in the Kernel function spans table above.

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

Four xUnit tests:

1. `Translates_synthetic_chat_completion_activity_to_openinference_attributes` -
   baseline happy path (provider/system, model, tokens, invocation params,
   indexed messages, `input.value` / `output.value`).
2. `Translates_synthetic_chat_with_tool_call_round_trip` - assistant tool call
   with `content: null`, tool result with `tool_call_id`, final assistant
   answer. Asserts indexed `tool_calls.0.tool_call.{id, function.name,
   function.arguments}`, `tool_call_id` on the tool message, and that
   `input.value` JSON round-trips the tool calls with `content` omitted.
3. `Translates_kernel_function_activity_to_tool_span_kind` - synthetic
   activity on `Microsoft.SemanticKernel` source becomes
   `openinference.span.kind = "TOOL"` with `tool.name` from the display name.
4. `Ignores_unrelated_activity_source` - guard test confirming the processor
   leaves arbitrary other ActivitySources untouched.

## Scope

POC. Optimised for "SK 1.54 + this processor + OTLP → working AX traces" and
nothing more. Known things explicitly out of scope:

- The third-party `OpenInference.NET` NuGet (constants are defined locally
  instead; swap in if Arize blesses that package).
- Package name `OpenInference.Instrumentation.SemanticKernel` is a placeholder
  pending Arize naming review.
- Enriching kernel function spans with arguments / return value. SK 1.54
  doesn't put these on the activity; the recommended workaround
  (`IFunctionInvocationFilter`) is documented above and lives on the customer
  side, not in this processor.
- `agent.*` attributes on `invoke_agent` activities (SK emits `gen_ai.agent.id`
  / `gen_ai.agent.name` / `gen_ai.agent.description`; processor currently maps
  only the LLM-shaped tags on those spans).
- DI registration helpers, retries, config systems, multiple processor instances.

## Verification

Demo trace (synthetic mode) sent to Arize space `LLM_test`, project
`sk-dotnet-openinference-demo`. AX rendered:

- Span kind `LLM`
- Model `gpt-4o-mini-2024-07-18` (response.model overrode request.model)
- Token counts (23 / 12 / 35), invocation parameters JSON
- Both input and output message panels populated
- Cost computed automatically by AX from model + token attribution

The tool-call and `TOOL` span-kind translations have been verified at the
unit-test level (xUnit) but not yet round-tripped to AX. The synthetic mode
in `samples/SkConsoleDemo` does not currently emit a tool-call scenario or a
kernel function span - extending it is a quick follow-up if a live AX render
of those is needed.
