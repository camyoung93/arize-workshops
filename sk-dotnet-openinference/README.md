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

The `tool_call_id?` and `tool_calls?` markers describe what the processor
handles, not what SK 1.54 actually emits. See [SK 1.54 limitations](#sk-154-limitations)
below for the specific gaps (notably `tool_call_id` on `gen_ai.tool.message`
events and `gen_ai.response.model` on the activity).

For `invoke_agent` activities the processor also passes through SK's
`gen_ai.agent.id` / `gen_ai.agent.name` / `gen_ai.agent.description` tags
unchanged. They are not remapped to OpenInference-prefixed attributes
because AX already renders the `gen_ai.agent.*` namespace natively in the
span Attributes panel. The AGENT span has no `input.value` / `output.value`
in SK 1.54; see [Enriching invoke_agent spans](#enriching-invoke_agent-spans-optional).

### Kernel function spans (ActivitySource `Microsoft.SemanticKernel`)

| SK                                                | OpenInference                                              |
| ------------------------------------------------- | ---------------------------------------------------------- |
| Any activity                                      | `openinference.span.kind = "TOOL"`                         |
| `activity.DisplayName` (= kernel function name)   | `tool.name`                                                |

**SK 1.54 limitation.** `KernelFunction.InvokeAsync` calls
`s_activitySource.StartActivity(this.Name)` and adds **no tags** about the
arguments or return value, so the translated `TOOL` span renders with only
`tool.name`. The richer per-call payload lives inside the parent LLM span's
`gen_ai.tool.message` event body, which the processor exposes via
`llm.input_messages.{i}.message.content` (and `.tool_call_id` when SK emits
it). To populate the `TOOL` span itself, see
[Enriching kernel function spans](#enriching-kernel-function-spans-optional).

### Enriching kernel function spans (optional)

To make a `TOOL` span carry the function's actual input and output, register
an `IFunctionInvocationFilter` that attaches them to `Activity.Current`
before the processor sees the span. The processor leaves any pre-existing
`input.value` / `output.value` tags alone:

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

### Enriching `invoke_agent` spans (optional)

SK 1.54's `StartAgentInvocationActivity` only attaches
`gen_ai.agent.{id, name, description}` to the AGENT span. It does **not**
write the user's input or the agent's final response, so AX's Input / Output
tab on an AGENT span is empty by default. The processor deliberately does
not paper over this by mapping `description` into `input.value`; description
is static metadata about what the agent is, not what was asked this turn,
and forcing it into the input slot would mislead anyone filtering or
searching traces by user input.

If you want the Input / Output tab populated, wrap the call so the actual
`ChatHistory` and final response land on `Activity.Current`:

```csharp
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;

static async Task<ChatMessageContent?> InvokeWithIoAsync(
    ChatCompletionAgent agent,
    ChatHistory history)
{
    var activity = Activity.Current;
    activity?.SetTag("input.value", JsonSerializer.Serialize(
        history.Select(m => new { role = m.Role.Label, content = m.Content })));
    activity?.SetTag("input.mime_type", "application/json");

    ChatMessageContent? final = null;
    await foreach (var msg in agent.InvokeAsync(history))
    {
        final = msg;
    }

    if (activity is not null && final is not null)
    {
        activity.SetTag("output.value", JsonSerializer.Serialize(
            new { role = final.Role.Label, content = final.Content }));
        activity.SetTag("output.mime_type", "application/json");
    }
    return final;
}
```

The helper runs inside the `invoke_agent` activity's scope (started by
`agent.InvokeAsync`), so `Activity.Current` is the AGENT span when the tags
are set. The processor leaves any pre-existing `input.value` / `output.value`
tags alone, so the explicit user-supplied values win.

### SK 1.54 limitations

A few attributes the OTel GenAI semconv defines aren't populated by SK 1.54.
The processor handles each one when present, so these resolve automatically
as SK evolves; they're listed here so the AX render isn't surprising:

- Operation values are `"chat.completions"` / `"text.completions"`, not
  `"chat"` / `"text_completion"`.
- SK does **not** emit `gen_ai.content.prompt` / `gen_ai.content.completion`
  with JSON-array bodies. It emits one event per message
  (`gen_ai.user.message`, `gen_ai.choice`, etc.), each carrying a
  `gen_ai.event.content` tag with the JSON body.
- SK only emits `temperature`, `top_p`, `max_tokens` invocation params. The
  processor picks up `frequency_penalty` / `presence_penalty` too, but SK
  doesn't produce them.
- `gen_ai.system = "openai"` is set for both the OpenAI and Azure OpenAI
  connectors. So Azure runs render as `llm.provider = "openai"`, not
  `"azure"`, in AX cost dashboards and span filters. The `"az.ai.openai"`
  branch in the processor is dead code today, kept for forward compatibility.
- `gen_ai.response.model` is declared in SK's `ModelDiagnosticsTags` but
  never written by `ModelDiagnostics`. So `llm.model_name` lands as the
  request model (e.g. `gpt-4o-mini`), not the dated revision
  (`gpt-4o-mini-2024-07-18`) the OpenAI API actually returns. The
  `response.model` preference path in the processor is dead code in 1.54.
- `gen_ai.tool.message` events carry `role` + `content` only. SK 1.54 does
  **not** include `tool_call_id` even though `FunctionResultContent.CallId`
  is available, so the assistant-call → tool-response correlation
  (`llm.input_messages.{i}.message.tool_call_id`) is absent on real traces.
- Kernel function spans are bare (display name only); see the table above.
- AGENT spans (`invoke_agent`) carry only `gen_ai.agent.{id, name, description}`.
  No `input.value` / `output.value`, no token totals across the agent
  invocation. See [Enriching invoke_agent spans](#enriching-invoke_agent-spans-optional).
- The (non-obsolete) `ChatCompletionAgent.InvokeAsync(ICollection,
  AgentThread, AgentInvokeOptions)` overload does **not** wrap the call in
  an `invoke_agent` activity in 1.54. Only the `[Obsolete]`
  `InvokeAsync(ChatHistory, KernelArguments, Kernel)` overload calls
  `ModelDiagnostics.StartAgentInvocationActivity`. The sample uses the
  obsolete overload for this reason; should be revisited when SK wires
  activity emission into the newer path.

### Streaming

Streaming chat completions (`GetStreamingChatMessageContentsAsync`) are
supported transparently by the processor. SK 1.54 starts the chat activity
once at the beginning of the call, accumulates the streamed chunks
internally, and at the end calls
[`activity.EndStreaming(streamedContents, ...)`](https://github.com/microsoft/semantic-kernel/blob/dotnet-1.54.0/dotnet/src/Connectors/Connectors.OpenAI/Core/ClientCore.ChatCompletion.cs#L365)
which routes through the same `SetCompletionResponse` path the
non-streaming call uses. By the time the activity ends, the tags and
`gen_ai.choice` event look identical to a non-streaming call (aggregated
token counts, single assistant message built from the chunks). The
processor's `OnEnd` cannot tell the two apart.

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

## Why `ChatCompletionAgent` (and not `IChatCompletionService`)

The sample uses `Microsoft.SemanticKernel.Agents.ChatCompletionAgent`
rather than calling `IChatCompletionService.GetChatMessageContentAsync`
directly. Both flows go through the same processor; the difference is the
trace shape SK actually produces.

**Calling `IChatCompletionService` directly** with `FunctionChoiceBehavior.Auto()`
produces three orphan root spans per turn:

```
chat.completions gpt-4o     (LLM, root)
GetWeather                  (TOOL, root)
chat.completions gpt-4o     (LLM, root)
```

This is because SK 1.54 ends the first `chat.completions` activity when the
API call returns, dispatches the auto-invoke loop outside that activity's
scope, and starts a fresh activity for the second API call. The processor
can't fix this; it's structural to how SK breaks up its work.

**Calling through `ChatCompletionAgent.InvokeAsync(ChatHistory, ...)` instead**
wraps the whole conversation in a `gen_ai.operation.name = invoke_agent`
activity, which the processor maps to `openinference.span.kind = AGENT`.
That gives the conventional tree:

```
invoke_agent WeatherConcierge   (AGENT, root)
├── chat.completions gpt-4o     (LLM)
├── GetWeather                  (TOOL, tool.name=GetWeather)
└── chat.completions gpt-4o     (LLM)
```

Adoption cost on the customer side is small: wrap an existing
`IChatCompletionService` user in `ChatCompletionAgent { Kernel = ..., Arguments = ... }`
and switch the call site. Plugins and `FunctionChoiceBehavior.Auto()` work
identically.

If raw `IChatCompletionService` usage cannot be changed, the customer can
wrap each request handler in their own outer `ActivitySource.StartActivity`
to get a shared parent across the three orphan spans. The processor will
ignore that wrapping activity (different source name), so its only effect
is providing a parent.

## Run the sample

The full list of environment variables (and their defaults) lives in
[`.env.example`](.env.example). Copy it to `.env`, fill in the credentials
you have, and source it before running. .NET console apps don't auto-load
`.env` files the way Python and Node demos do, so the sourcing step is
explicit:

```bash
cp .env.example .env
# edit .env with your Azure OpenAI or OpenAI key, plus Arize space/key
set -a && source .env && set +a
dotnet run --project samples/SkConsoleDemo
```

(`set -a` exports every variable defined while it's active; `set +a` turns
that back off after the source.)

Pass a question as a positional arg to override the default
("What's the weather in Paris right now?"):

```bash
dotnet run --project samples/SkConsoleDemo -- "What's the weather in Tokyo, and how does it compare to Reykjavik?"
```

The sample registers a small `WeatherPlugin`, wraps the call in a
`ChatCompletionAgent`, and uses `FunctionChoiceBehavior.Auto()`. A single
run lands one trace in AX containing the AGENT root, two child
`chat.completions` LLM spans, and the auto-invoked `GetWeather` TOOL span.

The TOOL span carries only `tool.name` by default (SK 1.54 limitation); the
AGENT span has empty Input / Output (SK 1.54 limitation). Two optional
wiring recipes ([function filter](#enriching-kernel-function-spans-optional),
[agent wrapper](#enriching-invoke_agent-spans-optional)) populate them
without modifying the processor.

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
- Enriching `TOOL` and `AGENT` spans with their actual input / output. SK 1.54
  doesn't put these on the activity; both recipes (`IFunctionInvocationFilter`
  for TOOL, agent-wrapper helper for AGENT) live in the customer's application
  code, not in this processor, because they require access to the
  `KernelArguments` / `ChatHistory` only the caller has.
- Remapping `gen_ai.agent.*` into OpenInference-prefixed attributes. AX
  renders the `gen_ai.agent.*` namespace natively, and `description` is
  metadata about what the agent is, not what was asked this turn (so it is
  intentionally not mapped to `input.value`).
- DI registration helpers, retries, config systems, multiple processor instances.

## Adapting this to a real .NET app

The sample is a single-file console quickstart. Four things to change when
lifting the wiring into a hosted ASP.NET Core / worker service:

1. **Drop the `ENABLE_OPENINFERENCE_PROCESSOR` toggle.** It exists in the
   sample only so you can compare with/without translation. In a real app
   the processor is always on - just one `.AddProcessor(new OpenInferenceSpanProcessor())`
   line on the `TracerProviderBuilder`, no conditional.
2. **Move the diagnostics switches into configuration.** Set
   `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE=true`
   via env var, launch profile, or `appsettings.json` rather than calling
   `AppContext.SetSwitch` in `Program.cs`. The flag still needs to be set
   before SK loads; configuration sources read at host startup are early
   enough.
3. **Split `service.name` from the Arize project name.** The sample
   conflates them as `projectName` for brevity. In a real deployment,
   `service.name` is per-process (e.g. `weather-api`) and the Arize project
   is a logical grouping (e.g. `weather-prod`); they should be separate
   config values bound via `IConfiguration`.
4. **Read all secrets and endpoints from `IConfiguration` rather than
   `Environment.GetEnvironmentVariable`.** That gives you `appsettings.json`
   + environment-variable overrides + secret-manager integration
   (Azure Key Vault, etc.) for free. The sample uses raw env vars only
   because it has no host.

Other things the sample omits that production code would add
(`ILoggerFactory` passed into the SK connector, `CancellationToken`
plumbing into `agent.InvokeAsync`, retry / error handling around the LLM
call, graceful `TracerProvider` shutdown via `IHost`) are standard
console-vs-service tradeoffs - none of them affect what the processor does
or how spans render in AX.

## Verification

Demo traces land in Arize space `LLM_test`, project
`sk-dotnet-openinference-demo`. One end-to-end scenario exercised:

**Agent + tool-call flow.** Real SK 1.54 call through
`ChatCompletionAgent` against Azure OpenAI, with `WeatherPlugin`
registered and `FunctionChoiceBehavior.Auto()` enabled. AX renders a
single trace tree:

```
invoke_agent WeatherConcierge   (AGENT, root)
├── chat.completions gpt-4o     (LLM)
├── GetWeather                  (TOOL, tool.name=GetWeather)
└── chat.completions gpt-4o     (LLM)
```

The LLM spans carry full `llm.input_messages.*` / `llm.output_messages.*`
panels including the assistant tool-call (with `tool_calls.0.tool_call.*`
attributes) and the tool-role response. Token counts and invocation
parameters land on both LLM spans; AX auto-computes cost from
`llm.model_name` + token counts.

The AGENT span has only `gen_ai.agent.{id, name, description}` populated;
the TOOL span has only `tool.name`. Both are SK 1.54 structural limitations
documented above with concrete enrichment recipes.
