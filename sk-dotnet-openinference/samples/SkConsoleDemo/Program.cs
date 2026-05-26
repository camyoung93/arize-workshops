using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using OpenInference.Instrumentation.SemanticKernel;
using OpenTelemetry;
using OpenTelemetry.Exporter;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

// MUST be set before SK loads; without it SK omits the prompt/completion event content.
// This is the single most common failure mode for empty input/output in Arize AX.
AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics", true);
AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnosticsSensitive", true);

var azureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
var azureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
var azureDeployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT");
var openAiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
var openAiModel = Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? "gpt-4o-mini";

var arizeSpaceId = Environment.GetEnvironmentVariable("ARIZE_SPACE_ID");
var arizeApiKey = Environment.GetEnvironmentVariable("ARIZE_API_KEY");
var arizeEndpoint = Environment.GetEnvironmentVariable("ARIZE_OTLP_ENDPOINT")
                    ?? "https://otlp.arize.com/v1/traces";
var projectName = Environment.GetEnvironmentVariable("ARIZE_PROJECT_NAME")
                  ?? "sk-dotnet-openinference-demo";

var enableProcessor = (Environment.GetEnvironmentVariable("ENABLE_OPENINFERENCE_PROCESSOR") ?? "false")
    .Equals("true", StringComparison.OrdinalIgnoreCase);
var syntheticMode = (Environment.GetEnvironmentVariable("SYNTHETIC") ?? "false")
    .Equals("true", StringComparison.OrdinalIgnoreCase);

var tracerBuilder = Sdk.CreateTracerProviderBuilder()
    .SetResourceBuilder(ResourceBuilder.CreateDefault()
        .AddService(serviceName: projectName)
        .AddAttributes(new[] { new KeyValuePair<string, object>("openinference.project.name", projectName) }))
    .AddSource("Microsoft.SemanticKernel*");

if (enableProcessor)
{
    Console.WriteLine("[demo] OpenInference processor enabled");
    tracerBuilder.AddProcessor(new OpenInferenceSpanProcessor());
}
else
{
    Console.WriteLine("[demo] OpenInference processor DISABLED (set ENABLE_OPENINFERENCE_PROCESSOR=true to enable)");
}

tracerBuilder.AddConsoleExporter();

if (!string.IsNullOrWhiteSpace(arizeSpaceId) && !string.IsNullOrWhiteSpace(arizeApiKey))
{
    Console.WriteLine($"[demo] OTLP exporter -> {arizeEndpoint} (space_id={arizeSpaceId})");
    tracerBuilder.AddOtlpExporter(opts =>
    {
        opts.Endpoint = new Uri(arizeEndpoint);
        opts.Protocol = OtlpExportProtocol.HttpProtobuf;
        opts.Headers = $"space_id={arizeSpaceId},api_key={arizeApiKey}";
    });
}

using var tracerProvider = tracerBuilder.Build();

if (syntheticMode)
{
    Console.WriteLine("[demo] SYNTHETIC=true: emitting hand-crafted SK-shaped activities (no LLM call)");
    EmitSyntheticActivity();
    EmitSyntheticToolCallFlow();
}
else
{
    var kernelBuilder = Kernel.CreateBuilder();
    if (!string.IsNullOrWhiteSpace(azureEndpoint) && !string.IsNullOrWhiteSpace(azureKey) && !string.IsNullOrWhiteSpace(azureDeployment))
    {
        Console.WriteLine($"[demo] Using Azure OpenAI deployment '{azureDeployment}' at {azureEndpoint}");
        kernelBuilder.AddAzureOpenAIChatCompletion(azureDeployment!, azureEndpoint!, azureKey!);
    }
    else if (!string.IsNullOrWhiteSpace(openAiKey))
    {
        Console.WriteLine($"[demo] Using OpenAI model '{openAiModel}'");
        kernelBuilder.AddOpenAIChatCompletion(openAiModel, openAiKey!);
    }
    else
    {
        Console.Error.WriteLine("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT, or OPENAI_API_KEY. Or set SYNTHETIC=true.");
        return 1;
    }

    var kernel = kernelBuilder.Build();
    var chat = kernel.GetRequiredService<IChatCompletionService>();

    var history = new ChatHistory();
    history.AddSystemMessage("You are a concise assistant. Reply in one sentence.");
    history.AddUserMessage("What is the capital of France?");

    var reply = await chat.GetChatMessageContentAsync(history);
    Console.WriteLine();
    Console.WriteLine($"[demo] Reply: {reply.Content}");
}

tracerProvider.ForceFlush();
return 0;

static void EmitSyntheticActivity()
{
    using var source = new ActivitySource("Microsoft.SemanticKernel.Diagnostics");
    using var activity = source.StartActivity("chat.completions gpt-4o-mini", ActivityKind.Client);
    if (activity is null)
    {
        Console.Error.WriteLine("[demo] Synthetic activity not sampled. Is the TracerProvider built and listening?");
        return;
    }

    activity.SetTag("gen_ai.operation.name", "chat.completions");
    activity.SetTag("gen_ai.system", "openai");
    activity.SetTag("gen_ai.request.model", "gpt-4o-mini");
    activity.SetTag("gen_ai.response.model", "gpt-4o-mini-2024-07-18");
    activity.SetTag("gen_ai.request.temperature", 0.7);
    activity.SetTag("gen_ai.request.top_p", 1.0);
    activity.SetTag("gen_ai.request.max_tokens", 256);
    activity.SetTag("gen_ai.usage.input_tokens", 23);
    activity.SetTag("gen_ai.usage.output_tokens", 12);
    activity.SetTag("server.address", "api.openai.com");

    activity.AddEvent(new ActivityEvent(
        "gen_ai.system.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"system\", \"content\": \"You are a concise assistant. Reply in one sentence.\"}",
        }));
    activity.AddEvent(new ActivityEvent(
        "gen_ai.user.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"user\", \"content\": \"What is the capital of France?\"}",
        }));
    activity.AddEvent(new ActivityEvent(
        "gen_ai.choice",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"The capital of France is Paris.\"}, \"tool_calls\": [], \"finish_reason\": \"stop\"}",
        }));

    Console.WriteLine($"[demo] Synthetic chat activity id={activity.TraceId}");
}

// Emits one trace that mirrors SK's auto-invoke tool-calling flow:
// a chat.completions LLM span that, mid-life, starts a child KernelFunction
// span for the tool execution, then continues with the tool result and the
// final assistant choice. This matches how SK 1.54 builds the activity tree
// when ToolCallBehavior.AutoInvokeKernelFunctions is enabled.
static void EmitSyntheticToolCallFlow()
{
    using var llmSource = new ActivitySource("Microsoft.SemanticKernel.Diagnostics");
    using var kernelSource = new ActivitySource("Microsoft.SemanticKernel");

    using var chat = llmSource.StartActivity("chat.completions gpt-4o-mini", ActivityKind.Client);
    if (chat is null)
    {
        Console.Error.WriteLine("[demo] Tool-call chat activity not sampled.");
        return;
    }

    chat.SetTag("gen_ai.operation.name", "chat.completions");
    chat.SetTag("gen_ai.system", "openai");
    chat.SetTag("gen_ai.request.model", "gpt-4o-mini");
    chat.SetTag("gen_ai.response.model", "gpt-4o-mini-2024-07-18");
    chat.SetTag("gen_ai.request.temperature", 0.2);
    chat.SetTag("gen_ai.usage.input_tokens", 87);
    chat.SetTag("gen_ai.usage.output_tokens", 24);
    chat.SetTag("server.address", "api.openai.com");

    chat.AddEvent(new ActivityEvent(
        "gen_ai.system.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"system\", \"content\": \"You can call tools to look up live weather.\"}",
        }));
    chat.AddEvent(new ActivityEvent(
        "gen_ai.user.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"user\", \"content\": \"What's the weather in Paris right now?\"}",
        }));
    chat.AddEvent(new ActivityEvent(
        "gen_ai.assistant.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"call_abc123\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"celsius\"}}}]}",
        }));

    // Child KernelFunction activity, opened while `chat` is still Activity.Current.
    using (var tool = kernelSource.StartActivity("get_weather", ActivityKind.Internal))
    {
        if (tool is not null)
        {
            // Simulates what an IFunctionInvocationFilter would attach in real customer code.
            tool.SetTag("input.value", "{\"city\":\"Paris\",\"units\":\"celsius\"}");
            tool.SetTag("input.mime_type", "application/json");
            tool.SetTag("output.value", "{\"temperature_c\":22,\"conditions\":\"Sunny\"}");
            tool.SetTag("output.mime_type", "application/json");
        }
    }

    chat.AddEvent(new ActivityEvent(
        "gen_ai.tool.message",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"role\": \"tool\", \"tool_call_id\": \"call_abc123\", \"content\": \"{\\\"temperature_c\\\": 22, \\\"conditions\\\": \\\"Sunny\\\"}\"}",
        }));
    chat.AddEvent(new ActivityEvent(
        "gen_ai.choice",
        tags: new ActivityTagsCollection
        {
            ["gen_ai.event.content"] = "{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"It is currently 22C and sunny in Paris.\"}, \"tool_calls\": [], \"finish_reason\": \"stop\"}",
        }));

    Console.WriteLine($"[demo] Synthetic tool-call flow trace id={chat.TraceId} (chat -> child get_weather)");
}
