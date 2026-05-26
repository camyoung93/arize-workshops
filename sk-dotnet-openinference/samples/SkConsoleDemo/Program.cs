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
    Console.WriteLine("[demo] SYNTHETIC=true: emitting a hand-crafted SK-shaped activity (no LLM call)");
    EmitSyntheticActivity();
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

    Console.WriteLine($"[demo] Synthetic activity id={activity.TraceId}");
}
