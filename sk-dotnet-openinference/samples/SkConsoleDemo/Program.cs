using System;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
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
    Console.Error.WriteLine("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT, or OPENAI_API_KEY.");
    return 1;
}

kernelBuilder.Plugins.AddFromType<WeatherPlugin>();

var kernel = kernelBuilder.Build();

// Use ChatCompletionAgent rather than IChatCompletionService directly: it wraps the
// whole conversation (including the auto-invoke loop for tool calls) in an
// invoke_agent activity, which the processor maps to openinference.span.kind = AGENT.
// That gives a proper parent-child trace tree in AX (agent -> chat -> execute_tool)
// instead of three orphan root spans.
var agent = new ChatCompletionAgent
{
    Name = "WeatherConcierge",
    Description = "Answers questions about current weather using the get_weather tool.",
    Instructions = "You are a concise assistant with access to a get_weather tool. Use it when asked about weather, then reply in one sentence.",
    Kernel = kernel,
    Arguments = new KernelArguments(new PromptExecutionSettings
    {
        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(),
    }),
};

var history = new ChatHistory();
history.AddUserMessage("What's the weather in Paris right now?");

// The (ChatHistory, KernelArguments, Kernel) overload is [Obsolete] but is the
// path that emits the invoke_agent activity in SK 1.54. Once the customer is on
// a newer SK that wires activity emission into the AgentThread-based overload,
// this should be swapped over.
ChatMessageContent? reply = null;
await foreach (var message in agent.InvokeAsync(history))
{
    reply = message;
}
Console.WriteLine();
Console.WriteLine($"[demo] Reply: {reply?.Content}");

tracerProvider.ForceFlush();
return 0;

internal sealed class WeatherPlugin
{
    [KernelFunction, Description("Get current weather for a city")]
    public string GetWeather(
        [Description("City name, e.g. 'Paris'")] string city)
        => $"Sunny, 22C in {city}";
}
