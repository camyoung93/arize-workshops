namespace OpenInference.Instrumentation.SemanticKernel;

/// <summary>
/// Source-side attribute keys emitted by Microsoft.SemanticKernel 1.54's
/// ModelDiagnostics (OTel GenAI semantic convention).
/// Verified against the SK source: dotnet-1.54.0/dotnet/src/InternalUtilities/src/Diagnostics/ModelDiagnostics.cs
/// </summary>
internal static class GenAiAttributes
{
    public const string ActivitySourceName = "Microsoft.SemanticKernel.Diagnostics";
    public const string KernelFunctionActivitySourceName = "Microsoft.SemanticKernel";

    public const string System = "gen_ai.system";
    public const string OperationName = "gen_ai.operation.name";

    public const string RequestModel = "gen_ai.request.model";
    public const string RequestMaxTokens = "gen_ai.request.max_tokens";
    public const string RequestTemperature = "gen_ai.request.temperature";
    public const string RequestTopP = "gen_ai.request.top_p";
    public const string RequestFrequencyPenalty = "gen_ai.request.frequency_penalty";
    public const string RequestPresencePenalty = "gen_ai.request.presence_penalty";

    public const string ResponseId = "gen_ai.response.id";
    public const string ResponseModel = "gen_ai.response.model";
    public const string ResponseFinishReason = "gen_ai.response.finish_reason";

    public const string UsageInputTokens = "gen_ai.usage.input_tokens";
    public const string UsageOutputTokens = "gen_ai.usage.output_tokens";
    public const string UsageTotalTokens = "gen_ai.usage.total_tokens";

    public const string AgentId = "gen_ai.agent.id";
    public const string AgentName = "gen_ai.agent.name";
    public const string AgentDescription = "gen_ai.agent.description";

    public const string EventContent = "gen_ai.event.content";

    public const string EventSystemMessage = "gen_ai.system.message";
    public const string EventUserMessage = "gen_ai.user.message";
    public const string EventAssistantMessage = "gen_ai.assistant.message";
    public const string EventToolMessage = "gen_ai.tool.message";
    public const string EventChoice = "gen_ai.choice";

    public const string OperationChatCompletions = "chat.completions";
    public const string OperationTextCompletions = "text.completions";
    public const string OperationInvokeAgent = "invoke_agent";

    public const string SystemOpenAi = "openai";
    public const string SystemAzureOpenAi = "az.ai.openai";
}
