namespace OpenInference.Instrumentation.SemanticKernel;

/// <summary>
/// Target-side OpenInference attribute keys. Mirrors the Python semconv enum
/// (openinference.semconv.trace.SpanAttributes / MessageAttributes).
/// </summary>
internal static class OpenInferenceAttributes
{
    public const string SpanKind = "openinference.span.kind";

    public const string LlmProvider = "llm.provider";
    public const string LlmSystem = "llm.system";
    public const string LlmModelName = "llm.model_name";
    public const string LlmInvocationParameters = "llm.invocation_parameters";

    public const string LlmTokenCountPrompt = "llm.token_count.prompt";
    public const string LlmTokenCountCompletion = "llm.token_count.completion";
    public const string LlmTokenCountTotal = "llm.token_count.total";

    public const string LlmInputMessagesPrefix = "llm.input_messages";
    public const string LlmOutputMessagesPrefix = "llm.output_messages";

    public const string MessageRoleSuffix = "message.role";
    public const string MessageContentSuffix = "message.content";

    public const string InputValue = "input.value";
    public const string InputMimeType = "input.mime_type";
    public const string OutputValue = "output.value";
    public const string OutputMimeType = "output.mime_type";

    public const string MimeTypeJson = "application/json";

    public const string SpanKindLlm = "LLM";
    public const string SpanKindAgent = "AGENT";
}
