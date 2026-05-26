namespace OpenInference.Instrumentation.SemanticKernel;

internal static class ProviderMapping
{
    /// <summary>
    /// Map an OTel GenAI <c>gen_ai.system</c> value to OpenInference (provider, system) pair.
    /// SK 1.54 ships "openai" for both the OpenAI and Azure OpenAI connectors today;
    /// the "az.ai.openai" branch is here so the processor is correct if SK switches.
    /// </summary>
    public static (string Provider, string System) Map(string genAiSystem)
    {
        return genAiSystem switch
        {
            GenAiAttributes.SystemOpenAi => ("openai", "openai"),
            GenAiAttributes.SystemAzureOpenAi => ("azure", "openai"),
            _ => (genAiSystem, genAiSystem),
        };
    }
}
