using System.Diagnostics;
using System.Text.Json;
using OpenInference.Instrumentation.SemanticKernel;

namespace OpenInference.Instrumentation.SemanticKernel.Tests;

public sealed class ProcessorTests
{
    private const string SourceName = "Microsoft.SemanticKernel.Diagnostics";

    [Fact]
    public void Translates_synthetic_chat_completion_activity_to_openinference_attributes()
    {
        using var source = new ActivitySource(SourceName);
        using var listener = new ActivityListener
        {
            ShouldListenTo = s => s.Name == SourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
        };
        ActivitySource.AddActivityListener(listener);

        using var activity = source.StartActivity("chat.completions gpt-4o-mini", ActivityKind.Client)!;
        Assert.NotNull(activity);

        activity.SetTag("gen_ai.operation.name", "chat.completions");
        activity.SetTag("gen_ai.system", "openai");
        activity.SetTag("gen_ai.request.model", "gpt-4o-mini");
        activity.SetTag("gen_ai.response.model", "gpt-4o-mini-2024-07-18");
        activity.SetTag("gen_ai.request.temperature", 0.7);
        activity.SetTag("gen_ai.request.top_p", 1.0);
        activity.SetTag("gen_ai.request.max_tokens", 256);
        activity.SetTag("gen_ai.usage.input_tokens", 23);
        activity.SetTag("gen_ai.usage.output_tokens", 12);

        activity.AddEvent(new ActivityEvent(
            "gen_ai.system.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"system\", \"content\": \"You are concise.\"}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.user.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"user\", \"content\": \"Capital of France?\"}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.choice",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"Paris.\"}, \"tool_calls\": [], \"finish_reason\": \"stop\"}",
            }));

        var processor = new OpenInferenceSpanProcessor();
        processor.OnEnd(activity);

        Assert.Equal("LLM", activity.GetTagItem("openinference.span.kind"));
        Assert.Equal("openai", activity.GetTagItem("llm.provider"));
        Assert.Equal("openai", activity.GetTagItem("llm.system"));
        Assert.Equal("gpt-4o-mini-2024-07-18", activity.GetTagItem("llm.model_name"));

        Assert.Equal(23L, ReadLong(activity, "llm.token_count.prompt"));
        Assert.Equal(12L, ReadLong(activity, "llm.token_count.completion"));
        Assert.Equal(35L, ReadLong(activity, "llm.token_count.total"));

        var invocationParamsJson = activity.GetTagItem("llm.invocation_parameters") as string;
        Assert.False(string.IsNullOrEmpty(invocationParamsJson));
        using (var doc = JsonDocument.Parse(invocationParamsJson!))
        {
            Assert.Equal(0.7, doc.RootElement.GetProperty("temperature").GetDouble());
            Assert.Equal(1.0, doc.RootElement.GetProperty("top_p").GetDouble());
            Assert.Equal(256, doc.RootElement.GetProperty("max_tokens").GetInt32());
        }

        Assert.Equal("system", activity.GetTagItem("llm.input_messages.0.message.role"));
        Assert.Equal("You are concise.", activity.GetTagItem("llm.input_messages.0.message.content"));
        Assert.Equal("user", activity.GetTagItem("llm.input_messages.1.message.role"));
        Assert.Equal("Capital of France?", activity.GetTagItem("llm.input_messages.1.message.content"));

        Assert.Equal("assistant", activity.GetTagItem("llm.output_messages.0.message.role"));
        Assert.Equal("Paris.", activity.GetTagItem("llm.output_messages.0.message.content"));

        Assert.Equal("application/json", activity.GetTagItem("input.mime_type"));
        Assert.Equal("application/json", activity.GetTagItem("output.mime_type"));

        var inputValue = activity.GetTagItem("input.value") as string;
        Assert.False(string.IsNullOrEmpty(inputValue));
        using (var doc = JsonDocument.Parse(inputValue!))
        {
            Assert.Equal(2, doc.RootElement.GetArrayLength());
            Assert.Equal("system", doc.RootElement[0].GetProperty("role").GetString());
            Assert.Equal("You are concise.", doc.RootElement[0].GetProperty("content").GetString());
        }

        var outputValue = activity.GetTagItem("output.value") as string;
        Assert.False(string.IsNullOrEmpty(outputValue));
        using (var doc = JsonDocument.Parse(outputValue!))
        {
            Assert.Equal(1, doc.RootElement.GetArrayLength());
            Assert.Equal("assistant", doc.RootElement[0].GetProperty("role").GetString());
            Assert.Equal("Paris.", doc.RootElement[0].GetProperty("content").GetString());
        }
    }

    private static long ReadLong(Activity activity, string key)
    {
        var raw = activity.GetTagItem(key);
        return raw switch
        {
            long l => l,
            int i => i,
            _ => throw new Xunit.Sdk.XunitException($"Tag {key} was {raw?.GetType().Name ?? "null"}, expected long/int"),
        };
    }
}
