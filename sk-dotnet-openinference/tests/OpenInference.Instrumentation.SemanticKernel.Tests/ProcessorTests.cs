using System.Diagnostics;
using System.Text.Json;
using OpenInference.Instrumentation.SemanticKernel;

namespace OpenInference.Instrumentation.SemanticKernel.Tests;

public sealed class ProcessorTests
{
    private const string SourceName = "Microsoft.SemanticKernel.Diagnostics";
    private const string KernelFunctionSourceName = "Microsoft.SemanticKernel";

    private static ActivityListener RegisterListener(params string[] sourceNames)
    {
        var listener = new ActivityListener
        {
            ShouldListenTo = s => System.Array.IndexOf(sourceNames, s.Name) >= 0,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
        };
        ActivitySource.AddActivityListener(listener);
        return listener;
    }

    [Fact]
    public void Translates_synthetic_chat_completion_activity_to_openinference_attributes()
    {
        using var source = new ActivitySource(SourceName);
        using var listener = RegisterListener(SourceName);

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

    [Fact]
    public void Translates_synthetic_chat_with_tool_call_round_trip()
    {
        using var source = new ActivitySource(SourceName);
        using var listener = RegisterListener(SourceName);

        using var activity = source.StartActivity("chat.completions gpt-4o-mini", ActivityKind.Client)!;
        Assert.NotNull(activity);

        activity.SetTag("gen_ai.operation.name", "chat.completions");
        activity.SetTag("gen_ai.system", "openai");
        activity.SetTag("gen_ai.request.model", "gpt-4o-mini");
        activity.SetTag("gen_ai.response.model", "gpt-4o-mini-2024-07-18");
        activity.SetTag("gen_ai.usage.input_tokens", 50);
        activity.SetTag("gen_ai.usage.output_tokens", 18);

        activity.AddEvent(new ActivityEvent(
            "gen_ai.system.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"system\", \"content\": \"You can call tools.\"}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.user.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"user\", \"content\": \"Weather in Paris?\"}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.assistant.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"call_abc\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}}]}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.tool.message",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"role\": \"tool\", \"tool_call_id\": \"call_abc\", \"content\": \"Sunny, 22C\"}",
            }));
        activity.AddEvent(new ActivityEvent(
            "gen_ai.choice",
            tags: new ActivityTagsCollection
            {
                ["gen_ai.event.content"] = "{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"It is sunny and 22C in Paris.\"}, \"tool_calls\": [], \"finish_reason\": \"stop\"}",
            }));

        var processor = new OpenInferenceSpanProcessor();
        processor.OnEnd(activity);

        Assert.Equal("LLM", activity.GetTagItem("openinference.span.kind"));

        Assert.Equal("system", activity.GetTagItem("llm.input_messages.0.message.role"));
        Assert.Equal("user", activity.GetTagItem("llm.input_messages.1.message.role"));
        Assert.Equal("assistant", activity.GetTagItem("llm.input_messages.2.message.role"));
        Assert.Null(activity.GetTagItem("llm.input_messages.2.message.content"));
        Assert.Equal("call_abc", activity.GetTagItem("llm.input_messages.2.message.tool_calls.0.tool_call.id"));
        Assert.Equal("get_weather", activity.GetTagItem("llm.input_messages.2.message.tool_calls.0.tool_call.function.name"));
        var args = activity.GetTagItem("llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments") as string;
        Assert.False(string.IsNullOrEmpty(args));
        using (var doc = JsonDocument.Parse(args!))
        {
            Assert.Equal("Paris", doc.RootElement.GetProperty("city").GetString());
        }

        Assert.Equal("tool", activity.GetTagItem("llm.input_messages.3.message.role"));
        Assert.Equal("call_abc", activity.GetTagItem("llm.input_messages.3.message.tool_call_id"));
        Assert.Equal("Sunny, 22C", activity.GetTagItem("llm.input_messages.3.message.content"));

        Assert.Equal("assistant", activity.GetTagItem("llm.output_messages.0.message.role"));
        Assert.Equal("It is sunny and 22C in Paris.", activity.GetTagItem("llm.output_messages.0.message.content"));

        var inputValue = activity.GetTagItem("input.value") as string;
        Assert.False(string.IsNullOrEmpty(inputValue));
        using (var doc = JsonDocument.Parse(inputValue!))
        {
            Assert.Equal(4, doc.RootElement.GetArrayLength());

            var assistantMsg = doc.RootElement[2];
            Assert.Equal("assistant", assistantMsg.GetProperty("role").GetString());
            Assert.False(assistantMsg.TryGetProperty("content", out _), "content should be omitted when source is null");
            var toolCalls = assistantMsg.GetProperty("tool_calls");
            Assert.Equal(1, toolCalls.GetArrayLength());
            Assert.Equal("call_abc", toolCalls[0].GetProperty("id").GetString());
            Assert.Equal("function", toolCalls[0].GetProperty("type").GetString());
            Assert.Equal("get_weather", toolCalls[0].GetProperty("function").GetProperty("name").GetString());

            var toolMsg = doc.RootElement[3];
            Assert.Equal("tool", toolMsg.GetProperty("role").GetString());
            Assert.Equal("call_abc", toolMsg.GetProperty("tool_call_id").GetString());
            Assert.Equal("Sunny, 22C", toolMsg.GetProperty("content").GetString());
        }
    }

    [Fact]
    public void Translates_kernel_function_activity_to_tool_span_kind()
    {
        using var source = new ActivitySource(KernelFunctionSourceName);
        using var listener = RegisterListener(KernelFunctionSourceName);

        using var activity = source.StartActivity("get_weather", ActivityKind.Internal)!;
        Assert.NotNull(activity);

        var processor = new OpenInferenceSpanProcessor();
        processor.OnEnd(activity);

        Assert.Equal("TOOL", activity.GetTagItem("openinference.span.kind"));
        Assert.Equal("get_weather", activity.GetTagItem("tool.name"));
    }

    [Fact]
    public void Ignores_unrelated_activity_source()
    {
        const string unrelated = "Some.Other.Source";
        using var source = new ActivitySource(unrelated);
        using var listener = RegisterListener(unrelated);

        using var activity = source.StartActivity("noop")!;
        Assert.NotNull(activity);
        activity.SetTag("preexisting", "value");

        var processor = new OpenInferenceSpanProcessor();
        processor.OnEnd(activity);

        Assert.Null(activity.GetTagItem("openinference.span.kind"));
        Assert.Equal("value", activity.GetTagItem("preexisting"));
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
