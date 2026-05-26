using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using OpenTelemetry;

namespace OpenInference.Instrumentation.SemanticKernel;

/// <summary>
/// OpenTelemetry SpanProcessor that translates Microsoft.SemanticKernel 1.54 GenAI activities
/// into OpenInference attributes so they render correctly in Arize AX.
/// </summary>
/// <remarks>
/// <para>Register this processor on the TracerProvider <b>before</b> any exporter so the mutation
/// lands before the span is exported.</para>
///
/// <para>Mapping (source verified against SK 1.54 ModelDiagnostics.cs):</para>
/// <list type="bullet">
///   <item>ActivitySource <c>Microsoft.SemanticKernel.Diagnostics</c> activities only.</item>
///   <item><c>gen_ai.operation.name</c> "chat.completions" / "text.completions" -&gt; <c>openinference.span.kind = LLM</c>.</item>
///   <item><c>gen_ai.operation.name</c> "invoke_agent" -&gt; <c>openinference.span.kind = AGENT</c>.</item>
///   <item><c>gen_ai.system</c> -&gt; (<c>llm.provider</c>, <c>llm.system</c>) via ProviderMapping.</item>
///   <item><c>gen_ai.request.model</c>, overridden by <c>gen_ai.response.model</c> -&gt; <c>llm.model_name</c>.</item>
///   <item><c>gen_ai.usage.input_tokens|output_tokens</c> -&gt; <c>llm.token_count.prompt|completion</c>; total computed if absent.</item>
///   <item><c>gen_ai.request.{temperature,top_p,max_tokens,frequency_penalty,presence_penalty}</c> packed into <c>llm.invocation_parameters</c> JSON.</item>
///   <item>Per-message events <c>gen_ai.{system,user,assistant,tool}.message</c> (each carrying a
///     <c>gen_ai.event.content</c> tag with JSON body) -&gt; <c>llm.input_messages.{i}.message.{role,content}</c> plus
///     <c>input.value</c> (JSON array) and <c>input.mime_type = application/json</c>.</item>
///   <item><c>gen_ai.choice</c> events -&gt; <c>llm.output_messages.{i}.message.{role,content}</c> plus
///     <c>output.value</c> and <c>output.mime_type</c>.</item>
/// </list>
/// </remarks>
public sealed class OpenInferenceSpanProcessor : BaseProcessor<Activity>
{
    private readonly Action<string>? _logger;

    public OpenInferenceSpanProcessor(Action<string>? logger = null)
    {
        _logger = logger;
    }

    public override void OnEnd(Activity activity)
    {
        try
        {
            if (activity == null) return;
            if (!string.Equals(activity.Source?.Name, GenAiAttributes.ActivitySourceName, StringComparison.Ordinal))
            {
                return;
            }

            var operation = activity.GetTagItem(GenAiAttributes.OperationName) as string;
            var spanKind = MapSpanKind(operation);
            if (spanKind is null)
            {
                return;
            }
            activity.SetTag(OpenInferenceAttributes.SpanKind, spanKind);

            if (activity.GetTagItem(GenAiAttributes.System) is string system && !string.IsNullOrEmpty(system))
            {
                var (provider, sys) = ProviderMapping.Map(system);
                activity.SetTag(OpenInferenceAttributes.LlmProvider, provider);
                activity.SetTag(OpenInferenceAttributes.LlmSystem, sys);
            }

            var model = activity.GetTagItem(GenAiAttributes.RequestModel) as string;
            if (activity.GetTagItem(GenAiAttributes.ResponseModel) is string responseModel && !string.IsNullOrEmpty(responseModel))
            {
                model = responseModel;
            }
            if (!string.IsNullOrEmpty(model))
            {
                activity.SetTag(OpenInferenceAttributes.LlmModelName, model);
            }

            ApplyTokenCounts(activity);
            ApplyInvocationParameters(activity);
            ApplyMessageEvents(activity);
        }
        catch (Exception ex)
        {
            Log($"OpenInferenceSpanProcessor.OnEnd swallowed exception: {ex}");
        }
    }

    private static string? MapSpanKind(string? operation) => operation switch
    {
        GenAiAttributes.OperationChatCompletions => OpenInferenceAttributes.SpanKindLlm,
        GenAiAttributes.OperationTextCompletions => OpenInferenceAttributes.SpanKindLlm,
        GenAiAttributes.OperationInvokeAgent => OpenInferenceAttributes.SpanKindAgent,
        _ => null,
    };

    private static void ApplyTokenCounts(Activity activity)
    {
        long? prompt = TryReadLong(activity, GenAiAttributes.UsageInputTokens);
        long? completion = TryReadLong(activity, GenAiAttributes.UsageOutputTokens);
        long? total = TryReadLong(activity, GenAiAttributes.UsageTotalTokens);

        if (prompt.HasValue)
        {
            activity.SetTag(OpenInferenceAttributes.LlmTokenCountPrompt, prompt.Value);
        }
        if (completion.HasValue)
        {
            activity.SetTag(OpenInferenceAttributes.LlmTokenCountCompletion, completion.Value);
        }
        if (total.HasValue)
        {
            activity.SetTag(OpenInferenceAttributes.LlmTokenCountTotal, total.Value);
        }
        else if (prompt.HasValue && completion.HasValue)
        {
            activity.SetTag(OpenInferenceAttributes.LlmTokenCountTotal, prompt.Value + completion.Value);
        }
    }

    private static long? TryReadLong(Activity activity, string key)
    {
        var raw = activity.GetTagItem(key);
        if (raw is null) return null;
        return raw switch
        {
            long l => l,
            int i => i,
            short s => s,
            double d => (long)d,
            float f => (long)f,
            string str when long.TryParse(str, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) => parsed,
            _ => null,
        };
    }

    private static void ApplyInvocationParameters(Activity activity)
    {
        var keys = new (string Source, string Target)[]
        {
            (GenAiAttributes.RequestTemperature, "temperature"),
            (GenAiAttributes.RequestTopP, "top_p"),
            (GenAiAttributes.RequestMaxTokens, "max_tokens"),
            (GenAiAttributes.RequestFrequencyPenalty, "frequency_penalty"),
            (GenAiAttributes.RequestPresencePenalty, "presence_penalty"),
        };

        var dict = new Dictionary<string, object?>();
        foreach (var (sourceKey, targetKey) in keys)
        {
            var value = activity.GetTagItem(sourceKey);
            if (value is not null)
            {
                dict[targetKey] = NormalizeJsonValue(value);
            }
        }
        if (dict.Count == 0) return;

        var json = JsonSerializer.Serialize(dict);
        activity.SetTag(OpenInferenceAttributes.LlmInvocationParameters, json);
    }

    private static object? NormalizeJsonValue(object value) => value switch
    {
        string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
        string s when long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var l) => l,
        _ => value,
    };

    private static void ApplyMessageEvents(Activity activity)
    {
        var inputs = new List<MessagePayload>();
        var outputs = new List<MessagePayload>();

        foreach (var ev in activity.Events)
        {
            string? roleFromName = ev.Name switch
            {
                GenAiAttributes.EventSystemMessage => "system",
                GenAiAttributes.EventUserMessage => "user",
                GenAiAttributes.EventAssistantMessage => "assistant",
                GenAiAttributes.EventToolMessage => "tool",
                _ => null,
            };
            bool isChoice = ev.Name == GenAiAttributes.EventChoice;
            if (roleFromName is null && !isChoice) continue;

            string? body = null;
            foreach (var tag in ev.Tags)
            {
                if (tag.Key == GenAiAttributes.EventContent)
                {
                    body = tag.Value?.ToString();
                    break;
                }
            }
            if (string.IsNullOrEmpty(body)) continue;

            if (isChoice)
            {
                var parsed = TryParseChoice(body!);
                if (parsed is not null)
                {
                    outputs.Add(parsed);
                }
            }
            else
            {
                var parsed = TryParseMessage(body!, roleFromName!);
                if (parsed is not null)
                {
                    inputs.Add(parsed);
                }
            }
        }

        WriteMessages(activity, inputs,
            OpenInferenceAttributes.LlmInputMessagesPrefix,
            OpenInferenceAttributes.InputValue,
            OpenInferenceAttributes.InputMimeType);

        WriteMessages(activity, outputs,
            OpenInferenceAttributes.LlmOutputMessagesPrefix,
            OpenInferenceAttributes.OutputValue,
            OpenInferenceAttributes.OutputMimeType);
    }

    private static MessagePayload? TryParseMessage(string json, string fallbackRole)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            string role = fallbackRole;
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String)
            {
                role = r.GetString() ?? fallbackRole;
            }
            string content = ExtractContent(root);
            return new MessagePayload(role, content);
        }
        catch
        {
            return new MessagePayload(fallbackRole, json);
        }
    }

    private static MessagePayload? TryParseChoice(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return new MessagePayload("assistant", json);

            JsonElement msg = root.TryGetProperty("message", out var m) ? m : root;
            string role = "assistant";
            if (msg.ValueKind == JsonValueKind.Object && msg.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String)
            {
                role = r.GetString() ?? role;
            }
            string content = ExtractContent(msg);
            return new MessagePayload(role, content);
        }
        catch
        {
            return new MessagePayload("assistant", json);
        }
    }

    private static string ExtractContent(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return element.ToString();
        if (element.TryGetProperty("content", out var c))
        {
            return c.ValueKind switch
            {
                JsonValueKind.String => c.GetString() ?? string.Empty,
                JsonValueKind.Null => string.Empty,
                _ => c.GetRawText(),
            };
        }
        return string.Empty;
    }

    private static void WriteMessages(
        Activity activity,
        List<MessagePayload> messages,
        string prefix,
        string valueKey,
        string mimeKey)
    {
        if (messages.Count == 0) return;

        for (int i = 0; i < messages.Count; i++)
        {
            activity.SetTag($"{prefix}.{i}.{OpenInferenceAttributes.MessageRoleSuffix}", messages[i].Role);
            activity.SetTag($"{prefix}.{i}.{OpenInferenceAttributes.MessageContentSuffix}", messages[i].Content);
        }

        var serialized = JsonSerializer.Serialize(messages, MessagePayloadJsonContext.Options);
        activity.SetTag(valueKey, serialized);
        activity.SetTag(mimeKey, OpenInferenceAttributes.MimeTypeJson);
    }

    private void Log(string message)
    {
        if (_logger is not null)
        {
            _logger(message);
        }
        else
        {
            Console.Error.WriteLine(message);
        }
    }

    private sealed record MessagePayload(string Role, string Content);

    private static class MessagePayloadJsonContext
    {
        public static readonly JsonSerializerOptions Options = new()
        {
            WriteIndented = false,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
    }
}
