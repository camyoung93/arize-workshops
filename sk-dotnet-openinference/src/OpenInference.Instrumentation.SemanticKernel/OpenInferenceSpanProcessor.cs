using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenTelemetry;

namespace OpenInference.Instrumentation.SemanticKernel;

/// <summary>
/// OpenTelemetry SpanProcessor that translates Microsoft.SemanticKernel 1.54 activities
/// into OpenInference attributes so they render correctly in Arize AX.
/// </summary>
/// <remarks>
/// <para>Register this processor on the TracerProvider <b>before</b> any exporter so the mutation
/// lands before the span is exported.</para>
///
/// <para>Two ActivitySources are handled (verified against SK 1.54 source):</para>
/// <list type="bullet">
///   <item><c>Microsoft.SemanticKernel.Diagnostics</c> - LLM / agent model spans from
///     <c>ModelDiagnostics</c>. Mapped to <c>openinference.span.kind = LLM</c> or <c>AGENT</c>,
///     plus provider/system/model/tokens/invocation parameters, plus per-message events
///     (including tool calls and tool result correlation) into
///     <c>llm.input_messages.*</c> / <c>llm.output_messages.*</c> and
///     <c>input.value</c> / <c>output.value</c>.</item>
///   <item><c>Microsoft.SemanticKernel</c> - kernel function invocation spans from
///     <c>KernelFunction.InvokeAsync</c>. Mapped to <c>openinference.span.kind = TOOL</c>
///     with <c>tool.name</c> from the activity display name. SK 1.54 does not attach
///     arguments or return value to these activities; see the README for the
///     <c>IFunctionInvocationFilter</c> enrichment recipe.</item>
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
            var source = activity.Source?.Name;
            if (source == GenAiAttributes.ActivitySourceName)
            {
                ApplyGenAi(activity);
            }
            else if (source == GenAiAttributes.KernelFunctionActivitySourceName)
            {
                ApplyKernelFunction(activity);
            }
        }
        catch (Exception ex)
        {
            Log($"OpenInferenceSpanProcessor.OnEnd swallowed exception: {ex}");
        }
    }

    private static void ApplyGenAi(Activity activity)
    {
        var operation = activity.GetTagItem(GenAiAttributes.OperationName) as string;
        var spanKind = MapSpanKind(operation);
        if (spanKind is null) return;
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

    private static void ApplyKernelFunction(Activity activity)
    {
        activity.SetTag(OpenInferenceAttributes.SpanKind, OpenInferenceAttributes.SpanKindTool);
        if (!string.IsNullOrEmpty(activity.DisplayName))
        {
            activity.SetTag(OpenInferenceAttributes.ToolName, activity.DisplayName);
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
        var inputs = new List<MessageDto>();
        var outputs = new List<MessageDto>();

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
                if (parsed is not null) outputs.Add(parsed);
            }
            else
            {
                var parsed = TryParseMessage(body!, roleFromName!);
                if (parsed is not null) inputs.Add(parsed);
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

    private static MessageDto TryParseMessage(string json, string fallbackRole)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            return BuildMessageDto(root, fallbackRole);
        }
        catch
        {
            return new MessageDto { Role = fallbackRole, Content = json };
        }
    }

    private static MessageDto TryParseChoice(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return new MessageDto { Role = "assistant", Content = json };
            }

            JsonElement msg = root.TryGetProperty("message", out var m) ? m : root;
            return BuildMessageDto(msg, "assistant");
        }
        catch
        {
            return new MessageDto { Role = "assistant", Content = json };
        }
    }

    private static MessageDto BuildMessageDto(JsonElement element, string fallbackRole)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            return new MessageDto { Role = fallbackRole, Content = element.ToString() };
        }

        string role = fallbackRole;
        if (element.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String)
        {
            role = r.GetString() ?? fallbackRole;
        }

        string? content = null;
        if (element.TryGetProperty("content", out var c))
        {
            content = c.ValueKind switch
            {
                JsonValueKind.String => c.GetString(),
                JsonValueKind.Null => null,
                _ => c.GetRawText(),
            };
        }

        string? toolCallId = null;
        if (element.TryGetProperty("tool_call_id", out var tci) && tci.ValueKind == JsonValueKind.String)
        {
            toolCallId = tci.GetString();
        }

        List<ToolCallDto>? toolCalls = null;
        if (element.TryGetProperty("tool_calls", out var tcs) && tcs.ValueKind == JsonValueKind.Array)
        {
            foreach (var tc in tcs.EnumerateArray())
            {
                var parsed = ParseToolCall(tc);
                if (parsed is null) continue;
                toolCalls ??= new List<ToolCallDto>();
                toolCalls.Add(parsed);
            }
        }

        return new MessageDto
        {
            Role = role,
            Content = content,
            ToolCallId = toolCallId,
            ToolCalls = toolCalls,
        };
    }

    private static ToolCallDto? ParseToolCall(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object) return null;

        string? id = element.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
            ? idEl.GetString()
            : null;

        string? name = null;
        string arguments = "{}";
        if (element.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object)
        {
            if (fn.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String)
            {
                name = n.GetString();
            }
            if (fn.TryGetProperty("arguments", out var args))
            {
                arguments = args.ValueKind switch
                {
                    JsonValueKind.String => args.GetString() ?? "{}",
                    JsonValueKind.Null => "{}",
                    _ => args.GetRawText(),
                };
            }
        }

        if (string.IsNullOrEmpty(id) && string.IsNullOrEmpty(name)) return null;

        return new ToolCallDto
        {
            Id = id ?? string.Empty,
            Function = new ToolFunctionDto { Name = name ?? string.Empty, Arguments = arguments },
        };
    }

    private static void WriteMessages(
        Activity activity,
        List<MessageDto> messages,
        string prefix,
        string valueKey,
        string mimeKey)
    {
        if (messages.Count == 0) return;

        for (int i = 0; i < messages.Count; i++)
        {
            var m = messages[i];
            activity.SetTag($"{prefix}.{i}.{OpenInferenceAttributes.MessageRoleSuffix}", m.Role);

            if (!string.IsNullOrEmpty(m.Content))
            {
                activity.SetTag($"{prefix}.{i}.{OpenInferenceAttributes.MessageContentSuffix}", m.Content);
            }

            if (!string.IsNullOrEmpty(m.ToolCallId))
            {
                activity.SetTag($"{prefix}.{i}.{OpenInferenceAttributes.MessageToolCallIdSuffix}", m.ToolCallId);
            }

            if (m.ToolCalls is not null)
            {
                for (int j = 0; j < m.ToolCalls.Count; j++)
                {
                    var tc = m.ToolCalls[j];
                    var callPrefix = $"{prefix}.{i}.{OpenInferenceAttributes.MessageToolCallsSuffix}.{j}";
                    if (!string.IsNullOrEmpty(tc.Id))
                    {
                        activity.SetTag($"{callPrefix}.{OpenInferenceAttributes.ToolCallIdSuffix}", tc.Id);
                    }
                    if (!string.IsNullOrEmpty(tc.Function.Name))
                    {
                        activity.SetTag($"{callPrefix}.{OpenInferenceAttributes.ToolCallFunctionNameSuffix}", tc.Function.Name);
                    }
                    activity.SetTag($"{callPrefix}.{OpenInferenceAttributes.ToolCallFunctionArgsSuffix}", tc.Function.Arguments);
                }
            }
        }

        var serialized = JsonSerializer.Serialize(messages, MessageJsonContext.Options);
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

    private sealed class MessageDto
    {
        [JsonPropertyName("role"), JsonPropertyOrder(0)]
        public string Role { get; init; } = string.Empty;

        [JsonPropertyName("content"), JsonPropertyOrder(1)]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? Content { get; init; }

        [JsonPropertyName("tool_calls"), JsonPropertyOrder(2)]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<ToolCallDto>? ToolCalls { get; init; }

        [JsonPropertyName("tool_call_id"), JsonPropertyOrder(3)]
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public string? ToolCallId { get; init; }
    }

    private sealed class ToolCallDto
    {
        [JsonPropertyName("id"), JsonPropertyOrder(0)]
        public string Id { get; init; } = string.Empty;

        [JsonPropertyName("type"), JsonPropertyOrder(1)]
        public string Type => "function";

        [JsonPropertyName("function"), JsonPropertyOrder(2)]
        public ToolFunctionDto Function { get; init; } = new();
    }

    private sealed class ToolFunctionDto
    {
        [JsonPropertyName("name"), JsonPropertyOrder(0)]
        public string Name { get; init; } = string.Empty;

        [JsonPropertyName("arguments"), JsonPropertyOrder(1)]
        public string Arguments { get; init; } = "{}";
    }

    private static class MessageJsonContext
    {
        public static readonly JsonSerializerOptions Options = new()
        {
            WriteIndented = false,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
    }
}
