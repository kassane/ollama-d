/**
 * MIT License
 *
 * Copyright (c) 2025 Matheus C. França
 *
 * Permission is granted to use, modify, and distribute this software
 * under the terms of the MIT License.
 */

/++
 + Module providing a D language binding for the Ollama REST API.
 +
 + This module defines the `OllamaClient` class, which facilitates interaction with an Ollama server
 + for tasks such as text generation, chat interactions, model management, embeddings, and tool calling.
 + It supports both native Ollama endpoints and OpenAI-compatible endpoints, using `etc.c.curl`
 + (libcurl C API) for HTTP requests and `std.json` for JSON processing.
 +
 + Examples:
 +     ---
 +     import ollama.client;
 +     import std.stdio;
 +
 +     void main() {
 +         auto client = new OllamaClient();
 +         auto response = client.generate("llama3", "What is the weather like?");
 +         writeln(response["response"].str);
 +     }
 +     ---
 +
 + See_Also:
 +     - $(LINK2 https://github.com/ollama/ollama/blob/main/docs/api.md, Ollama API Documentation)
 +     - $(LINK2 https://github.com/ollama/ollama/blob/main/docs/openai.md, OpenAI Compatibility)
 +/
module ollama.client;

import std.json      : JSONValue, JSONType, parseJSON;
import std.exception : enforce;
import std.string    : toStringz, fromStringz;
import std.datetime  : Duration, seconds;
import std.math      : isNaN;
import etc.c.curl;

@safe:

// ---------------------------------------------------------------------------
// Helper: build a JSONValue object from a JSONValue[string] AA.
// Using JSONValue(JSONValue[string]) constructor is @safe; the .object
// property setter is @system and must be avoided in @safe code.
// ---------------------------------------------------------------------------
private JSONValue makeObject(JSONValue[string] fields) @safe
{
    return JSONValue(fields);
}

/++
 + Typed options for controlling model generation behavior.
 +
 + Only non-default (explicitly set) fields are serialized to JSON.
 + Float fields use `float.nan` as the "unset" sentinel; integer fields use `0`.
 +
 + Examples:
 +     ---
 +     OllamaOptions opts;
 +     opts.temperature = 0.8f;
 +     opts.num_ctx     = 4096;
 +     opts.stop        = ["<|end|>", "\n\n"];
 +     ---
 +/
struct OllamaOptions
{
    float temperature    = float.nan; /// Sampling temperature (0 = deterministic).
    int   top_k          = 0;         /// Top-K sampling; 0 = disabled.
    float top_p          = float.nan; /// Nucleus sampling threshold.
    float min_p          = float.nan; /// Minimum probability threshold.
    float repeat_penalty = float.nan; /// Penalty for repeated tokens.
    int   repeat_last_n  = 0;         /// Tokens considered for repeat penalty.
    int   seed           = 0;         /// Random seed; 0 = random.
    int   num_predict    = 0;         /// Max tokens to generate; 0 = unlimited.
    int   num_ctx        = 0;         /// Context window size; 0 = model default.
    string[] stop;                    /// Stop sequences; empty = disabled.
    int   mirostat       = 0;         /// Mirostat strategy (0=off,1=v1,2=v2).
    float mirostat_tau   = float.nan; /// Mirostat target entropy.
    float mirostat_eta   = float.nan; /// Mirostat learning rate.

    /++
     + Serializes only non-default fields to a `JSONValue` object.
     +
     + Builds a `JSONValue[string]` AA (safe D operation) then wraps it.
     +
     + Returns: A `JSONValue` object containing only the explicitly-set fields.
     +/
    JSONValue toJson() const @safe
    {
        import std.math : isNaN;
        JSONValue[string] fields;
        if (!isNaN(temperature))    fields["temperature"]    = JSONValue(temperature);
        if (top_k > 0)              fields["top_k"]          = JSONValue(top_k);
        if (!isNaN(top_p))          fields["top_p"]          = JSONValue(top_p);
        if (!isNaN(min_p))          fields["min_p"]          = JSONValue(min_p);
        if (!isNaN(repeat_penalty)) fields["repeat_penalty"] = JSONValue(repeat_penalty);
        if (repeat_last_n > 0)      fields["repeat_last_n"]  = JSONValue(repeat_last_n);
        if (seed > 0)               fields["seed"]           = JSONValue(seed);
        if (num_predict > 0)        fields["num_predict"]    = JSONValue(num_predict);
        if (num_ctx > 0)            fields["num_ctx"]        = JSONValue(num_ctx);
        if (mirostat > 0)           fields["mirostat"]       = JSONValue(mirostat);
        if (!isNaN(mirostat_tau))   fields["mirostat_tau"]   = JSONValue(mirostat_tau);
        if (!isNaN(mirostat_eta))   fields["mirostat_eta"]   = JSONValue(mirostat_eta);
        if (stop.length > 0)
        {
            JSONValue[] arr;
            foreach (s; stop) arr ~= JSONValue(s);
            fields["stop"] = JSONValue(arr);
        }
        return makeObject(fields);
    }
}

///
unittest
{
    // Default options serialize to an empty JSON object
    OllamaOptions def;
    auto j0 = def.toJson();
    assert(j0.type == JSONType.object);
    assert(j0.objectNoRef.length == 0, "Default OllamaOptions should be empty");

    // Only set fields appear in output
    OllamaOptions opts;
    opts.temperature = 0.5f;  // 0.5 is exactly representable in float and double
    opts.top_k       = 40;
    opts.num_ctx     = 4096;
    opts.stop        = ["<|end|>"];
    auto j = opts.toJson();
    assert(j["temperature"].type == JSONType.float_);
    assert(j["temperature"].floating == 0.5);  // exact double comparison
    assert(j["top_k"].integer == 40);
    assert(j["num_ctx"].integer == 4096);
    assert(j["stop"].arrayNoRef[0].str == "<|end|>");
    assert("top_p"          !in j);
    assert("min_p"          !in j);
    assert("repeat_penalty" !in j);
    assert("mirostat"       !in j);

    // temperature = 0.0 is a valid explicit value and must be included
    OllamaOptions zeroTemp;
    zeroTemp.temperature = 0.0f;
    auto jz = zeroTemp.toJson();
    assert("temperature" in jz);
    assert(jz["temperature"].floating == 0.0);
}

/++
 + Function schema for tool/function calling definitions.
 +
 + Used inside `Tool` when registering callable tools with the model.
 +/
struct ToolFunction
{
    string    name;        /// Function name as called by the model.
    string    description; /// Human-readable description.
    JSONValue parameters;  /// JSON Schema object defining the function's parameters.

    /++
     + Converts to a JSON object for the Ollama API `tools` array.
     +
     + Returns: A `JSONValue` with "name", "description", and optionally "parameters".
     +/
    JSONValue toJson() const @safe
    {
        JSONValue[string] fields = [
            "name":        JSONValue(name),
            "description": JSONValue(description),
        ];
        if (parameters.type != JSONType.null_)
            fields["parameters"] = parameters;
        return makeObject(fields);
    }
}

/++
 + A tool (function) definition passed to `chat()` to enable tool/function calling.
 +
 + Examples:
 +     ---
 +     auto schema = parseJSON(`{
 +         "type": "object",
 +         "properties": {"location": {"type": "string"}},
 +         "required": ["location"]
 +     }`);
 +     auto tool = Tool("function", ToolFunction("get_weather", "Get current weather", schema));
 +     auto resp = client.chat("llama3", messages, JSONValue.init, false, [tool]);
 +     ---
 +/
struct Tool
{
    string       type = "function"; /// Tool type; currently always "function".
    ToolFunction function_;         /// The function definition.

    /++
     + Converts to a JSON object for the Ollama API `tools` array.
     +
     + Returns: A `JSONValue` with "type" and "function" fields.
     +/
    JSONValue toJson() const @safe
    {
        JSONValue[string] fields = [
            "type":     JSONValue(type),
            "function": function_.toJson(),
        ];
        return makeObject(fields);
    }
}

/++
 + Represents a tool/function call made by the model in a chat response.
 +
 + Access via `response["message"]["tool_calls"]` when the model calls a tool.
 +/
struct ToolCall
{
    string    id;        /// Optional tool call identifier.
    string    name;      /// Name of the function called.
    JSONValue arguments; /// Arguments passed to the function (JSON object).

    /++
     + Converts to a JSON object matching the Ollama API tool call format.
     +
     + Returns: A `JSONValue` with "function" containing "name" and "arguments".
     +/
    JSONValue toJson() const @safe
    {
        JSONValue[string] funcFields = ["name": JSONValue(name)];
        if (arguments.type != JSONType.null_)
            funcFields["arguments"] = arguments;

        JSONValue[string] fields = ["function": makeObject(funcFields)];
        if (id.length > 0)
            fields["id"] = JSONValue(id);
        return makeObject(fields);
    }
}

///
unittest
{
    // ToolFunction serialization
    auto tf = ToolFunction("get_weather", "Fetch weather data",
        parseJSON(`{"type":"object","properties":{"city":{"type":"string"}}}`));
    auto jtf = tf.toJson();
    assert(jtf["name"].str == "get_weather");
    assert(jtf["description"].str == "Fetch weather data");
    assert(jtf["parameters"]["type"].str == "object");

    // Tool serialization — JSON key must be "function" (not "function_")
    auto tool = Tool("function", tf);
    auto jt = tool.toJson();
    assert(jt["type"].str == "function");
    assert("function" in jt);
    assert(jt["function"]["name"].str == "get_weather");

    // ToolCall with id and arguments
    auto tc = ToolCall("call-1", "get_weather", parseJSON(`{"city":"Paris"}`));
    auto jtc = tc.toJson();
    assert(jtc["id"].str == "call-1");
    assert(jtc["function"]["name"].str == "get_weather");
    assert(jtc["function"]["arguments"]["city"].str == "Paris");

    // ToolCall without id — id key must be absent
    auto tc2 = ToolCall("", "sum", parseJSON(`{"a":1,"b":2}`));
    auto jtc2 = tc2.toJson();
    assert("id" !in jtc2);
    assert(jtc2["function"]["name"].str == "sum");
}

/++
 + Represents a single message in a chat interaction.
 +
 + Supports text, base64-encoded images (multimodal), and tool call results.
 + Backward compatible: `Message("user", "hello")` still compiles.
 +/
struct Message
{
    string     role;       /// Sender role: "user", "assistant", or "system".
    string     content;    /// Text content of the message.
    string[]   images;     /// Optional base64-encoded images for multimodal input.
    ToolCall[] tool_calls; /// Optional tool calls made by the assistant.

    /++
     + Converts the message to a JSON object for the Ollama API.
     +
     + Returns: A `JSONValue` with "role", "content", and optionally "images"
     + and "tool_calls".
     +/
    JSONValue toJson() const @safe
    {
        JSONValue[string] fields = [
            "role":    JSONValue(role),
            "content": JSONValue(content),
        ];
        if (images.length > 0)
        {
            JSONValue[] arr;
            foreach (img; images) arr ~= JSONValue(img);
            fields["images"] = JSONValue(arr);
        }
        if (tool_calls.length > 0)
        {
            JSONValue[] arr;
            foreach (tc; tool_calls) arr ~= tc.toJson();
            fields["tool_calls"] = JSONValue(arr);
        }
        return makeObject(fields);
    }
}

///
unittest
{
    // Basic message — no optional fields
    auto m = Message("user", "Hello, world!");
    auto j = m.toJson();
    assert(j["role"].str == "user");
    assert(j["content"].str == "Hello, world!");
    assert("images"     !in j);
    assert("tool_calls" !in j);

    // Message with images
    auto m2 = Message("user", "What is in this image?", ["aGVsbG8="]);
    auto j2 = m2.toJson();
    assert(j2["images"].arrayNoRef.length == 1);
    assert(j2["images"][0].str == "aGVsbG8=");

    // Message with tool_calls
    auto tc = ToolCall("id-1", "search", parseJSON(`{"query":"D language"}`));
    auto m3 = Message("assistant", "", null, [tc]);
    auto j3 = m3.toJson();
    assert(j3["tool_calls"].arrayNoRef.length == 1);
    assert(j3["tool_calls"][0]["function"]["name"].str == "search");

    // Backward compatibility: two-field initialization still compiles
    Message m4;
    m4.role    = "system";
    m4.content = "You are a helpful assistant.";
    auto j4 = m4.toJson();
    assert(j4["role"].str == "system");
}

/// Callback type used by the streaming methods.
/// Receives one fully-parsed NDJSON chunk per call; `chunk["done"]` is
/// `true` on the final chunk.
alias StreamCallback = void delegate(JSONValue chunk) @safe;

// ---------------------------------------------------------------------------
// libcurl C-API write callbacks — must be extern(C) nothrow @system.
// Using etc.c.curl directly avoids importing std.net.curl, whose D wrappers
// fail to compile under -preview=safer + -preview=dip1000 on DMD because
// unannotated Phobos functions become @safe by default and then fail
// __gshared / DIP1000 checks inside their bodies.
// ---------------------------------------------------------------------------
private struct _CurlBuf    { char[] data; }
private struct _CurlStream { char[] line; StreamCallback cb; }

extern(C) private @system nothrow
size_t _curlWriteBuf(char* p, size_t sz, size_t n, void* ud)
{
    try { (*cast(_CurlBuf*)ud).data ~= p[0 .. sz * n]; }
    catch (Throwable) {}
    return sz * n;
}

extern(C) private @system nothrow
size_t _curlWriteStream(char* p, size_t sz, size_t n, void* ud)
{
    try
    {
        auto sd = cast(_CurlStream*)ud;
        sd.line ~= p[0 .. sz * n];
        size_t start = 0;
        foreach (i; 0 .. sd.line.length)
        {
            if (sd.line[i] == '\n')
            {
                if (i > start)
                    try { sd.cb(parseJSON(sd.line[start .. i].idup)); }
                    catch (Exception) {}
                start = i + 1;
            }
        }
        if (start > 0)
            sd.line = sd.line[start .. $].dup;
    }
    catch (Throwable) {}
    return sz * n;
}

/++
 + A client class for interacting with the Ollama REST API.
 +
 + Provides methods for text generation, chat, embeddings, tool calling, and model
 + management using `etc.c.curl` (libcurl C API) for HTTP and `std.json` for JSON.
 +
 + Examples:
 +     ---
 +     auto client = new OllamaClient();
 +     auto resp = client.chat("llama3", [Message("user", "Hi there!")]);
 +     writeln(resp["message"]["content"].str);
 +     ---
 +/
class OllamaClient
{
    private string   host;
    private Duration timeout = 60.seconds;

    /++
     + Constructs a new Ollama client.
     +
     + Params:
     +     host = Base URL of the Ollama server. Defaults to `DEFAULT_HOST`.
     +/
    this(string host = DEFAULT_HOST) @safe
    {
        this.host = host;
    }

    /++
     + Sets the timeout for HTTP requests.
     +
     + Params:
     +     timeout = Duration to wait before timing out.
     +/
    void setTimeOut(Duration timeout) @safe
    {
        this.timeout = timeout;
    }

    // -----------------------------------------------------------------------
    // Private curl helpers — @trusted wrappers around the C libcurl API.
    // Using etc.c.curl directly (not std.net.curl) sidesteps Phobos template
    // compilation errors under -preview=safer + -preview=dip1000 on DMD.
    // -----------------------------------------------------------------------
    private auto _curlInit(string url) @trusted
    {
        auto curl = curl_easy_init();
        enforce(curl !is null, "curl_easy_init failed");
        curl_easy_setopt(curl, CurlOption.url,      url.toStringz);
        curl_easy_setopt(curl, CurlOption.timeout_ms,
                         cast(long) timeout.total!"msecs");
        return curl;
    }

    private JSONValue _curlFinish(CURL* curl, ref _CurlBuf buf) @trusted
    {
        auto rc = curl_easy_perform(curl);
        enforce(rc == CurlError.ok,
            "curl: " ~ fromStringz(curl_easy_strerror(rc)).idup);
        auto j = parseJSON(cast(string) buf.data);
        enforce("error" !in j,
            "HTTP request failed: " ~ ("message" in j["error"]
                ? j["error"]["message"].str : "Unknown error"));
        return j;
    }

    private JSONValue post(string url, JSONValue data, bool stream = false) @trusted
    {
        immutable js = data.toString();
        auto curl = _curlInit(url);
        scope(exit) curl_easy_cleanup(curl);

        curl_slist* hdrs;
        hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
        scope(exit) curl_slist_free_all(hdrs);

        _CurlBuf buf;
        curl_easy_setopt(curl, CurlOption.post,          1L);
        curl_easy_setopt(curl, CurlOption.postfields,    js.ptr);
        curl_easy_setopt(curl, CurlOption.postfieldsize, cast(long) js.length);
        curl_easy_setopt(curl, CurlOption.httpheader,    hdrs);
        curl_easy_setopt(curl, CurlOption.writefunction, &_curlWriteBuf);
        curl_easy_setopt(curl, CurlOption.writedata,     &buf);
        return _curlFinish(curl, buf);
    }

    private JSONValue get(string url) @trusted
    {
        auto curl = _curlInit(url);
        scope(exit) curl_easy_cleanup(curl);

        _CurlBuf buf;
        curl_easy_setopt(curl, CurlOption.writefunction, &_curlWriteBuf);
        curl_easy_setopt(curl, CurlOption.writedata,     &buf);
        return _curlFinish(curl, buf);
    }

    /++
     + HTTP DELETE with a JSON body, used by `deleteModel`.
     +
     + The Ollama API requires HTTP DELETE for `/api/delete`. We use libcurl's
     + CURLOPT_CUSTOMREQUEST to send DELETE with a request body.
     +/
    private JSONValue del(string url, JSONValue data) @trusted
    {
        immutable js = data.toString();
        auto curl = _curlInit(url);
        scope(exit) curl_easy_cleanup(curl);

        curl_slist* hdrs;
        hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
        scope(exit) curl_slist_free_all(hdrs);

        _CurlBuf buf;
        curl_easy_setopt(curl, CurlOption.customrequest, "DELETE".toStringz);
        curl_easy_setopt(curl, CurlOption.postfields,    js.ptr);
        curl_easy_setopt(curl, CurlOption.postfieldsize, cast(long) js.length);
        curl_easy_setopt(curl, CurlOption.httpheader,    hdrs);
        curl_easy_setopt(curl, CurlOption.writefunction, &_curlWriteBuf);
        curl_easy_setopt(curl, CurlOption.writedata,     &buf);

        auto rc = curl_easy_perform(curl);
        enforce(rc == CurlError.ok,
            "curl: " ~ fromStringz(curl_easy_strerror(rc)).idup);

        if (buf.data.length == 0)
            return JSONValue((JSONValue[string]).init); // empty 200 OK = success

        auto j = parseJSON(cast(string) buf.data);
        enforce("error" !in j,
            "HTTP request failed: " ~ ("message" in j["error"]
                ? j["error"]["message"].str : "Unknown error"));
        return j;
    }

    /++
     + Low-level streaming POST helper.
     +
     + Sends `data` to `url` and dispatches each newline-delimited JSON chunk
     + to `onChunk` as it arrives, enabling token-by-token streaming from
     + `/api/generate` and `/api/chat`.
     +/
    private void postStream(string url, JSONValue data,
        StreamCallback onChunk) @trusted
    {
        immutable js = data.toString();
        auto curl = _curlInit(url);
        scope(exit) curl_easy_cleanup(curl);

        curl_slist* hdrs;
        hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
        scope(exit) curl_slist_free_all(hdrs);

        auto sd = _CurlStream(null, onChunk);
        curl_easy_setopt(curl, CurlOption.post,          1L);
        curl_easy_setopt(curl, CurlOption.postfields,    js.ptr);
        curl_easy_setopt(curl, CurlOption.postfieldsize, cast(long) js.length);
        curl_easy_setopt(curl, CurlOption.httpheader,    hdrs);
        curl_easy_setopt(curl, CurlOption.writefunction, &_curlWriteStream);
        curl_easy_setopt(curl, CurlOption.writedata,     &sd);

        auto rc = curl_easy_perform(curl);
        enforce(rc == CurlError.ok,
            "curl: " ~ fromStringz(curl_easy_strerror(rc)).idup);

        if (sd.line.length > 0)
            try { onChunk(parseJSON(sd.line.idup)); } catch (Exception) {}
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /++
     + Generates text based on a prompt using the specified model.
     +
     + Params:
     +     model     = Model name (e.g. "llama3.1:8b").
     +     prompt    = Input text.
     +     options   = Raw `JSONValue` generation options (backward-compatible).
     +     stream    = Whether to stream the response (not fully supported).
     +     system    = Optional system prompt.
     +     images    = Optional base64-encoded images for multimodal input.
     +     format    = Structured output: `JSONValue("json")` or a JSON Schema.
     +     suffix    = Text appended after the generated response.
     +     keepAlive = How long to keep the model loaded (e.g. "5m", "0").
     +     opts      = Typed `OllamaOptions`; takes precedence over `options`.
     +
     + Returns: A `JSONValue` containing `"response"`, `"done"`, and metadata.
     +/
    JSONValue generate(
        string        model,
        string        prompt,
        JSONValue     options   = JSONValue.init,
        bool          stream    = false,
        string        system    = null,
        string[]      images    = null,
        JSONValue     format    = JSONValue.init,
        string        suffix    = null,
        string        keepAlive = null,
        OllamaOptions opts      = OllamaOptions.init,
    ) @safe
    {
        auto url = host ~ "/api/generate";

        JSONValue[string] fields = [
            "model":  JSONValue(model),
            "prompt": JSONValue(prompt),
            "stream": JSONValue(stream),
        ];

        // Typed OllamaOptions takes precedence over raw JSONValue options
        auto optsJson = opts.toJson();
        if (optsJson.objectNoRef.length > 0)
            fields["options"] = optsJson;
        else if (options.type != JSONType.null_)
            fields["options"] = options;

        if (system.length    > 0) fields["system"]     = JSONValue(system);
        if (suffix.length    > 0) fields["suffix"]     = JSONValue(suffix);
        if (keepAlive.length > 0) fields["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)
            fields["format"] = format;

        if (images.length > 0)
        {
            JSONValue[] arr;
            foreach (img; images) arr ~= JSONValue(img);
            fields["images"] = JSONValue(arr);
        }

        return post(url, makeObject(fields), stream);
    }

    /++
     + Streaming text generation — calls `onChunk` for every response token.
     +
     + Each call to `onChunk` receives one NDJSON chunk. The chunk contains a
     + `"response"` string token and a boolean `"done"`. The final chunk has
     + `"done": true` and carries usage/timing metadata.
     +
     + Params:
     +     model     = Model name (e.g. "llama3.1:8b").
     +     prompt    = Input prompt.
     +     onChunk   = Callback invoked per chunk; must be `@safe`.
     +     system    = Optional system prompt.
     +     images    = Optional base64-encoded images (multimodal).
     +     format    = Structured output: `JSONValue("json")` or JSON Schema.
     +     keepAlive = How long to keep the model loaded.
     +     opts      = Typed generation options.
     +/
    void generateStream(
        string        model,
        string        prompt,
        StreamCallback onChunk,
        string        system    = null,
        string[]      images    = null,
        JSONValue     format    = JSONValue.init,
        string        keepAlive = null,
        OllamaOptions opts      = OllamaOptions.init,
    ) @safe
    {
        auto url = host ~ "/api/generate";
        JSONValue[string] fields = [
            "model":  JSONValue(model),
            "prompt": JSONValue(prompt),
            "stream": JSONValue(true),
        ];
        auto optsJson = opts.toJson();
        if (optsJson.objectNoRef.length > 0) fields["options"]    = optsJson;
        if (system.length    > 0)            fields["system"]     = JSONValue(system);
        if (keepAlive.length > 0)            fields["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)   fields["format"]     = format;
        if (images.length > 0)
        {
            JSONValue[] arr;
            foreach (img; images) arr ~= JSONValue(img);
            fields["images"] = JSONValue(arr);
        }
        postStream(url, makeObject(fields), onChunk);
    }

    // -----------------------------------------------------------------------
    // Chat
    // -----------------------------------------------------------------------

    /++
     + Engages in a chat interaction using the specified model and message history.
     +
     + Params:
     +     model     = Model name.
     +     messages  = Array of `Message` structs (conversation history).
     +     options   = Raw `JSONValue` generation options (backward-compatible).
     +     stream    = Whether to stream the response (not fully supported).
     +     tools     = Optional tool definitions for tool/function calling.
     +     format    = Structured output schema or `JSONValue("json")`.
     +     keepAlive = How long to keep the model loaded.
     +     opts      = Typed `OllamaOptions`.
     +
     + Returns: A `JSONValue` with `"message"`, `"done"`, and metadata. When the
     +          model calls a tool, `response["message"]["tool_calls"]` is populated.
     +/
    JSONValue chat(
        string        model,
        Message[]     messages,
        JSONValue     options   = JSONValue.init,
        bool          stream    = false,
        Tool[]        tools     = null,
        JSONValue     format    = JSONValue.init,
        string        keepAlive = null,
        OllamaOptions opts      = OllamaOptions.init,
    ) @safe
    {
        auto url = host ~ "/api/chat";

        JSONValue[] msgArray;
        foreach (msg; messages) msgArray ~= msg.toJson();

        JSONValue[string] fields = [
            "model":    JSONValue(model),
            "messages": JSONValue(msgArray),
            "stream":   JSONValue(stream),
        ];

        auto optsJson = opts.toJson();
        if (optsJson.objectNoRef.length > 0)
            fields["options"] = optsJson;
        else if (options.type != JSONType.null_)
            fields["options"] = options;

        if (keepAlive.length > 0)
            fields["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)
            fields["format"] = format;

        if (tools.length > 0)
        {
            JSONValue[] arr;
            foreach (t; tools) arr ~= t.toJson();
            fields["tools"] = JSONValue(arr);
        }

        return post(url, makeObject(fields), stream);
    }

    /++
     + Streaming chat — calls `onChunk` for every assistant token.
     +
     + Each chunk contains `"message": {"role": "assistant", "content": "<token>"}`.
     + The final chunk has `"done": true` and carries usage metadata.
     +
     + Params:
     +     model     = Model name.
     +     messages  = Conversation history.
     +     onChunk   = Callback invoked per chunk; must be `@safe`.
     +     tools     = Optional tool definitions.
     +     format    = Structured output schema or `JSONValue("json")`.
     +     keepAlive = How long to keep the model loaded.
     +     opts      = Typed generation options.
     +/
    void chatStream(
        string        model,
        Message[]     messages,
        StreamCallback onChunk,
        Tool[]        tools     = null,
        JSONValue     format    = JSONValue.init,
        string        keepAlive = null,
        OllamaOptions opts      = OllamaOptions.init,
    ) @safe
    {
        auto url = host ~ "/api/chat";
        JSONValue[] msgArray;
        foreach (msg; messages) msgArray ~= msg.toJson();

        JSONValue[string] fields = [
            "model":    JSONValue(model),
            "messages": JSONValue(msgArray),
            "stream":   JSONValue(true),
        ];
        auto optsJson = opts.toJson();
        if (optsJson.objectNoRef.length > 0) fields["options"]    = optsJson;
        if (keepAlive.length > 0)            fields["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)   fields["format"]     = format;
        if (tools.length > 0)
        {
            JSONValue[] arr;
            foreach (t; tools) arr ~= t.toJson();
            fields["tools"] = JSONValue(arr);
        }
        postStream(url, makeObject(fields), onChunk);
    }

    // -----------------------------------------------------------------------
    // Embeddings
    // -----------------------------------------------------------------------

    /++
     + Generates an embedding vector for a single text input.
     +
     + Params:
     +     model     = Embedding model name (e.g. "nomic-embed-text").
     +     input     = Text to embed.
     +     keepAlive = How long to keep the model loaded.
     +
     + Returns: A `JSONValue` with an `"embeddings"` array containing one vector.
     +/
    JSONValue embed(string model, string input, string keepAlive = null) @safe
    {
        auto url = host ~ "/api/embed";
        JSONValue[string] fields = ["model": JSONValue(model), "input": JSONValue(input)];
        if (keepAlive.length > 0)
            fields["keep_alive"] = JSONValue(keepAlive);
        return post(url, makeObject(fields));
    }

    /++
     + Generates embedding vectors for a batch of text inputs.
     +
     + Params:
     +     model     = Embedding model name.
     +     inputs    = Array of texts to embed.
     +     keepAlive = How long to keep the model loaded.
     +
     + Returns: A `JSONValue` with an `"embeddings"` array, one vector per input.
     +/
    JSONValue embed(string model, string[] inputs, string keepAlive = null) @safe
    {
        auto url = host ~ "/api/embed";
        JSONValue[] arr;
        foreach (inp; inputs) arr ~= JSONValue(inp);
        JSONValue[string] fields = ["model": JSONValue(model), "input": JSONValue(arr)];
        if (keepAlive.length > 0)
            fields["keep_alive"] = JSONValue(keepAlive);
        return post(url, makeObject(fields));
    }

    // -----------------------------------------------------------------------
    // Model Management
    // -----------------------------------------------------------------------

    /++
     + Lists all locally available models.
     +
     + Returns: Pretty-printed JSON string of model details.
     +/
    string listModels() @safe
    {
        return get(host ~ "/api/tags").toPrettyString();
    }

    /++
     + Retrieves detailed information about a specific model.
     +
     + Params:
     +     model = Model name to query.
     +
     + Returns: Pretty-printed JSON string of model metadata.
     +/
    string showModel(string model) @safe
    {
        return post(host ~ "/api/show",
            makeObject(["name": JSONValue(model)])).toPrettyString();
    }

    /++
     + Creates a custom model from a modelfile.
     +
     + Params:
     +     name      = New model name.
     +     modelfile = Modelfile content string.
     +
     + Returns: A `JSONValue` with creation status.
     +/
    JSONValue createModel(string name, string modelfile) @safe
    {
        return post(host ~ "/api/create",
            makeObject(["name": JSONValue(name), "modelfile": JSONValue(modelfile)]));
    }

    /++
     + Copies an existing model to a new name.
     +
     + Params:
     +     source      = Source model name.
     +     destination = Destination model name.
     +
     + Returns: A `JSONValue` with copy status.
     +/
    JSONValue copy(string source, string destination) @safe
    {
        return post(host ~ "/api/copy",
            makeObject(["source": JSONValue(source), "destination": JSONValue(destination)]));
    }

    /++
     + Deletes a model from the Ollama server.
     +
     + Uses HTTP DELETE as required by the Ollama API specification.
     +
     + Params:
     +     name = Model name to delete.
     +
     + Returns: A `JSONValue` (empty object on success).
     +/
    JSONValue deleteModel(string name) @safe
    {
        return del(host ~ "/api/delete", makeObject(["name": JSONValue(name)]));
    }

    /++
     + Downloads a model from the Ollama registry.
     +
     + Params:
     +     name   = Model name to pull.
     +     stream = Whether to stream progress (not fully supported).
     +
     + Returns: A `JSONValue` with pull status.
     +/
    JSONValue pull(string name, bool stream = false) @safe
    {
        return post(host ~ "/api/pull",
            makeObject(["name": JSONValue(name), "stream": JSONValue(stream)]), stream);
    }

    /++
     + Uploads a model to the Ollama registry.
     +
     + Params:
     +     name   = Model name to push.
     +     stream = Whether to stream progress (not fully supported).
     +
     + Returns: A `JSONValue` with push status.
     +/
    JSONValue push(string name, bool stream = false) @safe
    {
        return post(host ~ "/api/push",
            makeObject(["name": JSONValue(name), "stream": JSONValue(stream)]), stream);
    }

    // -----------------------------------------------------------------------
    // Server Operations
    // -----------------------------------------------------------------------

    /++
     + Retrieves the Ollama server version string.
     +
     + Returns: Version string (e.g. "0.6.2").
     +/
    string getVersion() @safe
    {
        return get(host ~ "/api/version")["version"].str;
    }

    /++
     + Lists currently running (loaded) models.
     +
     + Returns: Pretty-printed JSON string with model names, sizes, and expiry.
     +/
    string ps() @safe
    {
        return get(host ~ "/api/ps").toPrettyString();
    }


    // -----------------------------------------------------------------------
    // OpenAI-Compatible Endpoints
    // -----------------------------------------------------------------------

    /++
     + Performs an OpenAI-style chat completion.
     +
     + Params:
     +     model       = Model name.
     +     messages    = Chat history as `Message` array.
     +     maxTokens   = Maximum tokens to generate (0 = unlimited).
     +     temperature = Sampling temperature (default 1.0).
     +     stream      = Whether to stream (not fully supported).
     +
     + Returns: A `JSONValue` in OpenAI `ChatCompletion` format.
     +/
    JSONValue chatCompletions(
        string    model,
        Message[] messages,
        int       maxTokens   = 0,
        float     temperature = 1.0,
        bool      stream      = false,
    ) @trusted
    {
        auto url = host ~ "/v1/chat/completions";
        JSONValue[] msgArray;
        foreach (msg; messages) msgArray ~= msg.toJson();

        JSONValue[string] fields = [
            "model":       JSONValue(model),
            "messages":    JSONValue(msgArray),
            "stream":      JSONValue(stream),
            "temperature": JSONValue(temperature),
        ];
        if (maxTokens > 0)
            fields["max_tokens"] = JSONValue(maxTokens);

        return post(url, makeObject(fields), stream);
    }

    /++
     + Performs an OpenAI-style text completion.
     +
     + Params:
     +     model       = Model name.
     +     prompt      = Input prompt.
     +     maxTokens   = Maximum tokens to generate (0 = unlimited).
     +     temperature = Sampling temperature (default 1.0).
     +     stream      = Whether to stream (not fully supported).
     +
     + Returns: A `JSONValue` in OpenAI `Completion` format.
     +/
    JSONValue completions(
        string model,
        string prompt,
        int    maxTokens   = 0,
        float  temperature = 1.0,
        bool   stream      = false,
    ) @trusted
    {
        auto url = host ~ "/v1/completions";
        JSONValue[string] fields = [
            "model":       JSONValue(model),
            "prompt":      JSONValue(prompt),
            "stream":      JSONValue(stream),
            "temperature": JSONValue(temperature),
        ];
        if (maxTokens > 0)
            fields["max_tokens"] = JSONValue(maxTokens);

        return post(url, makeObject(fields), stream);
    }

    /++
     + Lists models in OpenAI-compatible format.
     +
     + Returns: Pretty-printed JSON string of model data.
     +/
    string getModels() @safe
    {
        return get(host ~ "/v1/models").toPrettyString();
    }
}

/// Default host URL for the Ollama server.
enum DEFAULT_HOST = "http://127.0.0.1:11434";
