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
 + It supports both native Ollama endpoints and OpenAI-compatible endpoints, using `std.net.curl` for
 + HTTP requests and `std.json` for JSON processing.
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

import std;

@safe:

/++
 + Typed options for controlling model generation behavior.
 +
 + Only non-default (set) fields are serialized to JSON. Float fields use `float.nan` as the
 + "unset" sentinel; integer fields use `0` as unset (unless otherwise noted).
 +
 + Examples:
 +     ---
 +     OllamaOptions opts;
 +     opts.temperature = 0.8;
 +     opts.num_ctx = 4096;
 +     opts.stop = ["<|end|>", "\n\n"];
 +     auto response = client.generate("llama3", "Hello", JSONValue.init, false, null, null,
 +                                     JSONValue.init, null, null, opts);
 +     ---
 +/
struct OllamaOptions
{
    float temperature   = float.nan; /// Sampling temperature (0.0 = deterministic, higher = more creative).
    int   top_k         = 0;         /// Top-K sampling; 0 = disabled.
    float top_p         = float.nan; /// Nucleus sampling probability threshold.
    float min_p         = float.nan; /// Minimum probability threshold.
    float repeat_penalty = float.nan; /// Penalty for repeated tokens (default server-side: 1.1).
    int   repeat_last_n  = 0;        /// Number of last tokens considered for repeat penalty.
    int   seed           = 0;        /// Random seed; 0 = random.
    int   num_predict    = 0;        /// Max tokens to generate; 0 = unlimited.
    int   num_ctx        = 0;        /// Context window size; 0 = model default.
    string[] stop;                   /// Stop sequences; empty = disabled.
    int   mirostat       = 0;        /// Mirostat sampling strategy (0=off, 1=v1, 2=v2).
    float mirostat_tau   = float.nan; /// Mirostat target entropy.
    float mirostat_eta   = float.nan; /// Mirostat learning rate.

    /++
     + Serializes only non-default fields to a `JSONValue` object.
     +
     + Returns: A `JSONValue` object containing only the fields that were explicitly set.
     +/
    JSONValue toJson() const @safe
    {
        import std.math : isNaN;
        JSONValue[string] fields;
        if (!isNaN(temperature))          fields["temperature"]    = JSONValue(temperature);
        if (top_k > 0)                    fields["top_k"]          = JSONValue(top_k);
        if (!isNaN(top_p))                fields["top_p"]          = JSONValue(top_p);
        if (!isNaN(min_p))                fields["min_p"]          = JSONValue(min_p);
        if (!isNaN(repeat_penalty))       fields["repeat_penalty"] = JSONValue(repeat_penalty);
        if (repeat_last_n > 0)            fields["repeat_last_n"]  = JSONValue(repeat_last_n);
        if (seed > 0)                     fields["seed"]           = JSONValue(seed);
        if (num_predict > 0)              fields["num_predict"]    = JSONValue(num_predict);
        if (num_ctx > 0)                  fields["num_ctx"]        = JSONValue(num_ctx);
        if (!isNaN(mirostat_tau))         fields["mirostat_tau"]   = JSONValue(mirostat_tau);
        if (!isNaN(mirostat_eta))         fields["mirostat_eta"]   = JSONValue(mirostat_eta);
        if (mirostat > 0)                 fields["mirostat"]       = JSONValue(mirostat);
        if (stop.length > 0)
        {
            JSONValue[] arr;
            foreach (s; stop) arr ~= JSONValue(s);
            fields["stop"] = JSONValue(arr);
        }
        return JSONValue(fields);
    }
}

///
unittest
{
    import std.math : isNaN;

    // Default options produce an empty JSON object
    OllamaOptions def;
    auto j0 = def.toJson();
    assert(j0.type == JSONType.object);
    assert(j0.object.length == 0, "Default options should serialize to empty object");

    // Set individual fields
    OllamaOptions opts;
    opts.temperature  = 0.7f;
    opts.top_k        = 40;
    opts.num_ctx      = 4096;
    opts.stop         = ["<|end|>"];
    auto j = opts.toJson();
    assert(j["temperature"].floating == 0.7f);
    assert(j["top_k"].integer == 40);
    assert(j["num_ctx"].integer == 4096);
    assert(j["stop"].array[0].str == "<|end|>");

    // Unset float fields are absent
    assert("top_p"          !in j.object);
    assert("min_p"          !in j.object);
    assert("repeat_penalty" !in j.object);
    assert("mirostat"       !in j.object);

    // Explicit zero temperature is included (valid value)
    OllamaOptions zeroTemp;
    zeroTemp.temperature = 0.0f;
    auto jz = zeroTemp.toJson();
    assert("temperature" in jz.object);
    assert(jz["temperature"].floating == 0.0f);
}

/++
 + Function schema for tool/function calling definitions.
 +
 + Used inside `Tool` when registering callable tools with the model.
 +/
struct ToolFunction
{
    string    name;        /// Function name as called by the model.
    string    description; /// Human-readable description of what the function does.
    JSONValue parameters;  /// JSON Schema object defining the function's parameters.

    /++
     + Converts to a JSON object suitable for the Ollama API `tools` array.
     +
     + Returns: A `JSONValue` with "name", "description", and optionally "parameters".
     +/
    JSONValue toJson() const @safe
    {
        JSONValue j = ["name": JSONValue(name), "description": JSONValue(description)];
        if (parameters.type != JSONType.null_)
            j.object["parameters"] = parameters;
        return j;
    }
}

/++
 + A tool (function) definition passed to `chat()` enabling tool/function calling.
 +
 + Examples:
 +     ---
 +     auto schema = parseJSON(`{
 +         "type": "object",
 +         "properties": {"location": {"type": "string"}},
 +         "required": ["location"]
 +     }`);
 +     auto tool = Tool("function", ToolFunction("get_weather", "Get current weather", schema));
 +     auto resp = client.chat("llama3", messages, [tool]);
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
        return JSONValue(["type": JSONValue(type), "function": function_.toJson()]);
    }
}

/++
 + Represents a tool/function call made by the model in a chat response.
 +
 + Access via `response["message"]["tool_calls"]` when the model decides to call a tool.
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
        JSONValue func = ["name": JSONValue(name)];
        if (arguments.type != JSONType.null_)
            func.object["arguments"] = arguments;
        JSONValue j = ["function": func];
        if (id.length > 0)
            j.object["id"] = JSONValue(id);
        return j;
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

    // Tool serialization
    auto tool = Tool("function", tf);
    auto jt = tool.toJson();
    assert(jt["type"].str == "function");
    assert(jt["function"]["name"].str == "get_weather");

    // ToolCall serialization
    auto tc = ToolCall("call-1", "get_weather", parseJSON(`{"city":"Paris"}`));
    auto jtc = tc.toJson();
    assert(jtc["id"].str == "call-1");
    assert(jtc["function"]["name"].str == "get_weather");
    assert(jtc["function"]["arguments"]["city"].str == "Paris");

    // ToolCall without id
    auto tc2 = ToolCall("", "sum", parseJSON(`{"a":1,"b":2}`));
    auto jtc2 = tc2.toJson();
    assert("id" !in jtc2.object);
    assert(jtc2["function"]["name"].str == "sum");
}

/++
 + Represents a single message in a chat interaction.
 +
 + Supports text, base64-encoded images (multimodal), and tool call results.
 + Backward compatible: `Message("user", "hello")` still works.
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
     + Returns: A `JSONValue` with "role", "content", and optionally "images" and "tool_calls".
     +/
    JSONValue toJson() const @safe
    {
        JSONValue j = ["role": JSONValue(role), "content": JSONValue(content)];
        if (images.length > 0)
        {
            JSONValue[] arr;
            foreach (img; images) arr ~= JSONValue(img);
            j.object["images"] = JSONValue(arr);
        }
        if (tool_calls.length > 0)
        {
            JSONValue[] arr;
            foreach (tc; tool_calls) arr ~= tc.toJson();
            j.object["tool_calls"] = JSONValue(arr);
        }
        return j;
    }
}

///
unittest
{
    // Basic message
    auto m = Message("user", "Hello, world!");
    auto j = m.toJson();
    assert(j["role"].str == "user");
    assert(j["content"].str == "Hello, world!");
    assert("images"     !in j.object);
    assert("tool_calls" !in j.object);

    // Message with images
    auto m2 = Message("user", "What is in this image?", ["aGVsbG8="]);
    auto j2 = m2.toJson();
    assert(j2["images"].array.length == 1);
    assert(j2["images"][0].str == "aGVsbG8=");

    // Message with tool_calls
    auto tc  = ToolCall("id-1", "search", parseJSON(`{"query":"D language"}`));
    auto m3  = Message("assistant", "", null, [tc]);
    auto j3  = m3.toJson();
    assert(j3["tool_calls"].array.length == 1);
    assert(j3["tool_calls"][0]["function"]["name"].str == "search");

    // Backward compatibility: two-field struct literal
    Message m4;
    m4.role    = "system";
    m4.content = "You are a helpful assistant.";
    auto j4 = m4.toJson();
    assert(j4["role"].str == "system");
}

/++
 + A client class for interacting with the Ollama REST API.
 +
 + Provides methods for text generation, chat interactions, embeddings, tool calling,
 + and model management. Uses `std.net.curl` for HTTP and `std.json` for JSON.
 +
 + Examples:
 +     ---
 +     auto client = new OllamaClient();
 +     auto chatResp = client.chat("llama3", [Message("user", "Hi there!")]);
 +     writeln(chatResp["message"]["content"].str);
 +     ---
 +/
class OllamaClient
{
    private string   host;           /// Base URL of the Ollama server.
    private Duration timeout = 60.seconds; /// Default HTTP request timeout.

    /++
     + Constructs a new Ollama client instance.
     +
     + Params:
     +     host = Base URL of the Ollama server. Defaults to `DEFAULT_HOST`.
     +/
    this(string host = DEFAULT_HOST) @safe
    {
        this.host = host;
    }

    /++
     + Sets the timeout duration for HTTP requests.
     +
     + Params:
     +     timeout = Duration to wait before timing out.
     +/
    void setTimeOut(Duration timeout) @safe
    {
        this.timeout = timeout;
    }

    // -------------------------------------------------------------------------
    // Private HTTP helpers
    // -------------------------------------------------------------------------

    private JSONValue post(string url, JSONValue data, bool stream = false) @trusted
    {
        auto client = HTTP();
        client.addRequestHeader("Content-Type", "application/json");
        client.connectTimeout(timeout);

        auto jsonStr = data.toString();
        auto response = std.net.curl.post(url, jsonStr, client);
        auto jsonResponse = parseJSON(response);

        enforce("error" !in jsonResponse,
            "HTTP request failed: " ~ ("message" in jsonResponse["error"]
                ? jsonResponse["error"]["message"].str : "Unknown error"));
        return jsonResponse;
    }

    private JSONValue get(string url) @trusted
    {
        auto client = HTTP();
        client.connectTimeout(timeout);

        auto response = std.net.curl.get(url, client);
        auto jsonResponse = parseJSON(response);
        enforce("error" !in jsonResponse,
            "HTTP request failed: " ~ ("message" in jsonResponse["error"]
                ? jsonResponse["error"]["message"].str : "Unknown error"));
        return jsonResponse;
    }

    /++
     + HTTP DELETE request with a JSON body (used by `deleteModel`).
     +
     + Curl supports DELETE with a request body; we send the JSON payload via
     + postData and then override the method to DELETE.
     +/
    private JSONValue del(string url, JSONValue data) @trusted
    {
        auto jsonStr = data.toString();
        auto http = HTTP(url);
        http.addRequestHeader("Content-Type", "application/json");
        http.connectTimeout(timeout);
        // Set body first (internally switches to POST), then override to DELETE
        http.postData = cast(const(void)[]) jsonStr;
        http.method   = HTTP.Method.del;

        char[] respBuf;
        http.onReceive = (ubyte[] chunk) {
            respBuf ~= cast(char[]) chunk;
            return chunk.length;
        };
        http.perform();

        if (respBuf.length == 0)
            return JSONValue((JSONValue[string]).init); // empty 200 OK → success

        auto jsonResponse = parseJSON(respBuf);
        enforce("error" !in jsonResponse,
            "HTTP request failed: " ~ ("message" in jsonResponse["error"]
                ? jsonResponse["error"]["message"].str : "Unknown error"));
        return jsonResponse;
    }

    // -------------------------------------------------------------------------
    // Generation
    // -------------------------------------------------------------------------

    /++
     + Generates text based on a prompt using the specified model.
     +
     + Params:
     +     model     = Model name (e.g. "llama3.1:8b").
     +     prompt    = Input text to generate from.
     +     options   = Raw `JSONValue` generation options (backward-compatible).
     +     stream    = Whether to stream the response (not fully supported).
     +     system    = Optional system prompt to prepend.
     +     images    = Optional base64-encoded images for multimodal input.
     +     format    = Structured output schema: `JSONValue("json")` or a JSON Schema object.
     +     suffix    = Text appended after the generated response.
     +     keepAlive = How long to keep the model loaded (e.g. "5m", "0").
     +     opts      = Typed `OllamaOptions`; overrides `options` if non-empty.
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
        JSONValue data = [
            "model":  JSONValue(model),
            "prompt": JSONValue(prompt),
            "stream": JSONValue(stream),
        ];

        // Merge options: typed OllamaOptions take precedence over raw JSONValue
        auto optsJson = opts.toJson();
        if (optsJson.object.length > 0)
            data.object["options"] = optsJson;
        else if (options.type != JSONType.null_)
            data.object["options"] = options;

        if (system.length    > 0) data.object["system"]     = JSONValue(system);
        if (suffix.length    > 0) data.object["suffix"]     = JSONValue(suffix);
        if (keepAlive.length > 0) data.object["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)
            data.object["format"] = format;

        if (images.length > 0)
        {
            JSONValue[] arr;
            foreach (img; images) arr ~= JSONValue(img);
            data.object["images"] = JSONValue(arr);
        }

        return post(url, data, stream);
    }

    // -------------------------------------------------------------------------
    // Chat
    // -------------------------------------------------------------------------

    /++
     + Engages in a chat interaction using the specified model and message history.
     +
     + Params:
     +     model     = Model name.
     +     messages  = Array of `Message` structs (conversation history).
     +     options   = Raw `JSONValue` generation options (backward-compatible).
     +     stream    = Whether to stream the response (not fully supported).
     +     tools     = Optional tool definitions for tool/function calling.
     +     format    = Structured output schema: `JSONValue("json")` or a JSON Schema.
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

        JSONValue data = [
            "model":    JSONValue(model),
            "messages": JSONValue(msgArray),
            "stream":   JSONValue(stream),
        ];

        auto optsJson = opts.toJson();
        if (optsJson.object.length > 0)
            data.object["options"] = optsJson;
        else if (options.type != JSONType.null_)
            data.object["options"] = options;

        if (keepAlive.length > 0)
            data.object["keep_alive"] = JSONValue(keepAlive);
        if (format.type != JSONType.null_)
            data.object["format"] = format;

        if (tools.length > 0)
        {
            JSONValue[] arr;
            foreach (t; tools) arr ~= t.toJson();
            data.object["tools"] = JSONValue(arr);
        }

        return post(url, data, stream);
    }

    // -------------------------------------------------------------------------
    // Embeddings
    // -------------------------------------------------------------------------

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
        JSONValue data = ["model": JSONValue(model), "input": JSONValue(input)];
        if (keepAlive.length > 0)
            data.object["keep_alive"] = JSONValue(keepAlive);
        return post(url, data);
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
        JSONValue data = ["model": JSONValue(model), "input": JSONValue(arr)];
        if (keepAlive.length > 0)
            data.object["keep_alive"] = JSONValue(keepAlive);
        return post(url, data);
    }

    // -------------------------------------------------------------------------
    // Model Management
    // -------------------------------------------------------------------------

    /++
     + Lists all locally available models.
     +
     + Returns: Pretty-printed JSON string of model details.
     +/
    string listModels() @safe
    {
        auto url = host ~ "/api/tags";
        return get(url).toPrettyString();
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
        auto url = host ~ "/api/show";
        JSONValue data = ["name": JSONValue(model)];
        return post(url, data).toPrettyString();
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
        auto url = host ~ "/api/create";
        JSONValue data = ["name": JSONValue(name), "modelfile": JSONValue(modelfile)];
        return post(url, data);
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
        auto url = host ~ "/api/copy";
        JSONValue data = ["source": JSONValue(source), "destination": JSONValue(destination)];
        return post(url, data);
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
        auto url = host ~ "/api/delete";
        JSONValue data = ["name": JSONValue(name)];
        return del(url, data);
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
        auto url = host ~ "/api/pull";
        JSONValue data = ["name": JSONValue(name), "stream": JSONValue(stream)];
        return post(url, data, stream);
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
        auto url = host ~ "/api/push";
        JSONValue data = ["name": JSONValue(name), "stream": JSONValue(stream)];
        return post(url, data, stream);
    }

    // -------------------------------------------------------------------------
    // Server Operations
    // -------------------------------------------------------------------------

    /++
     + Retrieves the Ollama server version string.
     +
     + Returns: Version string (e.g. "0.6.2").
     +/
    string getVersion() @safe
    {
        auto url = host ~ "/api/version";
        return get(url)["version"].str;
    }

    /++
     + Lists currently running (loaded) models.
     +
     + Returns: Pretty-printed JSON string with model names, sizes, and expiry times.
     +/
    string ps() @safe
    {
        auto url = host ~ "/api/ps";
        return get(url).toPrettyString();
    }

    // -------------------------------------------------------------------------
    // OpenAI-Compatible Endpoints
    // -------------------------------------------------------------------------

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

        JSONValue data = [
            "model":    JSONValue(model),
            "messages": JSONValue(msgArray),
            "stream":   JSONValue(stream),
        ];
        data.object["temperature"] = JSONValue(temperature);
        if (maxTokens > 0)
            data.object["max_tokens"] = JSONValue(maxTokens);

        return post(url, data, stream);
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
        JSONValue data = [
            "model":  JSONValue(model),
            "prompt": JSONValue(prompt),
            "stream": JSONValue(stream),
        ];
        data.object["temperature"] = JSONValue(temperature);
        if (maxTokens > 0)
            data.object["max_tokens"] = JSONValue(maxTokens);

        return post(url, data, stream);
    }

    /++
     + Lists models in OpenAI-compatible format.
     +
     + Returns: Pretty-printed JSON string of model data.
     +/
    string getModels() @safe
    {
        auto url = host ~ "/v1/models";
        return get(url).toPrettyString();
    }
}

/// Default host URL for the Ollama server.
enum DEFAULT_HOST = "http://127.0.0.1:11434";
