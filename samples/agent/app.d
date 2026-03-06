/++
 + Minimal tool-calling agent demonstrating the agentic loop.
 +
 + Flow:
 +   1. Send the user's question to the model with available tools.
 +   2. If the model requests tool calls, execute them locally.
 +   3. Append the tool results and send back to the model.
 +   4. Repeat until the model returns a plain-text answer.
 +
 + Built-in tools:
 +   get_time()       — current UTC date and time
 +   add(a, b)        — arithmetic sum
 +   to_upper(text)   — TEXT
 +
 + Prerequisites:
 +     - Ollama server on http://127.0.0.1:11434  (ollama serve)
 +     - A tool-capable model, e.g. qwen3.5:2b   (ollama pull qwen3.5:2b)
 +
 + Usage:
 +     agent [--model qwen3.5:2b] [--host http://…] ["your question"]
 +/
import ollama;
import std;

// ── Tool dispatch ─────────────────────────────────────────────────────────────

JSONValue callTool(string name, JSONValue args) @safe
{
    switch (name)
    {
        case "get_time":
            import std.datetime.systime : Clock;
            return JSONValue(Clock.currTime.toUTC.toSimpleString());

        case "add":
            return JSONValue(args["a"].get!double + args["b"].get!double);

        case "to_upper":
            return JSONValue(args["text"].str.toUpper);

        default:
            return JSONValue("Unknown tool: " ~ name);
    }
}

// ── Tool definitions ──────────────────────────────────────────────────────────

Tool[] buildTools() @safe
{
    return [
        Tool("function", ToolFunction("get_time",
            "Return the current UTC date and time.",
            parseJSON(`{"type":"object","properties":{}}`))),

        Tool("function", ToolFunction("add",
            "Add two numbers and return the result.",
            parseJSON(`{
                "type": "object",
                "properties": {
                    "a": {"type":"number","description":"First operand"},
                    "b": {"type":"number","description":"Second operand"}
                },
                "required": ["a","b"]
            }`))),

        Tool("function", ToolFunction("to_upper",
            "Convert a string to upper case.",
            parseJSON(`{
                "type": "object",
                "properties": {
                    "text": {"type":"string","description":"The string to convert"}
                },
                "required": ["text"]
            }`))),
    ];
}

// ── Agent loop ────────────────────────────────────────────────────────────────

void runAgent(OllamaClient client, string model, string question) @safe
{
    writefln("User  : %s", question);

    Message[] history = [
        Message("system", "You are a helpful assistant. Use the provided tools when needed."),
        Message("user", question),
    ];

    auto tools = buildTools();

    for (;;)
    {
        auto r   = client.chat(model, history, JSONValue.init, false, tools);
        auto msg = r["message"];

        // No (or empty) tool_calls → final plain-text answer
        auto tcs = "tool_calls" in msg.objectNoRef;
        if (!tcs || (*tcs).arrayNoRef.length == 0)
        {
            writefln("Agent : %s", "content" in msg ? msg["content"].str : "");
            break;
        }

        // Record the assistant's tool-request message
        history ~= Message("assistant", "content" in msg ? msg["content"].str : "");

        // Execute each tool and feed results back as "tool" messages
        foreach (tc; (*tcs).arrayNoRef)
        {
            immutable fname = tc["function"]["name"].str;
            auto      fargs = tc["function"]["arguments"];

            // Some models serialise arguments as a JSON string; parse if needed
            if (fargs.type == JSONType.string)
                fargs = parseJSON(fargs.str);

            auto result = callTool(fname, fargs);
            writefln("  [%s(%s) → %s]", fname, fargs.toString, result.toString);

            history ~= Message("tool", result.toString);
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

void main(string[] args) @safe
{
    string model    = "qwen3.5:2b";
    string host     = DEFAULT_HOST;
    string question = "What is the current UTC time? Also, what is 123 plus 456?";

    for (int i = 1; i < args.length; i++)
    {
        if      (args[i] == "--model" && i + 1 < args.length) model    = args[++i];
        else if (args[i] == "--host"  && i + 1 < args.length) host     = args[++i];
        else question = args[i];
    }

    auto client = new OllamaClient(host);
    client.setTimeOut(120.seconds);

    try   runAgent(client, model, question);
    catch (Exception e) writeln("error: ", e.msg);
}
