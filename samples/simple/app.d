/++
 + Quick tour of the OllamaClient SDK.
 +
 + Prerequisites:
 +     - Ollama server on http://127.0.0.1:11434  (ollama serve)
 +     - Model qwen3.5:7b installed              (ollama pull qwen3.5:7b)
 +/
import ollama;
import std;

/// Run `body` inside a titled banner, printing any exception as an error line.
void section(string title, scope void delegate() @safe body) @safe
{
    writefln("\n=== %s ===", title);
    try   body();
    catch (Exception e) writefln("  error: %s", e.msg);
}

void main() @safe
{
    auto client = new OllamaClient();
    client.setTimeOut(120.seconds);
    writeln("host: ", DEFAULT_HOST);

    // ── Server ───────────────────────────────────────────────────────────────
    section("Server", () {
        writeln("version : ", client.getVersion());
        writeln("models  : ", client.listModels());
        writeln("running : ", client.ps());
    });

    section("Pull qwen3.5:7b", () {
        writeln(client.pull("qwen3.5:7b")["status"].str);
    });

    // ── Generate ─────────────────────────────────────────────────────────────
    enum model = "qwen3.5:7b";

    section("Generate — basic", () {
        writeln(client.generate(model, "Why is the sky blue?")["response"].str);
    });

    section("Generate — system prompt + options", () {
        OllamaOptions opts;
        opts.temperature = 0.5f;
        opts.num_predict = 80;
        auto r = client.generate(model,
            "One fun fact about D programming.",
            JSONValue.init, false,
            "You are a concise technical writer. Reply in one sentence.",
            null, JSONValue.init, null, null, opts);
        writeln(r["response"].str);
    });

    section("Generate — structured JSON", () {
        auto r = client.generate(model,
            "Capital and population of France as JSON.",
            JSONValue.init, false, null, null, JSONValue("json"));
        writeln(r["response"].str);
    });

    // ── Chat ─────────────────────────────────────────────────────────────────
    section("Chat", () {
        auto r = client.chat(model, [Message("user", "Hello!")]);
        writeln(r["message"]["content"].str);
    });

    section("Chat — tool calling", () {
        auto schema = parseJSON(`{
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit":     {"type": "string", "enum": ["celsius","fahrenheit"]}
            },
            "required": ["location"]
        }`);
        auto tools = [Tool("function",
            ToolFunction("get_current_weather",
                "Get the current weather for a location", schema))];
        auto r = client.chat(model,
            [Message("user", "What is the weather in Paris in celsius?")],
            JSONValue.init, false, tools);
        if (auto tc = "tool_calls" in r["message"].objectNoRef)
            foreach (c; (*tc).arrayNoRef)
                writefln("  call %s(%s)",
                    c["function"]["name"].str,
                    c["function"]["arguments"].toString);
        else
            writeln(r["message"]["content"].str);
    });

    // ── Embeddings ───────────────────────────────────────────────────────────
    section("Embeddings", () {
        auto vecs = client.embed(model, "The quick brown fox.")["embeddings"].arrayNoRef;
        writefln("  single vector length : %d", vecs[0].arrayNoRef.length);

        auto batch = client.embed(model, ["Hello world", "D is great", "Ollama rocks"]);
        writefln("  batch count          : %d", batch["embeddings"].arrayNoRef.length);
    });

    // ── OpenAI-compatible ─────────────────────────────────────────────────────
    section("OpenAI — chat completions", () {
        auto r = client.chatCompletions(model, [Message("user", "Hello!")], 50, 0.7f);
        writefln("  %s  (model: %s)",
            r["choices"][0]["message"]["content"].str, r["model"].str);
    });

    section("OpenAI — text completions", () {
        auto r = client.completions(model, "Once upon a time", 60, 0.9f);
        writefln("  %s  (model: %s)", r["choices"][0]["text"].str, r["model"].str);
    });

    section("OpenAI — list models", () {
        writeln(client.getModels());
    });

    // ── Streaming ─────────────────────────────────────────────────────────────
    section("Stream — generate", () {
        write("  ");
        client.generateStream(model, "Count from 1 to 5, one per line.",
            (JSONValue chunk) @safe {
                if (!chunk["done"].get!bool) write(chunk["response"].str);
                else writeln("\n  [done]");
            });
    });

    section("Stream — chat", () {
        write("  ");
        client.chatStream(model, [Message("user", "What is 2+2? Be brief.")],
            (JSONValue chunk) @safe {
                if (!chunk["done"].get!bool) write(chunk["message"]["content"].str);
                else writeln("\n  [done]");
            });
    });
}
