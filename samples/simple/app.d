/++
 + Example application demonstrating the full `OllamaClient` SDK.
 +
 + This module showcases all client capabilities including text generation, chat,
 + embeddings, tool calling, structured output, model management, and OpenAI-compatible
 + endpoints.
 +
 + Prerequisites:
 +     - Ollama server running on `http://127.0.0.1:11434` (start with `ollama serve`).
 +     - Model "llama3.1:8b" installed (`ollama pull llama3.1:8b`).
 +/

import ollama;
import std.stdio;
import core.time;

void main() @safe
{
    auto client = new OllamaClient();
    writeln("\n=== Server Info ===");
    writeln("Ollama Client initialized with host: ", DEFAULT_HOST);

    // Custom timeout
    client.setTimeOut(120.seconds);

    // -------------------------------------------------------------------------
    // 1. Server version
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Server Version ===");
        writeln("Version: ", client.getVersion());
    }
    catch (Exception e) { writeln("Exception in getVersion: ", e.msg); }

    // -------------------------------------------------------------------------
    // 2. List available models
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== List Models ===");
        writeln(client.listModels());
    }
    catch (Exception e) { writeln("Exception in listModels: ", e.msg); }

    // -------------------------------------------------------------------------
    // 3. Running models (ps)
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Running Models (ps) ===");
        writeln(client.ps());
    }
    catch (Exception e) { writeln("Exception in ps: ", e.msg); }

    // -------------------------------------------------------------------------
    // 4. Pull a model
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Pull Model ===");
        auto r = client.pull("llama3.1:8b");
        writeln("Pull status: ", r.toPrettyString());
    }
    catch (Exception e) { writeln("Exception in pull: ", e.msg); }

    // -------------------------------------------------------------------------
    // 5. Show model info
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Show Model Info ===");
        writeln(client.showModel("llama3.1:8b"));
    }
    catch (Exception e) { writeln("Exception in showModel: ", e.msg); }

    // -------------------------------------------------------------------------
    // 6. Text generation (basic)
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Generate Text ===");
        auto r = client.generate("llama3.1:8b", "Why is the sky blue?");
        writeln("Response: ", r["response"].str);
        writeln("Done: ", r["done"].get!bool);
    }
    catch (Exception e) { writeln("Exception in generate: ", e.msg); }

    // -------------------------------------------------------------------------
    // 7. Text generation with system prompt and typed options
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Generate with System Prompt & Typed Options ===");
        OllamaOptions opts;
        opts.temperature = 0.5f;
        opts.num_predict = 80;
        auto r = client.generate(
            "llama3.1:8b",
            "Give me a one-sentence fun fact about D programming.",
            JSONValue.init,   // options (raw, unused — opts takes precedence)
            false,            // stream
            "You are a concise technical writer. Reply in one sentence.", // system
            null,             // images
            JSONValue.init,   // format
            null,             // suffix
            null,             // keep_alive
            opts,
        );
        writeln("Response: ", r["response"].str);
    }
    catch (Exception e) { writeln("Exception in generate (with opts): ", e.msg); }

    // -------------------------------------------------------------------------
    // 8. Structured output (JSON schema format)
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Structured Output (JSON format) ===");
        auto r = client.generate(
            "llama3.1:8b",
            "Return the capital city and population of France as JSON.",
            JSONValue.init, false, null, null,
            JSONValue("json"), // format = "json" forces JSON output
        );
        writeln("JSON Response: ", r["response"].str);
    }
    catch (Exception e) { writeln("Exception in generate (structured): ", e.msg); }

    // -------------------------------------------------------------------------
    // 9. Chat interaction (basic)
    // -------------------------------------------------------------------------
    Message[] messages = [Message("user", "Hello, how are you?")];
    try
    {
        writeln("\n=== Chat Interaction ===");
        auto r = client.chat("llama3.1:8b", messages);
        writeln("Response: ", r["message"]["content"].str);
        writeln("Done: ", r["done"].get!bool);
    }
    catch (Exception e) { writeln("Exception in chat: ", e.msg); }

    // -------------------------------------------------------------------------
    // 10. Chat with tool calling
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Chat with Tool Calling ===");
        import std.json : parseJSON;
        auto schema = parseJSON(`{
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }`);
        auto tools = [Tool("function",
            ToolFunction("get_current_weather", "Get the current weather for a location", schema))];

        auto toolMessages = [Message("user", "What is the weather in Paris in celsius?")];
        auto r = client.chat("llama3.1:8b", toolMessages, JSONValue.init, false, tools);

        if ("tool_calls" in r["message"].object)
        {
            writeln("Model wants to call tools:");
            foreach (tc; r["message"]["tool_calls"].array)
                writeln("  Function: ", tc["function"]["name"].str,
                        " Args: ", tc["function"]["arguments"].toString());
        }
        else
        {
            writeln("Response: ", r["message"]["content"].str);
        }
    }
    catch (Exception e) { writeln("Exception in chat (tools): ", e.msg); }

    // -------------------------------------------------------------------------
    // 11. Embeddings — single input
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Embeddings (single) ===");
        auto r = client.embed("llama3.1:8b", "The quick brown fox jumps over the lazy dog.");
        auto vecs = r["embeddings"].array;
        writeln("Number of embedding vectors: ", vecs.length);
        if (vecs.length > 0)
            writeln("First vector length: ", vecs[0].array.length);
    }
    catch (Exception e) { writeln("Exception in embed (single): ", e.msg); }

    // -------------------------------------------------------------------------
    // 12. Embeddings — batch input
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Embeddings (batch) ===");
        auto r = client.embed("llama3.1:8b", ["Hello world", "D is great", "Ollama rocks"]);
        writeln("Batch embedding count: ", r["embeddings"].array.length);
    }
    catch (Exception e) { writeln("Exception in embed (batch): ", e.msg); }

    // -------------------------------------------------------------------------
    // 13. Push a model (may fail without registry auth)
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== Push Model ===");
        auto r = client.push("llama3.1:8b");
        writeln("Push Status: ", r.toPrettyString());
    }
    catch (Exception e) { writeln("Exception in push (expected without auth): ", e.msg); }

    // -------------------------------------------------------------------------
    // 14. OpenAI-compatible chat completions
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== OpenAI Chat Completions ===");
        auto r = client.chatCompletions("llama3.1:8b", messages, 50, 0.7f);
        writeln("Choice: ", r["choices"][0]["message"]["content"].str);
        writeln("Model: ",  r["model"].str);
    }
    catch (Exception e) { writeln("Exception in chatCompletions: ", e.msg); }

    // -------------------------------------------------------------------------
    // 15. OpenAI-compatible text completions
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== OpenAI Text Completions ===");
        auto r = client.completions("llama3.1:8b", "Once upon a time", 60, 0.9f);
        writeln("Text:  ", r["choices"][0]["text"].str);
        writeln("Model: ", r["model"].str);
    }
    catch (Exception e) { writeln("Exception in completions: ", e.msg); }

    // -------------------------------------------------------------------------
    // 16. OpenAI-compatible model listing
    // -------------------------------------------------------------------------
    try
    {
        writeln("\n=== OpenAI List Models ===");
        writeln(client.getModels());
    }
    catch (Exception e) { writeln("Exception in getModels: ", e.msg); }
}
