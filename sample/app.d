import ollama;
import std.stdio;
import vibe.d; // For Json and HTTP-related types

// Sample example demonstrating the full feature set of the Ollama Dlang library
// Prerequisites:
// - Ollama server must be running on http://127.0.0.1:11434
// - Models like "llama3.1:8b" should be installed (e.g., via `ollama pull llama3.1:8b`)

void main()
{
    // Initialize the Ollama client with default host
    auto client = new OllamaClient();
    // writeln("Ollama Client initialized with host: ", client.DEFAULT_HOST);

    // Set a custom timeout (optional)
    client.setTimeout(30.seconds);

    // --- Ollama-Specific Endpoints ---

    // 1. Generate text completion (non-streaming)
    try
    {
        writeln("\n=== Generate Text (Non-Streaming) ===");
        auto generateResponse = client.generate("llama3.1:8b", "Why is the sky blue?");
        if ("error" in generateResponse)
        {
            writeln("Error: ", generateResponse["error"].get!string);
        }
        else
        {
            writeln("Response: ", generateResponse["response"].get!string);
            writeln("Done: ", generateResponse["done"].get!bool);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in generate: ", e.msg);
    }

    // 2. Chat interaction (non-streaming)
    Message[] messages = [Message("user", "Hello, how are you?")];
    try
    {
        writeln("\n=== Chat Interaction (Non-Streaming) ===");
        auto chatResponse = client.chat("llama3.1:8b", messages);
        if ("error" in chatResponse)
        {
            writeln("Error: ", chatResponse["error"].get!string);
        }
        else
        {
            writeln("Response: ", chatResponse["message"]["content"].get!string);
            writeln("Done: ", chatResponse["done"].get!bool);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in chat: ", e.msg);
    }

    // 3. List available models
    try
    {
        writeln("\n=== List Models ===");
        auto models = client.listModels();
        writeln("Models: ", models["models"].toString());
    }
    catch (Exception e)
    {
        writeln("Exception in listModels: ", e.msg);
    }

    // 4. Show model information
    try
    {
        writeln("\n=== Show Model Info ===");
        auto modelInfo = client.showModel("llama3.1:8b");
        writeln("Model Info: ", modelInfo.toString());
    }
    catch (Exception e)
    {
        writeln("Exception in showModel: ", e.msg);
    }

    // --- OpenAI-Compatible Endpoints ---

    // 5. OpenAI-style chat completions (non-streaming)
    try
    {
        writeln("\n=== OpenAI Chat Completions (Non-Streaming) ===");
        auto chatCompResponse = client.chatCompletions("llama3.1:8b", messages, 50, 0.7);
        if ("error" in chatCompResponse)
        {
            writeln("Error: ", chatCompResponse["error"].get!string);
        }
        else
        {
            writeln("Choice: ", chatCompResponse["choices"][0]["message"]["content"].get!string);
            writeln("Model: ", chatCompResponse["model"].get!string);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in chatCompletions: ", e.msg);
    }

    // 6. OpenAI-style text completions (non-streaming)
    try
    {
        writeln("\n=== OpenAI Text Completions (Non-Streaming) ===");
        auto compResponse = client.completions("llama3.1:8b", "Once upon a time", 100, 0.9);
        if ("error" in compResponse)
        {
            writeln("Error: ", compResponse["error"].get!string);
        }
        else
        {
            writeln("Text: ", compResponse["choices"][0]["text"].get!string);
            writeln("Model: ", compResponse["model"].get!string);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in completions: ", e.msg);
    }

    // 7. OpenAI-style model listing
    try
    {
        writeln("\n=== OpenAI List Models ===");
        auto openaiModels = client.getModels();
        writeln("Models: ", openaiModels["data"].toString());
    }
    catch (Exception e)
    {
        writeln("Exception in getModels: ", e.msg);
    }

    // --- Streaming Example (Placeholder) ---
    // Note: Full streaming requires additional implementation
    try
    {
        writeln("\n=== Generate Text (Streaming) ===");
        auto streamResponse = client.generate("llama3.1:8b", "Tell me a story", Json.emptyObject, true);
        writeln("Streaming not fully implemented; response: ", streamResponse.toString());
    }
    catch (Exception e)
    {
        writeln("Exception in generate (streaming): ", e.msg);
    }
}
