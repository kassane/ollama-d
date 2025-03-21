/++
 + Example application demonstrating the usage of the `OllamaClient` class.
 +
 + This module provides a comprehensive example of how to use the `OllamaClient` to interact with
 + an Ollama server, including text generation, chat interactions, model management, and OpenAI-compatible
 + endpoints.
 +
 + Prerequisites:
 +     - Ollama server must be running on `http://127.0.0.1:11434` (start with `ollama serve`).
 +     - The model "llama3.1:8b" must be installed (e.g., `ollama pull llama3.1:8b`).
 +/

import ollama;
import std.stdio; // for writeln
import core.time; // for seconds

void main() @safe
{
    // Initialize the Ollama client with default host
    auto client = new OllamaClient();
    writeln("Ollama Client initialized with host: ", DEFAULT_HOST);

    // Set a custom timeout (optional)
    client.setTimeOut(30.seconds);

    // --- Ollama-Specific Endpoints ---

    // 1. Generate text completion (non-streaming)
    try
    {
        writeln("\n=== Generate Text (Non-Streaming) ===");
        auto generateResponse = client.generate("llama3.1:8b", "Why is the sky blue?");
        if ("error" in generateResponse)
        {
            writeln("Error: ", generateResponse["error"].str);
        }
        else
        {
            writeln("Response: ", generateResponse["response"].str);
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
            writeln("Error: ", chatResponse["error"].str);
        }
        else
        {
            writeln("Response: ", chatResponse["message"]["content"].str);
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
        writeln("Models: ", models);
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
        writeln("Model Info: ", modelInfo);
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
            writeln("Error: ", chatCompResponse["error"].str);
        }
        else
        {
            writeln("Choice: ", chatCompResponse["choices"][0]["message"]["content"].str);
            writeln("Model: ", chatCompResponse["model"].str);
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
            writeln("Error: ", compResponse["error"].str);
        }
        else
        {
            writeln("Text: ", compResponse["choices"][0]["text"].str);
            writeln("Model: ", compResponse["model"].str);
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
        writeln("Models: ", openaiModels);
    }
    catch (Exception e)
    {
        writeln("Exception in getModels: ", e.msg);
    }
}
