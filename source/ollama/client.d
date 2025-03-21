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
 + for tasks such as text generation, chat interactions, and model management. It supports both
 + native Ollama endpoints and OpenAI-compatible endpoints, using `std.net.curl` for HTTP requests
 + and `std.json` for JSON processing.
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
 + Represents a single message in a chat interaction.
 +/
struct Message
{
    string role; /// The role of the message sender (e.g., "user", "assistant").
    string content; /// The text content of the message.

    /++
     + Converts the message to a JSON object.
     +
     + Returns: A `JSONValue` object with "role" and "content" fields.
     +/
    JSONValue toJson() const
    {
        JSONValue j = ["role": JSONValue(role), "content": JSONValue(content)];
        return j;
    }
}

/++
 + A client class for interacting with the Ollama REST API.
 +
 + This class provides methods for text generation, chat interactions, and model management using
 + `std.net.curl` for HTTP requests and `std.json` for JSON handling. Streaming is not fully supported
 + in this version due to limitations in `std.net.curl`.
 +
 + Examples:
 +     ---
 +     auto client = new OllamaClient();
 +     auto chatResponse = client.chat("llama3", [Message("user", "Hi there!")]);
 +     writeln(chatResponse["message"]["content"].str);
 +     ---
 +/
class OllamaClient
{
    private string host; /// The base URL of the Ollama server.
    private Duration timeout = 60.seconds; /// Default timeout for HTTP requests.

    /++
     + Constructs a new Ollama client instance.
     +
     + Params:
     +     host = The base URL of the Ollama server. Defaults to `DEFAULT_HOST` if not specified.
     +/
    this(string host = DEFAULT_HOST)
    {
        this.host = host;
    }

    /++
     + Sets the timeout duration for HTTP requests.
     +
     + Params:
     +     timeout = The duration to wait before timing out HTTP requests.
     +/
    void setTimeOut(Duration timeout)
    {
        this.timeout = timeout;
    }

    /++
     + Private helper method for performing HTTP POST requests.
     +
     + Params:
     +     url = The endpoint URL to send the request to.
     +     data = The JSON data to send in the request body.
     +     stream = Whether to request a streaming response (ignored in this implementation).
     +
     + Returns: A `JSONValue` object with the response.
     +/
    private JSONValue post(string url, JSONValue data, bool stream = false) @trusted
    {
        auto client = HTTP();
        client.addRequestHeader("Content-Type", "application/json");
        client.connectTimeout(timeout);

        auto jsonStr = data.toString();
        auto response = std.net.curl.post(url, jsonStr, client);
        auto jsonResponse = parseJSON(response);

        enforce("error" !in jsonResponse, "HTTP request failed: " ~ ("message" in jsonResponse["error"] ? jsonResponse["error"]["message"]
                .str : "Unknown error"));
        return jsonResponse;
    }

    /++
     + Private helper method for performing HTTP GET requests.
     +
     + Params:
     +     url = The endpoint URL to send the request to.
     +
     + Returns: A `JSONValue` object with the response.
     +/
    private JSONValue get(string url) @trusted
    {
        auto client = HTTP();
        client.connectTimeout(timeout);

        auto response = std.net.curl.get(url, client);
        auto jsonResponse = parseJSON(response);
        enforce("error" !in jsonResponse, "HTTP request failed: " ~ ("message" in jsonResponse["error"] ? jsonResponse["error"]["message"]
                .str : "Unknown error"));
        return jsonResponse;
    }

    /++
     + Generates text based on a prompt using the specified model.
     +
     + Params:
     +     model = The name of the model to use (e.g., "llama3").
     +     prompt = The input text to generate from.
     +     options = Additional generation options (e.g., temperature, top_k).
     +     stream = Whether to stream the response (ignored in this implementation).
     +
     + Returns: A `JSONValue` object containing the generated text and metadata.
     +/
    JSONValue generate(string model, string prompt, JSONValue options = JSONValue.init, bool stream = false)
    {
        auto url = host ~ "/api/generate";
        JSONValue data = [
            "model": JSONValue(model),
            "prompt": JSONValue(prompt),
            "options": options,
            "stream": JSONValue(stream)
        ];
        return post(url, data, stream);
    }

    /++
     + Engages in a chat interaction using the specified model and message history.
     +
     + Params:
     +     model = The name of the model to use.
     +     messages = An array of `Message` structs representing the chat history.
     +     options = Additional chat options (e.g., temperature).
     +     stream = Whether to stream the response (ignored in this implementation).
     +
     + Returns: A `JSONValue` object containing the chat response and metadata.
     +/
    JSONValue chat(string model, Message[] messages, JSONValue options = JSONValue.init, bool stream = false)
    {
        auto url = host ~ "/api/chat";
        JSONValue[] msgArray;
        foreach (msg; messages)
        {
            msgArray ~= msg.toJson();
        }
        JSONValue data = [
            "model": JSONValue(model),
            "messages": JSONValue(msgArray),
            "options": options,
            "stream": JSONValue(stream)
        ];
        return post(url, data, stream);
    }

    /++
     + Retrieves a list of available models from the Ollama server in a formatted JSON string.
     +
     + Returns: A string containing the JSON-formatted list of model details, pretty-printed.
     +/
    string listModels()
    {
        auto url = host ~ "/api/tags";
        auto jsonResponse = get(url);
        return jsonResponse.toPrettyString(); // Adicionado: retorna JSON formatado
    }

    /++
     + Retrieves detailed information about a specific model in a formatted JSON string.
     +
     + Params:
     +     model = The name of the model to query.
     +
     + Returns: A string containing the JSON-formatted model metadata, pretty-printed.
     +/
    string showModel(string model)
    {
        auto url = host ~ "/api/show";
        JSONValue data = ["name": JSONValue(model)];
        auto jsonResponse = post(url, data);
        return jsonResponse.toPrettyString(); // Adicionado: retorna JSON formatado
    }

    /++
     + Creates a new model on the Ollama server using a modelfile.
     +
     + Params:
     +     name = The name of the new model.
     +     modelfile = The modelfile content defining the model.
     +
     + Returns: A `JSONValue` object with creation status.
     +/
    JSONValue createModel(string name, string modelfile)
    {
        auto url = host ~ "/api/create";
        JSONValue data = [
            "name": JSONValue(name),
            "modelfile": JSONValue(modelfile)
        ];
        return post(url, data);
    }

    /++
     + Performs an OpenAI-style chat completion.
     +
     + Params:
     +     model = The name of the model to use.
     +     messages = An array of `Message` structs representing the chat history.
     +     maxTokens = Maximum number of tokens to generate (0 for unlimited).
     +     temperature = Sampling temperature (default: 1.0).
     +     stream = Whether to stream the response (ignored in this implementation).
     +
     + Returns: A `JSONValue` object in OpenAI-compatible format.
     +/
    JSONValue chatCompletions(string model, Message[] messages, int maxTokens = 0, float temperature = 1.0, bool stream = false) @trusted
    {
        auto url = host ~ "/v1/chat/completions";
        JSONValue[] msgArray;
        foreach (msg; messages)
        {
            msgArray ~= msg.toJson();
        }
        JSONValue data = [
            "model": JSONValue(model),
            "messages": JSONValue(msgArray),
            "stream": JSONValue(stream)
        ];
        if (maxTokens > 0)
            data.object["max_tokens"] = JSONValue(maxTokens);
        data.object["temperature"] = JSONValue(temperature);
        return post(url, data, stream);
    }

    /++
     + Performs an OpenAI-style text completion.
     +
     + Params:
     +     model = The name of the model to use.
     +     prompt = The input prompt to complete.
     +     maxTokens = Maximum number of tokens to generate (0 for unlimited).
     +     temperature = Sampling temperature (default: 1.0).
     +     stream = Whether to stream the response (ignored in this implementation).
     +
     + Returns: A `JSONValue` object in OpenAI-compatible format.
     +/
    JSONValue completions(string model, string prompt, int maxTokens = 0, float temperature = 1.0, bool stream = false) @trusted
    {
        auto url = host ~ "/v1/completions";
        JSONValue data = [
            "model": JSONValue(model),
            "prompt": JSONValue(prompt),
            "stream": JSONValue(stream)
        ];
        if (maxTokens > 0)
            data.object["max_tokens"] = JSONValue(maxTokens);
        data.object["temperature"] = JSONValue(temperature);
        return post(url, data, stream);
    }

    /++
     + Lists models in an OpenAI-compatible format.
     +
     + Returns: A `JSONValue` object with model data in OpenAI style.
     +/
    string getModels()
    {
        auto url = host ~ "/v1/models";
        return get(url).toPrettyString();
    }
}

/// Default host URL for the Ollama server.
enum DEFAULT_HOST = "http://127.0.0.1:11434";
