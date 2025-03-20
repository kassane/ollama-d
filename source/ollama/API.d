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
 + native Ollama endpoints (e.g., `/api/generate`) and OpenAI-compatible endpoints (e.g., `/v1/chat/completions`).
 +
 + The client leverages the `vibe.d` library for HTTP requests and JSON handling, providing a robust
 + interface for integrating large language models into D applications.
 +
 + Examples:
 +     ---
 +     import ollama.client;
 +     import std.stdio;
 +
 +     void main() {
 +         auto client = new OllamaClient();
 +         auto response = client.generate("llama3", "What is the weather like?");
 +         writeln(response["response"].get!string);
 +     }
 +     ---
 +
 + See_Also:
 +     - $(LINK2 https://github.com/ollama/ollama/blob/main/docs/api.md, Ollama API Documentation)
 +     - $(LINK2 https://github.com/ollama/ollama/blob/main/docs/openai.md, OpenAI Compatibility)
 +/
module ollama.client;

import vibe.d;
import std.exception : enforce;

/++
 + Represents a single message in a chat interaction.
 +
 + This struct is used to structure chat messages with a role (e.g., "user" or "assistant") and content.
 + It provides a method to convert the message into a JSON format compatible with Ollama’s API.
 +/
struct Message
{
    string role; /// The role of the message sender (e.g., "user", "assistant").
    string content; /// The text content of the message.

    /++
     + Converts the message to a JSON object.
     +
     + Returns: A `Json` object with "role" and "content" fields.
     +
     + Examples:
     +     ---
     +     auto msg = Message("user", "Hello!");
     +     auto json = msg.toJson();
     +     assert(json["role"].get!string == "user");
     +     assert(json["content"].get!string == "Hello!");
     +     ---
     +/
    Json toJson() const
    {
        auto j = Json.emptyObject;
        j["role"] = role;
        j["content"] = content;
        return j;
    }
}

/++
 + A client class for interacting with the Ollama REST API.
 +
 + This class provides a comprehensive interface to Ollama’s functionality, including text generation,
 + chat interactions, and model management. It supports both Ollama-specific endpoints and OpenAI-compatible
 + endpoints, with configurable timeouts and basic error handling.
 +
 + Note: Streaming responses are partially supported; full implementation requires additional handling
 + of the response body.
 +
 + Examples:
 +     ---
 +     auto client = new OllamaClient();
 +     auto chatResponse = client.chat("llama3", [Message("user", "Hi there!")]);
 +     writeln(chatResponse["message"]["content"].get!string);
 +     ---
 +/
class OllamaClient
{
    private string host; /// The base URL of the Ollama server (e.g., "http://127.0.0.1:11434").
    private Duration timeout = 60.seconds; /// Default timeout for HTTP requests in seconds.

    /++
     + Constructs a new Ollama client instance.
     +
     + Params:
     +     host = The base URL of the Ollama server. Defaults to `DEFAULT_HOST` if not specified.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient("http://localhost:11434");
     +     ---
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
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     client.setTimeout(30.seconds);
     +     ---
     +/
    void setTimeout(Duration timeout)
    {
        this.timeout = timeout;
    }

    /++ 
     + Private helper method for performing HTTP POST requests.
     +
     + Params:
     +     url = The endpoint URL to send the request to.
     +     data = The JSON data to send in the request body.
     +     stream = Whether to request a streaming response (not fully implemented).
     +
     + Returns: A `Json` object with the response, or an empty object for streaming.
     +/
    private Json post(string url, Json data, bool stream = false)
    {
        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;

        scope HTTPClientResponse response;
        try
        {
            response = requestHTTP(url, (scope req) {
                req.method = HTTPMethod.POST;
                req.headers["Content-Type"] = "application/json";
                req.writeJsonBody(data);
            }, settings);

            enforce(response.statusCode == 200, "HTTP request failed: " ~ response.statusPhrase ~ " (" ~ response
                    .statusCode.to!string ~ ")");
            if (stream)
            {
                response.dropBody();
                return Json.emptyObject; // Placeholder for streaming
            }
            return response.readJson();
        }
        catch (Exception e)
        {
            throw e;
        }
    }

    /++ 
     + Private helper method for performing HTTP GET requests.
     +
     + Params:
     +     url = The endpoint URL to send the request to*.
     +
     + Returns: A `Json` object with the response.
     +/
    private Json get(string url)
    {
        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;

        scope HTTPClientResponse response = requestHTTP(url, (scope req) {
            req.method = HTTPMethod.GET;
        }, settings);

        enforce(response.statusCode == 200, "HTTP request failed: " ~ response.statusPhrase);
        return response.readJson();
    }

    /++
     + Generates text based on a prompt using the specified model.
     +
     + This method calls the `/api/generate` endpoint for text completion.
     +
     + Params:
     +     model = The name of the model to use (e.g., "llama3").
     +     prompt = The input text to generate from.
     +     options = Additional generation options (e.g., temperature, top_k).
     +     stream = Whether to stream the response (not fully implemented).
     +
     + Returns: A `Json` object containing the generated text and metadata.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto response = client.generate("llama3", "Tell me a story");
     +     writeln(response["response"].get!string);
     +     ---
     +/
    Json generate(string model, string prompt, Json options = Json.emptyObject, bool stream = false)
    {
        auto url = host ~ "/api/generate";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["prompt"] = prompt;
        data["options"] = options;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /++
     + Engages in a chat interaction using the specified model and message history.
     +
     + This method calls the `/api/chat` endpoint for conversational responses.
     +
     + Params:
     +     model = The name of the model to use.
     +     messages = An array of `Message` structs representing the chat history.
     +     options = Additional chat options (e.g., temperature).
     +     stream = Whether to stream the response (not fully implemented).
     +
     + Returns: A `Json` object containing the chat response and metadata.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     Message[] msgs = [Message("user", "Hi!")];
     +     auto response = client.chat("llama3", msgs);
     +     writeln(response["message"]["content"].get!string);
     +     ---
     +/
    Json chat(string model, Message[] messages, Json options = Json.emptyObject, bool stream = false)
    {
        auto url = host ~ "/api/chat";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["messages"] = Json.emptyArray;
        foreach (msg; messages)
        {
            data["messages"] ~= msg.toJson();
        }
        data["options"] = options;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /++
     + Retrieves a list of available models from the Ollama server.
     +
     + This method calls the `/api/tags` endpoint.
     +
     + Returns: A `Json` object containing an array of model details.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto models = client.listModels();
     +     writeln(models["models"].toString());
     +     ---
     +/
    Json listModels()
    {
        auto url = host ~ "/api/tags";
        return get(url);
    }

    /++
     + Retrieves detailed information about a specific model.
     +
     + This method calls the `/api/show` endpoint.
     +
     + Params:
     +     model = The name of the model to query.
     +
     + Returns: A `Json` object with model metadata.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto info = client.showModel("llama3");
     +     writeln(info.toString());
     +     ---
     +/
    Json showModel(string model)
    {
        auto url = host ~ "/api/show";
        auto data = Json.emptyObject;
        data["name"] = model;
        return post(url, data);
    }

    /++
     + Creates a new model on the Ollama server using a modelfile.
     +
     + This method calls the `/api/create` endpoint.
     +
     + Params:
     +     name = The name of the new model.
     +     modelfile = The modelfile content defining the model.
     +
     + Returns: A `Json` object with creation status.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto result = client.createModel("myModel", "FROM llama3");
     +     writeln(result.toString());
     +     ---
     +/
    Json createModel(string name, string modelfile)
    {
        auto url = host ~ "/api/create";
        auto data = Json.emptyObject;
        data["name"] = name;
        data["modelfile"] = modelfile;
        return post(url, data);
    }

    /++
     + Performs an OpenAI-style chat completion.
     +
     + This method calls the `/v1/chat/completions` endpoint, mimicking OpenAI’s API.
     +
     + Params:
     +     model = The name of the model to use.
     +     messages = An array of `Message` structs representing the chat history.
     +     maxTokens = Maximum number of tokens to generate (0 for unlimited).
     +     temperature = Sampling temperature (default: 1.0).
     +     stream = Whether to stream the response (not fully implemented).
     +
     + Returns: A `Json` object in OpenAI-compatible format.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     Message[] msgs = [Message("user", "What’s the weather?")];
     +     auto response = client.chatCompletions("llama3", msgs, 50, 0.7);
     +     writeln(response["choices"][0]["message"]["content"].get!string);
     +     ---
     +/
    Json chatCompletions(string model, Message[] messages, int maxTokens = 0, float temperature = 1.0, bool stream = false)
    {
        auto url = host ~ "/v1/chat/completions";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["messages"] = Json.emptyArray;
        foreach (msg; messages)
        {
            data["messages"] ~= msg.toJson();
        }
        if (maxTokens > 0)
            data["max_tokens"] = maxTokens;
        data["temperature"] = temperature;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /++
     + Performs an OpenAI-style text completion.
     +
     + This method calls the `/v1/completions` endpoint, mimicking OpenAI’s API.
     +
     + Params:
     +     model = The name of the model to use.
     +     prompt = The input prompt to complete.
     +     maxTokens = Maximum number of tokens to generate (0 for unlimited).
     +     temperature = Sampling temperature (default: 1.0).
     +     stream = Whether to stream the response (not fully implemented).
     +
     + Returns: A `Json` object in OpenAI-compatible format.
     ~
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto response = client.completions("llama3", "Once upon a time", 100, 0.9);
     +     writeln(response["choices"][0]["text"].get!string);
     +     ---
     +/
    Json completions(string model, string prompt, int maxTokens = 0, float temperature = 1.0, bool stream = false)
    {
        auto url = host ~ "/v1/completions";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["prompt"] = prompt;
        if (maxTokens > 0)
            data["max_tokens"] = maxTokens;
        data["temperature"] = temperature;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /++
     + Lists models in an OpenAI-compatible format.
     +
     + This method calls the `/v1/models` endpoint.
     +
     + Returns: A `Json` object with model data in OpenAI style.
     +
     + Examples:
     +     ---
     +     auto client = new OllamaClient();
     +     auto models = client.getModels();
     +     writeln(models["data"].toString());
     +     ---
     +/
    Json getModels()
    {
        auto url = host ~ "/v1/models";
        return get(url);
    }
}

/// Default host URL for the Ollama server (http://127.0.0.1:11434).
enum DEFAULT_HOST = "http://127.0.0.1:11434";
