/**
 * MIT License
 * 
 * Copyright (c) 2025 Matheus C. França
 * 
 * Permission is granted to use, modify, and distribute this software
 * under the terms of the MIT License.
 */

module ollama.client;
import vibe.d;

// chat messages
struct Message
{
    string role;
    string content;

    Json toJson() const
    {
        auto j = Json.emptyObject;
        j["role"] = role;
        j["content"] = content;
        return j;
    }
}

class OllamaClient
{
    private string host;
    private Duration timeout = 60.seconds;

    this(string host = DEFAULT_HOST)
    {
        this.host = host;
    }

    /// Set the timeout for HTTP requests
    void setTimeout(Duration timeout)
    {
        this.timeout = timeout;
    }

    /// Generate text completion (non-streaming)
    Json generate(string model, string prompt, Json options = Json.emptyObject)
    {
        auto url = host ~ "/api/generate";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["prompt"] = prompt;
        data["options"] = options;
        data["stream"] = false; // Disable streaming

        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;

        auto response = requestHTTP(url, (scope req) {
            req.method = HTTPMethod.POST;
            req.headers["Content-Type"] = "application/json";
            req.writeJsonBody(data);
        }, settings);

        auto jsonResponse = response.readJson();
        return jsonResponse;
    }

    /// Chat interaction (non-streaming)
    Json chat(string model, Message[] messages, Json options = Json.emptyObject)
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
        data["stream"] = false; // Disable streaming

        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;
        auto response = requestHTTP(url, (scope req) {
            req.method = HTTPMethod.POST;
            req.headers["Content-Type"] = "application/json";
            req.writeJsonBody(data);
        }, settings);
        auto jsonResponse = response.readJson();
        return jsonResponse;
    }
}

enum  DEFAULT_HOST = "http://127.0.0.1:11434";