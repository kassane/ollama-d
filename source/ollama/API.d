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
import std.exception : enforce;

// Chat message structure
struct Message {
    string role;
    string content;

    Json toJson() const {
        auto j = Json.emptyObject;
        j["role"] = role;
        j["content"] = content;
        return j;
    }
}

class OllamaClient {
    private string host;
    private Duration timeout = 60.seconds;

    this(string host = DEFAULT_HOST) {
        this.host = host;
    }

    /// Set the timeout for HTTP requests
    void setTimeout(Duration timeout) {
        this.timeout = timeout;
    }

    /// Helper method for HTTP POST requests
    private Json post(string url, Json data, bool stream = false) {
        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;

        scope HTTPClientResponse response;
        try {
            response = requestHTTP(url, (scope req) {
                req.method = HTTPMethod.POST;
                req.headers["Content-Type"] = "application/json";
                req.writeJsonBody(data);
            }, settings);

            enforce(response.statusCode == 200, "HTTP request failed: " ~ response.statusPhrase ~ " (" ~ response.statusCode.to!string ~ ")");
            if (stream) {
                response.dropBody();
                return Json.emptyObject; // Placeholder for streaming
            }
            return response.readJson();
        } catch (Exception e) {
            throw e;
        }
    }

    /// Helper method for HTTP GET requests
    private Json get(string url) {
        auto settings = new HTTPClientSettings();
        settings.connectTimeout = timeout;
        settings.readTimeout = timeout;

        scope HTTPClientResponse response = requestHTTP(url, (scope req) {
            req.method = HTTPMethod.GET;
        }, settings);

        enforce(response.statusCode == 200, "HTTP request failed: " ~ response.statusPhrase);
        return response.readJson();
    }

    /// Generate text completion (non-streaming)
    Json generate(string model, string prompt, Json options = Json.emptyObject, bool stream = false) {
        auto url = host ~ "/api/generate";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["prompt"] = prompt;
        data["options"] = options;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /// Chat interaction (non-streaming)
    Json chat(string model, Message[] messages, Json options = Json.emptyObject, bool stream = false) {
        auto url = host ~ "/api/chat";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["messages"] = Json.emptyArray;
        foreach (msg; messages) {
            data["messages"] ~= msg.toJson();
        }
        data["options"] = options;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /// List available models
    Json listModels() {
        auto url = host ~ "/api/tags";
        return get(url);
    }

    /// Show model information
    Json showModel(string model) {
        auto url = host ~ "/api/show";
        auto data = Json.emptyObject;
        data["name"] = model;
        return post(url, data);
    }

    /// Create a new model (not used in main, but included for completeness)
    Json createModel(string name, string modelfile) {
        auto url = host ~ "/api/create";
        auto data = Json.emptyObject;
        data["name"] = name;
        data["modelfile"] = modelfile;
        return post(url, data);
    }

    /// OpenAI-style chat completions
    Json chatCompletions(string model, Message[] messages, int maxTokens = 0, float temperature = 1.0, bool stream = false) {
        auto url = host ~ "/v1/chat/completions";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["messages"] = Json.emptyArray;
        foreach (msg; messages) {
            data["messages"] ~= msg.toJson();
        }
        if (maxTokens > 0) data["max_tokens"] = maxTokens;
        data["temperature"] = temperature;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /// OpenAI-style text completions
    Json completions(string model, string prompt, int maxTokens = 0, float temperature = 1.0, bool stream = false) {
        auto url = host ~ "/v1/completions";
        auto data = Json.emptyObject;
        data["model"] = model;
        data["prompt"] = prompt;
        if (maxTokens > 0) data["max_tokens"] = maxTokens;
        data["temperature"] = temperature;
        data["stream"] = stream;
        return post(url, data, stream);
    }

    /// OpenAI-style model listing
    Json getModels() {
        auto url = host ~ "/v1/models";
        return get(url);
    }
}

enum DEFAULT_HOST = "http://127.0.0.1:11434";