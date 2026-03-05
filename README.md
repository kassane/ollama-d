# ollama-d

[![Static Badge](https://img.shields.io/badge/v2.110.0%20(stable)-f8240e?logo=d&logoColor=f8240e&label=runtime)](https://dlang.org/download.html)
![Latest release](https://img.shields.io/github/v/release/kassane/ollama-d?include_prereleases&label=latest)
[![Artifacts](https://github.com/kassane/ollama-d/actions/workflows/ci.yml/badge.svg)](https://github.com/kassane/ollama-d/actions/workflows/ci.yml)

D language bindings for the Ollama REST API — seamless integration with local AI models.

## Features

- Text generation with native Ollama API
- Chat interactions with conversation history
- **Embeddings** — single and batch (`/api/embed`)
- **Tool calling** — pass function definitions to `chat()`
- **Multimodal** — base64 image input for vision models
- **Structured output** — JSON schema `format` enforcement
- **Typed `OllamaOptions`** — temperature, top_k, num_ctx, stop sequences, and more
- Model management — list, create, show, pull, push, copy, delete
- Running models inspection (`/api/ps`)
- Configurable timeout settings
- Built-in unit test suite (`dub test`)
- OpenAI-compatible API endpoints (`/v1/…`)
- Docker-based local development environment
- Zero external dependencies — only `std.net.curl` and `std.json`

## Prerequisites

- [D compiler](https://dlang.org/download.html) (v2.110.0 stable or compatible)
- Ollama server running locally (default: `http://127.0.0.1:11434`)
- An installed model (e.g. `ollama pull llama3.2`)

## Quick Start

```d
import ollama;
import std.stdio;

void main() {
    auto client = new OllamaClient();

    // Text generation
    auto gen = client.generate("llama3.2", "Why is the sky blue?");
    writeln(gen["response"].str);

    // Chat
    auto resp = client.chat("llama3.2", [Message("user", "Hello!")]);
    writeln(resp["message"]["content"].str);

    // Embeddings
    auto emb = client.embed("nomic-embed-text", "The quick brown fox");
    writeln("Vector length: ", emb["embeddings"][0].array.length);

    // Server version
    writeln("Ollama: ", client.getVersion());
}
```

## API Reference

### Generation

```d
// Basic generation
JSONValue generate(string model, string prompt,
    JSONValue options = JSONValue.init, bool stream = false,
    string system = null, string[] images = null,
    JSONValue format = JSONValue.init, string suffix = null,
    string keepAlive = null, OllamaOptions opts = OllamaOptions.init)

// With typed options and system prompt
OllamaOptions opts;
opts.temperature = 0.7f;
opts.num_ctx     = 4096;
opts.stop        = ["<|end|>"];
auto r = client.generate("llama3.2", "Hello",
    JSONValue.init, false, "You are a helpful assistant.",
    null, JSONValue.init, null, null, opts);
writeln(r["response"].str);

// Structured JSON output
auto r = client.generate("llama3.2", "Capital of France as JSON",
    JSONValue.init, false, null, null, JSONValue("json"));
```

### Chat

```d
// Basic chat
JSONValue chat(string model, Message[] messages,
    JSONValue options = JSONValue.init, bool stream = false,
    Tool[] tools = null, JSONValue format = JSONValue.init,
    string keepAlive = null, OllamaOptions opts = OllamaOptions.init)

// Tool calling
import std.json : parseJSON;
auto schema = parseJSON(`{
    "type": "object",
    "properties": {"location": {"type": "string"}},
    "required": ["location"]
}`);
auto tools = [Tool("function",
    ToolFunction("get_weather", "Get current weather", schema))];
auto r = client.chat("llama3.2",
    [Message("user", "Weather in Paris?")],
    JSONValue.init, false, tools);
// Check r["message"]["tool_calls"] for model's tool call request
```

### Embeddings

```d
// Single text
JSONValue embed(string model, string input, string keepAlive = null)

// Batch
JSONValue embed(string model, string[] inputs, string keepAlive = null)

auto r = client.embed("nomic-embed-text", "Hello, world!");
writeln(r["embeddings"][0].array.length); // vector dimension

auto batch = client.embed("nomic-embed-text", ["text one", "text two"]);
writeln(batch["embeddings"].array.length); // 2 vectors
```

### Model Management

```d
string   listModels()                            // GET  /api/tags
string   showModel(string model)                 // POST /api/show
JSONValue createModel(string name, string file)  // POST /api/create
JSONValue copy(string src, string dst)           // POST /api/copy
JSONValue deleteModel(string name)               // DELETE /api/delete
JSONValue pull(string name, bool stream = false) // POST /api/pull
JSONValue push(string name, bool stream = false) // POST /api/push
```

### Server

```d
string getVersion()  // GET /api/version  → "0.6.2"
string ps()          // GET /api/ps       → running models JSON
```

### OpenAI-Compatible

```d
JSONValue chatCompletions(string model, Message[] messages,
    int maxTokens = 0, float temperature = 1.0, bool stream = false)

JSONValue completions(string model, string prompt,
    int maxTokens = 0, float temperature = 1.0, bool stream = false)

string getModels()  // GET /v1/models
```

### OllamaOptions

```d
OllamaOptions opts;
opts.temperature   = 0.8f;   // Creativity (0.0 = deterministic)
opts.top_k         = 40;     // Top-K sampling
opts.top_p         = 0.9f;   // Nucleus sampling
opts.repeat_penalty = 1.1f;  // Penalize repeated tokens
opts.num_predict   = 200;    // Max tokens to generate
opts.num_ctx       = 8192;   // Context window size
opts.seed          = 42;     // Reproducible output
opts.stop          = ["</s>", "\n\n"];  // Stop sequences
```

## Running Tests

```bash
# Unit tests (no Ollama server required)
dub test

# Build and run samples against a live server
dub build -b release
dub run -b release :simple
dub run -b release :coder -- --prompt "Sort a list in D" --model llama3.2
```

## Docker

```bash
# Unit tests only (no Ollama needed)
docker build --target builder -t ollama-d .
docker run --rm ollama-d dub test

# Full integration tests with Ollama
docker compose up --build

# Tear down
docker compose down -v
```

## License

MIT License
