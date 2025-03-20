# ollama-d

[![Static Badge](https://img.shields.io/badge/v2.110.0%20(stable)-f8240e?logo=d&logoColor=f8240e&label=runtime)](https://dlang.org/download.html)
![Latest release](https://img.shields.io/github/v/release/kassane/ollama-d?include_prereleases&label=latest)
[![Artifacts](https://github.com/kassane/ollama-d/actions/workflows/ci.yml/badge.svg)](https://github.com/kassane/ollama-d/actions/workflows/ci.yml)

D language bindings for the Ollama AI API - Seamless integration with local AI models

## Features

- Text generation with streaming and non-streaming API
- Chat interactions with local AI models
- OpenAI-compatible endpoints support
- Model management (list, create, show models)
- Configurable timeout settings
- Simple and intuitive API design

## Prerequisites

- Ollama server running locally (default: http://127.0.0.1:11434)
- Installed AI model (e.g., "llama3.2")
- Vibe.d library

## Quick Examples

```d
import ollama;
import std.stdio;

void main() {
    // Initialize Ollama client
    auto client = new OllamaClient();

    // Text generation
    auto generateResponse = client.generate("llama3.2", "Why is the sky blue?");
    writeln("Generate Response: ", generateResponse["response"].get!string);

    // Chat interaction
    Message[] messages = [Message("user", "Hello, how are you?")];
    auto chatResponse = client.chat("llama3.2", messages);
    writeln("Chat Response: ", chatResponse["message"]["content"].get!string);

    // List available models
    auto models = client.listModels();
    writeln("Available Models: ", models["models"].toString());

    // OpenAI-compatible chat completions
    auto openaiResponse = client.chatCompletions("llama3.2", messages, 50, 0.7);
    writeln("OpenAI-style Response: ", openaiResponse["choices"][0]["message"]["content"].get!string);
}
```

## Additional Methods

- `generate()`: Text generation with custom options
- `chat()`: Conversational interactions
- `listModels()`: Retrieve available models
- `showModel()`: Get detailed model information
- `createModel()`: Create custom models
- `chatCompletions()`: OpenAI-compatible chat endpoint
- `completions()`: OpenAI-compatible text completion

## License

MIT License
