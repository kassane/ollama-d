# ollama-d

[![Static Badge](https://img.shields.io/badge/v2.110.0%20(stable)-f8240e?logo=d&logoColor=f8240e&label=runtime)](https://dlang.org/download.html)
![Latest release](https://img.shields.io/github/v/release/kassane/ollama-d?include_prereleases&label=latest)
[![Artifacts](https://github.com/kassane/ollama-d/actions/workflows/ci.yml/badge.svg)](https://github.com/kassane/ollama-d/actions/workflows/ci.yml)

D language bindings for the Ollama REST API - Seamless integration with local AI models

## Features

- Text generation with native and OpenAI-compatible endpoints
- Chat interactions with local AI models
- Model management (list, create, show models)
- Configurable timeout settings
- Simple and intuitive API design using `std.net.curl` and `std.json`

## Prerequisites

- [D compiler](https://dlang.org/download.html) installed on your system
- Ollama server running locally (default: "http://127.0.0.1:11434")
- Installed AI model (e.g., "llama3.2")

## Quick Examples

```d
import ollama;
import std.stdio;

void main() {
    // Initialize Ollama client on localhost at port 11434
    auto client = new OllamaClient();

    // Text generation
    auto generateResponse = client.generate("llama3.2", "Why is the sky blue?");
    writeln("Generate Response: ", generateResponse["response"].str);

    // Chat interaction
    Message[] messages = [Message("user", "Hello, how are you?")];
    auto chatResponse = client.chat("llama3.2", messages);
    writeln("Chat Response: ", chatResponse["message"]["content"].str);

    // List available models
    auto models = client.listModels();
    writeln("Available Models: ", models);

    // OpenAI-compatible chat completions
    auto openaiResponse = client.chatCompletions("llama3.2", messages, 50, 0.7);
    writeln("OpenAI-style Response: ", openaiResponse["choices"][0]["message"]["content"].str);
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
- `getModels()`: List models in OpenAI-compatible format
- `setTimeOut()`: Configure request timeout duration

## License

MIT License
