# ollama-d

D language bindings for the Ollama AI API - Seamless integration with local AI models

## Features

- Text generation with non-streaming API
- Chat interactions with local AI models
- Configurable timeout settings
- Simple and intuitive API design

## Prerequisites

- Ollama server running locally (default: http://127.0.0.1:11434)
- Installed AI model (e.g., "llama3.2")

## Quick Example

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
}
```

## License

MIT License
