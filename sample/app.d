import ollama;
import std.stdio;

// Sample example for using the Ollama Dlang library
// Prerequisites:
// - Ollama server must be running locally on http://localhost:11434
// - The model "llama3.2" must be installed on the server

void main()
{
    // Initialize the Ollama client
    auto client = new OllamaClient();

    // Demonstrate text completion with the 'generate' function
    try
    {
        auto generateResponse = client.generate("llama3.2", "Why is the sky blue?");
        if ("error" in generateResponse)
        {
            writeln("Error in generate: ", generateResponse["error"].get!string);
        }
        else
        {
            writeln("Generate Response: ", generateResponse["response"].get!string);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in generate: ", e.msg);
    }

    // Demonstrate chat interaction with the 'chat' function
    Message[] messages = [Message("user", "Hello, how are you?")];
    try
    {
        auto chatResponse = client.chat("llama3.2", messages);
        if ("error" in chatResponse)
        {
            writeln("Error in chat: ", chatResponse["error"].get!string);
        }
        else
        {
            writeln("Chat Response: ", chatResponse["message"]["content"].get!string);
        }
    }
    catch (Exception e)
    {
        writeln("Exception in chat: ", e.msg);
    }
}
