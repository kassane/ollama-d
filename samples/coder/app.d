/++
 + CLI tool to generate code using the Ollama API's chat endpoint and save it to a .md file.
 +
 + Usage:
 +     coder --prompt "your prompt" [--model model] [--output output.c] [--host host] [--verbose]
 +
 + Examples:
 +     coder --prompt "Create a C function to sort an array" --output sort.c --verbose
 +     coder --prompt "Write a Java program to calculate fibonacci numbers" --host http://localhost:11434
 +/
import ollama;
import std;

void main(string[] args) @safe
{
    // Variables for command-line arguments
    string prompt; // Required: Prompt for code generation
    string model; // Required: Model to use for code generation
    string output = "generated.md"; // Default: Output file name
    string host = DEFAULT_HOST;
    bool verbose;

    // Parse command-line arguments
    for (int i = 1; i < args.length; i++)
    {
        if (args[i] == "--prompt" && i + 1 < args.length)
        {
            prompt = args[++i];
        }
        else if (args[i] == "--model" && i + 1 < args.length)
        {
            model = args[++i];
        }
        else if (args[i] == "--output" && i + 1 < args.length)
        {
            output = args[++i];
        }
        else if (args[i] == "--host" && i + 1 < args.length)
        {
            host = args[++i];
        }
        else if (args[i] == "--verbose")
        {
            verbose = true;
        }
        else
        {
            writefln("Invalid argument: %s", args[i]);
            return;
        }
    }

    // Show help and exit if prompt is missing
    if (prompt.empty || model.empty)
    {
        writefln(
            "Usage: coder --prompt \"your prompt\" [--model model] [--output %s] [--host %s] [--verbose]",
            output, host);
        return;
    }

    // Initialize Ollama client with specified host
    auto client = new OllamaClient(host);
    client.setTimeOut(30.seconds);

    try
    {
        // Prepare chat message for the API
        Message[] messages = [
            Message("user", "Generate code: " ~ prompt)
        ];

        // Call the chat endpoint to generate Code
        auto response = client.chat(model, messages);

        // Extract the generated code from the response
        string cCode = response["message"]["content"].str;

        // Write the code to the output file
        std.file.write(output, cCode);
        writeln("Code successfully generated and saved to ", output);

        if (verbose)
        {
            writeln("\nFull API Response:");
            writeln(response.toPrettyString());
        }
    }
    catch (Exception e)
    {
        writeln("Error: ", e.msg);
    }
}
