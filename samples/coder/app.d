/++
 + CLI tool to generate code using the Ollama API's chat streaming endpoint
 + and save it to a file.
 +
 + Usage:
 +     coder --prompt "your prompt" --model model [--output file] [--host host] [--verbose]
 +
 + Examples:
 +     coder --prompt "Create a C function to sort an array" --output sort.c --verbose
 +     coder --prompt "Write a Java program to calculate fibonacci numbers" --model llama3.1:8b
 +/
import ollama;
import std;

void main(string[] args) @safe
{
    string prompt;
    string model;
    string output = "generated.md";
    string host   = DEFAULT_HOST;
    bool   verbose;

    for (int i = 1; i < args.length; i++)
    {
        if      (args[i] == "--prompt" && i + 1 < args.length) prompt  = args[++i];
        else if (args[i] == "--model"  && i + 1 < args.length) model   = args[++i];
        else if (args[i] == "--output" && i + 1 < args.length) output  = args[++i];
        else if (args[i] == "--host"   && i + 1 < args.length) host    = args[++i];
        else if (args[i] == "--verbose")                        verbose = true;
        else
        {
            writefln("Unknown argument: %s", args[i]);
            return;
        }
    }

    if (prompt.empty || model.empty)
    {
        writefln(
            "Usage: coder --prompt \"your prompt\" --model <model> " ~
            "[--output %s] [--host %s] [--verbose]",
            output, host);
        return;
    }

    auto client = new OllamaClient(host);
    client.setTimeOut(120.seconds);

    Message[] messages = [
        Message("system", "You are a senior software engineer. " ~
                           "Respond only with code and brief inline comments."),
        Message("user",   "Generate code: " ~ prompt),
    ];

    string code;

    try
    {
        if (verbose)
        {
            writefln("Model   : %s", model);
            writefln("Prompt  : %s", prompt);
            writefln("Output  : %s", output);
            writeln("Streaming response:");
            writeln("─".replicate(60));
        }

        client.chatStream(model, messages, (JSONValue chunk) @safe {
            if (!chunk["done"].get!bool)
            {
                auto tok = chunk["message"]["content"].str;
                code ~= tok;
                if (verbose) write(tok);
            }
            else if (verbose)
            {
                writeln("\n" ~ "─".replicate(60));
            }
        });

        std.file.write(output, code);
        writefln("Code saved to %s (%d bytes)", output, code.length);
    }
    catch (Exception e)
    {
        writeln("Error: ", e.msg);
    }
}
