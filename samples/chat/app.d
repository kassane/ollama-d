/++
 + Interactive multi-turn chatbot powered by Qwen3 0.6B via Ollama.
 +
 + Demonstrates:
 +   - Streaming output (tokens appear character-by-character)
 +   - Multi-turn conversation with full history
 +   - Qwen3 thinking mode (`/think` — chain-of-thought reasoning)
 +   - System prompt customisation at startup
 +
 + The default model is `qwen3.5:0.6b` — tiny (~500 MB), CPU-friendly,
 + fast enough for interactive use. Override with `--model`.
 +
 + Usage:
 +     chat [--model qwen3.5:0.6b] [--host http://127.0.0.1:11434]
 +          [--system "You are…"] [--think]
 +
 + In-session commands:
 +     /help    show this list
 +     /clear   erase history and start a new conversation
 +     /think   toggle Qwen3 chain-of-thought mode
 +     /quit    exit
 +/
import ollama;
import std;

// ── Model defaults ─────────────────────────────────────────────────────────
enum DEFAULT_MODEL  = "qwen3.5:0.6b";
enum DEFAULT_SYSTEM = "You are a helpful, concise assistant.";

// stdout.flush() and readln() access __gshared stdin/stdout under
// -preview=safer, making them @system.  Thin @trusted wrappers let the
// @safe StreamCallback lambda call them without relaxing the whole file.
private void    flushOut() @trusted { stdout.flush(); }
private string  nextLine() @trusted { return readln(); }

void main(string[] args) @safe
{
    // ── CLI arguments ──────────────────────────────────────────────────────
    string model      = DEFAULT_MODEL;
    string host       = DEFAULT_HOST;
    string systemText = DEFAULT_SYSTEM;
    bool   think;                         // Qwen3 chain-of-thought off by default

    for (int i = 1; i < args.length; i++)
    {
        if      (args[i] == "--model"  && i + 1 < args.length) model      = args[++i];
        else if (args[i] == "--host"   && i + 1 < args.length) host       = args[++i];
        else if (args[i] == "--system" && i + 1 < args.length) systemText = args[++i];
        else if (args[i] == "--think")                          think      = true;
        else
        {
            writefln("Unknown argument: %s\n"
                ~ "Usage: chat [--model M] [--host H] [--system S] [--think]",
                args[i]);
            return;
        }
    }

    auto client = new OllamaClient(host);
    client.setTimeOut(120.seconds);

    // ── Banner ─────────────────────────────────────────────────────────────
    writeln("╔══════════════════════════════════════════════╗");
    writefln("║  Ollama Chat  ·  %-28s║", model);
    writeln("║  /help  /clear  /think  /quit               ║");
    writeln("╚══════════════════════════════════════════════╝");
    if (think)
        writeln("[thinking mode ON — Qwen3 will show chain-of-thought]");

    // ── Conversation state ─────────────────────────────────────────────────
    // Prepend a persistent system message; it is never removed by /clear.
    Message[] history = [Message("system", systemText)];

    // ── REPL loop ──────────────────────────────────────────────────────────
    while (true)
    {
        write("\nYou: ");
        flushOut();

        string line = nextLine();
        if (line is null) { writeln(); break; }     // EOF (Ctrl-D / pipe end)

        import std.string : strip;
        line = line.strip();
        if (line.length == 0) continue;

        // ── Commands ───────────────────────────────────────────────────────
        if (line == "/quit" || line == "/q")
        {
            writeln("Bye!");
            break;
        }
        if (line == "/help")
        {
            writeln("  /clear   erase conversation (keeps system prompt)");
            writeln("  /think   toggle Qwen3 chain-of-thought");
            writeln("  /quit    exit");
            continue;
        }
        if (line == "/clear")
        {
            history = [Message("system", systemText)];
            writeln("[conversation cleared]");
            continue;
        }
        if (line == "/think")
        {
            think = !think;
            writefln("[thinking mode: %s]", think ? "ON" : "OFF");
            continue;
        }

        // ── Qwen3 thinking directive ───────────────────────────────────────
        // Prepend /think or /no_think to steer the model's reasoning budget.
        // Qwen3 interprets these prefixes natively and wraps its reasoning
        // inside <think>…</think> tags (visible in the streamed output).
        immutable userText = (think ? "/think " : "/no_think ") ~ line;
        history ~= Message("user", userText);

        // ── Streamed response ──────────────────────────────────────────────
        write("Assistant: ");
        flushOut();

        string reply;
        bool   ok = true;

        try
        {
            client.chatStream(model, history, (JSONValue chunk) @safe {
                if (chunk["done"].get!bool) return;
                immutable tok = chunk["message"]["content"].str;
                reply ~= tok;
                write(tok);
                flushOut();
            });
            writeln();
        }
        catch (Exception e)
        {
            writefln("\n[error: %s]", e.msg);
            ok = false;
        }

        if (ok && reply.length > 0)
            history ~= Message("assistant", reply);
        else if (!ok)
            history = history[0 .. $ - 1]; // drop the failed user message
    }
}
