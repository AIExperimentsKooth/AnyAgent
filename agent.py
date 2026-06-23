#!/usr/bin/env python3
"""
AnyAgent — Minimal Python-only agentic system.

A pure-stdlib tool-calling agent loop for i686 Debian and other
constrained environments. No pip packages required.

Usage:
    python agent.py "Write a Python script that prints 'hello from i686'"
    python agent.py --model qwen2.5:1.5b --url http://localhost:11434 "task..."
    python agent.py --interactive   # REPL-like single-turn mode

Environment:
    MINIMA_MODEL   — model name (default: qwen2.5-0.5b-q4, env: MINIMA_MODEL)
    MINIMA_URL     — llama.cpp or OpenAI-compatible API (default: http://localhost:8080, env: MINIMA_URL)
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from llm import LLMClient, LLMError
from tools import TOOL_REGISTRY, format_tool_doc, run_tool
from tool_parser import parse, FinishSignal, ToolCall


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous coding agent called AnyAgent. You run on a 32-bit Debian system.

You have tools to interact with the environment. To call a tool, output exactly:

    TOOL name(param1="value", param2=123)

Parameters must be key=value inside parentheses. Use double quotes for strings.
When your task is complete, output:

    FINISH: your final message here

Think step by step. You can call multiple tools sequentially — the system
will execute your tool call and return the result, then you get another turn.

Available tools:
{tools_doc}

Rules:
- Read files before editing them.
- Run shell commands to install/build/test things.
- When done, always use FINISH with a summary of what was accomplished.
- If you need more information from the user, use ask_user and they will respond.
- Never fabricate file contents or command output. Always use the tools.
- Do not try to call tools that don't exist in the list above.
- Tool results are appended as separate messages. Only output one tool call
  per turn, or multiple — the system runs them all and returns results.
"""


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

class Agent:
    """Main agent loop."""

    def __init__(self, model: str, base_url: str, max_iterations: int = 50):
        self.llm = LLMClient(base_url=base_url, model=model)
        self.max_iterations = max_iterations
        self.tool_doc = format_tool_doc()

    def _build_system(self) -> str:
        return SYSTEM_PROMPT.format(tools_doc=self.tool_doc)

    def run(self, task: str) -> str:
        """Execute a task to completion."""
        messages: list[dict] = [
            {"role": "system", "content": self._build_system()},
            {"role": "user", "content": task},
        ]

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n─── [iter {iteration}/{self.max_iterations}] ───", file=sys.stderr)

            # --- Think ---
            try:
                response = self.llm.chat(messages)
            except LLMError as e:
                err_msg = str(e)
                if "timed out" in err_msg or "Connection refused" in err_msg:
                    return (
                        f"[fatal] Cannot reach the LLM backend at {self.llm.base_url}.\n"
                        f"  Make sure the llama.cpp server is running:\n"
                        f"    curl http://localhost:8080/v1/chat/completions -d '{{\"model\":\"test\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
                        f"  Start it with: ./build-llama.sh\n"
                        f"  Or check the log: tail -50 /tmp/anyagent-llama-8080.log\n"
                        f"  Or use a different URL: --url http://other-host:8080"
                    )
                return f"[fatal] LLM call failed: {e}"
            except KeyboardInterrupt:
                return "[interrupted]"

            print(f"\033[33m[assistant]\033[0m {response}\n", file=sys.stderr)

            # --- Parse ---
            results = parse(response)
            if not results:
                # No tool call — treat response as final answer
                return response

            has_finish = any(isinstance(r, FinishSignal) for r in results)
            tool_results: list[str] = []

            for item in results:
                if isinstance(item, FinishSignal):
                    print(f"\033[32m[FINISH]\033[0m {item.message}", file=sys.stderr)
                    return item.message

                if isinstance(item, ToolCall):
                    print(f"\033[34m[tool]\033[0m {item.name}({item.kwargs})", file=sys.stderr)
                    result = run_tool(item.name, item.kwargs)
                    print(f"\033[90m[result: {len(result)} chars]\033[0m", file=sys.stderr)
                    _print_result_preview(result, file=sys.stderr)
                    tool_results.append(result)

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": response})

            if has_finish:
                break

            # Append tool results — handle ask_user specially
            for tr in tool_results:
                if tr.startswith("[need_input]"):
                    # Pause and get user input
                    print(f"\n\033[33m*** AGENT NEEDS INPUT ***\033[0m", file=sys.stderr)
                    question = tr.replace("[need_input] ", "", 1) if tr.startswith("[need_input] ") else tr
                    print(f"QUESTION: {question}", file=sys.stderr)
                    try:
                        user_input = input(">> ")
                    except (EOFError, KeyboardInterrupt):
                        user_input = "[user interrupted]"
                    messages.append({"role": "tool", "content": tr})
                    messages.append({"role": "user", "content": user_input})
                else:
                    messages.append({"role": "tool", "content": tr[:8000]})

        return f"[timeout] Reached max iterations ({self.max_iterations})"


def _print_result_preview(result: str, file=sys.stderr) -> None:
    """Print the first 3 lines of a result for visibility."""
    lines = result.split("\n")
    for line in lines[:3]:
        print(f"  | {line}", file=file)
    if len(lines) > 3:
        print(f"  | ... ({len(lines) - 3} more lines)", file=file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AnyAgent — minimal Python-only agentic system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python agent.py "List all files in /tmp"
              python agent.py --model qwen2.5-1.5b-q4 "Write hello.py"
              python agent.py --url http://192.168.1.5:8080 "task"
              python agent.py --list-tools
        """),
    )
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument(
        "--model",
        default=os.environ.get("MINIMA_MODEL", "qwen2.5-0.5b-q4"),
        help="LLM model name (default: qwen2.5-0.5b-q4, env: MINIMA_MODEL)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MINIMA_URL", "http://localhost:8080"),
        help="API base URL (default: http://localhost:8080 — llama.cpp, env: MINIMA_URL)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=50,
        help="Max agent loop iterations (default: 50)",
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Print tool descriptions and exit",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Enter interactive single-turn mode (one task per line)",
    )
    parser.add_argument(
        "--check-api", action="store_true",
        help="Test LLM API connectivity and exit",
    )

    args = parser.parse_args()

    if args.list_tools:
        print(format_tool_doc())
        return

    if args.check_api:
        agent = Agent(model=args.model, base_url=args.url)
        import urllib.request, urllib.error
        # First check basic TCP connectivity
        print(f"Checking TCP connectivity to {args.url}...", file=sys.stderr)
        tcp_ok = False
        try:
            url_parts = args.url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            port = int(url_parts[1].split("/")[0]) if len(url_parts) > 1 else 80
            import socket
            s = socket.create_connection((host, port), timeout=5)
            s.close()
            tcp_ok = True
            print(f"  TCP: OK (port {port} open)", file=sys.stderr)
        except Exception as e:
            print(f"  TCP: FAILED — {e}", file=sys.stderr)
            print(f"  Is the server running? Try: ./build-llama.sh", file=sys.stderr)
            sys.exit(1)

        if tcp_ok:
            try:
                resp = agent.llm.chat(
                    [{"role": "user", "content": "Say exactly: API OK"}],
                    max_tokens=16,
                )
                print(f"  API: OK", file=sys.stderr)
                print(f"  Model: {args.model}")
                print(f"  Response: {resp}")
            except LLMError as e:
                print(f"  API call failed: {e}", file=sys.stderr)
                sys.exit(1)
        return

    agent = Agent(model=args.model, base_url=args.url, max_iterations=args.max_iterations)

    if args.interactive:
        print("AnyAgent interactive mode. Type your task (Ctrl+D to exit).",
              file=sys.stderr)
        try:
            for line in sys.stdin:
                task = line.strip()
                if not task:
                    continue
                print(f"\n\033[36m--- TASK: {task} ---\033[0m", file=sys.stderr)
                result = agent.run(task)
                print(f"\n\033[32m=== RESULT ===\033[0m\n{result}")
                print(file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
        return

    if not args.task:
        parser.print_help()
        sys.exit(1)

    result = agent.run(args.task)
    print(f"\n=== RESULT ===\n{result}")


if __name__ == "__main__":
    main()
