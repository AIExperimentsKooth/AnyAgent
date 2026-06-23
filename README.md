# minima — Minimal Python-Only Agentic System

**Zero dependencies.** Pure Python 3 stdlib. Runs on i686 (32-bit) Debian
and other CPU-constrained environments.

A lightweight tool-calling agent loop inspired by Hermes Agent and Codex CLI.
No pip packages, no wheels, no C compilation needed for the agent itself.

## Architecture

```
agent.py       — main loop + CLI parser
llm.py         — LLM API client (urllib, OpenAI-compatible)
tools.py       — tool implementations + registry
tool_parser.py — parse TOOL name(args) / FINISH from LLM output
```

The loop:
1. Sends conversation to LLM via OpenAI-compatible HTTP API
2. Parses response for `TOOL name(key="val", ...)` calls
3. Executes each tool, appends results to conversation
4. Repeats until `FINISH: message` or max iterations

## Requirements

- **Python 3.8+** (stdlib only — no pip packages needed)
- **An LLM backend** with an OpenAI-compatible `/v1/chat/completions` endpoint

## LLM Backend on i686

**Ollama does not support 32-bit architectures.** For local inference on i686:

### Option 1: llama.cpp server (recommended for local)

llama.cpp compiles from source on i686 with cmake and g++. Use the
built-in `start_llama` tool in the agent to automate the process:

```
TOOL start_llama(model_path="~/.minima/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
```

Or do it manually:
```bash
sudo apt-get install cmake make g++ git
git clone https://github.com/ggerganov/llama.cpp ~/.minima/llama.cpp
cd ~/.minima/llama.cpp && mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF -DLLAMA_BLAS=OFF
make -j1
# Download a model:
wget -O ~/.minima/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
# Start the server:
./build/bin/server -m ~/.minima/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --host 0.0.0.0 --port 8080
```

Then run the agent:
```bash
python agent.py "Your task here"
```

### Option 2: Remote API

Point minima at any existing OpenAI-compatible endpoint (a more powerful
machine on your network, a hosted API, etc.):

```bash
python agent.py --url http://192.168.1.50:8080 "Your task"
```

## Usage

```bash
# One-shot task
python agent.py "List all files in /tmp and show their sizes"

# Custom model/endpoint
python agent.py --model qwen2.5-1.5b-q4 --url http://192.168.1.5:8080 "..."

# Check API connectivity
python agent.py --check-api

# List available tools
python agent.py --list-tools

# Interactive mode (one task per line)
python agent.py --interactive
```

Environment variables: `MINIMA_MODEL`, `MINIMA_URL`.

## Tools

| Tool | Description |
|---|---|
| `start_llama` | Build & start a local llama.cpp server from source |
| `shell` | Run a shell command |
| `read_file` | Read lines from a file |
| `write_file` | Write content to a file (overwrites) |
| `patch` | Find-and-replace text in a file |
| `search_files` | Grep contents or find files by glob |
| `ask_user` | Pause and ask the user a question |

## Calling Convention

```
TOOL read_file(path="/etc/hostname")
TOOL shell(command="uname -a", timeout=30)
FINISH: Task complete — wrote 3 files and ran tests

# Multi-line also works:
TOOL write_file
  path="hello.py"
  content="print('hello from i686')"
```

## Why no pip dependencies?

The target environment is **32-bit Debian** (i686). Many Python packages
with C extensions fail to build on i686 or require complex cross-compilation
toolchains. Minima uses only Python stdlib modules:

- `urllib.request` — HTTP client for LLM API
- `subprocess` — shell commands
- `pathlib`, `json`, `re`, `shlex` — file I/O and parsing

The `start_llama` tool does require build tools (`cmake`, `g++`, `make`)
to compile llama.cpp from source — but those are OS-level packages
installed via `apt-get`, not pip dependencies.
