"""
tools — Built-in tool implementations for the agent loop.

All tools are pure-stdlib. Each is a callable that takes **kwargs
and returns a string result.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_shell(command: str, timeout: int = 120, workdir: str | None = None) -> str:
    """Run a shell command and return stdout+stderr.

    Args:
        command: Shell command string.
        timeout: Max seconds to wait (default 120).
        workdir: Working directory (default None = cwd).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir or os.getcwd(),
        )
        output = []
        if result.stdout:
            output.append(result.stdout.rstrip())
        if result.stderr:
            output.append(f"[stderr]\n{result.stderr.rstrip()}")
        output.append(f"[exit: {result.returncode}]")
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"[error] Command not found: {e}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def tool_read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """Read lines from a text file.

    Args:
        path: Absolute or relative file path.
        offset: Starting line (1-indexed, default 1).
        limit: Max lines to read (default 500).
    """
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return f"[error] File not found: {resolved}"
        if not resolved.is_file():
            return f"[error] Not a file: {resolved}"

        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        if offset < 1:
            offset = 1
        start = offset - 1
        end = start + limit
        selected = lines[start:end]

        out = []
        for i, line in enumerate(selected, start=offset):
            out.append(f"{i}|{line}")
        summary = f"[{resolved}] {total} lines total, showing {offset}-{min(end, total)}"
        return summary + "\n" + "\n".join(out)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def tool_write_file(path: str, content: str) -> str:
    """Write content to a file (overwrites).

    Args:
        path: Absolute or relative file path.
        content: Full file content.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"[ok] Wrote {len(content)} bytes to {resolved}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def tool_patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Find and replace text in a file.

    Args:
        path: File to edit.
        old_string: Text to find (must be unique unless replace_all=True).
        new_string: Replacement text.
        replace_all: Replace all occurrences (default False).
    """
    try:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return f"[error] File not found: {resolved}"

        text = resolved.read_text(encoding="utf-8", errors="replace")
        if replace_all:
            count = text.count(old_string)
            if count == 0:
                return f"[error] old_string not found in {resolved}"
            text = text.replace(old_string, new_string)
        else:
            count = text.count(old_string)
            if count == 0:
                return f"[error] old_string not found in {resolved}"
            if count > 1:
                return (f"[error] old_string appears {count} times in {resolved} "
                        f"(set replace_all=True to replace all)")
            text = text.replace(old_string, new_string)

        resolved.write_text(text, encoding="utf-8")
        return f"[ok] Replaced {count} occurrence(s) in {resolved}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def tool_search_files(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
    mode: str = "content",
) -> str:
    """Search file contents or find files by name.

    Args:
        pattern: Regex (content mode) or glob (files mode) pattern.
        path: Root directory to search.
        file_glob: When mode=content, filter by file glob (e.g. '*.py').
        limit: Max results.
        mode: 'content' to grep, 'files' to find files.
    """
    try:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"[error] Not a directory: {root}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

    results: list[str] = []
    count = 0

    if mode == "files":
        # Find files matching glob pattern
        for p in root.rglob(pattern):
            if count >= limit:
                results.append(f"  ... (truncated at {limit})")
                break
            results.append(str(p.relative_to(root)))
            count += 1
    else:
        # Grep content (simple recursive scan)
        import re as _re
        try:
            regex = _re.compile(pattern)
        except _re.error as e:
            return f"[error] Invalid regex '{pattern}': {e}"

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if file_glob and not p.match(file_glob):
                continue

            if count >= limit:
                results.append(f"  ... (truncated at {limit})")
                break

            try:
                for line_no, line_text in enumerate(
                    p.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if regex.search(line_text):
                        if count >= limit:
                            break
                        rel = p.relative_to(root)
                        results.append(f"{rel}:{line_no}: {line_text.rstrip()}")
                        count += 1
            except (UnicodeDecodeError, OSError):
                continue  # skip binary

    if not results:
        return f"[ok] No matches for '{pattern}' in {root}"

    return f"[ok] {count} match(es) in {root}\n" + "\n".join(results)


def tool_start_llama(
    model_path: str | None = None,
    port: int = 8080,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
) -> str:
    """Build (if needed) and start a llama.cpp server from source.

    This tool helps set up the local LLM backend on i686. It will:
      1. Check for required build tools (cmake, make, g++).
      2. Clone or update llama.cpp from source.
      3. Compile the server binary.
      4. Start it as a background server process.

    Args:
        model_path: Path/URL to a GGUF model file. If None, searches
                    for .gguf files in common locations.
        port: Server port (default 8080).
        n_ctx: Context size (default 2048).
        n_gpu_layers: GPU offload layers (default 0 = CPU only).
    """
    import glob
    import http.client
    import os
    import signal
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    LLAMA_SRC = os.path.expanduser("~/.anyagent/llama.cpp")
    BUILD_DIR = os.path.join(LLAMA_SRC, "build")
    SERVER_BIN = os.path.join(BUILD_DIR, "bin", "server")

    def _log(msg: str) -> None:
        print(f"  [llama] {msg}", file=sys.stderr)

    def _check_arch() -> str:
        """Return arch string for diagnostic purposes."""
        try:
            return subprocess.run(
                ["uname", "-m"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:
            return "unknown"

    def _check_tools() -> list[str]:
        missing = []
        for tool in ["cmake", "make", "g++", "git"]:
            if subprocess.run(
                ["which", tool], capture_output=True, timeout=5
            ).returncode != 0:
                missing.append(tool)
        return missing

    def _clone_or_update() -> str | None:
        if not os.path.exists(LLAMA_SRC):
            _log(f"Cloning llama.cpp into {LLAMA_SRC}...")
            r = subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_SRC],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                return f"Clone failed: {r.stderr[:500]}"
        else:
            _log("Updating llama.cpp...")
            subprocess.run(
                ["git", "-C", LLAMA_SRC, "pull", "--ff-only"],
                capture_output=True, timeout=30,
            )
        return None

    def _build() -> str | None:
        if os.path.exists(SERVER_BIN):
            _log("Server binary already built, skipping build.")
            return None

        os.makedirs(BUILD_DIR, exist_ok=True)

        _log("Configuring with cmake (this may take a while on i686)...")
        cfg = subprocess.run(
            ["cmake", "..", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CUDA=OFF",
             "-DLLAMA_METAL=OFF", "-DLLAMA_BLAS=OFF", "-DCMAKE_BUILD_TYPE=Release"],
            capture_output=True, text=True, timeout=600,
            cwd=BUILD_DIR,
        )
        if cfg.returncode != 0:
            return f"cmake config failed: {cfg.stderr[:500]}"

        _log("Building server (this takes a while on i686)...")
        bld = subprocess.run(
            ["make", "-j", "1"],  # single-thread for i686
            capture_output=True, text=True, timeout=7200,
            cwd=BUILD_DIR,
        )
        if bld.returncode != 0:
            return f"Build failed: {bld.stderr[:500]}"

        _log("Build complete!")
        return None

    def _find_model() -> str | None:
        if model_path:
            return model_path
        # Search common locations
        search_dirs = [
            os.path.expanduser("~/.anyagent/models/"),
            os.path.expanduser("~/models/"),
            "/tmp/models/",
            ".",
        ]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*.gguf")):
                _log(f"Found model: {f}")
                return f
        return None

    def _ensure_model(path: str) -> tuple[str | None, bool]:
        """Download model if path is a URL. Returns (local_path, was_downloaded)."""
        if path.startswith(("http://", "https://", "hf.co/", "huggingface.co/")):
            url = path
            if url.startswith("hf.co/"):
                url = "https://" + url
            elif url.startswith("huggingface.co/"):
                url = "https://" + url
            model_dir = os.path.expanduser("~/.anyagent/models/")
            os.makedirs(model_dir, exist_ok=True)
            # Derive filename from URL
            fname = url.rstrip("/").rsplit("/", 1)[-1]
            if not fname.endswith(".gguf"):
                fname += ".gguf"
            local = os.path.join(model_dir, fname)
            if os.path.exists(local):
                _log(f"Model already cached: {local}")
                return local, False
            _log(f"Downloading model from {url}...")
            try:
                urllib.request.urlretrieve(url, local)
                return local, True
            except Exception as e:
                return None, False
        return path, False

    arch = _check_arch()
    _log(f"Architecture: {arch}")

    missing_tools = _check_tools()
    if missing_tools:
        return (
            f"[error] Build tools not found: {', '.join(missing_tools)}.\n"
            "Install them on Debian with:\n"
            f"  sudo apt-get install {' '.join(missing_tools)}\n"
            f"Or use a remote API endpoint instead: --url http://..."
        )

    # Step 1: get llama.cpp source
    err = _clone_or_update()
    if err:
        return f"[error] {err}"

    # Step 2: build
    err = _build()
    if err:
        return f"[error] {err}"

    # Step 3: find / download model
    model_file = _find_model()
    if not model_file:
        return (
            "[error] No GGUF model found and no model_path provided.\n"
            "Download one with:\n"
            "  wget -O ~/.anyagent/models/qwen2.5-0.5b-q4.gguf \\\n"
            "    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf\n"
            Then re-run: TOOL start_llama(model_path="~/.anyagent/models/
        )

    local_model, _ = _ensure_model(model_file)
    if not local_model:
        return f"[error] Could not resolve model: {model_file}"

    # Step 4: start server
    _log(f"Starting llama.cpp server on port {port} with model: {local_model}")
    logfile = f"/tmp/llama-server-{port}.log"
    pidfile = f"/tmp/llama-server-{port}.pid"

    # Check if already running
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=2)
        conn.request("GET", "/health")
        if conn.getresponse().status < 500:
            return f"[ok] Server already running on port {port}"
    except Exception:
        pass

    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            [SERVER_BIN, "-m", local_model, "--host", "0.0.0.0",
             "--port", str(port), "--ctx-size", str(n_ctx),
             "--n-gpu-layers", str(n_gpu_layers)],
            stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    with open(pidfile, "w") as pf:
        pf.write(str(proc.pid))

    # Wait for startup
    _log("Waiting for server to start...")
    for _ in range(60):
        time.sleep(1)
        try:
            conn = http.client.HTTPConnection("localhost", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status < 500:
                _log(f"Server is running on port {port} (pid {proc.pid})")
                return (
                    f"[ok] llama.cpp server started on port {port} (pid {proc.pid}).\n"
                    f"Logs: {logfile}\n"
                    f"Model: {local_model}"
                )
        except Exception:
            pass

    # Check if process died
    if proc.poll() is not None:
        with open(logfile) as lf:
            tail = lf.read()[-1000:]
        return f"[error] Server exited (code {proc.returncode}). Tail of log:\n{tail}"

    return f"[ok] Server process running but not yet responding. Check: {logfile}"


def tool_ask_user(question: str, choices: list[str] | None = None) -> str:
    """Ask the user for input — agent pauses and waits for reply.

    Args:
        question: The question to present.
        choices: Optional list of up to 4 choices.
    """
    # The agent loop calls this, which triggers the clarify tool.
    # We return a placeholder; the loop will handle the actual interaction.
    parts = [f"QUESTION: {question}"]
    if choices:
        parts.append(f"CHOICES: {', '.join(choices)}")
    return "[need_input] " + " | ".join(parts)


def tool_done(result: str = "") -> str:
    """Signal that the task is complete. DEPRECATED — use FINISH instead."""
    return f"[done] {result}".strip()


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "start_llama": {
        "fn": tool_start_llama,
        "description": "Build (from source) and start a llama.cpp server. Clones llama.cpp, compiles it, downloads a GGUF model if needed, and starts the server. The recommended way to bootstrap the local LLM backend on i686.",
        "params": {
            "model_path": "string (optional) — path or URL to a .gguf model file",
            "port": "int (optional, default 8080) — server port",
            "n_ctx": "int (optional, default 2048) — context size",
            "n_gpu_layers": "int (optional, default 0) — GPU offload (0 = CPU)",
        },
    },
    "shell": {
        "fn": tool_shell,
        "description": "Run a shell command. Returns stdout, stderr, and exit code.",
        "params": {
            "command": "string (required) — shell command",
            "timeout": "int (optional, default 120) — max seconds",
            "workdir": "string (optional) — working directory",
        },
    },
    "read_file": {
        "fn": tool_read_file,
        "description": "Read lines from a file. Lines are 1-indexed.",
        "params": {
            "path": "string (required) — file path",
            "offset": "int (optional, default 1) — starting line",
            "limit": "int (optional, default 500) — max lines",
        },
    },
    "write_file": {
        "fn": tool_write_file,
        "description": "Write content to a file (overwrites entire file).",
        "params": {
            "path": "string (required) — file path",
            "content": "string (required) — full file content",
        },
    },
    "patch": {
        "fn": tool_patch,
        "description": "Find and replace text in a file.",
        "params": {
            "path": "string (required) — file path",
            "old_string": "string (required) — text to find",
            "new_string": "string (required) — replacement text",
            "replace_all": "bool (optional, default False) — replace all occurrences",
        },
    },
    "search_files": {
        "fn": tool_search_files,
        "description": "Search file contents (mode=content) or find files by glob (mode=files).",
        "params": {
            "pattern": "string (required) — regex (content) or glob (files) pattern",
            "path": "string (optional, default '.') — root directory",
            "file_glob": "string (optional) — filter by glob when mode=content",
            "limit": "int (optional, default 50) — max results",
            "mode": "string (optional, default 'content') — 'content' or 'files'",
        },
    },
    "ask_user": {
        "fn": tool_ask_user,
        "description": "Ask the user a question. Agent pauses and waits for reply.",
        "params": {
            "question": "string (required) — the question",
            "choices": "list (optional) — up to 4 choices",
        },
    },
}


def format_tool_doc() -> str:
    """Return tool documentation for the system prompt."""
    lines = [
        "## Available Tools",
        "",
        "Call tools using:  TOOL name(param1=\"value\", param2=value)",
        "Signal completion: FINISH: your final answer or result message",
        "",
    ]
    for name, info in sorted(TOOL_REGISTRY.items()):
        lines.append(f"### {name}")
        lines.append(info["description"])
        for pname, pdesc in info["params"].items():
            lines.append(f"  - {pname}: {pdesc}")
        lines.append("")
    return "\n".join(lines)


def run_tool(name: str, kwargs: dict) -> str:
    """Execute a named tool with kwargs. Returns result string."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return f"[error] Unknown tool: {name}. Available: {', '.join(sorted(TOOL_REGISTRY))}"
    try:
        return entry["fn"](**kwargs)
    except TypeError as e:
        return f"[error] Bad args to {name}: {e}"
    except Exception as e:
        return f"[error] {name} raised {type(e).__name__}: {e}"
