#!/usr/bin/env bash
# build-llama.sh — Build llama.cpp server from source on i686 Debian
#
# Usage:
#   ./build-llama.sh                    # build + auto-download a tiny model
#   ./build-llama.sh --model URL        # download a specific GGUF model
#   ./build-llama.sh --model /path/to/model.gguf  # use existing file
#   ./build-llama.sh --port 8080        # custom port
#   ./build-llama.sh --rebuild          # force recompile from scratch
#   ./build-llama.sh --help             # this message
#
# This script:
#   1. Installs build tools if missing (cmake, make, g++, git)
#   2. Clones llama.cpp into ~/.anyagent/llama.cpp/
#   3. Compiles the server binary (~/.anyagent/llama.cpp/build/bin/server)
#   4. Downloads a GGUF model to ~/.anyagent/models/ if none exists
#   5. Starts the server on the specified port
#
# Designed for 32-bit (i686) Debian. The build uses -j1 and disables
# all GPU/accelerator backends. Expect 10-30 minutes for compilation.

set -euo pipefail

# ─── Config ─────────────────────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.anyagent"
LLAMA_SRC="${INSTALL_DIR}/llama.cpp"
MODELS_DIR="${INSTALL_DIR}/models"
BUILD_DIR="${LLAMA_SRC}/build"
# Modern llama.cpp builds the server as 'llama-server'.
# Older versions used 'server'.  We check both at runtime.
SERVER_BIN="${BUILD_DIR}/bin/llama-server"
SERVER_BIN_OLD="${BUILD_DIR}/bin/server"
PORT="${PORT:-8080}"
MODEL_URL=""
MODEL_PATH=""
FORCE_REBUILD=0

# Default tiny model — Qwen2.5-0.5B Instruct, Q4_K_M, ~350 MB
DEFAULT_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Helpers ─────────────────────────────────────────────────────────────────
log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*" >&2; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

usage() {
    sed -n '2,18p' "$0" | sed 's/^# //'
    exit 0
}

# ─── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --model) shift; MODEL_URL="$1" ;;
        --port)  shift; PORT="$1" ;;
        --rebuild) FORCE_REBUILD=1 ;;
        *) err "Unknown option: $1"; usage ;;
    esac
    shift
done

# If a local file path was given as --model value, use it directly
if [[ -n "$MODEL_URL" && -f "$MODEL_URL" ]]; then
    MODEL_PATH="$MODEL_URL"
    MODEL_URL=""
fi

# ─── Banner ──────────────────────────────────────────────────────────────────
cat <<'EOF'
 ╔══════════════════════════════════════════════════╗
 ║         AnyAgent — llama.cpp Builder             ║
 ║     Builds a local LLM server for i686 Debian    ║
 ╚══════════════════════════════════════════════════╝
EOF
echo "  Install dir:  ${INSTALL_DIR}"
echo "  Build dir:    ${BUILD_DIR}"
echo "  Server:       ${SERVER_BIN}"
echo "  Port:         ${PORT}"
echo "  Architecture: $(uname -m)"
echo ""

# ─── Step 1: Check / Install build tools ────────────────────────────────────
log "Checking build tools..."

MISSING=()
for tool in cmake make g++ git wget; do
    if command -v "$tool" &>/dev/null; then
        info "  $tool: found ($(command -v "$tool"))"
    else
        warn "  $tool: NOT FOUND"
        MISSING+=("$tool")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    log "Installing missing tools: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${MISSING[@]}"
    log "Build tools installed."
fi

# Verify C++ compiler works
if ! g++ --version &>/dev/null; then
    err "g++ is not functional. Try: sudo apt-get install g++"
    exit 1
fi

# ─── Step 2: Get llama.cpp source ────────────────────────────────────────────
if [[ -d "$LLAMA_SRC" ]]; then
    log "Updating existing llama.cpp checkout..."
    git -C "$LLAMA_SRC" pull --ff-only --quiet || warn "git pull failed, continuing with existing source"
else
    log "Cloning llama.cpp into ${LLAMA_SRC}..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_SRC"
fi

# ─── Step 3: Build server binary ─────────────────────────────────────────────
if [[ -f "$SERVER_BIN" && "$FORCE_REBUILD" -eq 0 ]]; then
    log "Server binary already exists: ${SERVER_BIN}"
    log "Pass --rebuild to force recompilation."
else
    if [[ "$FORCE_REBUILD" -eq 1 ]]; then
        log "Force rebuild requested. Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    log "Configuring with cmake (i686 — no GPU, no BLAS)..."
    echo ""
    warn "╔══════════════════════════════════════════════════════════════╗"
    warn "║  This will take a while on i686. Expect 10-30 minutes.     ║"
    warn "║  Grab a coffee.                                            ║"
    warn "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    cmake .. \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CUDA=OFF \
        -DLLAMA_METAL=OFF \
        -DLLAMA_BLAS=OFF \
        -DLLAMA_CUBLAS=OFF \
        -DLLAMA_VULKAN=OFF \
        -DLLAMA_OPENCL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON

    echo ""
    log "Building server (single-threaded, this is the slow part)..."
    echo ""

    # Build only the server target, not everything.
    # Run directly (no pipe) so all compiler warnings and errors
    # are visible and the exit code is captured correctly.
    # Show a heartbeat timestamp every 60s during silence.
    set +e
    (
        make -j1 llama-server 2>&1 &
        MAKE_PID=$!

        # Heartbeat: print a timestamp every 60s so the user knows
        # the build is still alive during long single-file compiles
        while kill -0 "$MAKE_PID" 2>/dev/null; do
            sleep 60
            kill -0 "$MAKE_PID" 2>/dev/null || break
            echo "  [still building... $(date +%H:%M:%S)]"
        done

        wait "$MAKE_PID"
        exit $?
    )
    MAKE_EXIT=$?
    set -e

    if [[ "$MAKE_EXIT" -ne 0 ]]; then
        echo ""
        err "Build failed (exit code $MAKE_EXIT)."
        err "Scroll up to find the error — look for 'error:' or 'FAILED' in the output."
        err "Common issues on i686 Debian:"
        err "  - Out of memory (try adding swap: sudo fallocate -l 2G /swap && sudo mkswap /swap && sudo swapon /swap)"
        err "  - Missing dependencies: sudo apt-get install build-essential"
        err "  - Full log: ${BUILD_DIR}/CMakeFiles/CMakeOutput.log"
        exit 1
    fi

    echo ""
    echo ""
    log "Build complete!"
fi

# Verify binary exists — check modern name, then old name
if [[ ! -f "$SERVER_BIN" ]]; then
    if [[ -f "$SERVER_BIN_OLD" ]]; then
        log "Using older server binary at ${SERVER_BIN_OLD}"
        SERVER_BIN="$SERVER_BIN_OLD"
    else
        err "Server binary not found."
        err "  Tried: ${SERVER_BIN}"
        err "  Tried: ${SERVER_BIN_OLD}"
        err "Build may have failed. Scroll up for errors."
        exit 1
    fi
fi

# ─── Step 4: Download model ──────────────────────────────────────────────────
mkdir -p "$MODELS_DIR"

# If user provided a URL, download it
if [[ -n "$MODEL_URL" ]]; then
    MODEL_NAME="${MODEL_URL##*/}"
    MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"
    if [[ ! -f "$MODEL_PATH" ]]; then
        log "Downloading model from ${MODEL_URL}..."
        wget -O "$MODEL_PATH" "$MODEL_URL"
    else
        log "Model already cached: ${MODEL_PATH}"
    fi
fi

# If no model specified, find or download default
if [[ -z "$MODEL_PATH" ]]; then
    # Check for any existing .gguf in models dir
    EXISTING=$(find "$MODELS_DIR" -name '*.gguf' -type f 2>/dev/null | head -1)
    if [[ -n "$EXISTING" ]]; then
        MODEL_PATH="$EXISTING"
        log "Found existing model: ${MODEL_PATH}"
    else
        MODEL_PATH="${MODELS_DIR}/${DEFAULT_MODEL_NAME}"
        if [[ ! -f "$MODEL_PATH" ]]; then
            log "Downloading default model (~350 MB):"
            echo "  ${DEFAULT_MODEL_URL}"
            echo ""
            wget -O "$MODEL_PATH" "$DEFAULT_MODEL_URL"
            log "Download complete."
        fi
    fi
fi

# Verify model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    err "Model file not found: ${MODEL_PATH}"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
log "Using model: ${MODEL_PATH} (${MODEL_SIZE})"

# ─── Step 5: Start server ────────────────────────────────────────────────────
# Check if something is already listening on the port
if command -v ss &>/dev/null; then
    LISTENING=$(ss -tlnp "sport = :${PORT}" 2>/dev/null | grep -c LISTEN || true)
elif command -v lsof &>/dev/null; then
    LISTENING=$(lsof -i :${PORT} -sTCP:LISTEN 2>/dev/null | wc -l) || true
else
    LISTENING=0
fi

if [[ "$LISTENING" -gt 0 ]]; then
    warn "Port ${PORT} is already in use."
    warn "If this is a llama.cpp server, you can use it now."
    warn "Otherwise, stop the existing process or use --port to pick a different port."
else
    LOGFILE="/tmp/anyagent-llama-${PORT}.log"
    PIDFILE="/tmp/anyagent-llama-${PORT}.pid"

    log "Starting server on port ${PORT}..."
    log "Logs: ${LOGFILE}"
    log "PID:  ${PIDFILE}"
    echo ""

    "${SERVER_BIN}" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --ctx-size 2048 \
        --n-gpu-layers 0 \
        > "$LOGFILE" 2>&1 &

    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE"

    # Wait for server to respond
    echo -n "[i] Waiting for server to become ready..."
    for i in $(seq 1 120); do
        if command -v curl &>/dev/null; then
            if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
                echo " ready! (${i}s)"
                break
            fi
        else
            # Use /dev/tcp if curl isn't available
            if exec 3<>/dev/tcp/localhost/${PORT} 2>/dev/null; then
                echo -e "GET /health HTTP/1.0\r\n\r\n" >&3
                read -t 2 -r resp <&3 2>/dev/null || true
                exec 3>&-
                if [[ "$resp" == *"200"* ]]; then
                    echo " ready! (${i}s)"
                    break
                fi
            fi
        fi
        if [[ $((i % 10)) -eq 0 ]]; then
            echo -n " ${i}s..."
        fi
        sleep 1
    done

    # Check if process died during startup
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        err "Server process exited during startup. Check logs:"
        err "  tail -50 ${LOGFILE}"
        exit 1
    fi

    echo ""
    log "╔══════════════════════════════════════════╗"
    log "║  llama.cpp server is RUNNING             ║"
    log "║                                          ║"
    log "║  API:   http://localhost:${PORT}/v1/chat/completions"
    log "║  Model: $(basename "$MODEL_PATH")"
    log "║  PID:   ${SERVER_PID}"
    log "║                                          ║"
    log "║  Test:  curl http://localhost:${PORT}/health"
    log "║                                          ║"
    log "║  Stop:  kill ${SERVER_PID}              ║"
    log "╚══════════════════════════════════════════╝"
    echo ""
    log "Now run AnyAgent:"
    log "  python agent.py \"Your task here\""
fi
