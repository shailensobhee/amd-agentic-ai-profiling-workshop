#!/bin/bash

# ===========================================================================
# Configuration
# ===========================================================================
HERMES_WORKSPACE_DIR="$HOME/.hermes/workspace"
mkdir -p "$HERMES_WORKSPACE_DIR"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
WORKSPACE_DIR=$(pwd)

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"

# ---------------------------------------------------------------------------
# Interpreter resolution
# ---------------------------------------------------------------------------
# The project venv created in README step 2. Everything this script installs or
# runs for MLflow/Streamlit must use THIS interpreter, not the system python3:
# the AMD AI/ML Ready Image ships a system python3 with no pip and no ensurepip,
# so bare `python3 -m pip` / `python3 -m mlflow` fail with
# "No module named pip" / "No module named mlflow".
VENV_PY="$WORKSPACE_DIR/env/bin/python"
HERMES_ENV="$HOME/.hermes/.env"
HERMES_ENV_TMP="$HOME/.hermes/.env.managed.$$"
if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] Project venv not found at $WORKSPACE_DIR/env." >&2
    echo "        Run the README setup first:" >&2
    echo "          sudo apt-get update && sudo apt-get install -y python3-venv" >&2
    echo "          python3 -m venv env && source env/bin/activate" >&2
    echo "          python -m pip install --upgrade pip && python -m pip install -r requirements.txt" >&2
    exit 1
fi

# Preflight: docker is required (this script runs 'sudo docker' many times) and
# is NOT present on the stock AMD AI/ML Ready Image.
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker is not installed, but this script requires it." >&2
    echo "        Install it first:" >&2
    echo "          sudo apt-get update && sudo apt-get install -y docker.io" >&2
    echo "          sudo systemctl enable --now docker" >&2
    exit 1
fi

HERMES_GPU="0"   # Muse-Glimmer-30B runs on GPU 0

# vLLM image: built locally from the ROCm nightly with PR #51655 (Muse-Glimmer
# support) overlaid. Built once below if not already present.
IMAGE_NAME="vllm-muse-glimmer:rocm"
VLLM_HERMES_PORT=8001
VLLM_DEVICE_METRICS_EXPORTER_PORT=5050

SYSTEM_IP=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || ip route get 8.8.8.8 | awk '{print $7; exit}')

# Clear caches before starting.
bash clear_cache.sh

HERMES_MODEL="meta-models/Muse-Glimmer-30B"

# Local Kokoro TTS server (sequential + batched inference modes).
KOKORO_PORT=8092
KOKORO_ENV="$WORKSPACE_DIR/env"
KOKORO_SERVER="$WORKSPACE_DIR/kokoro_server.py"

# ===========================================================================
# Helpers and lifecycle management
# ===========================================================================

cleanup() {
    # Exit code: 0 from the trap (clean Ctrl+C / TERM), non-zero when called by fail().
    local exit_code="${1:-0}"
    echo -e "\n[INFO] Cleaning up containers and background services..."

    echo "[INFO] Stopping Streamlit dashboard..."
    if [ -n "$STREAMLIT_PID" ]; then
        kill "$STREAMLIT_PID" >/dev/null 2>&1
    fi
    sudo fuser -k 8501/tcp >/dev/null 2>&1

    echo "[INFO] Stopping MLflow server..."
    if [ -n "$MLFLOW_PID" ]; then
        kill "$MLFLOW_PID" >/dev/null 2>&1
    fi
    sudo fuser -k 5004/tcp >/dev/null 2>&1

    echo "[INFO] Stopping Kokoro TTS server..."
    if [ -n "$KOKORO_PID" ]; then
        kill "$KOKORO_PID" >/dev/null 2>&1
    fi
    sudo fuser -k ${KOKORO_PORT}/tcp >/dev/null 2>&1

    echo "[INFO] Stopping hermes_service container..."
    sudo docker stop hermes_service >/dev/null 2>&1
    sudo docker rm hermes_service >/dev/null 2>&1

    echo "[INFO] Stopping device-metrics-exporter container..."
    sudo docker stop device-metrics-exporter >/dev/null 2>&1
    sudo docker rm device-metrics-exporter >/dev/null 2>&1

    echo "[INFO] Removing profiling artifacts cache..."
    rm -rf "$HOME/profiling_cache" >/dev/null 2>&1
    echo "[INFO] Cleanup complete. Exiting."
    exit "$exit_code"
}

# Report a fatal setup failure, dump the offending log, tear everything down,
# and exit non-zero so the caller knows the run did not come up cleanly.
fail() {
    local service_name="$1"
    local log_file="$2"
    echo -e "\n[FATAL] $service_name failed to start properly. Aborting setup." >&2
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "----- last 40 lines of $log_file -----" >&2
        tail -n 40 "$log_file" >&2
        echo "--------------------------------------" >&2
    fi
    cleanup 1
}

# Declared up front so cleanup can reference them safely even if Ctrl+C arrives
# before the corresponding server is started.
MLFLOW_PID=""
KOKORO_PID=""
STREAMLIT_PID=""

# Catch Ctrl+C and termination so containers are always cleaned up.
trap cleanup INT TERM

wait_for_vllm_readiness() {
    local port=$1
    local service_name=$2
    local timeout=600
    local counter=0

    echo "[INFO] Waiting for $service_name to load weights and start its API on port $port..."
    while true; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/v1/models || echo "000")

        if [ "$status_code" -eq 200 ]; then
            echo "[OK] $service_name is active and responsive."
            return 0
        fi

        sleep 5
        counter=$((counter + 5))
        if [ $counter -ge $timeout ]; then
            echo "[ERROR] Timeout waiting for $service_name to respond."
            return 1
        fi
    done
}

# ===========================================================================
# vLLM engine
# ===========================================================================

# Remove any leftover containers from a previous run.
sudo docker rm -f hermes_service device-metrics-exporter >/dev/null 2>&1

# Build the image locally if it is missing: clone the PR, overlay its Python
# files onto the nightly's installed vLLM (no kernel compile), then commit.
BASE_IMAGE="vllm/vllm-openai-rocm:nightly"
if [ -z "$(sudo docker images -q $IMAGE_NAME)" ]; then
    echo "[INFO] Image $IMAGE_NAME not found. Building from $BASE_IMAGE + PR #51655..."
    sudo docker pull "$BASE_IMAGE"
    sudo docker rm -f muse_build >/dev/null 2>&1

    # Overlay PR #51655's Python files and verify the parsers register before
    # committing.
    sudo docker run --name muse_build \
        --device=/dev/kfd --device=/dev/dri \
        --security-opt seccomp=unconfined --group-add video --privileged \
        --entrypoint /bin/bash "$BASE_IMAGE" -c '
            set -e
            apt-get update && apt-get install -y git rsync
            git clone https://github.com/vllm-project/vllm.git /tmp/vllm-src
            cd /tmp/vllm-src
            git fetch origin pull/51655/head:muse
            git checkout muse
            cd /root
            VLLM_PKG=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null | tail -1)
            echo "Overlaying PR #51655 python files onto: $VLLM_PKG"
            rsync -a --include="*/" --include="*.py" --exclude="*" /tmp/vllm-src/vllm/ "$VLLM_PKG/"
            python3 -c "from vllm.tool_parsers import ToolParserManager; assert ToolParserManager.get_tool_parser(\"muse_glimmer\"); print(\"tool parser OK\")"
            python3 -c "from vllm.reasoning import ReasoningParserManager; assert ReasoningParserManager.get_reasoning_parser(\"muse_glimmer\"); print(\"reasoning parser OK\")"
        '
    BUILD_RC=$?
    if [ "$BUILD_RC" -ne 0 ]; then
        echo "[ERROR] Build of $IMAGE_NAME failed (overlay step). See output above."
        sudo docker rm -f muse_build >/dev/null 2>&1
        exit 1
    fi

    echo "[INFO] Committing patched container to image $IMAGE_NAME..."
    sudo docker commit muse_build "$IMAGE_NAME"
    sudo docker rm -f muse_build >/dev/null 2>&1
    echo "[OK] Built $IMAGE_NAME"
else
    echo "[OK] Image $IMAGE_NAME found locally. Skipping build."
fi


echo "[INFO] Launching hermes_service (vLLM)..."

sudo docker run -d \
    --ipc=host \
    --network=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    --name hermes_service \
    -e HIP_VISIBLE_DEVICES=$HERMES_GPU \
    -e VLLM_ROCM_USE_AITER=1 \
    -v "$HERMES_WORKSPACE_DIR":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -c \
    "python3 -m vllm.entrypoints.openai.api_server \
        --model $HERMES_MODEL \
        --port $VLLM_HERMES_PORT \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.6 \
        --enable-auto-tool-choice \
        --tool-call-parser muse_glimmer \
        --reasoning-parser muse_glimmer \
        --attention-backend ROCM_AITER_FA \
        --generation-config auto \
        --enable-prefix-caching \
        --host 0.0.0.0"

echo "[INFO] Verifying container runtimes..."
if ! wait_for_vllm_readiness $VLLM_HERMES_PORT "hermes_service vLLM engine"; then
    echo "----- last 40 lines of hermes_service container logs -----" >&2
    sudo docker logs --tail 40 hermes_service >&2 2>&1
    echo "---------------------------------------------------------" >&2
    fail "hermes_service vLLM engine" ""
fi

echo "[INFO] Launching device-metrics-exporter to track GPU usage..."
sudo docker run -d \
    --device=/dev/dri \
    --device=/dev/kfd \
    -v /sys:/sys:ro \
    -p $VLLM_DEVICE_METRICS_EXPORTER_PORT:5000 \
    --name device-metrics-exporter \
    rocm/device-metrics-exporter:v1.5.0

# The exporter serves Prometheus metrics at /metrics (not /v1/models), so it
# needs its own readiness check rather than wait_for_vllm_readiness.
echo "[INFO] Waiting for device-metrics-exporter on port $VLLM_DEVICE_METRICS_EXPORTER_PORT..."
exporter_ready=0
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://localhost:$VLLM_DEVICE_METRICS_EXPORTER_PORT/metrics" || echo "000")
    if [ "$code" -eq 200 ]; then
        echo "[OK] device-metrics-exporter is serving metrics."
        exporter_ready=1
        break
    fi
    sleep 2
done
if [ "$exporter_ready" -ne 1 ]; then
    echo "[WARN] device-metrics-exporter did not respond on /metrics; GPU columns may be empty."
fi

# ===========================================================================
# Hermes toolchain and MLflow integration
# ===========================================================================
echo "[INFO] Installing MLflow and OpenTelemetry dependencies..."
"$VENV_PY" -m pip install -q mlflow==3.13.0 opentelemetry-sdk==1.42.1

echo "[INFO] Launching MLflow server on port 5004..."
"$VENV_PY" -m mlflow server \
  --host 0.0.0.0 \
  --port 5004 \
  --backend-store-uri sqlite:///mlflow.db \
  --allowed-hosts "*" > mlflow_server.log 2>&1 &

MLFLOW_PID=$!
echo "[INFO] MLflow server started (PID $MLFLOW_PID)."

echo "[INFO] Waiting for MLflow server /health on port 5004..."
mlflow_ready=0
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:5004/health" || echo "000")
    if [ "$code" -eq 200 ]; then
        echo "[OK] MLflow server is up."
        mlflow_ready=1
        break
    fi
    # Fail fast if the background process already died.
    if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [ "$mlflow_ready" -ne 1 ]; then
    fail "MLflow server" "$WORKSPACE_DIR/mlflow_server.log"
fi

# ===========================================================================
# Hermes Installation & Configuration
# ===========================================================================
sudo chown -R $(whoami):$(whoami) "$HOME/.hermes"
if ! command -v hermes &> /dev/null && [ ! -f "$HOME/.local/bin/hermes" ]; then
    echo "[INFO] Installing Hermes agent..."
    curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup
    echo "[OK] Hermes agent installed."
else
    echo "[INFO] Hermes agent already available. Skipping."
fi

echo "[INFO] Applying local backend configuration..."
hermes config set model.provider custom
hermes config set model.base_url "http://localhost:$VLLM_HERMES_PORT/v1"
hermes config set model.default "$HERMES_MODEL"
hermes config set compression.enabled false
hermes config set model.max_tokens 8192
hermes config set terminal.cwd "$WORKSPACE_DIR"
hermes config set tool_output.max_bytes 150000
hermes config set tool_output.max_lines 5000
hermes config set tool_output.max_line_length 5000

# ===========================================================================
# Hermes OpenTelemetry Plugin & Patch Setup
# ===========================================================================
echo "[INFO] Installing Hermes OpenTelemetry plugin..."

rm -rf "$HOME/.hermes/plugins/hermes_otel"
mkdir -p "$HOME/.hermes/plugins"
git clone https://github.com/briancaffey/hermes-otel.git "$HOME/.hermes/plugins/hermes_otel"

cd "$HOME/.hermes/plugins/hermes_otel"
git fetch origin --tags --depth=1
git checkout hermes-otel-v0.10.0

# Apply the advanced profiling patch once while inside the plugin directory
echo "[INFO] Applying advanced profiling patch..."
PATCH_FILE="$WORKSPACE_DIR/hermes_advanced_profiling.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "[ERROR] Patch file not found at $PATCH_FILE; profiling not installed."
elif git apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git apply "$PATCH_FILE"
    echo "[OK] Advanced profiling patch applied."
elif git apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[INFO] Patch already applied. Skipping."
else
    echo "[WARN] Patch did not apply cleanly (wrong plugin version or conflict)."
fi

# Install the plugin package in editable mode. Use the project venv explicitly:
# the system python3 on this image has no pip, and in the verified run this step
# resolved to the venv interpreter because the venv was active.
echo "[INFO] Installing plugin package in editable mode..."
"$VENV_PY" -m pip install -e .

cd "$WORKSPACE_DIR"

# Write the plugin config
cat << 'EOF' > "$HOME/.hermes/plugins/hermes_otel/config.yaml"
enabled: true
force_flush_on_session_end: true
backends:
  - type: otlp
    name: mlflow
    endpoint: http://127.0.0.1:5004/v1/traces
    metrics: false
    logs: false
    headers:
      x-mlflow-experiment-id: "0"
EOF

hermes plugins enable hermes_otel --allow-tool-override

# Install the OTel runtime deps into the venv that ACTUALLY runs Hermes.
# The launcher at $(command -v hermes) execs a specific interpreter; resolve it
# from the launcher instead of guessing a path. The old hardcoded
# "$HOME/.hermes/hermes-agent/venv" does not exist for a standard install, so
# these packages silently went nowhere and the plugin failed at runtime with
# "OpenTelemetry import error: No module named 'opentelemetry.exporter'",
# leaving MLflow with ZERO traces while every health check still passed.
HERMES_BIN="$(command -v hermes || echo "$HOME/.local/bin/hermes")"
HERMES_PY="$(grep -oE '"[^"]*/venv/bin/python"' "$HERMES_BIN" 2>/dev/null | head -1 | tr -d '"')"
if [ -z "$HERMES_PY" ] || [ ! -x "$HERMES_PY" ]; then
    for cand in /usr/local/lib/hermes-agent/venv/bin/python \
                "$HOME/.hermes/hermes-agent/venv/bin/python"; do
        [ -x "$cand" ] && HERMES_PY="$cand" && break
    done
fi

if [ -n "$HERMES_PY" ] && [ -x "$HERMES_PY" ]; then
    echo "[INFO] Installing OTel runtime into the Hermes venv: $HERMES_PY"
    # This venv ships without pip, so bootstrap it before installing.
    "$HERMES_PY" -m ensurepip --upgrade >/dev/null 2>&1
    "$HERMES_PY" -m pip install -q opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 opentelemetry-exporter-otlp-proto-http==1.42.1 pyrsmi==1.1.0 amdsmi==7.0.2 mlflow==3.13.0 psutil requests cryptography
    # Verify the import the plugin actually performs, rather than trusting pip's exit code.
    if "$HERMES_PY" -c "import opentelemetry.exporter.otlp.proto.http" >/dev/null 2>&1; then
        echo "[OK] OpenTelemetry exporter importable by Hermes: traces will reach MLflow."
    else
        echo "[WARN] OTel exporter still not importable by $HERMES_PY. MLflow will show NO traces." >&2
    fi
else
    echo "[WARN] Could not locate the Hermes venv interpreter. MLflow will show NO traces." >&2
fi
echo "[INFO] MLflow tracking available at http://${SYSTEM_IP}:5004/"

cat << EOF > "$HERMES_ENV_TMP"
# >>> hermes-audio-notebook managed block >>>
# MLflow and vLLM observability configuration
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=1
MLFLOW_TRACKING_URI=http://127.0.0.1:5004
MLFLOW_EXPERIMENT_NAME=Default
VLLM_HERMES_PORT=8001
MLFLOW_RUN_NAME=Hermes_Profiling
MLFLOW_KEEP_RUN_ACTIVE=false
MLFLOW_LOGGING_LEVEL=ERROR
MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=1
HERMES_TOOL_PROFILING=1
HERMES_GPU_EXPORTER_URL=http://localhost:5050/metrics
HERMES_PROFILING_DEBUG=0
MLFLOW_DISABLE_TELEMETRY=true
HERMES_CPU_DEBUG=0
HERMES_PROFILING_OUTPUT_DIR=${WORKSPACE_DIR}/outputs
# <<< hermes-audio-notebook managed block <<<
EOF

# Rewrite the managed block in place so repeated runs stay idempotent. The old
# version used '>>' unconditionally, so every run appended another copy of every
# variable (after 3 runs, .env was 528 lines with MLFLOW_TRACKING_URI twice).
touch "$HERMES_ENV"
awk '/^# >>> hermes-audio-notebook managed block >>>$/{skip=1} !skip{print} /^# <<< hermes-audio-notebook managed block <<<$/{skip=0}' \
    "$HERMES_ENV" > "$HERMES_ENV.stripped"
cat "$HERMES_ENV.stripped" "$HERMES_ENV_TMP" > "$HERMES_ENV"
rm -f "$HERMES_ENV.stripped" "$HERMES_ENV_TMP"

# ===========================================================================
# Custom Kokoro TTS tool
# ===========================================================================
# This is the workshop's "extend Hermes with your own tool" lesson. Without it
# the agent cannot call kokoro_tts, and instead falls back to raw shell curl
# against :8092. That still produces audio, so it looks like it worked, but the
# custom-tool step never happened and the MLflow trace shows a pile of terminal
# spans instead of one clean kokoro_tts tool span.
echo "[INFO] Deploying custom Kokoro TTS tool..."
HERMES_TOOLS_DIR="$HOME/.hermes/hermes-agent/tools"
mkdir -p "$HERMES_TOOLS_DIR"
if [ -f "$WORKSPACE_DIR/custom_tools/kokoro_tts_tool.py" ]; then
    cp "$WORKSPACE_DIR/custom_tools/kokoro_tts_tool.py" "$HERMES_TOOLS_DIR/"
    echo "[OK] Copied kokoro_tts_tool.py -> $HERMES_TOOLS_DIR"
else
    echo "[WARN] custom_tools/kokoro_tts_tool.py not found; the kokoro_tts tool will be unavailable." >&2
fi

# ===========================================================================
# ROCm headers required by MIOpen's runtime kernel compilation
# ===========================================================================
# Kokoro's LSTM path makes MIOpen JIT-compile its dropout kernel, which
# #includes <rocrand/rocrand_xorwow.h> and <hip/hip_runtime.h>. The AMD AI/ML
# Ready Image ships neither header tree under /opt/rocm/include, so the compile
# fails and the server dies with a bare, misleading:
#     RuntimeError: miopenStatusUnknownError
# (A plain torch LSTM on GPU works fine, which is why this looks like a Kokoro
# bug rather than a missing-headers problem.)
if [ ! -f /opt/rocm/include/rocrand/rocrand_xorwow.h ] || [ ! -f /opt/rocm/include/hip/hip_runtime.h ]; then
    echo "[INFO] ROCm dev headers (rocrand/hip) missing; installing so MIOpen can compile its kernels..."
    sudo apt-get install -y -qq rocrand-dev hip-dev >/dev/null 2>&1 || true
fi

# Fallback: the AMD AI/ML Ready Image has no ROCm apt repo configured, so the
# apt install above is usually a no-op. The vLLM ROCm image we just built does
# ship the full ROCm 7.2.3 header tree, so copy the headers out of it.
if [ ! -f /opt/rocm/include/rocrand/rocrand_xorwow.h ] || [ ! -f /opt/rocm/include/hip/hip_runtime.h ]; then
    echo "[INFO] Extracting ROCm headers from $IMAGE_NAME ..."
    HDR_CID=$(sudo docker create "$IMAGE_NAME" 2>/dev/null)
    if [ -n "$HDR_CID" ]; then
        sudo mkdir -p /opt/rocm/include
        for hdr in rocrand hip hsa half; do
            if [ ! -e "/opt/rocm/include/$hdr" ]; then
                sudo docker cp "$HDR_CID:/opt/rocm/include/$hdr" /opt/rocm/include/ >/dev/null 2>&1 \
                    && echo "[INFO]   installed /opt/rocm/include/$hdr"
            fi
        done
        sudo docker rm -f "$HDR_CID" >/dev/null 2>&1
    fi
fi

if [ ! -f /opt/rocm/include/rocrand/rocrand_xorwow.h ] || [ ! -f /opt/rocm/include/hip/hip_runtime.h ]; then
    echo "[WARN] rocrand/hip headers still absent under /opt/rocm/include." >&2
    echo "       Kokoro will likely fail with 'miopenStatusUnknownError'." >&2
    echo "       Install the ROCm dev packages for this image, or copy the header" >&2
    echo "       trees (rocrand/, hip/, hsa/, half/) into /opt/rocm/include." >&2
else
    echo "[OK] ROCm rocrand/hip headers present."
fi

# ===========================================================================
# Kokoro TTS server
# ===========================================================================
echo "[INFO] Setting up the Kokoro TTS server environment ($KOKORO_ENV)..."
if [ ! -d "$KOKORO_ENV" ]; then
    echo "[INFO] Creating Python venv at $KOKORO_ENV..."
    python3 -m venv "$KOKORO_ENV"
fi
"$KOKORO_ENV/bin/python" -m pip install -q --upgrade pip
echo "[INFO] Installing PyTorch (ROCm 7.2)..."
"$KOKORO_ENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
echo "[INFO] Installing kokoro, soundfile, fastapi, uvicorn, streamlit..."
"$KOKORO_ENV/bin/pip" install kokoro soundfile fastapi uvicorn
"$KOKORO_ENV/bin/pip" install 'streamlit>=1.30'

# MIOpen needs this lock directory to persist its kernel DB and avoid errors on
# new shapes; create it before the server starts.
mkdir -p "$HOME/.config/miopen/miopen-lockfiles"

echo "[INFO] Launching Kokoro TTS server on port $KOKORO_PORT..."
KOKORO_PORT=$KOKORO_PORT "$KOKORO_ENV/bin/python" "$KOKORO_SERVER" > "$WORKSPACE_DIR/kokoro_server.log" 2>&1 &
KOKORO_PID=$!
echo "[INFO] Kokoro server started (PID $KOKORO_PID, logs: $WORKSPACE_DIR/kokoro_server.log)."

echo "[INFO] Waiting for Kokoro server /health on port $KOKORO_PORT..."
kokoro_ready=0
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$KOKORO_PORT/health" || echo "000")
    if [ "$code" -eq 200 ]; then
        echo "[OK] Kokoro TTS server is active (sequential + batched)."
        kokoro_ready=1
        break
    fi
    # Fail fast if the background process already died.
    if ! kill -0 "$KOKORO_PID" 2>/dev/null; then
        break
    fi
    sleep 3
done
if [ "$kokoro_ready" -ne 1 ]; then
    fail "Kokoro TTS server" "$WORKSPACE_DIR/kokoro_server.log"
fi

# ===========================================================================
# Telemetry dashboard (Streamlit)
# ===========================================================================
# Runs for the whole session; --server.address 0.0.0.0 makes it reachable from
# other machines.
DASHBOARD_APP="$WORKSPACE_DIR/hermes_profiler.py"
if [ -f "$DASHBOARD_APP" ]; then
    echo "[INFO] Launching telemetry dashboard on port 8501..."
    # Invoke via the Kokoro env's interpreter (that is where streamlit was
    # installed, line ~513). A bare `streamlit` only resolves when a venv
    # happens to be active, so unactivated runs failed with
    # "streamlit: command not found".
    "$KOKORO_ENV/bin/python" -m streamlit run "$DASHBOARD_APP" \
        --server.address 0.0.0.0 \
        --server.port 8501 \
        --server.headless true > streamlit_dashboard.log 2>&1 &
    STREAMLIT_PID=$!
    echo "[INFO] Dashboard started (PID $STREAMLIT_PID)."

    echo "[INFO] Waiting for Streamlit dashboard /_stcore/health on port 8501..."
    streamlit_ready=0
    for i in $(seq 1 30); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8501/_stcore/health" || echo "000")
        if [ "$code" -eq 200 ]; then
            echo "[OK] Streamlit dashboard is up."
            streamlit_ready=1
            break
        fi
        # Fail fast if the background process already died.
        if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            break
        fi
        sleep 2
    done
    if [ "$streamlit_ready" -ne 1 ]; then
        fail "Streamlit telemetry dashboard" "$WORKSPACE_DIR/streamlit_dashboard.log"
    fi
else
    fail "Streamlit telemetry dashboard" ""
fi

echo -e "\n========================================================================="
echo "[OK] Setup complete."
echo "  vLLM endpoint:        http://${SYSTEM_IP}:$VLLM_HERMES_PORT"
echo "  Kokoro TTS server:    http://${SYSTEM_IP}:$KOKORO_PORT"
echo "  MLflow tracking:      http://${SYSTEM_IP}:5004"
echo "  Telemetry dashboard:  http://${SYSTEM_IP}:8501"
echo "========================================================================="
echo "[INFO] Holding the session open. Press Ctrl+C to stop all services and exit."

# Keep the shell process alive so the cleanup trap stays active.
while true; do
    sleep 60
done
