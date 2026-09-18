#!/bin/bash
#
# start_kokoro_server.sh
#
# Starts the local Kokoro TTS server and waits until it answers /health.
#
# Called from the workshop notebook, after the Edge TTS baseline has been
# profiled, so the local GPU engine comes up as its own visible step.
# utils/helper.sh on a bare host and utils/docker-entrypoint.sh inside the
# workshop image both prepare what this depends on (the env/ interpreter, the
# dependencies, the GPU environment) and leave the server itself to this script.
#
# Safe to re-run: a healthy server already on the port is left alone rather
# than a second one being started.
#
# Usage:
#     bash utils/start_kokoro_server.sh
#     KOKORO_PORT=9000 bash utils/start_kokoro_server.sh

set -uo pipefail

# Resolve siblings relative to this script, not the caller's CWD, so the script
# works from the repo root, from utils/, or from a notebook kernel started
# anywhere else.
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$UTILS_DIR/.." && pwd)"

KOKORO_PORT="${KOKORO_PORT:-8092}"
APP_ENV="$WORKSPACE_DIR/env"
KOKORO_SERVER="$UTILS_DIR/kokoro_server.py"
KOKORO_LOG="$WORKSPACE_DIR/kokoro_server.log"

if [ ! -f "$KOKORO_SERVER" ]; then
    echo "[ERROR] $KOKORO_SERVER not found."
    exit 1
fi

if [ ! -x "$APP_ENV/bin/python" ]; then
    echo "[ERROR] $APP_ENV/bin/python not found."
    echo "        On a bare host it is created by utils/helper.sh: check that it"
    echo "        has finished and reported no errors."
    echo "        In the workshop container it ships in the image, so check that"
    echo "        no volume is mounted over /workshop."
    exit 1
fi

# Idempotent: if a healthy server is already on the port, keep it.
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$KOKORO_PORT/health")" = "200" ]; then
    echo "[OK] Kokoro server already running on port $KOKORO_PORT."
    exit 0
fi

echo "[INFO] Launching Kokoro TTS server on port $KOKORO_PORT..."
KOKORO_PORT=$KOKORO_PORT nohup "$APP_ENV/bin/python" "$KOKORO_SERVER" \
    > "$KOKORO_LOG" 2>&1 &
KOKORO_PID=$!
echo "[INFO] Started (PID $KOKORO_PID, logs: $KOKORO_LOG)."

echo "[INFO] Waiting for /health"
for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$KOKORO_PORT/health")
    if [ "${code:-000}" = "200" ]; then
        # Report the device instead of the raw /health JSON. Worth one line:
        # the engine falls back to CPU when ROCm is not visible, and a CPU run
        # still works, just far slower, which would quietly ruin the profiling.
        # torch.cuda.get_device_name() returns an empty string on some ROCm
        # builds, so the GPU name is only appended when there is one.
        device=$(curl -s "http://localhost:$KOKORO_PORT/health" |
            "$APP_ENV/bin/python" -c 'import json,sys; h=json.load(sys.stdin); g=(h.get("gpu") or "").strip(); print(h["device"] + (" (" + g + ")" if g else ""))' 2>/dev/null)
        echo "[OK] Kokoro TTS server is active on ${device:-unknown device}."
        exit 0
    fi
    # Fail fast if the process already died rather than waiting out the loop.
    if ! kill -0 "$KOKORO_PID" 2>/dev/null; then
        echo "[ERROR] Server exited during startup. Last lines of $KOKORO_LOG:"
        tail -n 20 "$KOKORO_LOG"
        exit 1
    fi
    sleep 3
done

echo "[ERROR] Timed out after 360s. Last lines of $KOKORO_LOG:"
tail -n 20 "$KOKORO_LOG"
exit 1
