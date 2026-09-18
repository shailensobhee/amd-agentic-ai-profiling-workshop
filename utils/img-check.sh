#!/usr/bin/env bash
# Build-time parity guard for the workshop image.
#
# The bare-host installer (utils/helper.sh) and the Dockerfile configure the
# SAME agent two different ways, so they drift. Every drift found so far was
# silent: the image built clean, all services reported healthy, and the notebook
# ran with zero errors while the telemetry it is supposed to teach was dead.
#
# This runs INSIDE the image during build and fails it loudly instead.
set -euo pipefail

OTEL_CFG=/root/.hermes/plugins/hermes_otel/config.yaml
fail=0

note() { printf '  %s\n' "$1"; }

echo "[img-check] hermes-otel config backends and metrics"
# The plugin reads config.yaml (matching utils/helper.sh); no ~/.hermes/.env is
# needed. Traces must go to the mlflow backend and CPU/GPU metrics to the lgtm
# backend, with host metrics enabled, or the dashboard renders empty.
for needle in 'host_metrics: true' 'name: mlflow' 'name: lgtm'; do
    if grep -qF "$needle" "$OTEL_CFG"; then
        note "OK   ${needle}"
    else
        note "MISS '${needle}' absent from ${OTEL_CFG}"
        fail=1
    fi
done

echo "[img-check] OTLP experiment header"
# MLflow 3.x answers 422 to OTLP spans with no experiment header. The exporter
# only prints that on stderr, so the sole symptom is an empty Traces tab.
if grep -q 'x-mlflow-experiment-id' "$OTEL_CFG"; then
    note "OK   x-mlflow-experiment-id present"
else
    note "MISS x-mlflow-experiment-id absent from ${OTEL_CFG}"
    fail=1
fi

echo "[img-check] dashboard assets resolve"
# hermes_profiler.py lives in utils/ but assets/ sits at the workshop root, so a
# path anchored on __file__ alone silently misses and the AMD logo vanishes.
python3 - <<'PY' || fail=1
import os, sys
sys.path.insert(0, "/workshop/utils")
here = "/workshop/utils"
cands = [os.path.join(here, os.pardir, "assets", "images", "amd_logo.png"),
         os.path.join(here, "assets", "images", "amd_logo.png")]
hit = next((os.path.normpath(p) for p in cands if os.path.exists(p)), None)
print(f"  {'OK   logo ' + hit if hit else 'MISS amd_logo.png not found'}")
sys.exit(0 if hit else 1)
PY

echo "[img-check] streamlit theme resolves from the dashboard CWD"
# Streamlit reads .streamlit/config.toml from the PROCESS CWD, not the script
# dir. docker-entrypoint.sh must therefore cd into utils/ before running the
# app, or the AMD theme is silently dropped.
if [ -f /workshop/utils/.streamlit/config.toml ]; then
    resolved=$(cd /workshop/utils && streamlit config show 2>/dev/null \
               | grep -E '^primaryColor' | head -1 || true)
    if [ -n "$resolved" ]; then
        note "OK   ${resolved}"
    else
        note "MISS primaryColor unset when resolved from /workshop/utils"
        fail=1
    fi
    if grep -qE 'cd "\$\{UTILS_DIR\}" && streamlit run' /usr/local/bin/docker-entrypoint.sh; then
        note "OK   entrypoint launches streamlit from UTILS_DIR"
    else
        note "MISS entrypoint does not cd to UTILS_DIR; theme will not load"
        fail=1
    fi
else
    note "MISS /workshop/utils/.streamlit/config.toml absent"
    fail=1
fi

echo "[img-check] TTS server handoff to the notebook"
# The notebook starts the Kokoro server, as it does on a bare host under
# utils/helper.sh. Three things in the image carry that handoff; any one of them
# missing surfaces only when a participant runs the cell that starts the server.
#
# 1. The entrypoint leaves the server to the notebook. cleanup()'s pkill is its
#    only reference to kokoro_server.py; any other one launches the server.
if grep -n 'kokoro_server\.py' /usr/local/bin/docker-entrypoint.sh \
   | grep -qv 'pkill'; then
    note "MISS entrypoint launches kokoro_server.py; the notebook starts it"
    fail=1
else
    note "OK   entrypoint leaves the TTS server to the notebook"
fi

# 2. The script that cell runs.
if [ -f /workshop/utils/start_kokoro_server.sh ]; then
    note "OK   utils/start_kokoro_server.sh present"
else
    note "MISS /workshop/utils/start_kokoro_server.sh absent; the notebook cannot start the server"
    fail=1
fi

# 3. The interpreter the cell and that script call, carrying GPU PyTorch and
#    Kokoro. helper.sh provides it on a bare host; the Dockerfile provides it here.
for bin in /workshop/env/bin/python /workshop/env/bin/pip; do
    if [ -x "$bin" ]; then
        note "OK   ${bin}"
    else
        note "MISS ${bin} absent; the notebook's './env/bin/...' cells cannot run"
        fail=1
    fi
done

if /workshop/env/bin/python -c 'import torch, kokoro' 2>/dev/null; then
    note "OK   env/ interpreter imports torch and kokoro"
else
    note "MISS env/ interpreter cannot import torch and kokoro"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "[img-check] FAILED"
    exit 1
fi
echo "[img-check] all checks passed"
