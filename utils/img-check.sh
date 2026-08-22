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

ENV_FILE=/root/.hermes/.env
OTEL_CFG=/root/.hermes/plugins/hermes_otel/config.yaml
fail=0

note() { printf '  %s\n' "$1"; }

echo "[img-check] .env required keys"
# HERMES_TOOL_PROFILING gates tool_profiler._enabled(). Without it the CPU/GPU
# poller never starts, no timeline CSVs are written, and every MLflow run has
# zero artifacts while still logging vLLM metrics, so nothing looks wrong.
for key in \
    MLFLOW_TRACKING_URI \
    HERMES_PROFILING_OUTPUT_DIR \
    HERMES_GPU_EXPORTER_URL \
    KOKORO_SERVER_URL \
    MLFLOW_EXPERIMENT_NAME \
    MLFLOW_KEEP_RUN_ACTIVE \
    HERMES_TOOL_PROFILING
do
    if grep -qE "^${key}=" "$ENV_FILE"; then
        note "OK   ${key}=$(grep -m1 -E "^${key}=" "$ENV_FILE" | cut -d= -f2-)"
    else
        note "MISS ${key} absent from ${ENV_FILE}"
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

echo "[img-check] model weights are baked into the image"
# The whole point of baking is that vLLM never downloads at start. If a future
# edit reorders or drops the bake step, the image still builds and still boots,
# it just quietly downloads 60 GB on a live workshop machine. Assert the shards
# are on disk, resolved WITHOUT network access.
if python3 - <<'PY'
import json
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    sys.exit(f"huggingface_hub unavailable: {exc}")

model = os.environ.get("HERMES_MODEL")
kokoro = os.environ.get("KOKORO_MODEL")
if not model or not kokoro:
    sys.exit("HERMES_MODEL or KOKORO_MODEL unset")

# local_files_only proves the bytes are present rather than fetchable.
path = snapshot_download(model, local_files_only=True)
index = os.path.join(path, "model.safetensors.index.json")
with open(index) as fh:
    shards = sorted({v for v in json.load(fh)["weight_map"].values()})

total = 0
for s in shards:
    f = os.path.join(path, s)
    if not os.path.exists(f):
        sys.exit(f"missing shard {s}")
    total += os.path.getsize(f)

if total < 40e9:
    sys.exit(f"shards total only {total / 1e9:.2f} GB")

# The TTS voice model is equally load-bearing: kokoro_server.py calls
# hf_hub_download at startup and dies offline without it, which hangs the
# entrypoint on the Kokoro health check.
kpath = snapshot_download(kokoro, local_files_only=True)
ktotal = sum(
    os.path.getsize(os.path.join(dp, n))
    for dp, _, ns in os.walk(kpath)
    for n in ns
)
if ktotal < 100e6:
    sys.exit(f"{kokoro} only {ktotal / 1e6:.1f} MB")

print(f"LLM {len(shards)} shard(s) {total / 1e9:.2f} GB; TTS {ktotal / 1e9:.2f} GB")
PY
then
    note "OK   weights present locally (no runtime download)"
else
    note "MISS model weights are NOT baked in; vLLM would download at start"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "[img-check] FAILED"
    exit 1
fi
echo "[img-check] all checks passed"
