#!/usr/bin/env bash
# One-shot backend setup for the single-container tutorial. Started from the
# notebook (Environment setup, step 2). Everything runs under the system
# python3 that already carries vLLM / PyTorch / ROCm in the official image.
set -uo pipefail
log(){ echo "[$(date +%H:%M:%S)] $*"; }

HERMES_MODEL="meta-models/Muse-Glimmer-30B"
VLLM_PORT=8001
MLFLOW_PORT=5004
PROM_PORT=9090
GRAFANA_PORT=3000
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore DEBIAN_FRONTEND=noninteractive

# --- OS + Python packages the official image does not ship -----------------
log "Installing OS audio packages for Kokoro..."
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq espeak-ng libsndfile1 procps lsof ripgrep >/dev/null 2>&1 || true

log "Installing Python packages (Kokoro, MLflow, dashboard, telemetry)..."
python3 -m pip install -q \
  kokoro soundfile fastapi "uvicorn[standard]" \
  "streamlit>=1.30" "mlflow>=3.0.0" plotly pandas ipywidgets matplotlib kaleido \
  "opentelemetry-sdk==1.44.0" "opentelemetry-exporter-otlp-proto-http==1.44.0" \
  psutil requests

# --- Metrics: upstream Prometheus (native OTLP receiver) + Grafana ---------
# Resolve the latest upstream releases so the tutorial tracks current versions.
cd /root
PROM_VER="$(curl -fsSL https://api.github.com/repos/prometheus/prometheus/releases/latest | python3 -c 'import sys,json;print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"
GRAF_VER="$(curl -fsSL https://api.github.com/repos/grafana/grafana/releases/latest | python3 -c 'import sys,json;print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"
log "Prometheus ${PROM_VER}, Grafana ${GRAF_VER}"

if [ ! -x /root/prom/prometheus ]; then
  curl -fsSL -o /root/prom.tgz "https://github.com/prometheus/prometheus/releases/download/v${PROM_VER}/prometheus-${PROM_VER}.linux-amd64.tar.gz"
  tar xzf /root/prom.tgz -C /root && rm -rf /root/prom && mv "/root/prometheus-${PROM_VER}.linux-amd64" /root/prom
fi
cat > /root/prom/prometheus.yml <<'CFG'
global:
  scrape_interval: 15s
otlp:
  promote_resource_attributes: [service.instance.id, service.name]
storage:
  tsdb:
    out_of_order_time_window: 30m
CFG
pkill -f '/root/prom/prometheus' 2>/dev/null || true; sleep 1
( cd /root/prom && nohup ./prometheus --config.file=prometheus.yml \
    --web.listen-address=0.0.0.0:${PROM_PORT} --web.enable-otlp-receiver \
    --storage.tsdb.path=/root/prom/data > /root/prom.log 2>&1 & )

if [ ! -x /root/grafana/bin/grafana ]; then
  curl -fsSL -o /root/graf.tgz "https://dl.grafana.com/oss/release/grafana-${GRAF_VER}.linux-amd64.tar.gz"
  tar xzf /root/graf.tgz -C /root && rm -rf /root/grafana
  { mv "/root/grafana-v${GRAF_VER}" /root/grafana 2>/dev/null || mv "/root/grafana-${GRAF_VER}" /root/grafana; }
fi
mkdir -p /root/grafana/conf/provisioning/datasources
cat > /root/grafana/conf/provisioning/datasources/prom.yaml <<DS
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:${PROM_PORT}
    isDefault: true
DS
cat > /root/grafana.ini <<INI
[server]
http_addr = 0.0.0.0
http_port = ${GRAFANA_PORT}
[auth.anonymous]
enabled = true
[security]
admin_user = admin
admin_password = admin
INI
pkill -f 'grafana server' 2>/dev/null || true; sleep 1
( cd /root/grafana && nohup ./bin/grafana server --config=/root/grafana.ini \
    --homepath=/root/grafana > /root/grafana.log 2>&1 & )

# --- vLLM: the agent's model, served with AITER on the MI300X --------------
log "Starting vLLM (${HERMES_MODEL}); the first start downloads ~60 GB of weights..."
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true; sleep 1
HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER=1 \
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "${HERMES_MODEL}" --served-model-name "${HERMES_MODEL}" \
  --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer --attention-backend ROCM_AITER_FA \
  --generation-config auto --enable-prefix-caching \
  --host 0.0.0.0 --port ${VLLM_PORT} > /root/vllm.log 2>&1 &

# --- MLflow: stores the execution traces -----------------------------------
log "Starting MLflow on :${MLFLOW_PORT}..."
pkill -f 'mlflow server' 2>/dev/null || true; sleep 1
nohup python3 -m mlflow server --host 0.0.0.0 --port ${MLFLOW_PORT} \
  --backend-store-uri sqlite:////root/mlflow.db --allowed-hosts '*' > /root/mlflow.log 2>&1 &

# --- Hermes + hermes-otel plugin -------------------------------------------
if ! command -v hermes >/dev/null 2>&1; then
  log "Installing the Hermes agent..."
  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup >/dev/null 2>&1 || true
fi

# Resolve the interpreter the hermes launcher actually runs. The installer
# creates an isolated venv that has neither pip nor the system site-packages,
# so the plugin and amdsmi must be installed into THAT interpreter or the agent
# emits no telemetry and the dashboard stays empty with no error.
HVPY="$(python3 - <<'PY'
import shutil, re, os
launcher = shutil.which("hermes")
py = ""
if launcher:
    try:
        txt = open(launcher).read()
        m = re.search(r'exec\s+"?([^"\s]+/venv/bin/python)"?', txt)
        if m: py = m.group(1)
    except Exception:
        pass
for cand in (py, "/usr/local/lib/hermes-agent/venv/bin/python",
             os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python")):
    if cand and os.access(cand, os.X_OK):
        print(cand); break
PY
)"
log "Hermes venv python: ${HVPY:-not found}"
if [ -n "${HVPY}" ]; then
  "${HVPY}" -m ensurepip --upgrade >/dev/null 2>&1 || {
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /root/get-pip.py && "${HVPY}" /root/get-pip.py >/dev/null 2>&1; }
fi

# Install the hermes-otel plugin (latest release tag) and enable it.
rm -rf /root/.hermes/plugins/hermes_otel; mkdir -p /root/.hermes/plugins
git clone -q https://github.com/briancaffey/hermes-otel.git /root/.hermes/plugins/hermes_otel
( cd /root/.hermes/plugins/hermes_otel
  HOTEL="$(git tag -l 'hermes-otel-v*' | sort -V | tail -1)"; [ -n "$HOTEL" ] && git checkout -q "$HOTEL"
  log "hermes-otel ${HOTEL:-default-branch}"
  python3 -m pip install -q -e . >/dev/null 2>&1 || true
  if [ -n "${HVPY}" ]; then
    "${HVPY}" -m pip install -q -e . psutil requests \
      "opentelemetry-sdk==1.44.0" "opentelemetry-exporter-otlp-proto-http==1.44.0" >/dev/null 2>&1 || true
  fi
)
# amdsmi (GPU metric source) ships with ROCm, not on the venv path. Expose it.
if [ -n "${HVPY}" ]; then
  AMDSMI_PARENT="$(python3 -c 'import amdsmi,os;print(os.path.dirname(os.path.dirname(amdsmi.__file__)))' 2>/dev/null || true)"
  SITE="$("${HVPY}" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null || true)"
  [ -n "${AMDSMI_PARENT}" ] && [ -n "${SITE}" ] && echo "${AMDSMI_PARENT}" > "${SITE}/amdsmi_system.pth"
fi

# Point the plugin's metrics at Prometheus (OTLP) and its traces at MLflow.
# hermes-otel rewrites the /v1/traces suffix per signal, so aiming a metrics
# backend at Prometheus's /api/v1/otlp/v1/traces makes metrics land on
# /api/v1/otlp/v1/metrics, which is the Prometheus OTLP metrics endpoint.
cat > /root/.hermes/plugins/hermes_otel/config.yaml <<YML
host_metrics: true
host_metrics_gpu: amd
host_metrics_interval_ms: 100
flush_interval_ms: 3000
backends:
  - type: otlp
    name: mlflow-traces
    endpoint: http://127.0.0.1:${MLFLOW_PORT}/v1/traces
    metrics: false
    # MLflow 3.x OTLP trace ingestion requires the target experiment id as an
    # HTTP header; without it every span batch is rejected with HTTP 422 and no
    # traces are stored (the profiling dashboard would stay empty). "0" is the
    # Default experiment MLflow creates on first start.
    headers:
      x-mlflow-experiment-id: "0"
  - type: lgtm
    name: prometheus-metrics
    endpoint: http://127.0.0.1:${PROM_PORT}/api/v1/otlp/v1/traces
    traces: false
    metrics: true
YML

hermes config set model.provider custom >/dev/null 2>&1 || true
hermes config set model.base_url "http://localhost:${VLLM_PORT}/v1" >/dev/null 2>&1 || true
hermes config set model.default "${HERMES_MODEL}" >/dev/null 2>&1 || true
hermes config set compression.enabled false >/dev/null 2>&1 || true
hermes config set model.max_tokens 16384 >/dev/null 2>&1 || true
hermes config set display.show_reasoning false >/dev/null 2>&1 || true
hermes plugins enable hermes_otel --allow-tool-override >/dev/null 2>&1 || true

# --- Wait for the services the notebook needs next -------------------------
wait_http(){ local url="$1" name="$2" t="${3:-600}" w=0 c
  while [ "$w" -lt "$t" ]; do
    c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null)"; c="${c:-000}"
    [[ "$c" =~ ^2 ]] && { log "  ${name} is up (HTTP ${c})."; return 0; }
    sleep 5; w=$((w+5)); [ $((w % 60)) -eq 0 ] && log "  waiting for ${name}... ${w}s (HTTP ${c})"
  done; log "  ${name} did not become ready in ${t}s (see /root/*.log)"; return 1; }

wait_http "http://localhost:${PROM_PORT}/-/ready" "Prometheus" 120 || true
wait_http "http://localhost:${GRAFANA_PORT}/api/health" "Grafana" 120 || true
wait_http "http://localhost:${MLFLOW_PORT}/health" "MLflow" 180 || true
log "Waiting for vLLM to load the model (first run downloads weights)..."
wait_http "http://localhost:${VLLM_PORT}/v1/models" "vLLM" 3600 || true

# --- Profiling dashboard (Streamlit) ------------------------------------
# The show_session_overview() cells render an inline overview, but the full
# five-tab dashboard is a Streamlit app. Launch it from utils/ so Streamlit
# resolves utils/.streamlit/config.toml (the AMD theme); from any other CWD the
# theme is silently dropped. Port 8501 matches the links the notebook prints.
log "Launching the profiling dashboard on :8501..."
pkill -f "streamlit run" 2>/dev/null || true; sleep 1
( cd "${UTILS_DIR:-utils}" 2>/dev/null || cd utils
  nohup "$(command -v streamlit)" run hermes_profiler.py \
    --server.address 0.0.0.0 --server.port 8501 --server.headless true \
    > /root/streamlit_dashboard.log 2>&1 & )
wait_http "http://localhost:8501/_stcore/health" "Dashboard" 120 || true

log "Backend ready. Dashboard :8501  JupyterLab :8888  vLLM :8001  MLflow :5004  Grafana :3000  Prometheus :9090"
