# =============================================================================
# AMD Agentic AI Profiling Workshop
# =============================================================================
# All-in-one workshop image for AMD Instinct MI300X.
#
# Everything runs INSIDE this single container: vLLM (Muse-Glimmer-30B), the
# Hermes agent with the hermes-otel OpenTelemetry plugin, the Kokoro TTS server,
# MLflow, Grafana otel-lgtm, the Streamlit telemetry dashboard, and JupyterLab.
# There is no nested Docker, unlike utils/helper.sh which orchestrates sibling
# containers on a bare host. See utils/docker-entrypoint.sh for the in-container
# service manager.
#
# BUILD (from the repo root):
#   docker build -f utils/Dockerfile -t amd-agentic-ai-profiling:mi300x .
#
# RUN:
#   docker run -d --name amd-agentic-ai-profiling \
#     --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined --group-add video --ipc=host \
#     --shm-size 16G -p 8888:8888 -p 8501:8501 -p 5004:5004 \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     amd-agentic-ai-profiling:mi300x
#
# The model weights (about 60 GB) are downloaded on first start. Mount the
# Hugging Face cache as shown so the download is paid only once per host.
# =============================================================================

# -----------------------------------------------------------------------------
# LATEST-SOFTWARE-ALWAYS: ROCm 7.14.1 + vLLM 0.29.x, built from source
# -----------------------------------------------------------------------------
# The prebuilt vllm/vllm-openai-rocm release images (even v0.29.0) are still
# built on ROCm 7.2.3 / Ubuntu 22.04 (verified 2026-09-16 from vLLM's own
# docker/Dockerfile.rocm_base at the v0.29.0 tag: BASE_IMAGE=rocm/dev-ubuntu-22.04:7.2.3-complete).
# To ship BOTH the newest ROCm (7.14.1) AND the newest vLLM (0.29.x) we build
# vLLM from its released source tag on the ROCm 7.14.1 dev base, using a torch
# wheel from the pytorch.org rocm7.14 channel. This exact recipe was proven end
# to end on a live MI300X on 2026-09-16 (vllm 0.29.0+rocm714 on torch
# 2.14.0+rocm7.14 serving over the OpenAI API with real GPU inference).
#
# Muse-Glimmer-30B support was merged upstream in PR #51655 (2026-08-14, shipped
# in vLLM >= 0.28.1), so it is NATIVE in 0.29.x: no source overlay is needed,
# unlike earlier revisions of this image that cloned vLLM and rsynced PR #51655's
# .py files onto an older release.
#
# The ROCm 7.14.1 dev base ships system Python 3.12 with NO conda, so a single
# venv holds torch, vLLM, Kokoro TTS, MLflow, Streamlit and JupyterLab. Kokoro
# requires Python <3.13, which 3.12 satisfies, so no second interpreter is
# needed. Prepending the venv to PATH makes a bare `python3` resolve to it
# everywhere (entrypoint, Jupyter kernels), matching how helper.sh expects it.

# The Grafana otel-lgtm metrics backend is pulled in as a build stage so it ships
# inside this image and runs as a local process. utils/helper.sh runs the same
# `grafana/otel-lgtm` image as a sibling container; there is no nested Docker
# here, so its payload is copied in and started by the entrypoint. It receives
# the plugin's OTLP metrics on :4318 and serves them from Prometheus on :9090,
# which the dashboard queries.
#
# COPY --from only copies files, not the image's own ENV vars. otel-lgtm's
# run-*.sh scripts require several *_VERSION vars (set via ENV in its own
# Dockerfile) or they abort with "unbound variable", so those are captured here
# and sourced by the entrypoint before starting it.
FROM grafana/otel-lgtm:latest AS lgtm
RUN env | grep -E '_VERSION=' > /otel-lgtm.env

FROM rocm/dev-ubuntu-24.04:7.14.1-full

LABEL org.opencontainers.image.title="AMD Agentic AI Profiling Workshop"
LABEL org.opencontainers.image.description="Profile and optimize an AI agent on AMD Instinct MI300X: Hermes Agent, MLflow telemetry, Kokoro TTS, JupyterLab."
LABEL org.opencontainers.image.vendor="AMD"
LABEL org.opencontainers.image.source="https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The vLLM venv is first on PATH so a bare `python3` resolves to it everywhere.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_VENV=/opt/vllm-venv \
    PATH="/opt/vllm-venv/bin:/opt/rocm/bin:/root/.local/bin:${PATH}" \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PYTORCH_ROCM_ARCH=gfx942 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    LD_LIBRARY_PATH="/opt/rocm/lib" \
    HF_HOME=/root/.cache/huggingface \
    WORKSHOP_DIR=/workshop \
    HERMES_PROFILING_CACHE_DIR=/workshop/profiling_cache \
    HERMES_GPU=0 \
    VLLM_HERMES_PORT=8001 \
    KOKORO_PORT=8092 \
    MLFLOW_PORT=5004 \
    DASHBOARD_PORT=8501 \
    JUPYTER_PORT=8888 \
    HERMES_MODEL=meta-models/Muse-Glimmer-30B \
    HERMES_PROXY_BASE=https://notebooks.amd.com

# HERMES_PROXY_BASE is the base for the notebook's "detailed view" links and the
# URLs the entrypoint prints. It is inherited by the JupyterLab kernels the
# entrypoint launches, so the notebook picks it up automatically. Default is the
# AMD hosted-notebook proxy; override at run time for other environments, e.g.:
#   docker run -e HERMES_PROXY_BASE="" ...            # direct http://127.0.0.1:<port>/
#   docker run -e HERMES_PROXY_BASE="10.0.0.5" ...    # direct http://10.0.0.5:<port>/
#   docker run -e HERMES_PROXY_BASE="https://my-proxy" ...

# -----------------------------------------------------------------------------
# 1. OS packages (runtime + vLLM build toolchain)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates procps lsof espeak-ng libsndfile1 \
        ripgrep tini rsync \
        python3.12-venv python3.12-dev build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Grafana otel-lgtm payload (OTel collector + Prometheus + Grafana), started by
# the entrypoint to receive OTLP metrics and serve Prometheus on :9090.
COPY --from=lgtm /otel-lgtm /otel-lgtm
COPY --from=lgtm /otel-lgtm.env /otel-lgtm/lgtm.env

# -----------------------------------------------------------------------------
# 2. vLLM 0.29.x built from source on ROCm 7.14.1 (native Muse-Glimmer)
# -----------------------------------------------------------------------------
# Pinned to a released tag for reproducibility; bump VLLM_VERSION to move it.
# torch/torchvision come from the pytorch.org rocm7.14 channel, which serves the
# newest torch built for ROCm 7.14. That wheel is self-contained (it bundles
# triton-rocm and a full rocm-sdk 7.14 pip stack), so no separate ROCm/triton
# source build is needed.
ARG VLLM_VERSION=0.29.0
RUN python3 -m venv "${VLLM_VENV}" \
    && "${VLLM_VENV}/bin/pip" install --no-cache-dir --upgrade pip wheel setuptools setuptools_scm \
    && echo "===== torch from the rocm7.14 channel =====" \
    && "${VLLM_VENV}/bin/pip" install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm7.14 \
    && TORCH_VER="$("${VLLM_VENV}/bin/python" -c 'import torch; print(torch.__version__)')" \
    && TORCH_HIP="$("${VLLM_VENV}/bin/python" -c 'import torch; print(torch.version.hip or "")')" \
    && echo "resolved torch: ${TORCH_VER} (hip ${TORCH_HIP})" \
    && if [ -z "${TORCH_HIP}" ]; then echo "ERROR: torch has no HIP build" >&2; exit 1; fi \
    && echo "===== amd-smi python wheel (from the ROCm tree) =====" \
    && if [ -d /opt/rocm/share/amd_smi ]; then \
         "${VLLM_VENV}/bin/pip" install --no-cache-dir /opt/rocm/share/amd_smi; \
       else echo "warn: /opt/rocm/share/amd_smi missing"; fi \
    && echo "===== vLLM build-time deps =====" \
    && "${VLLM_VENV}/bin/pip" install --no-cache-dir --upgrade \
        numba scipy "huggingface-hub[cli]" hf_transfer \
    && echo "===== clone + build vLLM v${VLLM_VERSION} =====" \
    && git clone -q --branch "v${VLLM_VERSION}" --depth 1 \
        https://github.com/vllm-project/vllm.git /opt/vllm-src \
    && cd /opt/vllm-src \
    && "${VLLM_VENV}/bin/pip" install --no-cache-dir -r requirements/rocm.txt \
    # requirements/rocm.txt must not knock torch off its ROCm HIP build.
    && _hip_after="$("${VLLM_VENV}/bin/python" -c 'import torch; print(torch.version.hip or "")')" \
    && if [ -z "${_hip_after}" ]; then echo "ERROR: ROCm torch lost after rocm.txt" >&2; exit 1; fi \
    # pip install . is broken for vLLM on ROCm; setup.py develop is the path.
    && "${VLLM_VENV}/bin/python" setup.py develop \
    && VLLM_BUILT="$("${VLLM_VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')" \
    && echo "built vllm: ${VLLM_BUILT}" \
    && if [ "${VLLM_BUILT%%+*}" != "${VLLM_VERSION}" ]; then \
         echo "ERROR: vLLM version mismatch: built ${VLLM_BUILT}, asked ${VLLM_VERSION}" >&2; exit 1; \
       fi \
    # Muse-Glimmer is native in >=0.28.1; assert the parsers register (no overlay).
    && "${VLLM_VENV}/bin/python" -c "from vllm.tool_parsers import ToolParserManager; assert ToolParserManager.get_tool_parser('muse_glimmer'); print('tool parser OK')" \
    && "${VLLM_VENV}/bin/python" -c "from vllm.reasoning import ReasoningParserManager; assert ReasoningParserManager.get_reasoning_parser('muse_glimmer'); print('reasoning parser OK')"

# -----------------------------------------------------------------------------
# 3. Kokoro TTS, dashboard, MLflow and JupyterLab dependencies
# -----------------------------------------------------------------------------
# Installed into the same venv so Kokoro reuses the ROCm 7.14 PyTorch that vLLM
# uses, rather than a second multi-gigabyte copy. Python 3.12 satisfies Kokoro's
# `>=3.10,<3.13` constraint, so no separate interpreter is required.
RUN python3 -m pip install --no-cache-dir \
        kokoro soundfile fastapi 'uvicorn[standard]' \
        'streamlit>=1.30' mlflow \
        opentelemetry-sdk==1.44.0 opentelemetry-api==1.44.0 \
        opentelemetry-exporter-otlp-proto-http==1.44.0 \
        psutil requests \
        jupyterlab==4.5.7 'tornado<6.5.9' numpy==2.4.6 'plotly>=5.18' 'pandas>=2.0' \
        matplotlib ipywidgets kaleido \
    && python3 -m pip install --no-cache-dir --upgrade "mlflow>=3.0.0" fastapi uvicorn pydantic \
    && python3 -c "import kokoro, soundfile, streamlit, mlflow, jupyterlab; print('workshop deps OK')"

# -----------------------------------------------------------------------------
# 4. Hermes Agent
# -----------------------------------------------------------------------------
RUN curl -fsSL \
        https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
        | bash -s -- --skip-setup --skip-browser \
    # --skip-browser omits Playwright and Chromium (about 170 MB plus a long
    # download). The workshop drives TTS and file tools, never browser tools.
    # The installer links hermes into /usr/local/bin (not ~/.local/bin), so
    # resolve it via PATH instead of hardcoding a location that can move.
    && hermes --version

# -----------------------------------------------------------------------------
# 5. hermes-otel plugin
# -----------------------------------------------------------------------------
# Install the hermes-otel plugin (per-span CPU/GPU profiling) into the venv and
# the Hermes venv, since the plugin runs inside Hermes. Checks out the latest
# release tag. Tested on hermes-otel-v1.3.0.
#
# GPU sampling must use the `rocm-smi` subprocess path, NOT the in-process
# amdsmi Python binding: on ROCm 7.14 / glibc 2.39 (Ubuntu 24.04) calling
# amdsmi.amdsmi_init() on the host-metrics sampler thread concurrently with
# Hermes agent init heap-corrupts the process (glibc `corrupted size vs.
# prev_size` abort, verified on an MI300X VF 2026-09-16). Upstream v1.3.0 has
# no toggle for this: _amd_init() calls amdsmi_init() unconditionally and the
# rocm-smi path is only a per-field fallback AFTER init. So we apply
# utils/hermes_otel_rocm_smi_gpu.patch.py, which rewrites _amd_init() and
# _amd_gpu_stats() to source detection and per-tick busy%/power/VRAM purely
# from a single `rocm-smi` subprocess call (identical metrics, no native
# amdsmi in the agent process). The patch is idempotent and self-verifies.
COPY utils/hermes_otel_rocm_smi_gpu.patch.py /tmp/hermes_otel_rocm_smi_gpu.patch.py
RUN git clone https://github.com/briancaffey/hermes-otel.git \
        /root/.hermes/plugins/hermes_otel \
    && cd /root/.hermes/plugins/hermes_otel \
    && HERMES_OTEL_VERSION="$(git tag -l 'hermes-otel-v*' | sort -V | tail -1)" \
    && echo "Using latest hermes-otel release: ${HERMES_OTEL_VERSION:-<none, default branch>}" \
    && if [ -n "${HERMES_OTEL_VERSION}" ]; then git checkout -q "${HERMES_OTEL_VERSION}"; fi \
    && python3 /tmp/hermes_otel_rocm_smi_gpu.patch.py \
        /root/.hermes/plugins/hermes_otel/hermes_otel/gpu_probe.py \
    && grep -q PURE_ROCM_SMI_PATCH /root/.hermes/plugins/hermes_otel/hermes_otel/gpu_probe.py \
    && rm -f /tmp/hermes_otel_rocm_smi_gpu.patch.py \
    && python3 -m pip install --no-cache-dir -e . \
    # Install the plugin's dependencies into the Hermes venv as well. Discover
    # the venv location rather than hardcoding it.
    && HERMES_VENV_PY="$( \
         for c in /usr/local/lib/hermes-agent/venv/bin/python \
                  /root/.hermes/hermes-agent/venv/bin/python; do \
             [ -x "$c" ] && echo "$c" && break; \
         done)" \
    && if [ -z "${HERMES_VENV_PY}" ]; then \
         echo "ERROR: could not locate the Hermes venv python" >&2; exit 1; \
       fi \
    && echo "Hermes venv python: ${HERMES_VENV_PY}" \
    && "${HERMES_VENV_PY}" -m ensurepip --upgrade \
    && "${HERMES_VENV_PY}" -m pip install --no-cache-dir -q \
        opentelemetry-api==1.44.0 opentelemetry-sdk==1.44.0 \
        opentelemetry-exporter-otlp-proto-http==1.44.0 \
        psutil requests

# -----------------------------------------------------------------------------
# 6. Workshop content
# -----------------------------------------------------------------------------
WORKDIR /workshop
COPY tts.ipynb tts_executed.ipynb README.md ./
COPY assets/ ./assets/
COPY custom_tools/ ./custom_tools/
COPY utils/ ./utils/

# kokoro_tts_tool.py is intentionally NOT installed into Hermes here. The
# notebook's own cell installs it, matching the bare-host workflow, so the
# Step 1 (Edge TTS baseline) is not skewed by having kokoro_tts already
# registered and preferred before the notebook reaches that point.
RUN mkdir -p /workshop/outputs /root/.hermes/workspace \
             /root/.config/miopen/miopen-lockfiles

# -----------------------------------------------------------------------------
# 7. Preconfigure Hermes and the profiling plugin
# -----------------------------------------------------------------------------
# Baked in so the container starts already pointing at the local vLLM server.
RUN hermes config set model.provider custom \
    && hermes config set model.base_url "http://localhost:${VLLM_HERMES_PORT}/v1" \
    && hermes config set model.default "${HERMES_MODEL}" \
    && hermes config set compression.enabled false \
    && hermes config set model.max_tokens 16384 \
    && hermes config set terminal.cwd "${WORKSHOP_DIR}" \
    && hermes config set tool_output.max_bytes 150000 \
    && hermes config set tool_output.max_lines 5000 \
    && hermes config set tool_output.max_line_length 5000 \
    && hermes plugins enable hermes_otel --allow-tool-override

# hermes-otel plugin config, identical to the one utils/helper.sh writes: traces
# go to MLflow, CPU/GPU metrics to Grafana otel-lgtm, sampled every 100 ms.
RUN printf '%s\n' \
    'enabled: true' \
    'force_flush_on_session_end: true' \
    'capture_previews: true' \
    'capture_full_prompts: false' \
    'capture_full_responses: false' \
    'host_metrics: true' \
    'host_metrics_gpu: amd' \
    'host_metrics_interval_ms: 100' \
    'flush_interval_ms: 100' \
    'backends:' \
    '  - type: otlp' \
    '    name: mlflow' \
    "    endpoint: http://127.0.0.1:${MLFLOW_PORT}/v1/traces" \
    '    metrics: false' \
    '    logs: false' \
    '    headers:' \
    '      x-mlflow-experiment-id: "0"' \
    '  - type: otlp' \
    '    name: lgtm' \
    '    endpoint: http://127.0.0.1:4318/v1/traces' \
    '    traces: false' \
    '    metrics: true' \
    '    logs: false' \
    > /root/.hermes/plugins/hermes_otel/config.yaml

# MLflow 3.x OTLP ingestion rejects spans that carry no experiment header with
# HTTP 422, so the x-mlflow-experiment-id header above is what makes traces land.
# The plugin reads config.yaml above; the rocm-smi GPU sampler is baked in by
# the patch applied at plugin-install time, so no ~/.hermes/.env is needed here.

# -----------------------------------------------------------------------------
# 8. Entrypoint
# -----------------------------------------------------------------------------
COPY utils/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Build-time check that asserts the telemetry wiring, so a misconfigured image
# fails the build rather than at run time.
COPY utils/img-check.sh /usr/local/bin/img-check.sh
RUN chmod +x /usr/local/bin/img-check.sh && /usr/local/bin/img-check.sh

EXPOSE 8888 8501 5004 8001 8092 9090 3000

# JupyterLab is the workshop's front door; the rest come up behind it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD curl -fsS "http://localhost:${JUPYTER_PORT}/api" >/dev/null || exit 1


ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
