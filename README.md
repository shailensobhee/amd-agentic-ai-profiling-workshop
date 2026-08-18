# Hermes Autonomous Agent: TTS Optimization Tutorial

This repository contains a practical, hands-on tutorial for working with **[Hermes Agent](https://hermes-agent.nousresearch.com/docs/getting-started/installation)**, an open-source, autonomous AI framework developed by Nous Research.

Unlike standard conversational chatbots, Hermes operates as a persistent, self-improving assistant capable of autonomous planning, tool execution, and dynamic problem-solving. In this tutorial, you will walk through a real-world engineering workflow: baseline TTS, performance bottleneck identification via MLflow observability, and hardware-accelerated pipeline optimization on AMD hardware.

---

## 📋 Before you start

**Hardware and OS**

| Requirement | Validated configuration |
| :--- | :--- |
| GPU | 1x AMD Instinct MI300X (gfx942), ROCm |
| OS | Ubuntu 24.04 LTS (AMD AI/ML Ready Image) |
| Python | 3.12 |
| Free disk | **~200 GB.** The vLLM ROCm image alone is ~48 GB, and a full run took the droplet from 28 GB to 159 GB used |
| Time | **~30 minutes** for the first `bash helper.sh` (image pull, vLLM overlay build, 30B weight download, Kokoro model). It is not hung, it is downloading |

**Install the system prerequisites first.** The stock AMD AI/ML Ready Image does
**not** ship Docker, and `helper.sh` needs it:

```bash
sudo apt-get update
sudo apt-get install -y docker.io python3-venv
sudo systemctl enable --now docker
```

> **ROCm dev headers.** Kokoro's LSTM makes MIOpen compile a kernel at runtime that
> needs the `rocrand` and `hip` headers. If `/opt/rocm/include/rocrand/` and
> `/opt/rocm/include/hip/` are missing, the TTS server dies with a misleading
> `RuntimeError: miopenStatusUnknownError`. `helper.sh` checks for these and tries
> to install `rocrand-dev` / `hip-dev`, and warns you clearly if they are still absent.

---

## 🛠️ Quick Start Setup

Follow these steps to set up your local environment and launch the interactive tutorial notebook.

### 1. Clone the Repository
Open your terminal and clone this repository.
```bash
git clone https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop.git
cd amd-agentic-ai-profiling-workshop
```

### 2. Create and Activate a Virtual Environment
Isolate your project dependencies by spinning up a clean Python virtual environment:
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
Install all required frameworks, libraries, and tracking tools automatically from the requirements manifest:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `requirements.txt` covers the notebook side only. `helper.sh` additionally installs
> the ROCm build of PyTorch, Kokoro, FastAPI/uvicorn, Streamlit, MLflow and
> OpenTelemetry into this same venv.

---

## 🛠️ Start the backend

**Run `helper.sh` with the virtual environment active.** It uses the venv
interpreter for MLflow and Streamlit; the system `python3` on this image has no
`pip`, so running it without the venv aborts partway through and tears down the
containers it just built.

```bash
source env/bin/activate     # required
bash helper.sh
```

Wait for `[OK] Setup complete.` The script brings up five services:

| Port | Service |
| :--- | :--- |
| 8001 | vLLM (Muse-Glimmer-30B), the agent's brain |
| 8092 | Kokoro TTS server |
| 8501 | Streamlit profiling dashboard |
| 5004 | MLflow tracking and traces |
| 5050 | AMD device metrics exporter |

Leave this terminal running. In a second terminal you can talk to the agent directly with `hermes`.

---

## 🌐 Reaching the web UIs from your laptop

The services bind to `0.0.0.0`, but on a cloud GPU instance those ports are
typically **not reachable from the public internet** (verified: port 8888 was
unreachable externally while the droplet was healthy). Use an SSH tunnel:

```bash
ssh -L 8888:localhost:8888 \
    -L 8501:localhost:8501 \
    -L 5004:localhost:5004 \
    <user>@<instance-ip>
```

Then open `http://localhost:8888` (Jupyter), `http://localhost:8501` (dashboard),
and `http://localhost:5004` (MLflow) in your own browser.

> **Security note:** this setup exposes Jupyter on `0.0.0.0` and runs MLflow with
> `--allowed-hosts "*"`. That is fine for a short workshop on a disposable
> instance, but do not leave it running unattended.

---

## 🛠️ Running the Tutorial Notebook

JupyterLab defaults to a light theme, but the Hermes environment is designed with a dark theme. To ensure a consistent and comfortable experience, we recommend forcing JupyterLab into dark mode before launching.

Run the following commands to configure your environment and begin interacting with the codebase:

```bash
# 1. Ensure the setting directories exist
mkdir -p ./env/share/jupyter/lab/settings
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension

# 2. Apply the dark theme to both the virtual environment and global settings
echo '{"@jupyterlab/apputils-extension:themes": {"theme": "JupyterLab Dark"}}' > ./env/share/jupyter/lab/settings/overrides.json
echo '{"theme": "JupyterLab Dark"}' > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# 3. Launch JupyterLab
#    --allow-root is required when running as root, which is the default user
#    on most cloud GPU instances. Without it, jupyter refuses to start.
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Copy the `token=...` value from the launch output, then open
`http://localhost:8888/lab?token=<token>` through the SSH tunnel above.

Open `tts.ipynb` and work through it.

---

## ✅ What a good run looks like

Measured on 1x MI300X, so you know whether your numbers are sane:

| Step | Expected |
| :--- | :--- |
| `!which hermes` | prints a path, then `Hermes is ready.` |
| Kokoro health (`:8092/health`) | `{"status":"ok","device":"cuda:0","gpu":"AMD Instinct MI300X VF"}` |
| Sequential TTS | ~8s inference for 60s of audio (~7x real-time) on a cold MIOpen cache |
| Batched TTS | ~2.5s inference for the same 60s (~24x real-time) |
| MLflow | at least one trace visible at `http://localhost:5004` after an agent run |

The point of the exercise is the **sequential vs batched** gap. First-run numbers
include MIOpen kernel compilation, so a warm rerun is much faster (a second batched
run measured 0.31s, ~196x real-time). Use your own measured values in the final
chart cell rather than the placeholders.

**If the MLflow UI is empty**, the OpenTelemetry exporter is not installed in the
venv that runs Hermes. `helper.sh` now resolves that interpreter from the `hermes`
launcher and verifies the import, printing either
`[OK] OpenTelemetry exporter importable by Hermes` or a clear warning.

---

## 🛠️ Running the Test on terminal

1. In Terminal 1 run: `source env/bin/activate && bash helper.sh`
2. In Terminal 2 run: `hermes`
