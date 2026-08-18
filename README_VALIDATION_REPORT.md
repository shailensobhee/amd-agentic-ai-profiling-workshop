# README replication report: MI300X validation

**Date:** 2026-08-18
**Droplet:** `agentic-ai-profiling-readme-test` (id `593281601`), `gpu-mi300x1-192gb-contracted`, atl1
**Image:** AMD AI/ML Ready Image (`gpu-amd-base`, id 239657705), Ubuntu 24.04.4 LTS, Python 3.12.3
**GPU:** 1x AMD Instinct MI300X VF, gfx942
**Repo state tested:** `shailensobhee/amd-agentic-ai-profiling-workshop` @ `ed8d2003`

> **Scope note.** The run was executed against commit `ed8d2003`. Commit `04baa497`
> ("Sync main with upstream tts branch") landed mid-run. `README.md` is **byte-identical**
> between the two (`git diff ed8d200 origin/main -- README.md` is empty), and I re-checked
> every blocker below against the new `helper.sh`: **B2, B3, B5, M9 all still present**.
> The sync also introduced one NEW defect, logged as **B6**.

Every line below is from observed terminal output on that droplet. Nothing inferred.

---

## Verdict

Following the README **verbatim on a clean MI300X, the workshop cannot be completed.**
It fails at step 1 and then at four further points. After applying the fixes below, the
full stack came up and the headline result reproduced:
**sequential 8.29s vs batched 2.45s inference (7.2x vs 24.4x real-time).**

---

## BLOCKERS (stop a participant dead)

### B1. Step 1 clone fails: repo is private, and the URL is the wrong repo
README says:
```
git clone https://github.com/Jereshea/hermes-audio-notebook.git
cd hermes-audio-notebook
git checkout tts
```
Observed:
```
fatal: could not read Username for 'https://github.com': No such device or address
```
Two defects: (a) `Jereshea/hermes-audio-notebook` is PRIVATE, so an unauthenticated clone
cannot work for any attendee; (b) it points at the upstream personal repo, not ours. Also
`git checkout tts` is now wrong: our default branch is `main`.

**Fix:**
```bash
git clone https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop.git
cd amd-agentic-ai-profiling-workshop
```
Decide and state whether the repo is public for attendees, or document a token/SSH path.

### B2. `bash helper.sh` needs Docker, which the README never mentions and the image lacks
Observed on the clean image:
```
docker: MISSING
sudo: docker: command not found
[ERROR] Build of vllm-muse-glimmer:rocm failed (overlay step).
```
helper.sh runs `sudo docker` in **15** places (count unchanged on new main). The AMD AI/ML
Ready Image has **no docker binary**.

**Fix:** add a prerequisites section.
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
```
Verified: installed `Docker version 29.1.3`, `systemctl is-active docker` -> `active`.

### B3. helper.sh launches MLflow with system `python3`, not the venv, so it always aborts
Observed:
```
[INFO] Installing MLflow and OpenTelemetry dependencies...
/usr/bin/python3: No module named pip
[INFO] Launching MLflow server on port 5004...
/usr/bin/python3: No module named mlflow
[FATAL] MLflow server failed to start properly. Aborting setup.
```
helper.sh lines 242 and 245 call bare `python3` (still true on new main). The image's system
Python has **no pip and no ensurepip** (`ModuleNotFoundError: No module named 'ensurepip'`),
and MLflow was installed into `env/` by README step 3. helper.sh then tears down every
container it just built, so the participant loses the 47.6 GB vLLM image work and sees only
`[FATAL]`.

**Fix (either):**
- Document that helper.sh MUST be run with the venv active: `source env/bin/activate && bash helper.sh`, or
- better, make helper.sh self-contained: `PY="$WORKSPACE_DIR/env/bin/python"` and use `$PY -m mlflow ...`.

Verified: with the venv active, `[OK] MLflow server is up.`

### B4. Kokoro TTS crashes on ROCm: MIOpen cannot compile, missing ROCm dev headers
Observed, and this kills the workshop's core demo:
```
RuntimeError: miopenStatusUnknownError
ERROR:    Application startup failed. Exiting.
[FATAL] Kokoro TTS server failed to start properly. Aborting setup.
```
With `MIOPEN_FIND_MODE=NORMAL` the real cause surfaces:
```
MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6), source file: MIOpenDropoutHIP.cpp
/tmp/.../miopen_rocrand.hpp:45:10: fatal error: 'rocrand/rocrand_xorwow.h' file not found
1 error generated when compiling for gfx942.
```
Diagnosed with a positive control so this is not a guess: a bare `torch.nn.LSTM` on GPU
**works** (`LSTM OK torch.Size([2, 10, 64])`), as does a bidirectional + packed-sequence
LSTM. MIOpen/ROCm is functional in general. Kokoro's LSTM path triggers MIOpen's **dropout**
kernel, JIT-compiled at runtime, which needs `rocrand` + `hip` headers.
`/opt/rocm/include` on this image ships **neither** `rocrand/` nor `hip/`.

**Fix applied and verified:** after installing the missing headers,
`INFER OK, audio samples = (88200,)`. The clean documented fix is to install ROCm dev headers
(`rocrand-dev`, `hip-dev`). Note `apt-cache policy rocrand-dev` returned **empty**, so the
ROCm apt repo is not configured on this image either, that needs adding too.
Related: `clear_cache.sh` wipes `~/.cache/miopen` on every run, forcing this JIT compile
every single time. Not wiping it would let a warm cache survive.

### B5. Hermes OTel plugin installs into a venv that does not exist, so MLflow captures ZERO traces
This silently guts the entire point of a *profiling* workshop. helper.sh (line 346-347 on new
main) does:
```
$HOME/.hermes/hermes-agent/venv/bin/python -m pip install ... opentelemetry-exporter-otlp-proto-http ...
```
Observed:
```
ls: cannot access '/root/.hermes/hermes-agent/venv': No such file or directory
```
The real interpreter is `/usr/local/lib/hermes-agent/venv/bin/python` (from
`/usr/local/bin/hermes` line 4). At runtime:
```
[hermes-otel] ✗ OpenTelemetry packages not available
[hermes-otel] OpenTelemetry import error: No module named 'opentelemetry.exporter'
```
and `trace_info rows = 0` in mlflow.db, while every service showed green and helper.sh
printed `[OK] Setup complete.` A textbook false green: the dashboard is empty and nothing
says why.

**Fix applied and verified:**
```bash
V=/usr/local/lib/hermes-agent/venv
$V/bin/python -m ensurepip --upgrade      # this venv has no pip either
$V/bin/python -m pip install opentelemetry-api==1.42.1 opentelemetry-sdk==1.42.1 \
    opentelemetry-exporter-otlp-proto-http==1.42.1 pyrsmi==1.1.0 mlflow==3.13.0 psutil requests cryptography
```
After the fix:
```
[hermes-otel] ✓ mlflow at http://127.0.0.1:5004/v1/traces (traces only)
[hermes-otel] Registered 13 hooks
trace_info rows = 1
```
Note helper.sh's own `ensurepip` call is on the SAME wrong path, so it never ran.

### B6. NEW on `04baa497`: removing the kokoro tool copy step breaks the custom-tool lesson
Commit `04baa497` deleted this block from helper.sh:
```bash
-HERMES_TOOLS_DIR="$HOME/.hermes/hermes-agent/tools"
-cp "$WORKSPACE_DIR/custom_tools/kokoro_tts_tool.py" "$HERMES_TOOLS_DIR/"
```
Nothing replaces it: `custom_tools/kokoro_tts_tool.py` is still in the repo, and the notebook
still has a markdown cell teaching "`kokoro_tts` is a custom tool we added to Hermes", but no
cell installs it.

Tested both ways (first probe with `hermes tools` was invalid, it needs a TTY, so I used real
agent runs):
- **Positive control**, tool file present: `Done [batched] 5 sentence(s) - 0.31s inference (195.9x real-time)`
- **Negative**, tool file removed: the agent burns 8+ shell calls hunting for it
  (`find /root -type f -name "*kokoro*"`, `xargs grep -l "kokoro_tts"`) and then falls back to
  a raw `curl http://localhost:8092/...`

It "works" either way, which makes this **worse, not better**: the participant sees a result
while the custom-tool lesson silently did not happen, and the MLflow trace shows a pile of
terminal calls instead of one clean `kokoro_tts` tool span. On a profiling workshop that is
exactly the wrong trace.

**Fix:** restore the copy in helper.sh, or add an explicit notebook cell that installs the
tool so the participant sees the extension step.

---

## MISSING / WRONG STEPS (not fatal, but wrong as written)

### M1. `sudo apt install python3-venv` aborts non-interactively
```
Do you want to continue? [Y/n] Abort.
```
No `apt update` beforehand and no `-y`. Fix: `sudo apt-get update && sudo apt-get install -y python3-venv`.

### M2. `jupyter lab` as documented exits 1 for the root user
The droplet's default (and only) user is `root`. Verbatim command output:
```
[C 2026-08-18 14:49:42.656 ServerApp] Running as root is not recommended. Use --allow-root to bypass.
verbatim exit=1
```
Fix: add `--allow-root`, or document creating a non-root user. With `--allow-root` it binds
fine (`HTTP 302` on `/lab`, token issued).

### M3. Nothing tells the participant how to actually REACH JupyterLab
The README launches on `0.0.0.0:8888` and stops. Measured from my machine:
```
external :8888 -> HTTP 000     (unreachable)
external :80   -> HTTP 200     (reachable)
port 22        -> OPEN
```
The box is alive and port 80 is fine, so **8888 is blocked upstream of the droplet**. ufw is
active but only DENYs 6601/50061, so ufw is NOT the cause. Same will apply to 5004, 8501, 8092.
Add an access section, SSH port-forward is the reliable path:
```bash
ssh -L 8888:localhost:8888 -L 8501:localhost:8501 -L 5004:localhost:5004 root@<droplet-ip>
```

### M4. No prerequisites / hardware / timing section at all
Never states: AMD GPU + ROCm required, disk needed, or time cost. Measured: disk went
**28 GB -> 159 GB** used; `vllm/vllm-openai-rocm:nightly` alone is **47.6 GB**; first
`bash helper.sh` takes **~25-30 minutes** (image pull, PR #51655 overlay build, 30B weight
download, Playwright chromium 167 MB, Kokoro model). With no warning a participant will
assume it hung.

### M5. `requirements.txt` is not the real dependency list
It pins 6 packages. helper.sh separately pip-installs torch (ROCm 7.2), kokoro, soundfile,
fastapi, uvicorn, streamlit, mlflow, opentelemetry, and a spacy model. Nothing documents the
split, and `numpy==2.4.6` + `mlflow==3.13.0` are hard pins that can fight the torch install.

### M6. Notebook cell builds a dashboard link that is wrong for a remote droplet
It computes `system_ip` via a socket trick then **ignores it**:
`direct_link = f"http://localhost:{DASHBOARD_PORT}/"`. On a remote droplet `localhost` is the
participant's own laptop. Use `system_ip`, or document the SSH tunnel (M3).

### M7. Final chart cell hardcodes fake benchmark numbers
`edge_time = 14`, `seq_time = 43.3`, `batched_time = 4.4` are literals with a comment to
replace them. My measured run on MI300X: **sequential 8.29s, batched 2.45s**. The chart shows
numbers unrelated to what the participant just ran. Should read from the tool output.

### M8. Stale absolute path from the author's machine
`kokoro_server.py:11`: `cd /home/sabiras/hermes-audio-notebook/workspace`. Neither that path
nor a `workspace` subdir exists in the repo.

### M9. helper.sh appends to `~/.hermes/.env` with `>>` on every run
After 3 runs `MLFLOW_TRACKING_URI` appears twice and `.env` is 528 lines. Still 2 append sites
on new main. Should be a marker-delimited block that is rewritten, not appended.

### M10. `git checkout tts` no longer matches our repo
Our default branch is now `main`. Remove the line.

### M11. Exposure note
`--allowed-hosts "*"` on MLflow, `--ip=0.0.0.0` on Jupyter, and the token printed to a log.
Fine for a sandboxed workshop, worth an explicit "this box is exposed, do not leave it running"
line, especially since participants will tunnel rather than firewall.

---

## WHAT WORKS (verified, so we know the fixes are sufficient)

With B1-B5 + M1/M2 applied, `bash helper.sh` reached `[OK] Setup complete.` with all five
services listening: 8001 (vLLM), 8092 (Kokoro), 8501 (Streamlit), 5004 (MLflow), 5050 (metrics).

| Capability | Evidence |
|---|---|
| vLLM serves the model | `GET /v1/models` -> `meta-models/Muse-Glimmer-30B`, `max_model_len: 131072` |
| Real LLM inference | chat/completions returned reasoning text. Note it is a reasoning model: `content` is null and text lands in `reasoning`, so a low `max_tokens` looks like an empty reply |
| Muse-Glimmer PR overlay | `tool parser OK` / `reasoning parser OK`, image `vllm-muse-glimmer:rocm` committed |
| Kokoro on GPU | `/health` -> `{"status":"ok","device":"cuda:0","gpu":"AMD Instinct MI300X VF"}` |
| Real TTS, both modes | 55.5s audio generated, sequential RTF 0.0386, batched RTF 0.0376 |
| Agent writes input_text.txt | 842-byte file produced by the local 30B via `hermes chat --yolo` |
| Custom kokoro_tts tool | sequential: `8.29s inference, 60.0s audio (7.2x real-time)`, 2.8 MB wav |
| Batched optimization | `2.45s inference, 60.0s audio (24.4x real-time)`, 2.8 MB wav |
| MLflow trace capture | `trace_info rows = 1`, trace `tr-16377a237f1dc546db7aeeb7a116766b`, `execution_duration 144.198s`, state OK |

The workshop's core teaching point (sequential vs batched on AMD hardware) **does reproduce on
MI300X**: 8.29s -> 2.45s, a 3.4x speedup, 7.2x -> 24.4x real-time. A later warm-cache run hit
0.31s (195.9x), so first-run MIOpen JIT cost is a large part of the sequential number and is
worth mentioning in the notebook.

---

## Suggested README skeleton

1. Prerequisites: AMD Instinct GPU + ROCm, Ubuntu 24.04, ~200 GB free disk, ~30 min first run
2. Install prereqs: `docker.io`, `python3-venv`, ROCm dev headers (`rocrand-dev`, `hip-dev`)
3. Clone OUR repo (`main`, no checkout line)
4. venv + `pip install -r requirements.txt`
5. **`source env/bin/activate`** then `bash helper.sh` (call out that it is required)
6. Access section: SSH tunnel for 8888 / 8501 / 5004
7. Jupyter with `--allow-root`
8. Expected first-run timings so nobody thinks it hung
