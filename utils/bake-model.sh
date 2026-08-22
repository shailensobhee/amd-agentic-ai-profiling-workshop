#!/usr/bin/env bash
# Prefetch every model the container needs at runtime, so nothing is downloaded
# on first start and the workshop works with no network access to Hugging Face.
#
# Split into a script (not an inline RUN heredoc) because this Dockerfile has no
# `# syntax=docker/dockerfile:1.4` directive, so the classic builder would treat
# a heredoc as literal text and silently skip the verification.
set -euo pipefail

: "${HERMES_MODEL:?HERMES_MODEL must be set}"
: "${KOKORO_MODEL:?KOKORO_MODEL must be set}"

echo "Baking models into the image:"
echo "  LLM   : ${HERMES_MODEL}"
echo "  TTS   : ${KOKORO_MODEL}"

# hf_transfer must be importable for HF_HUB_ENABLE_HF_TRANSFER to do anything.
# Install it explicitly and report which downloader is actually in use, so a
# fallback to the slow path shows up in the build log instead of being a mystery.
python3 -m pip install --no-cache-dir huggingface_hub hf_transfer >/dev/null
if python3 -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "  downloader: hf_transfer (accelerated)"
else
    echo "  downloader: standard (hf_transfer unavailable)"
fi

# Download BOTH models. Baking only the LLM leaves the container still dependent
# on Hugging Face: kokoro_server.py calls hf_hub_download at startup, fails with
# LocalEntryNotFoundError when offline, and the entrypoint then blocks forever
# on the Kokoro health check.
python3 - <<'PY'
import os

from huggingface_hub import snapshot_download

for var in ("HERMES_MODEL", "KOKORO_MODEL"):
    repo = os.environ[var]
    path = snapshot_download(repo, max_workers=8)
    print(f"  downloaded {repo}")
    print(f"    -> {path}")
PY

# Verify what actually landed on disk. local_files_only resolves paths WITHOUT
# network access, so this checks bytes rather than re-downloading. This is the
# gate: a truncated or partial fetch must fail the BUILD, never ship as a
# "baked" image that silently downloads at runtime.
python3 - <<'PY'
import json
import os
import sys

from huggingface_hub import snapshot_download

# ---- LLM ----------------------------------------------------------------
llm = os.environ["HERMES_MODEL"]
path = snapshot_download(llm, local_files_only=True)

index = os.path.join(path, "model.safetensors.index.json")
if not os.path.exists(index):
    sys.exit(f"BAKE FAILED: no {index}")

with open(index) as fh:
    shards = sorted({v for v in json.load(fh)["weight_map"].values()})

missing, total = [], 0
for shard in shards:
    target = os.path.join(path, shard)
    if os.path.exists(target):
        total += os.path.getsize(target)
    else:
        missing.append(shard)

if missing:
    sys.exit(f"BAKE FAILED: missing shard(s): {missing}")

# A 30B model is tens of GB. Anything far smaller means a metadata-only fetch.
if total < 40e9:
    sys.exit(f"BAKE FAILED: shards total only {total / 1e9:.2f} GB")

for extra in ("config.json", "tokenizer.json", "chat_template.jinja"):
    if not os.path.exists(os.path.join(path, extra)):
        sys.exit(f"BAKE FAILED: missing {extra}")

print(f"  {llm}: {len(shards)} shard(s), {total / 1e9:.2f} GB")

# ---- Kokoro TTS voice model --------------------------------------------
# Smaller, but equally load-bearing: without it the TTS server exits at startup
# and the entrypoint blocks on its health check.
kokoro = os.environ["KOKORO_MODEL"]
kpath = snapshot_download(kokoro, local_files_only=True)

kfiles = [
    os.path.join(dirpath, name)
    for dirpath, _, names in os.walk(kpath)
    for name in names
]
ktotal = sum(os.path.getsize(f) for f in kfiles if os.path.exists(f))

if ktotal < 100e6:
    sys.exit(f"BAKE FAILED: {kokoro} only {ktotal / 1e6:.1f} MB")

print(f"  {kokoro}: {len(kfiles)} file(s), {ktotal / 1e9:.2f} GB")
PY

echo "Models baked successfully."
