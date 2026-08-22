#!/usr/bin/env bash
# Prefetch the model weights into the image.
#
# Split into a script (not an inline RUN heredoc) because this Dockerfile has no
# `# syntax=docker/dockerfile:1.4` directive, so the classic builder would treat
# a heredoc as literal text and silently skip the verification.
set -euo pipefail

: "${HERMES_MODEL:?HERMES_MODEL must be set}"

echo "Baking ${HERMES_MODEL} into the image..."

python3 -m pip install --no-cache-dir 'huggingface_hub[hf_transfer]' >/dev/null
python3 -c "import hf_transfer" 2>/dev/null \
  && export HF_HUB_ENABLE_HF_TRANSFER=1 \
  || echo "  hf_transfer unavailable, falling back to the standard downloader"

# Download. max_workers parallelises the two large shards.
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(os.environ["HERMES_MODEL"], max_workers=8)
print("  snapshot at", path)
PY

# Verify what actually landed on disk. local_files_only resolves the path
# WITHOUT network access, so this checks the bytes rather than re-downloading.
# This is the gate: a truncated or partial fetch must fail the BUILD, never ship
# as a "baked" image that silently downloads at runtime.
python3 - <<'PY'
import json
import os
import sys

from huggingface_hub import snapshot_download

path = snapshot_download(os.environ["HERMES_MODEL"], local_files_only=True)

index = os.path.join(path, "model.safetensors.index.json")
if not os.path.exists(index):
    sys.exit(f"BAKE FAILED: no {index}")

shards = sorted({v for v in json.load(open(index))["weight_map"].values()})
missing, total = [], 0
for s in shards:
    f = os.path.join(path, s)
    if os.path.exists(f):
        total += os.path.getsize(f)
    else:
        missing.append(s)

if missing:
    sys.exit(f"BAKE FAILED: missing shard(s): {missing}")

# A 30B model is tens of GB. Anything far smaller means a metadata-only fetch.
if total < 40e9:
    sys.exit(f"BAKE FAILED: shards total only {total / 1e9:.2f} GB")

for extra in ("config.json", "tokenizer.json", "chat_template.jinja"):
    if not os.path.exists(os.path.join(path, extra)):
        sys.exit(f"BAKE FAILED: missing {extra}")

print(f"  baked {len(shards)} shard(s), {total / 1e9:.2f} GB")
print(f"  resolved snapshot: {path}")
PY

echo "Model weights baked successfully."
