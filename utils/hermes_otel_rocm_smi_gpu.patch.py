"""Patch hermes_otel gpu_probe.py to use a pure rocm-smi subprocess path on AMD,
avoiding in-process amdsmi which corrupts the glibc heap when its init runs on a
sampler thread concurrently with Hermes agent startup (observed: 'corrupted size
vs. prev_size' / 'Fatal glibc error: malloc.c:2599 (sysmalloc)') on Ubuntu 24.04
glibc 2.39 + ROCm 7.14.1 MI300X VF. rocm-smi gives the same busy%/power/VRAM.
"""
import re, io, os, sys

# Path to gpu_probe.py; overridable as argv[1] so this runs both at image build
# time and against an installed plugin. Defaults to the in-image plugin path.
p = sys.argv[1] if len(sys.argv) > 1 else "/root/.hermes/plugins/hermes_otel/hermes_otel/gpu_probe.py"
s = open(p).read()

if "PURE_ROCM_SMI_PATCH" in s:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

# 1) Replace _amd_init: detect AMD GPUs via rocm-smi (no in-process amdsmi).
old_init = '''def _amd_init():
    """Initialize AMD SMI and return its GPU handles, or None if amdsmi is not
    installed or no AMD GPU is present."""
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            amdsmi.amdsmi_shut_down()
            return None
        return handles
    except Exception:
        return None'''

new_init = '''def _amd_init():
    """PURE_ROCM_SMI_PATCH: detect AMD GPUs via the rocm-smi CLI instead of the
    in-process amdsmi Python binding. amdsmi_init() on a background sampler
    thread corrupts the glibc heap during concurrent Hermes agent startup on
    Ubuntu 24.04 (glibc 2.39) + ROCm 7.14.1 MI300X VF. The CLI path is
    subprocess-isolated and returns identical busy%/power/VRAM. Returns a list
    of synthetic integer handles (one per detected GPU index) or None."""
    snap = _amd_cli_snapshot()
    if not snap:
        return None
    # handles are just GPU indices; _amd_gpu_stats reads via the CLI snapshot.
    return sorted(snap.keys())'''

assert old_init in s, "old_init anchor missing"
s = s.replace(old_init, new_init)

# 2) Rewrite _amd_gpu_stats to read purely from _amd_cli_snapshot (one call).
#    Find the function start and replace its whole body up to the next 'def '.
start = s.index("def _amd_gpu_stats(handles):")
end = s.index("\ndef _detect_gpu_vendor(")
new_stats = '''def _amd_gpu_stats(handles):
    """PURE_ROCM_SMI_PATCH: read every AMD GPU's busy%/power/VRAM from a single
    rocm-smi snapshot, keyed by GPU index. No in-process amdsmi is used, so no
    heap corruption on the sampler thread. handles is the index list from
    _amd_init; a None field means that metric was absent this tick."""
    snap = _amd_cli_snapshot()
    stats = []
    for idx in (handles or []):
        e = snap.get(idx, {"busy_pct": None, "power_w": None, "vram_used_mb": None})
        stats.append({
            "busy_pct": e.get("busy_pct"),
            "power_w": e.get("power_w"),
            "vram_used_mb": e.get("vram_used_mb"),
        })
    return stats

'''
s = s[:start] + new_stats + s[end+1:]

# 3) _amd_cli_snapshot converts VRAM 'digits' as bytes -> MB via /1024**2.
#    Keep as-is; rocm-smi --showmeminfo prints bytes. Fine.

open(p, "w").write(s)
print("PATCHED gpu_probe.py (pure rocm-smi)")
# sanity compile
import py_compile
py_compile.compile(p, doraise=True)
print("COMPILE_OK")
