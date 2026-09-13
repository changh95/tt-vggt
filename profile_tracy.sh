#!/usr/bin/env bash
# Tracy profiling run for the VGGT ttnn port.
#
# Produces per-op SFPU/FPU/pack/unpack utilisation, NoC traces, DRAM
# bandwidth, and a Tracy GUI-compatible dump.
#
# Output lands under
#   $TT_METAL_HOME/generated/profiler/reports/
# with a timestamped directory containing ops_perf_results_*.csv +
# raw Tracy .tracy file.
#
# Env:
#   PROFILE_SEQ   sequence length to profile (default: 1 — known-safe
#                 after BF0 option 1 ruled out; raise only after
#                 BF0 option 2 lands).
#   PROFILE_RUNS  timed runs (default: 1 — Tracy overhead is large, one
#                 run is enough for the op breakdown).
set -euo pipefail

# TT_METAL_HOME must point at a Tracy-enabled tt-metal build; VGGT_REF at the
# pinned facebookresearch/vggt checkout if it is not already on PYTHONPATH.
TT_ROOT="${TT_METAL_HOME:?set TT_METAL_HOME to a Tracy-enabled tt-metal tree}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${TT_ROOT}"

SEQ="${PROFILE_SEQ:-1}"
RUNS="${PROFILE_RUNS:-1}"
NAME="vggt_s${SEQ}_$(date +%Y%m%d_%H%M%S)"

# The default .tenstorrent-venv pins pi0_5's tt-metal tree (Tracy-OFF)
# via a .pth file in site-packages. python3 -m tracy would pick that up
# and fail with "TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build".
# Force medgemma's tree (Tracy-ON) first on PYTHONPATH and TT_METAL_HOME.
export PYTHONPATH="${CODE_DIR}:${VGGT_REF:+${VGGT_REF}:}${TT_ROOT}:${TT_ROOT}/ttnn:${TT_ROOT}/tools:${PYTHONPATH:-}"
export TT_METAL_HOME="${TT_ROOT}"

# -r = generate ops report (CSV with per-op device time).
# --sync-host-device = precise timing for host-initiated ops.
#
# Why no --collect-noc-traces / --profiler-capture-perf-counters=all:
# Both fail on Blackhole (p150a) in this tt-metal build. NoC tracing
# raises TT_FATAL "Invalid NoC transfer type on device: 2" mid-run
# (noc_xfer_type validation only covers Wormhole IDs as of this build),
# and perf-counter capture triggers a pandas dtype error in the Tracy
# Python post-processor. Both flags are worth re-enabling once tt-metal
# lands Blackhole support — track as a separate TODO.
exec python3 -m tracy \
    -r \
    --sync-host-device \
    -n "${NAME}" \
    "${CODE_DIR}/test_vggt.py" \
    --seq "${SEQ}" --runs "${RUNS}" --device-id "${TT_DEVICE_ID:-0}"
