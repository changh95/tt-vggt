#!/usr/bin/env python3
"""VGGT end-to-end benchmark on a single Tenstorrent Blackhole (p150a) chip.

Outputs fields PROGRAM.md expects (greppable):
    inference_speed: <frames/sec>
    accuracy:        <PCC * 100, threshold 99.0>
    peak_dram:       <MB, best-effort>
    pcc:             <raw PCC, for context>
    latency_ms:      <wall-clock per call>

Usage:
    python3 test_vggt.py                              # end_to_end, B=1 S=1
    python3 test_vggt.py --layer end_to_end --runs 3
    python3 test_vggt.py --seq 2                      # S=2 views
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Our tt-metal build is at experiments/vggt/tt-metal. The venv ttnn is
# pinned to pi0_5's checkout (without matching kernel source), so reuse
# medgemma's tt-metal the way sibling experiments (mast3r) do.
_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
if _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)
    sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
    sys.path.insert(2, os.path.join(_TT_METAL_ROOT, "tools"))
os.chdir(_TT_METAL_ROOT)

# Model code lives under this experiment's tt-metal tree. Put the demo
# dir directly on sys.path so imports stay short ("reference.*", "tt.*")
# and don't collide with medgemma's sibling models/ tree.
_VGGT_DEMO = "/home/ttuser/experiments/vggt/tt-metal/models/demos/vggt"
if _VGGT_DEMO not in sys.path:
    sys.path.insert(0, _VGGT_DEMO)

import torch  # noqa: E402

from reference.torch_vggt import load_vggt  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float((a @ b).item() / denom)


def multi_pcc(ref: dict, out: dict, keys) -> tuple[float, dict]:
    scores = {}
    for k in keys:
        if k in ref and k in out and isinstance(ref[k], torch.Tensor) and isinstance(out[k], torch.Tensor):
            scores[k] = pcc(ref[k], out[k])
    if not scores:
        return 0.0, scores
    return min(scores.values()), scores


def device_peak_dram_mb(device) -> float:
    """Best-effort peak DRAM in MB. Returns 0.0 if not accessible."""
    try:
        import ttnn  # noqa
        try:
            import ttnn._ttnn as _t  # type: ignore
            stats = _t.device.allocator_statistics(device, _t.tensor.BufferType.DRAM)
            return float(stats.peak_bytes) / (1024 * 1024)
        except Exception:
            pass
        try:
            stats = ttnn.get_memory_per_bank_dram_allocation_stats(device)  # type: ignore
            return float(stats.peak_bytes) / (1024 * 1024)
        except Exception:
            pass
    except Exception:
        pass
    return 0.0


def print_result(layer: str, pcc_val: float, latency_ms: float,
                 status: str, peak_dram_mb: float = 0.0,
                 per_key_pcc: dict | None = None):
    speed = (1000.0 / latency_ms) if latency_ms > 0 else 0.0
    accuracy = max(0.0, min(100.0, pcc_val * 100.0))
    print(f"--- layer: {layer}")
    print(f"pcc: {pcc_val:.4f}")
    if per_key_pcc:
        for k, v in per_key_pcc.items():
            print(f"pcc_{k}: {v:.4f}")
    print(f"latency_ms: {latency_ms:.2f}")
    print(f"inference_speed: {speed:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"peak_dram: {peak_dram_mb:.2f}")
    print(f"status: {status}")
    print("---")


# ---------- layer runners ----------

def _make_inputs(batch: int, seq: int, img_size: int, seed: int = 0):
    torch.manual_seed(seed)
    images = torch.rand(batch, seq, 3, img_size, img_size)
    return images


# Primary geometric outputs we care about for PCC. Track outputs require
# query_points; skip those in the default path.
_PCC_KEYS = ("depth", "depth_conf", "world_points", "world_points_conf", "pose_enc")


def run_end_to_end(device, runs: int, batch: int, seq: int, img_size: int):
    from tt.ttnn_vggt import vggt_forward as tt_vggt_forward

    images = _make_inputs(batch, seq, img_size)
    ref_model = load_vggt(eval_mode=True)
    with torch.no_grad():
        ref_out = ref_model(images)
    del ref_model

    _ = tt_vggt_forward(images, device=device)  # warmup
    times = []
    tt_out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_vggt_forward(images, device=device)
        times.append((time.perf_counter() - t0) * 1000)

    score, per_key = multi_pcc(ref_out, tt_out, _PCC_KEYS)
    return score, per_key, min(times)


LAYER_DISPATCH = {
    "end_to_end": run_end_to_end,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="end_to_end")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed runs (best-of-N reported)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1,
                        help="Number of views (S). VGGT supports variable S.")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--device-id", type=int, default=0,
                        help="Tenstorrent UMD chip id. Assigned PROGRAM.md number is 7 "
                             "but this host only exposes UMD ids 0-3 (2xp300c).")
    args = parser.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        if args.layer not in LAYER_DISPATCH:
            print_result(args.layer, 0.0, 0.0, "crash")
            print(f"ERROR: unknown layer '{args.layer}'")
            return 2
        try:
            pcc_val, per_key, latency_ms = LAYER_DISPATCH[args.layer](
                device, args.runs, args.batch, args.seq, args.img_size
            )
            peak = device_peak_dram_mb(device)
            status = "PASS" if pcc_val >= 0.99 else "FAIL"
            print_result(args.layer, pcc_val, latency_ms, status, peak, per_key)
            return 0 if status == "PASS" else 1
        except Exception:
            traceback.print_exc()
            print_result(args.layer, 0.0, 0.0, "crash")
            return 3
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
