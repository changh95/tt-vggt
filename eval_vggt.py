#!/usr/bin/env python3
"""VGGT correctness evaluation on CO3Dv2 scenes.

Two metrics per scene:

  1. Relative PCC (port vs torch reference) — catches port regressions under
     real image statistics. Same formula as test_vggt.py but fed real photos
     instead of torch.rand() noise.

  2. Camera-pose error vs CO3D ground-truth viewpoints:
       - RRA@5/15 deg  (relative rotation accuracy, fraction of pairs below thr)
       - RTA@5/15 deg  (relative translation-direction accuracy)
       - AUC@30 deg    (Area Under the cumulative-error Curve up to 30 deg,
                        averaged over RRA and RTA)
     Metrics follow the VGGT paper convention — pair-wise relative poses
     because absolute pose + scale are ambiguous.

Usage:
    python3 eval_vggt.py --co3d-root co3d_data \\
        --category apple --seqs 110_13051_23361,189_20393_38136 \\
        --num-views 8 --runs 1
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

# Match test_vggt.py's import shim: use medgemma's tt-metal build + point at
# our demo tree.
_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
if _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)
    sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
    sys.path.insert(2, os.path.join(_TT_METAL_ROOT, "tools"))
os.chdir(_TT_METAL_ROOT)
_VGGT_DEMO = "/home/ttuser/experiments/vggt/tt-metal/models/demos/vggt"
if _VGGT_DEMO not in sys.path:
    sys.path.insert(0, _VGGT_DEMO)
_VGGT_REF = "/home/ttuser/experiments/vggt/vggt_ref"
if _VGGT_REF not in sys.path:
    sys.path.insert(0, _VGGT_REF)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from reference.torch_vggt import load_vggt  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


# ---------- CO3D annotation loading ----------

def load_co3d_annotations(co3d_root: Path, category: str):
    """Return dict keyed by seq_name -> list of frame annotations."""
    ann_path = co3d_root / category / "frame_annotations.jgz"
    with gzip.open(ann_path, "rb") as f:
        annos = json.load(f)
    by_seq = {}
    for entry in annos:
        by_seq.setdefault(entry["sequence_name"], []).append(entry)
    for seq in by_seq:
        by_seq[seq].sort(key=lambda e: e["frame_number"])
    return by_seq


def co3d_to_opencv_extrinsic(viewpoint: dict) -> np.ndarray:
    """CO3Dv2 viewpoint (PyTorch3D convention) -> OpenCV 3x4 [R|t].

    PyTorch3D stores R such that x_cam = x_world @ R + T (row-vector), and
    the camera axes are x-left, y-up, z-forward. OpenCV uses column-vectors
    (x_cam = R' @ x_world + T') with x-right, y-down, z-forward.

    The conversion:
        R_py3d_standard = R_stored.T      # column-vector rotation
        flip = diag(-1, -1, 1)            # swap x/y axes
        R_opencv = flip @ R_py3d_standard
        T_opencv = flip @ T_stored
    """
    R = np.array(viewpoint["R"], dtype=np.float64).T
    T = np.array(viewpoint["T"], dtype=np.float64)
    flip = np.diag([-1.0, -1.0, 1.0])
    R_cv = flip @ R
    T_cv = flip @ T
    extri = np.concatenate([R_cv, T_cv[:, None]], axis=-1)  # (3, 4)
    return extri


# ---------- metrics ----------

def rel_rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (deg) between two 3x3 rotation matrices."""
    R_rel = R1 @ R2.T
    tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def rel_translation_angle_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Angle (deg) between two translation directions. Translation norms are
    collapsed to unit length because VGGT is scale-ambiguous. Returns 0 if
    either vector is degenerate (near-zero)."""
    n1 = np.linalg.norm(t1); n2 = np.linalg.norm(t2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos = float(np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def extrinsic_to_rel(extri_i: np.ndarray, extri_j: np.ndarray):
    """Relative pose from camera i to camera j given world->cam extrinsics.
    Returns (R_rel, t_rel) where camera_j_from_camera_i."""
    Ri, ti = extri_i[:, :3], extri_i[:, 3]
    Rj, tj = extri_j[:, :3], extri_j[:, 3]
    R_rel = Rj @ Ri.T
    t_rel = tj - R_rel @ ti
    return R_rel, t_rel


def pairwise_pose_errors(pred_extri: np.ndarray, gt_extri: np.ndarray):
    """Per-pair (i, j) rotation + translation-direction errors (deg).
    pred_extri, gt_extri: (S, 3, 4)."""
    S = pred_extri.shape[0]
    rot_errs, tr_errs = [], []
    for i in range(S):
        for j in range(i + 1, S):
            R_pred, t_pred = extrinsic_to_rel(pred_extri[i], pred_extri[j])
            R_gt, t_gt = extrinsic_to_rel(gt_extri[i], gt_extri[j])
            rot_errs.append(rel_rotation_angle_deg(R_pred, R_gt))
            tr_errs.append(rel_translation_angle_deg(t_pred, t_gt))
    return np.array(rot_errs), np.array(tr_errs)


def rra_rta_at(errs_deg: np.ndarray, thresholds=(5, 15)) -> dict:
    out = {}
    for t in thresholds:
        out[f"at_{t}"] = float((errs_deg < t).mean() * 100.0)
    return out


def auc_deg(errs_deg: np.ndarray, max_thr: float = 30.0, nbins: int = 30) -> float:
    """Area under cumulative-error curve up to max_thr, normalised to [0, 100]."""
    if len(errs_deg) == 0:
        return 0.0
    thrs = np.linspace(0.0, max_thr, nbins + 1)
    cdf = np.array([(errs_deg < t).mean() for t in thrs])
    # Trapezoidal integration, normalise to percentage.
    auc = float(np.trapz(cdf, thrs) / max_thr * 100.0)
    return auc


# ---------- relative PCC (port vs ref) ----------

def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom > 0 else 0.0


# ---------- per-scene eval ----------

def eval_scene(category: str, seq_name: str, seq_anns: list,
               co3d_root: Path, num_views: int,
               ref_model, tt_forward, device, seed: int = 0):
    rng = random.Random(seed)
    if len(seq_anns) < num_views:
        return None
    # Even spacing through the sequence — VGGT trains with diverse viewpoints.
    step = max(1, len(seq_anns) // num_views)
    picks = [seq_anns[i * step] for i in range(num_views)]

    img_paths = [str(co3d_root / p["image"]["path"]) for p in picks]
    images = load_and_preprocess_images(img_paths, mode="pad")
    # Harness expects (B, S, 3, H, W).
    images_bSCHW = images.unsqueeze(0)  # B=1

    # GT extrinsics in OpenCV convention.
    gt_extri = np.stack([co3d_to_opencv_extrinsic(p["viewpoint"]) for p in picks])  # (S, 3, 4)

    H, W = images.shape[-2], images.shape[-1]

    # --- reference ---
    with torch.no_grad():
        ref_out = ref_model(images_bSCHW)
    pose_enc_ref = ref_out["pose_enc"].cpu()  # (1, S, 9)
    ref_extri, _ = pose_encoding_to_extri_intri(
        pose_enc_ref, image_size_hw=(H, W), build_intrinsics=False,
    )
    ref_extri = ref_extri[0].numpy()  # (S, 3, 4)

    # --- port ---
    tt_out = tt_forward(images_bSCHW, device=device)
    pose_enc_tt = tt_out["pose_enc"].cpu()
    tt_extri, _ = pose_encoding_to_extri_intri(
        pose_enc_tt, image_size_hw=(H, W), build_intrinsics=False,
    )
    tt_extri = tt_extri[0].numpy()

    # Relative PCC on key outputs.
    pcc_scores = {}
    for k in ("pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"):
        if k in ref_out and k in tt_out:
            pcc_scores[k] = pcc(ref_out[k], tt_out[k])

    # Pair-wise pose errors vs GT.
    ref_rot, ref_tr = pairwise_pose_errors(ref_extri, gt_extri)
    tt_rot, tt_tr = pairwise_pose_errors(tt_extri, gt_extri)

    return {
        "seq": seq_name,
        "num_views": num_views,
        "num_pairs": len(ref_rot),
        "pcc": pcc_scores,
        "ref_rra": rra_rta_at(ref_rot),
        "ref_rta": rra_rta_at(ref_tr),
        "ref_auc30": auc_deg(np.minimum(ref_rot, ref_tr)),  # joint AUC
        "ref_rot_med": float(np.median(ref_rot)),
        "ref_tr_med": float(np.median(ref_tr)),
        "tt_rra": rra_rta_at(tt_rot),
        "tt_rta": rra_rta_at(tt_tr),
        "tt_auc30": auc_deg(np.minimum(tt_rot, tt_tr)),
        "tt_rot_med": float(np.median(tt_rot)),
        "tt_tr_med": float(np.median(tt_tr)),
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d-root", type=Path, default=Path("/home/ttuser/experiments/vggt/co3d_data"))
    ap.add_argument("--category", default="apple")
    ap.add_argument("--seqs", default="",
                    help="Comma-separated sequence names. Empty = all in category.")
    ap.add_argument("--num-views", type=int, default=8)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"# loading CO3Dv2 annotations: {args.category}")
    by_seq = load_co3d_annotations(args.co3d_root, args.category)
    seqs = args.seqs.split(",") if args.seqs else list(by_seq.keys())
    seqs = [s for s in seqs if s in by_seq]
    print(f"# sequences: {seqs}")

    # Open device + install ttnn port on the cached model. Separately load
    # a fresh, un-ported VGGT for the reference path. Block.forward is
    # class-patched but gates on _tt_block_ready (only set on the
    # install-preloaded instance), so a fresh VGGT falls through to the
    # original forward automatically.
    import ttnn
    from tt.ttnn_vggt import vggt_forward, _ensure_installed
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        _ensure_installed(device)
        print("# loading separate un-ported reference model")
        ref_model = load_vggt(eval_mode=True)

        results = []
        for seq in seqs:
            anns = by_seq[seq]
            if len(anns) < args.num_views:
                print(f"# skip {seq}: only {len(anns)} frames")
                continue
            print(f"# eval seq {seq} ({len(anns)} frames, picking {args.num_views})")
            r = eval_scene(args.category, seq, anns, args.co3d_root,
                           args.num_views, ref_model, vggt_forward, device,
                           seed=args.seed)
            if r is None:
                continue
            results.append(r)
            _print_row(r)

        if results:
            _print_summary(results)
    finally:
        ttnn.close_device(device)


def _fmt_pcc(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())


def _print_row(r):
    print(f"\n--- seq: {r['seq']}  views: {r['num_views']}  pairs: {r['num_pairs']}")
    print(f"pcc (port vs ref): {_fmt_pcc(r['pcc'])}")
    print(f"ref  rot_med={r['ref_rot_med']:.2f}deg  tr_med={r['ref_tr_med']:.2f}deg  "
          f"RRA@5={r['ref_rra']['at_5']:.1f} @15={r['ref_rra']['at_15']:.1f}  "
          f"RTA@5={r['ref_rta']['at_5']:.1f} @15={r['ref_rta']['at_15']:.1f}  "
          f"AUC30={r['ref_auc30']:.1f}")
    print(f"port rot_med={r['tt_rot_med']:.2f}deg  tr_med={r['tt_tr_med']:.2f}deg  "
          f"RRA@5={r['tt_rra']['at_5']:.1f} @15={r['tt_rra']['at_15']:.1f}  "
          f"RTA@5={r['tt_rta']['at_5']:.1f} @15={r['tt_rta']['at_15']:.1f}  "
          f"AUC30={r['tt_auc30']:.1f}")


def _print_summary(results):
    print("\n=== summary across {} scene(s) ===".format(len(results)))
    def mean(key_path):
        vals = []
        for r in results:
            v = r
            for k in key_path.split("."):
                v = v[k]
            vals.append(v)
        return sum(vals) / len(vals)
    pcc_keys = list(results[0]["pcc"].keys())
    for k in pcc_keys:
        print(f"mean pcc_{k:<22} : {mean(f'pcc.{k}'):.4f}")
    print("ref  mean RRA@5  = {:.1f}   RRA@15 = {:.1f}   RTA@5  = {:.1f}   RTA@15 = {:.1f}   AUC30 = {:.1f}".format(
        mean("ref_rra.at_5"), mean("ref_rra.at_15"),
        mean("ref_rta.at_5"), mean("ref_rta.at_15"),
        mean("ref_auc30"),
    ))
    print("port mean RRA@5  = {:.1f}   RRA@15 = {:.1f}   RTA@5  = {:.1f}   RTA@15 = {:.1f}   AUC30 = {:.1f}".format(
        mean("tt_rra.at_5"), mean("tt_rra.at_15"),
        mean("tt_rta.at_5"), mean("tt_rta.at_15"),
        mean("tt_auc30"),
    ))
    print("Δ(port-ref) RRA@5 = {:+.1f}  RRA@15 = {:+.1f}  RTA@5 = {:+.1f}  RTA@15 = {:+.1f}  AUC30 = {:+.1f}".format(
        mean("tt_rra.at_5") - mean("ref_rra.at_5"),
        mean("tt_rra.at_15") - mean("ref_rra.at_15"),
        mean("tt_rta.at_5") - mean("ref_rta.at_5"),
        mean("tt_rta.at_15") - mean("ref_rta.at_15"),
        mean("tt_auc30") - mean("ref_auc30"),
    ))


if __name__ == "__main__":
    main()
