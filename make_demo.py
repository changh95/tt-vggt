#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-view 3D point-map demo for the README / model card.

Two views of one scene go in, ONE figure comes out (``media/pointmap.png``): the two
inputs on the left, the fused 3D point map on the right from two viewpoints. The point
map is the union of both views' ``world_points`` (VGGT predicts every view's points in
the frame of camera 1 = frame 0), coloured by the source pixels and confidence-filtered.

Default = ask a running server (the same request `POST /predict` any client sends, NPZ
response); ``--local`` runs the ttnn port directly on the device instead.

    python code/make_demo.py                                  # media/source_1.png + source_2.png -> http://127.0.0.1:20000
    python code/make_demo.py a.png b.png --url http://host:20000 --save-response out.json
    TT_METAL_HOME=/path/to/tt-metal VGGT_REF=/path/to/vggt python code/make_demo.py --local

Client side needs only numpy + pillow + matplotlib (``--local`` additionally torch, ttnn and
the upstream ``vggt`` package, exactly like the server).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _CODE_ROOT.parent
IMG_SIZE, PATCH = 518, 14


# --------------------------------------------------------------------------- preprocessing

def pad_geometry(w: int, h: int) -> dict:
    """The server's ``preprocess.per_view`` entry for a w x h image (longer side -> 518,
    shorter side -> round(x/14)*14, centred white pad to 518x518)."""
    if w >= h:
        new_w = IMG_SIZE
        new_h = round(h * (new_w / w) / PATCH) * PATCH
    else:
        new_h = IMG_SIZE
        new_w = round(w * (new_h / h) / PATCH) * PATCH
    pad_h, pad_w = IMG_SIZE - new_h, IMG_SIZE - new_w
    return {"orig_w": w, "orig_h": h, "resized_w": new_w, "resized_h": new_h,
            "scale_x": new_w / w, "scale_y": new_h / h, "pad_left": pad_w // 2, "pad_top": pad_h // 2}


def padded_rgb(path: Path, geo: dict) -> tuple[np.ndarray, np.ndarray]:
    """(518,518,3) float RGB in [0,1] laid out like the server's canvas, and the (518,518)
    bool mask of real (non-padding) pixels."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA"))
    im = im.convert("RGB").resize((geo["resized_w"], geo["resized_h"]), Image.BICUBIC)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    canvas.paste(im, (geo["pad_left"], geo["pad_top"]))
    valid = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    valid[geo["pad_top"]:geo["pad_top"] + geo["resized_h"], geo["pad_left"]:geo["pad_left"] + geo["resized_w"]] = True
    return np.asarray(canvas, dtype=np.float32) / 255.0, valid


# --------------------------------------------------------------------------- inference

def predict_server(paths: list[Path], url: str, timeout: float, save_response: Path | None) -> dict:
    """POST the views to a running server and return the decoded prediction."""
    b64s = [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths]
    payload = {"images": b64s, "output_format": "npz", "dtype": "float32"}
    req = urllib.request.Request(f"{url.rstrip('/')}/predict", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    out = json.loads(raw)
    if save_response is not None:
        save_response.parent.mkdir(parents=True, exist_ok=True)
        save_response.write_bytes(raw)
        z = np.load(io.BytesIO(base64.b64decode(out["dense"]["data"])))
        np.savez_compressed(save_response.with_suffix(".npz"), **{k: z[k] for k in z.files})
    z = np.load(io.BytesIO(base64.b64decode(out["dense"]["data"])))
    return {
        "world_points": z["world_points"].astype(np.float32),        # (S, 518, 518, 3)
        "world_points_conf": z["world_points_conf"].astype(np.float32),
        "depth": z["depth"].astype(np.float32),                      # (S, 518, 518)
        "extrinsic": np.asarray(out["extrinsic"], dtype=np.float64),  # (S, 3, 4) cam-from-world
        "intrinsic": np.asarray(out["intrinsic"], dtype=np.float64),  # (S, 3, 3)
        "per_view": out["preprocess"]["per_view"],
        "timing_ms": dict(out["timing_ms"], client_wall=round(wall_ms, 1)),
        "source": f"server {url} ({out.get('model')})",
    }


def load_response(path: Path) -> dict:
    """Re-render from a raw ``/predict`` JSON response saved earlier with ``--save-response``."""
    out = json.loads(path.read_bytes())
    z = np.load(io.BytesIO(base64.b64decode(out["dense"]["data"])))
    return {
        "world_points": z["world_points"].astype(np.float32),
        "world_points_conf": z["world_points_conf"].astype(np.float32),
        "depth": z["depth"].astype(np.float32),
        "extrinsic": np.asarray(out["extrinsic"], dtype=np.float64),
        "intrinsic": np.asarray(out["intrinsic"], dtype=np.float64),
        "per_view": out["preprocess"]["per_view"],
        "timing_ms": out["timing_ms"],
        "source": f"saved response {path} ({out.get('model')})",
    }


def predict_local(paths: list[Path]) -> dict:
    """Run the ttnn port directly (needs a device, torch, ttnn and the `vggt` package)."""
    for var, sub in (("TT_METAL_HOME", ("", "ttnn", "tools")), ("VGGT_REF", ("",))):
        root = os.environ.get(var)
        if root:
            for s in sub:
                p = os.path.join(root, s) if s else root
                if p not in sys.path:
                    sys.path.append(p)
    if str(_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(_CODE_ROOT))
    import torch
    import ttnn
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from models.demos.vggt.tt.ttnn_vggt import vggt_forward, _ensure_installed, trace_region_bytes, l1_small_bytes

    images = load_and_preprocess_images([str(p) for p in paths], mode="pad").unsqueeze(0)  # (1, S, 3, 518, 518)
    trace_kw = {"trace_region_size": trace_region_bytes()} if trace_region_bytes() else {}
    device = ttnn.open_device(device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
                              l1_small_size=l1_small_bytes(), **trace_kw)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        _ensure_installed(device)
        with torch.no_grad():
            t0 = time.perf_counter()
            out = vggt_forward(images, device=device)
            fwd_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        ttnn.close_device(device)
    pose_enc = out["pose_enc"].detach().float().cpu()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(IMG_SIZE, IMG_SIZE))
    geos = [pad_geometry(*Image.open(p).size) for p in paths]
    return {
        "world_points": out["world_points"][0].detach().float().cpu().numpy(),
        "world_points_conf": out["world_points_conf"][0].detach().float().cpu().numpy(),
        "depth": out["depth"][0, ..., 0].detach().float().cpu().numpy(),
        "extrinsic": extrinsic[0].numpy().astype(np.float64),
        "intrinsic": intrinsic[0].numpy().astype(np.float64),
        "per_view": geos,
        "timing_ms": {"forward": round(fwd_ms, 1)},
        "source": "local ttnn port (vggt_forward)",
    }


# --------------------------------------------------------------------------- metrics

def cross_view_consistency(pred: dict, valid: list[np.ndarray], keep: list[np.ndarray]) -> dict:
    """How well view 2's points (predicted in the camera-1 frame) agree with view 1's own
    depth map: project them with camera 1's intrinsics, compare z against depth_1 at the
    landing pixel. Median relative difference ~ a few % for one coherent scene; ~O(1) for
    two unrelated sheets."""
    K1 = pred["intrinsic"][0]
    p2 = pred["world_points"][1][keep[1]]                     # (N, 3) in the camera-1 frame
    z = p2[:, 2]
    ok = z > 1e-6
    u = K1[0, 0] * p2[ok, 0] / z[ok] + K1[0, 2]
    v = K1[1, 1] * p2[ok, 1] / z[ok] + K1[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (ui >= 0) & (ui < IMG_SIZE) & (vi >= 0) & (vi < IMG_SIZE)
    ui, vi, z_ok = ui[inside], vi[inside], z[ok][inside]
    landed = valid[0][vi, ui] & keep[0][vi, ui]
    d1 = pred["depth"][0][vi[landed], ui[landed]]
    rel = np.abs(z_ok[landed] - d1) / np.maximum(d1, 1e-6)
    return {
        "view2_points_tested": int(p2.shape[0]),
        "frac_inside_view1_frustum": float(inside.mean()) if inside.size else 0.0,
        "frac_landing_on_kept_view1_pixels": float(landed.mean()) if landed.size else 0.0,
        "median_rel_depth_diff": float(np.median(rel)) if rel.size else None,
        "frac_rel_depth_diff_lt_10pct": float((rel < 0.10).mean()) if rel.size else None,
    }


def relative_pose(pred: dict) -> dict:
    E1, E2 = pred["extrinsic"][0], pred["extrinsic"][1]          # cam-from-world, world = cam 1
    R_rel = E2[:, :3] @ E1[:, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
    c2_in_world = -E2[:, :3].T @ E2[:, 3]                          # camera-2 centre in the camera-1 frame
    return {"rotation_deg": float(ang), "camera2_centre_in_cam1": [float(x) for x in c2_in_world],
            "baseline": float(np.linalg.norm(c2_in_world))}


# --------------------------------------------------------------------------- rendering

def splat(P: np.ndarray, cols: np.ndarray, yaw_deg: float, pitch_deg: float, size: tuple[int, int],
          box: tuple[np.ndarray, np.ndarray], px: int = 2, margin: float = 0.04) -> np.ndarray:
    """Orthographic z-buffered splat of points ``P`` (plot frame: x right, y into the page,
    z up) seen from a virtual camera turned ``yaw`` degrees to the right of camera 1 and
    ``pitch`` degrees above it. Equal axes: one scale for both screen directions, chosen so
    the 3D box ``box`` fits the canvas. Returns an (H, W, 3) float RGB image on white."""
    W, H = size
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Rv = np.stack([r, u, f])                                   # rows: screen-right, screen-up, depth
    Q = P @ Rv.T
    corners = np.array([[x, y, z] for x in (box[0][0], box[1][0]) for y in (box[0][1], box[1][1])
                        for z in (box[0][2], box[1][2])]) @ Rv.T
    lo, hi = corners.min(0), corners.max(0)
    scale = (1 - 2 * margin) * min(W / max(hi[0] - lo[0], 1e-9), H / max(hi[1] - lo[1], 1e-9))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    us = np.round((Q[:, 0] - cx) * scale + W / 2).astype(int)
    vs = np.round(H / 2 - (Q[:, 1] - cy) * scale).astype(int)
    depth = Q[:, 2]
    img = np.ones((H, W, 3), dtype=np.float32)
    zbuf = np.full(H * W, np.inf)
    for du in range(px):
        for dv in range(px):
            uu, vv = us + du, vs + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            lin = vv[ok] * W + uu[ok]
            d, c = depth[ok], cols[ok]
            order = np.lexsort((d, lin))                       # by pixel, nearest first
            lin_s = lin[order]
            first = np.r_[True, lin_s[1:] != lin_s[:-1]]
            sel = order[first]
            tgt = lin[sel]
            nearer = d[sel] < zbuf[tgt]
            zbuf[tgt[nearer]] = d[sel][nearer]
            img.reshape(-1, 3)[tgt[nearer]] = c[sel][nearer]
    return img


def render(paths: list[Path], pred: dict, out_png: Path, conf_min: float, max_points: int,
           seed: int = 0) -> dict:
    S = pred["world_points"].shape[0]
    assert S == 2, f"demo expects two views, got S={S}"
    rgbs, valids, keeps, thresholds = [], [], [], []
    for i in range(S):
        rgb, valid = padded_rgb(paths[i], pred["per_view"][i])
        conf = pred["world_points_conf"][i]
        # One absolute threshold for both views (expp1 confidence, >= 1): a per-view quantile would
        # cut the fine-structured object out of the view whose flat mats push its quantiles up.
        keep = valid & (conf >= conf_min) & np.isfinite(pred["world_points"][i]).all(-1)
        rgbs.append(rgb); valids.append(valid); keeps.append(keep); thresholds.append(float(conf_min))

    pts = np.concatenate([pred["world_points"][i][keeps[i]] for i in range(S)])
    cols = np.concatenate([rgbs[i][keeps[i]] for i in range(S)]).clip(0, 1)
    n_kept = [int(k.sum()) for k in keeps]
    rng = np.random.default_rng(seed)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    # OpenCV camera-1 frame (x right, y down, z forward) -> plot frame (x right, depth into the page, up).
    P = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], 1)
    box = (np.quantile(P, 0.02, axis=0), np.quantile(P, 0.98, axis=0))   # framing box; outliers stay drawn if on-canvas
    views = [("front-left of camera 1, slightly above", -25, 15), ("top-down from the left", -45, 45)]
    canvas = (760, 760)
    renders = [splat(P, cols, yaw, pitch, canvas, box) for _, yaw, pitch in views]

    fig = plt.figure(figsize=(15.0, 6.6), dpi=130)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.55, 1.55], wspace=0.03, hspace=0.12,
                          left=0.01, right=0.99, top=0.88, bottom=0.02)
    for i in range(S):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(np.asarray(Image.open(paths[i]).convert("RGB")))
        ax.set_title(f"view {i + 1}  ({paths[i].name})", fontsize=10, pad=3)
        ax.set_axis_off()
    for j, ((label, _, _), img) in enumerate(zip(views, renders)):
        ax = fig.add_subplot(gs[:, j + 1])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"fused point map, {label}", fontsize=10, pad=3)
        ax.set_axis_off()
    fig.suptitle("VGGT-1B on Blackhole p150a: two views -> one point map in the camera-1 frame "
                 f"(union of both views' world_points, {pts.shape[0]:,} points, confidence-filtered)",
                 fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"conf_min": conf_min, "conf_thresholds": thresholds, "points_kept_per_view": n_kept,
            "points_drawn": int(pts.shape[0]), "valid": valids, "keep": keeps}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", nargs="?", type=Path, default=REPO_ROOT / "media" / "source_1.png")
    ap.add_argument("image2", nargs="?", type=Path, default=REPO_ROOT / "media" / "source_2.png")
    ap.add_argument("--url", default="http://127.0.0.1:20000", help="running server (default)")
    ap.add_argument("--local", action="store_true", help="run the ttnn port directly instead of the server")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "media" / "pointmap.png")
    ap.add_argument("--save-response", type=Path, default=None,
                    help="server mode: save the raw JSON response here (and its dense arrays as <stem>.npz)")
    ap.add_argument("--from-response", type=Path, default=None,
                    help="re-render from a raw JSON response saved with --save-response (no server, no device)")
    ap.add_argument("--metrics", type=Path, default=None, help="write the printed numbers as JSON")
    ap.add_argument("--conf-min", type=float, default=1.5,
                    help="drop points whose world_points_conf is below this (expp1 confidence >= 1; default 1.5 "
                         "removes the saturated-low background, keeps the object and the table)")
    ap.add_argument("--max-points", type=int, default=260_000)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    paths = [args.image1, args.image2]
    for p in paths:
        if not p.is_file():
            ap.error(f"missing input image {p}")
    if args.from_response:
        pred = load_response(args.from_response)
    elif args.local:
        pred = predict_local(paths)
    else:
        pred = predict_server(paths, args.url, args.timeout, args.save_response)

    info = render(paths, pred, args.out, args.conf_min, args.max_points)
    metrics = {
        "inputs": [str(p) for p in paths],
        "source": pred["source"],
        "timing_ms": pred["timing_ms"],
        "conf_min": info["conf_min"],
        "points_kept_per_view": info["points_kept_per_view"],
        "points_drawn": info["points_drawn"],
        "relative_pose_cam2_wrt_cam1": relative_pose(pred),
        "cross_view_consistency": cross_view_consistency(pred, info["valid"], info["keep"]),
        "depth_range_per_view": [[float(np.nanmin(pred["depth"][i][info["valid"][i]])),
                                  float(np.nanmax(pred["depth"][i][info["valid"][i]]))] for i in range(2)],
        "output": str(args.out),
    }
    print(json.dumps(metrics, indent=1))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
