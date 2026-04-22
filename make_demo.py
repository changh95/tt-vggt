#!/usr/bin/env python3
"""Run the ttnn port on one image and write three demo artefacts into
media/: the input, a depth colormap, and a re-rendered point cloud from a
different azimuth. Used to populate the README demo section.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
if _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)
    sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
os.chdir(_TT_METAL_ROOT)
_VGGT_DEMO = "/home/ttuser/experiments/vggt/tt-metal/models/demos/vggt"
sys.path.insert(0, _VGGT_DEMO)
_VGGT_REF = "/home/ttuser/experiments/vggt/vggt_ref"
sys.path.insert(0, _VGGT_REF)

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def main():
    img_src = Path("/home/ttuser/experiments/vggt/vggt_ref/examples/kitchen/images/00.png")
    media_dir = Path("/home/ttuser/experiments/vggt/tt-vggt/media")
    media_dir.mkdir(exist_ok=True)

    images = load_and_preprocess_images([str(img_src)], mode="pad")  # (1, 3, 518, 518)
    images_bSCHW = images.unsqueeze(0)  # (B=1, S=1, 3, 518, 518)

    import ttnn
    from tt.ttnn_vggt import vggt_forward, _ensure_installed
    device = ttnn.open_device(device_id=2, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        _ensure_installed(device)
        with torch.no_grad():
            out = vggt_forward(images_bSCHW, device=device)
    finally:
        ttnn.close_device(device)

    # The preprocessed image is padded to 518x518 with white. Use that as
    # the colour source so the projected points line up with the geometry
    # predicted at 518x518.
    img_arr = images[0].permute(1, 2, 0).cpu().numpy()  # (518, 518, 3) in [0, 1]
    depth = out["depth"][0, 0, :, :, 0].detach().cpu().numpy()  # (518, 518)
    world_points = out["world_points"][0, 0].detach().cpu().numpy()  # (518, 518, 3)
    wp_conf = out["world_points_conf"][0, 0].detach().cpu().numpy()  # (518, 518)

    # ---- input ----
    Image.fromarray((img_arr * 255).clip(0, 255).astype(np.uint8)).save(
        media_dir / "input.png"
    )

    # ---- depth colormap ----
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=120)
    ax.imshow(depth, cmap="turbo")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(media_dir / "depth.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ---- point cloud re-rendered from a different azimuth ----
    pts = world_points.reshape(-1, 3)
    colors = img_arr.reshape(-1, 3).clip(0.0, 1.0)
    conf = wp_conf.reshape(-1)
    # Filter: drop the bottom quartile of confidence (background + sky in the
    # kitchen scene look noisy otherwise).
    thr = np.quantile(conf, 0.25)
    keep = conf >= thr
    pts = pts[keep]
    colors = colors[keep]

    # Subsample for interactive-render speed and file size.
    rng = np.random.default_rng(0)
    n_target = 30000
    if pts.shape[0] > n_target:
        idx = rng.choice(pts.shape[0], n_target, replace=False)
        pts = pts[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(5.6, 5.6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    # VGGT world_points use OpenCV convention (x-right, y-down, z-forward).
    # Remap so plot-up is world-up: plot(X, Z, -Y).
    plot_x = pts[:, 0]
    plot_y = pts[:, 2]   # depth -> plot's into-the-page direction
    plot_z = -pts[:, 1]  # -Y -> plot-up
    ax.scatter(plot_x, plot_y, plot_z, c=colors, s=0.6, marker=".")
    # Rotate the virtual camera slightly off the capture viewpoint.
    ax.view_init(elev=5, azim=-15)
    ax.set_axis_off()
    # Trim to the 2..98 percentile box so outliers don't squash the frame.
    for setter, arr in (
        (ax.set_xlim, plot_x), (ax.set_ylim, plot_y), (ax.set_zlim, plot_z),
    ):
        lo, hi = np.quantile(arr, [0.02, 0.98])
        setter(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0)
    fig.savefig(media_dir / "point_cloud_reprojected.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"wrote {media_dir}/input.png")
    print(f"wrote {media_dir}/depth.png")
    print(f"wrote {media_dir}/point_cloud_reprojected.png")
    print(f"depth range: {depth.min():.3f} .. {depth.max():.3f}")
    print(f"points kept: {pts.shape[0]} (from {wp_conf.size} total)")


if __name__ == "__main__":
    main()
