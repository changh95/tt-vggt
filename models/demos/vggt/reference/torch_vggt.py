"""Torch reference loader for facebook/VGGT-1B.

Builds the upstream ``vggt.models.vggt.VGGT`` module (the pinned
facebookresearch/vggt checkout must be importable as top-level ``vggt``;
in the tt-model image it lives at ``/opt/tt-metal/vggt``) and loads the
pretrained ``model.safetensors``.

Weights are resolved, in order, from:

1. ``VGGT_WEIGHTS_DIR`` -- a local directory holding ``model.safetensors``
   (offline / host validation override);
2. ``huggingface_hub.hf_hub_download(HF_MODEL, "model.safetensors",
   revision=TT_WEIGHTS_REVISION)`` -- ``HF_MODEL`` defaults to
   ``facebook/VGGT-1B``; ``TT_WEIGHTS_REVISION`` is the sha the tt-model
   manifest pins (``tt-model serve`` pre-downloads exactly that snapshot
   into the HF cache mounted at ``/hf``, so this is a cache hit there).

We only need forward+eval for PCC validation and serving; training
utilities are not wired.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

DEFAULT_WEIGHTS_REPO = "facebook/VGGT-1B"
WEIGHTS_FILENAME = "model.safetensors"


def weights_repo() -> str:
    return os.environ.get("HF_MODEL") or DEFAULT_WEIGHTS_REPO


def weights_revision() -> Optional[str]:
    return os.environ.get("TT_WEIGHTS_REVISION") or None


def resolve_weights_path(filename: str = WEIGHTS_FILENAME) -> Path:
    """Return the local path of ``filename`` without loading it."""
    local_dir = os.environ.get("VGGT_WEIGHTS_DIR")
    if local_dir:
        p = Path(local_dir).expanduser() / filename
        if not p.is_file():
            raise FileNotFoundError(f"VGGT_WEIGHTS_DIR={local_dir!r} has no {filename}")
        return p
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=weights_repo(), filename=filename, revision=weights_revision())
    )


def load_state_dict(weights_path: Optional[os.PathLike] = None) -> dict[str, torch.Tensor]:
    path = Path(weights_path) if weights_path is not None else resolve_weights_path()
    return load_file(str(path))


def load_vggt(eval_mode: bool = True, enable_track: Optional[bool] = None,
              weights_path: Optional[os.PathLike] = None):
    """Build VGGT and load the pretrained weights.

    ``enable_track`` (default: env ``VGGT_ENABLE_TRACK``, off) controls the
    tracking head. It is not used by the ttnn port (``query_points`` is
    never passed) and its ~46M parameters only cost host RAM and dead bf16
    uploads, so it is skipped by default and its ``track_head.*`` keys are
    dropped before the strict bookkeeping check below.
    """
    from vggt.models.vggt import VGGT  # type: ignore

    if enable_track is None:
        enable_track = os.environ.get("VGGT_ENABLE_TRACK", "0") not in ("", "0")

    model = VGGT(enable_track=enable_track)
    sd = load_state_dict(weights_path)
    if not enable_track:
        sd = {k: v for k, v in sd.items() if not k.startswith("track_head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict: missing={missing[:3]} unexpected={unexpected[:3]}")
    if eval_mode:
        model.eval()
    return model
