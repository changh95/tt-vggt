"""Torch-only constant tables and exact reformulations for the ``TT_FUSED=1`` path.

Everything in this module is plain ``torch`` (no ``ttnn`` import) so that the
reformulations the device-resident wrapper (``ttnn_vggt_fused.py``) relies on can be
proven against the reference math on any host (``models/tests/test_fused_host.py``).

Contents (lever ids refer to ``reports/megakernel/vggt-1b-p150.md``):

* knob parsing: ``fused_enabled`` (``TT_FUSED``) and the sub-knobs ``fused_option``.
* B2 -- fold ``1/sqrt(Dh)`` into the Q projection (no qk-norm) or into the q-norm affine
  (qk-norm).  Exact only when the scale is a power of two (``Dh = 64`` -> ``1/8``);
  ``scale_fold_is_exact`` says so, the camera trunk (``Dh = 128``) keeps its multiply.
* B3 -- VGGT's 2-D RoPE (two independent 32-dim rotate-half rotations on the vertical /
  horizontal halves of a 64-dim head) rewritten as ONE standard 64-dim HF RoPE
  (``x * cos + rotate_half(x) * sin``, midpoint 32) after a fixed permutation of the
  head columns.  ``ROPE_PERM = [v0..15, h0..15, v16..31, h16..31]``; the permutation is
  folded into the Q/K output columns of ``qkv`` and into the q/k-norm affine, the merged
  angle table is ``[theta_y(16), theta_x(16), theta_y(16), theta_x(16)]``.  Because Q and K
  share the permutation ``Q.K^T`` is invariant (up to fp32 reassociation), V is untouched.
* stage 3 -- aggregator token assembly (drop the 5 DINOv2 special rows, prepend the
  camera + register constants) as ``x * row_mask + T`` with a ``(1, P, 1)`` 0/1 mask and a
  constant ``(S, P, C)`` table (two exact eltwise ops instead of slice + padded concat).
* B6 -- bilinear ``align_corners=True`` resize as two constant matmuls
  ``A_h @ X @ A_w^T`` (``interp_matrix``); the ttnn bilinear kernel implements the
  half-pixel convention and cannot be used.
* B7 -- DPT positional-embedding constants in the layouts the device graph adds them in.

Model constants are those of VGGT-1B at 518x518 (patch 14 -> 37x37 = 1369 patches,
1 camera + 4 register tokens -> P = 1374 tokens per frame, C = 1024, 16 heads x 64).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

IMG_SIZE = 518
PATCH = 14
GRID = IMG_SIZE // PATCH            # 37
N_PATCHES = GRID * GRID             # 1369
N_SPECIAL = 5                       # 1 camera + 4 register tokens (== DINOv2's 1 cls + 4 reg)
P_TOKENS = N_PATCHES + N_SPECIAL    # 1374
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = EMBED_DIM // NUM_HEADS   # 64
ROPE_BASE = 100.0
TILE = 32

# DPT geometry: resize_layers outputs 148/74/37/19, refinenets 19->37->74->148->296,
# custom_interpolate 296->518.
DPT_UPSAMPLES: Tuple[Tuple[int, int], ...] = ((19, 37), (37, 74), (74, 148), (148, 296), (296, 518))


# ----------------------------------------------------------------------------- knobs

_TRUE = ("1", "true", "yes", "on")


def fused_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """``TT_FUSED`` -> bool.  Default ON since the p150a validation of 2026-09-13 (device-resident,
    metal-traced port); ``TT_FUSED=0`` (or false/no/off) selects the legacy monkey-patch path,
    unchanged.  An empty value counts as unset."""
    env = os.environ if env is None else env
    raw = str(env.get("TT_FUSED", "1")).strip().lower()
    return (raw or "1") in _TRUE


def fused_option(name: str, default: str, choices: Sequence[str],
                 env: Optional[Mapping[str, str]] = None) -> str:
    """Read sub-knob ``VGGT_FUSED_<name>`` (lower-cased) and validate it against ``choices``."""
    env = os.environ if env is None else env
    raw = str(env.get(f"VGGT_FUSED_{name}", default)).strip().lower() or default
    if raw not in choices:
        raise RuntimeError(f"VGGT_FUSED_{name}={raw!r}: expected one of {list(choices)}")
    return raw


def fused_int(name: str, default: int, env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    raw = str(env.get(f"VGGT_FUSED_{name}", "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"VGGT_FUSED_{name}={raw!r} is not an integer") from None


def trace_region_bytes(env: Optional[Mapping[str, str]] = None) -> int:
    """Trace region to reserve when opening the device: 0 when the knob is off (legacy
    ``ttnn.open_device`` call unchanged), else ``VGGT_TRACE_REGION_MB`` (default 1536) MiB.
    Multiple coexisting traces need a pre-allocated region (tt-metal issue 48869)."""
    env = os.environ if env is None else env
    if not fused_enabled(env):
        return 0
    raw = str(env.get("VGGT_TRACE_REGION_MB", "1536")).strip() or "1536"
    return int(raw) * 1024 * 1024


def l1_small_bytes(env: Optional[Mapping[str, str]] = None) -> int:
    """``l1_small_size`` for ``ttnn.open_device``: the port's 32 KiB when the knob is off, else
    ``VGGT_L1_SMALL_KB`` KiB (default 64).  conv2d keeps its sliding-window config tensors in the
    L1_SMALL region per distinct conv shape; with the batched DPT chain the second pre-warmed S
    overflowed 32 KiB (``Out of Memory: ... L1_SMALL ... allocated: 32704 B, free: 64 B``, p150a
    2026-09-13).  The DPT chain now runs per frame (S = 1 shapes only); 64 KiB is margin."""
    env = os.environ if env is None else env
    if not fused_enabled(env):
        return 32 * 1024
    raw = str(env.get("VGGT_L1_SMALL_KB", "64")).strip() or "64"
    return int(raw) * 1024


def pad_to_tile(n: int, tile: int = TILE) -> int:
    return -(-n // tile) * tile


# ----------------------------------------------------------------------------- B2 scale fold

def attention_scale(head_dim: int) -> float:
    return head_dim ** -0.5


def scale_fold_is_exact(head_dim: int) -> bool:
    """True iff ``1/sqrt(head_dim)`` is a power of two, i.e. multiplying bf16/fp32 weights by
    it is an exponent shift (bit-exact, denormals aside).  ``Dh = 64 -> 1/8`` yes,
    ``Dh = 128`` (camera trunk) no."""
    if head_dim <= 0:
        return False
    log2 = math.log2(head_dim)
    return abs(log2 - round(log2)) < 1e-9 and int(round(log2)) % 2 == 0


# ----------------------------------------------------------------------------- B3 RoPE

def rope_perm(head_dim: int = HEAD_DIM) -> torch.Tensor:
    """Column permutation ``[v0..q-1, h0..q-1, vq..2q-1, hq..2q-1]`` with ``q = head_dim/4``;
    ``x[..., perm]`` puts the two rotate-half pairs of VGGT's 2-D RoPE at HF midpoint distance.
    perm[j] is the ORIGINAL column that lands at permuted position j."""
    assert head_dim % 4 == 0, head_dim
    q = head_dim // 4
    v_lo = list(range(0, q))                 # vertical half, first quarter
    v_hi = list(range(q, 2 * q))             # vertical half, second quarter
    h_lo = list(range(2 * q, 3 * q))         # horizontal half, first quarter
    h_hi = list(range(3 * q, 4 * q))         # horizontal half, second quarter
    return torch.tensor(v_lo + h_lo + v_hi + h_hi, dtype=torch.long)


def rope_inv_freq(dim_half: int = HEAD_DIM // 2, base: float = ROPE_BASE) -> torch.Tensor:
    """Frequency bands of one 1-D RoPE of width ``dim_half`` (VGGT: ``exponents = arange(0, D, 2)/D``)."""
    exponents = torch.arange(0, dim_half, 2).float() / dim_half
    return 1.0 / (base ** exponents)  # (dim_half/2,)


def frame_positions(grid: int = GRID, n_special: int = N_SPECIAL) -> torch.Tensor:
    """``(P, 2)`` long (y, x) positions of one frame exactly as ``Aggregator.forward`` builds them:
    ``cartesian_prod(arange(grid), arange(grid)) + 1`` for patches, ``0`` for the special tokens."""
    y = torch.arange(grid)
    x = torch.arange(grid)
    pos = torch.cartesian_prod(y, x) + 1
    special = torch.zeros(n_special, 2, dtype=pos.dtype)
    return torch.cat([special, pos], dim=0)


def rope_angles_merged(pos: torch.Tensor, head_dim: int = HEAD_DIM, base: float = ROPE_BASE) -> torch.Tensor:
    """``(N, head_dim)`` fp32 merged angle table ``[theta_y(q), theta_x(q), theta_y(q), theta_x(q)]``
    for positions ``pos (N, 2)`` = (y, x); ``theta = pos * inv_freq`` computed exactly like
    ``RotaryPositionEmbedding2D._compute_frequency_components`` (fp32 outer product)."""
    inv_freq = rope_inv_freq(head_dim // 2, base)                     # (q,)
    theta_y = pos[:, 0].to(inv_freq.dtype)[:, None] * inv_freq[None, :]
    theta_x = pos[:, 1].to(inv_freq.dtype)[:, None] * inv_freq[None, :]
    return torch.cat([theta_y, theta_x, theta_y, theta_x], dim=-1)     # (N, head_dim)


def rope_tables(S: int, head_dim: int = HEAD_DIM, base: float = ROPE_BASE,
                grid: int = GRID, n_special: int = N_SPECIAL) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` each ``(1, 1, S*P, head_dim)`` fp32 for the merged-layout attention over
    ``S`` frames (``S = 1`` is also the per-frame table: positions are identical in every
    frame, so the ``(S, H, P, Dh)`` frame attention reads the same ``(1, 1, P, Dh)`` table)."""
    pos = frame_positions(grid, n_special)                              # (P, 2)
    ang = rope_angles_merged(pos, head_dim, base)                       # (P, Dh)
    ang = ang.repeat(S, 1)                                              # (S*P, Dh)
    return ang.cos()[None, None], ang.sin()[None, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope_standard(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF-style RoPE on the full last dim (what ``ttnn.experimental.rotary_embedding`` computes).
    ``x (B, H, N, D)``, ``cos/sin (1, 1, N, D)``."""
    return x * cos + rotate_half(x) * sin


def permute_head_columns(w_t: torch.Tensor, num_heads: int, head_dim: int, perm: torch.Tensor,
                         sections: Sequence[int] = (0, 1)) -> torch.Tensor:
    """Permute the per-head columns of a transposed fused-qkv weight ``w_t (in, 3*H*Dh)`` (or a
    bias ``(3*H*Dh,)``) for the sections listed (0 = Q, 1 = K, 2 = V).  Column
    ``sec*H*Dh + h*Dh + j`` of the result is column ``sec*H*Dh + h*Dh + perm[j]`` of the input."""
    C = num_heads * head_dim
    idx = torch.arange(3 * C)
    for sec in sections:
        for h in range(num_heads):
            base = sec * C + h * head_dim
            idx[base:base + head_dim] = base + perm
    return w_t[..., idx].clone()


def fold_qkv(weight_t: torch.Tensor, bias: Optional[torch.Tensor], num_heads: int, head_dim: int,
             *, fold_scale: bool, perm: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fold B2/B3 into a transposed qkv weight ``(in, 3*C)`` and bias ``(3*C,)``:
    ``fold_scale`` multiplies the Q columns by ``1/sqrt(head_dim)`` (only call with
    ``scale_fold_is_exact(head_dim)``), ``perm`` permutes the Q and K head columns.
    Returns new tensors (inputs untouched)."""
    C = num_heads * head_dim
    w = weight_t.clone()
    b = bias.clone() if bias is not None else None
    if fold_scale:
        s = attention_scale(head_dim)
        w[..., :C] = w[..., :C] * s
        if b is not None:
            b[..., :C] = b[..., :C] * s
    if perm is not None:
        w = permute_head_columns(w, num_heads, head_dim, perm, sections=(0, 1))
        if b is not None:
            b = permute_head_columns(b, num_heads, head_dim, perm, sections=(0, 1))
    return w, b


def fold_qk_norm(gamma: torch.Tensor, beta: torch.Tensor, perm: Optional[torch.Tensor],
                 scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k-norm affine ``(Dh,)`` after the column permutation and (for q) the scale fold.
    LayerNorm over the head dim is permutation-equivariant, so ``LN(x[perm]) * g[perm] + b[perm]
    == (LN(x) * g + b)[perm]``; multiplying g and b by a power of two commutes with every
    rounding step of the kernel."""
    g = gamma.clone()
    b = beta.clone()
    if perm is not None:
        g = g[perm]
        b = b[perm]
    if scale != 1.0:
        g = g * scale
        b = b * scale
    return g, b


# ----------------------------------------------------------------------------- token assembly

def special_token_table(camera_token: torch.Tensor, register_token: torch.Tensor, S: int) -> torch.Tensor:
    """``(S, 5, C)`` = ``cat([slice_expand_and_flatten(camera_token, 1, S),
    slice_expand_and_flatten(register_token, 1, S)], dim=1)`` for B = 1: frame 0 takes index 0 of
    the ``(1, 2, X, C)`` parameters, frames 1..S-1 take index 1."""
    cam = camera_token.detach()
    reg = register_token.detach()
    assert cam.shape[:2] == (1, 2) and reg.shape[:2] == (1, 2), (cam.shape, reg.shape)
    cams = torch.cat([cam[:, 0:1].expand(1, 1, *cam.shape[2:]), cam[:, 1:].expand(1, S - 1, *cam.shape[2:])], dim=1)
    regs = torch.cat([reg[:, 0:1].expand(1, 1, *reg.shape[2:]), reg[:, 1:].expand(1, S - 1, *reg.shape[2:])], dim=1)
    cams = cams.reshape(S, *cam.shape[2:])
    regs = regs.reshape(S, *reg.shape[2:])
    return torch.cat([cams, regs], dim=1).contiguous()


def assembly_row_mask(P: int = P_TOKENS, n_special: int = N_SPECIAL) -> torch.Tensor:
    """``(1, P, 1)`` fp32: 0 on the first ``n_special`` rows (DINOv2 cls + registers, dropped),
    1 on the patch rows."""
    m = torch.ones(1, P, 1, dtype=torch.float32)
    m[:, :n_special] = 0.0
    return m


def assembly_add_table(camera_token: torch.Tensor, register_token: torch.Tensor, S: int,
                       P: int = P_TOKENS) -> torch.Tensor:
    """``(S, P, C)`` fp32: the special-token constants in rows 0..4, zeros elsewhere, so that
    ``tokens = x_norm * assembly_row_mask() + assembly_add_table()`` equals
    ``cat([camera, register, x_norm[:, 5:]], dim=1)`` exactly (``x*1 == x``, ``x*0 + t == t``)."""
    spec = special_token_table(camera_token, register_token, S).to(torch.float32)
    C = spec.shape[-1]
    T = torch.zeros(S, P, C, dtype=torch.float32)
    T[:, :spec.shape[1]] = spec
    return T


def assemble_tokens_torch(x_norm: torch.Tensor, camera_token: torch.Tensor,
                          register_token: torch.Tensor) -> torch.Tensor:
    """Reference-shaped host implementation of the device assembly (for tests)."""
    S, P, _ = x_norm.shape
    return x_norm * assembly_row_mask(P) + assembly_add_table(camera_token, register_token, S, P)


def patch_gather_matrix(P: int = P_TOKENS, n_special: int = N_SPECIAL) -> torch.Tensor:
    """``(P - n_special, P)`` 0/1 row-selection matrix: ``G @ X == X[n_special:]``.  Exact as a
    bf16 HiFi4 matmul (one 1.0 term per output row; the rf-detr port verified this bit-exact on
    the p150a)."""
    n = P - n_special
    G = torch.zeros(n, P, dtype=torch.float32)
    G[torch.arange(n), torch.arange(n_special, P)] = 1.0
    return G


# ----------------------------------------------------------------------------- B6 bilinear

def interp_matrix(n_in: int, n_out: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``(n_out, n_in)`` weights of 1-D linear interpolation with ``align_corners=True``
    (``src = i * (n_in - 1) / (n_out - 1)``), i.e. ``F.interpolate(mode="bilinear",
    align_corners=True)`` separates into ``A_h @ X @ A_w^T``.  Rows sum to 1."""
    A = torch.zeros(n_out, n_in, dtype=dtype)
    if n_out == 1 or n_in == 1:
        A[:, 0] = 1.0
        return A
    scale = (n_in - 1) / (n_out - 1)
    for i in range(n_out):
        src = i * scale
        i0 = min(int(math.floor(src)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        w1 = src - i0
        A[i, i0] += 1.0 - w1
        A[i, i1] += w1
    return A


def interp_gather_tables(n_in: int, n_out: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``A = G0 * w0 + G1 * w1`` (rows): the ``align_corners=True`` interpolation matrix as two 0/1
    row-selection matrices ``G0, G1 (n_out, n_in)`` (the lower / upper source index of every output
    row, ``G1 == G0`` where the source hits the last row) and the fp32 weight columns ``w0, w1
    (n_out, 1)``.  ``G0/G1`` are bf16-exact, so ``Y = (G0 @ X) * w0 + (G1 @ X) * w1`` reproduces
    ``F.interpolate`` to fp32 rounding when the gathers are exact and the eltwise math is fp32
    (the device fp32 x fp32 matmul rounds its inputs tf32-class -- measured wp_conf cost 0.0017)."""
    G0 = torch.zeros(n_out, n_in, dtype=torch.float32)
    G1 = torch.zeros(n_out, n_in, dtype=torch.float32)
    w0 = torch.zeros(n_out, 1, dtype=torch.float32)
    w1 = torch.zeros(n_out, 1, dtype=torch.float32)
    if n_out == 1 or n_in == 1:
        G0[:, 0] = 1.0
        G1[:, 0] = 1.0
        w0[:] = 1.0
        return G0, G1, w0, w1
    scale = (n_in - 1) / (n_out - 1)
    for i in range(n_out):
        src = i * scale
        i0 = min(int(math.floor(src)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        lam = src - i0
        G0[i, i0] = 1.0
        G1[i, i1] = 1.0
        w0[i, 0] = 1.0 - lam
        w1[i, 0] = lam
    return G0, G1, w0, w1


def bilinear_via_matmul(x_nchw: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """Torch reference of the device formulation: ``Y = A_h @ X @ A_w^T`` per (n, c) plane."""
    H, W = x_nchw.shape[-2:]
    A_h = interp_matrix(H, out_hw[0], dtype=x_nchw.dtype)
    A_w = interp_matrix(W, out_hw[1], dtype=x_nchw.dtype)
    return torch.matmul(torch.matmul(A_h, x_nchw), A_w.t())


def interp_weights_bf16_exact(n_in: int, n_out: int) -> bool:
    A = interp_matrix(n_in, n_out, dtype=torch.float32)
    return bool(torch.equal(A.to(torch.bfloat16).to(torch.float32), A))


# ----------------------------------------------------------------------------- B7 DPT constants

def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    # == vggt.heads.utils.make_sincos_pos_embed (CPU: float64 omega, einsum, .float())
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def _position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    # == vggt.heads.utils.position_grid_to_embed
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def _create_uv_grid(width: int, height: int, aspect_ratio: Optional[float] = None,
                    dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    # == vggt.heads.utils.create_uv_grid (returns (height, width, 2) despite its docstring)
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def dpt_pos_embed_hwc(H: int, W: int, C: int, img_w: int = IMG_SIZE, img_h: int = IMG_SIZE,
                      ratio: float = 0.1) -> torch.Tensor:
    """``DPTHead._apply_pos_embed(x, W=img_w, H=img_h, ratio)`` for a feature map ``(_, C, H, W)``
    fp32, returned channels-last as ``(H, W, C)`` (row = y, col = x)."""
    grid = _create_uv_grid(W, H, aspect_ratio=img_w / img_h, dtype=torch.float32)  # (H, W, 2)
    return (_position_grid_to_embed(grid, C) * ratio).contiguous()


def dpt_pos_embed_tokens(C: int, grid: int = GRID, ratio: float = 0.1) -> torch.Tensor:
    """``(1, grid*grid, C)`` for the prelude add on the ``(Bs, N_patches, C)`` token layout
    (the same table the legacy ``_install_ttnn_dpt_prelude`` uploads)."""
    return dpt_pos_embed_hwc(grid, grid, C, ratio=ratio).reshape(1, grid * grid, C)


def dpt_pos_embed_rows(H: int, W: int, C: int, ratio: float = 0.1) -> torch.Tensor:
    """``(1, H, W*C)``: the ``(H, W, C)`` table with W and C merged, i.e. the layout of the
    upsample H-pass output ``(S, Ho, Wo*C)`` the device graph adds it to (batch-broadcast over S)."""
    return dpt_pos_embed_hwc(H, W, C, ratio=ratio).reshape(1, H, W * C)


# ----------------------------------------------------------------------------- op-count bookkeeping

def legacy_block_ops(qk_norm: bool, rope: bool, decomposed_softmax: bool = False) -> int:
    """Launch count of one shipped (legacy) Block, from reading ``tt_block_forward``."""
    n = 22
    if qk_norm:
        n += 4
    if rope:
        n += 17 + 19
    if decomposed_softmax:
        n += 7
    return n


def fused_block_ops(qk_norm: bool, rope: bool, decomposed_softmax: bool = False,
                    scale_folded: bool = True, attn: str = "matmul") -> int:
    """Launch count of one fused Block (B2/B3/B4 applied), from ``TtVggt._block``:
    typecast LN1 qkv heads [LNq LNk] [ropeq ropek] [k-permute] scores softmax ctx concat_heads
    typecast proj mul add typecast LN2 fc1 gelu fc2 mul add (views and deallocates not counted)."""
    n = 19 if attn == "matmul" else 17   # sdpa: no separate softmax, ctx already bf16 (no typecast)
    if not scale_folded and attn == "matmul":
        n += 1
    if qk_norm:
        n += 2
    if rope:
        n += 2
    if (rope or qk_norm) and attn == "matmul":
        n += 1  # k^T permute (k is needed un-transposed for LN / RoPE first)
    if decomposed_softmax and attn == "matmul":
        n += 7
    return n
