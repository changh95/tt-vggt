"""Host-only (torch, no device, no ttnn) proofs for the ``TT_FUSED`` (default) reformulations.

Every exact reformulation the device graph in ``ttnn_vggt_fused.py`` relies on is checked
here against the reference math -- the upstream ``vggt`` modules when they are importable
(``VGGT_REF`` or ``<tt-models>/ports/vggt-upstream``), otherwise the tests that need them
skip -- plus the constant tables and the knob plumbing.

Run (tree python has pytest 9):

    TREE=/home/deepgadget/experiments/gbp-tt/tt-metal
    cd <repo>/code && VGGT_REF=<tt-models>/ports/vggt-upstream \\
        $TREE/python_env/bin/python -m pytest models/tests/test_fused_host.py -q

or as a plain script: ``python models/tests/test_fused_host.py`` (runs every test with asserts).
"""
from __future__ import annotations

import math
import os
import sys

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)
_VGGT_REF = os.environ.get("VGGT_REF") or os.path.abspath(
    os.path.join(_CODE_ROOT, "..", "..", "..", "ports", "vggt-upstream"))
if os.path.isdir(os.path.join(_VGGT_REF, "vggt")) and _VGGT_REF not in sys.path:
    sys.path.insert(0, _VGGT_REF)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.demos.vggt.tt import fused_tables as ft  # noqa: E402
from models.demos.vggt.tt import ttnn_vggt  # noqa: E402  (imports torch only at module level)

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - plain-script mode
    pytest = None

try:
    import vggt  # type: ignore  # noqa: F401
    HAVE_VGGT = True
except Exception:
    HAVE_VGGT = False


def _skip_without_vggt():
    if not HAVE_VGGT:
        if pytest is not None:
            pytest.skip("upstream vggt package not importable (set VGGT_REF)")
        raise RuntimeError("upstream vggt package not importable (set VGGT_REF)")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


# =============================================================================== knob plumbing

def test_knob_default_off_selects_legacy():
    assert ft.fused_enabled({}) is True                      # default ON (device-validated 2026-09-13)
    assert ft.fused_enabled({"TT_FUSED": "0"}) is False
    assert ft.fused_enabled({"TT_FUSED": "false"}) is False
    assert ft.fused_enabled({"TT_FUSED": ""}) is True        # empty == unset
    assert ft.fused_enabled({"TT_FUSED": "1"}) is True
    assert ft.fused_enabled({"TT_FUSED": "true"}) is True
    assert ft.trace_region_bytes({"TT_FUSED": "0"}) == 0
    assert ft.l1_small_bytes({"TT_FUSED": "0"}) == 32 * 1024 and ft.l1_small_bytes({}) == 64 * 1024
    assert ft.trace_region_bytes({"TT_FUSED": "1"}) == 1536 * 1024 * 1024
    assert ft.trace_region_bytes({"TT_FUSED": "1", "VGGT_TRACE_REGION_MB": "64"}) == 64 * 1024 * 1024
    # module-level plumbing: nothing decided yet -> follows the environment; wrapper absent
    env_backup = os.environ.pop("TT_FUSED", None)
    try:
        import models.demos.vggt.tt.ttnn_vggt as ttnn_vggt
        ttnn_vggt._FUSED_ACTIVE = None
        assert ttnn_vggt.fused_enabled() is True             # unset -> fused (default)
        assert ttnn_vggt.trace_region_bytes() > 0
        os.environ["TT_FUSED"] = "0"
        assert ttnn_vggt.fused_enabled() is False            # the knob restores legacy
        assert ttnn_vggt.trace_region_bytes() == 0 and ttnn_vggt.l1_small_bytes() == 32 * 1024
    finally:
        os.environ.pop("TT_FUSED", None)
        if env_backup is not None:
            os.environ["TT_FUSED"] = env_backup
    # the legacy surfaces are still there and untouched in name
    for name in ("_ensure_installed", "vggt_forward", "release_device_state", "_install_ttnn_block",
                 "_install_ttnn_dpt_scratch", "_install_ttnn_dpt_output_conv2", "_get_model"):
        assert callable(getattr(ttnn_vggt, name)), name


def test_fused_config_parsing():
    from models.demos.vggt.tt.ttnn_vggt_fused import FusedConfig
    cfg = FusedConfig(env={})
    assert cfg.as_dict() == {
        "trace": "multi", "attn": "matmul", "matmul": "linear", "softmax": "legacy", "camera": "device",
        "input": "rowmajor", "gather": "matmul", "interp": "exact", "ln32": "kernel", "prelude": "bf16", "reshape": "rm", "oc2_fp32": False, "mm_blocks": [8, 4, 4, 2, 2],
        "sdpa_chunks": [128, 256], "large_softmax_n": 4000,
    }
    cfg = FusedConfig(env={"VGGT_FUSED_ATTN": "SDPA", "VGGT_FUSED_TRACE": "off", "VGGT_FUSED_MM_BLOCKS": "4,4,4,2,2"})
    assert cfg.attn == "sdpa" and cfg.trace_mode == "off" and cfg.mm_blocks == (4, 4, 4, 2, 2)
    try:
        FusedConfig(env={"VGGT_FUSED_ATTN": "flash"})
    except RuntimeError as e:
        assert "VGGT_FUSED_ATTN" in str(e)
    else:
        raise AssertionError("invalid knob value must raise")


# =============================================================================== B2 scale fold

def test_scale_fold_exactness_rule():
    assert ft.scale_fold_is_exact(64) is True      # 1/8
    assert ft.scale_fold_is_exact(16) is True      # 1/4
    assert ft.scale_fold_is_exact(128) is False    # 2^-3.5: camera trunk keeps its multiply
    assert ft.scale_fold_is_exact(96) is False
    assert ft.attention_scale(64) == 0.125


def test_scale_fold_into_qkv_is_bit_identical_fp32_and_bf16():
    torch.manual_seed(0)
    H, Dh, C, N = 16, 64, 1024, 96
    lin = nn.Linear(C, 3 * C)
    x = torch.randn(N, C)
    w_t, b = ft.fold_qkv(lin.weight.detach().t().contiguous(), lin.bias.detach(), H, Dh, fold_scale=True, perm=None)
    # NB: torch's addmm (F.linear) and `x @ W + b` round differently on CPU, so both sides must
    # use the same primitive -- on device both paths are the same ttnn.linear kernel.
    ref = F.linear(x, lin.weight.detach(), lin.bias.detach())
    q_ref, k_ref = ref[:, :C], ref[:, C:2 * C]
    out = F.linear(x, w_t.t().contiguous(), b)
    q_new, k_new = out[:, :C], out[:, C:2 * C]
    assert torch.equal(k_new, k_ref)                       # K, V columns untouched
    assert torch.equal(out[:, 2 * C:], ref[:, 2 * C:])
    assert torch.equal(q_new, q_ref * 0.125)               # fp32 power-of-two scaling is exact
    s_ref = (q_ref @ k_ref.t()) * 0.125                    # legacy: matmul then multiply
    s_new = q_new @ k_new.t()                              # fused: no multiply
    assert torch.equal(s_new, s_ref)
    # bf16 weights (what the device holds): scaling the weights by 1/8 is an exponent shift
    assert torch.equal(w_t[:, :C].to(torch.bfloat16), (lin.weight.detach().t()[:, :C]).to(torch.bfloat16) * 0.125)
    assert torch.equal(b[:C].to(torch.bfloat16), lin.bias.detach()[:C].to(torch.bfloat16) * 0.125)


def test_scale_fold_into_qk_norm_affine_is_bit_identical_bf16():
    torch.manual_seed(1)
    Dh = 64
    ln = nn.LayerNorm(Dh)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(Dh))
        ln.bias.copy_(torch.randn(Dh))
    q = torch.randn(2, 16, 64, Dh).to(torch.bfloat16)
    g, b = ft.fold_qk_norm(ln.weight.detach(), ln.bias.detach(), None, scale=0.125)
    normed = F.layer_norm(q.float(), (Dh,), eps=ln.eps)
    ref = (normed * ln.weight.detach().to(torch.bfloat16).float() + ln.bias.detach().to(torch.bfloat16).float()).to(torch.bfloat16)
    new = (normed * g.to(torch.bfloat16).float() + b.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert torch.equal(new.float(), ref.float() * 0.125)


# =============================================================================== B3 RoPE

def test_rope_perm_is_a_permutation_and_matches_layout():
    perm = ft.rope_perm(64)
    assert perm.tolist() == list(range(0, 16)) + list(range(32, 48)) + list(range(16, 32)) + list(range(48, 64))
    assert sorted(perm.tolist()) == list(range(64))


def test_rope_tables_shapes_and_special_rows():
    for S in (1, 2, 3, 4):
        cos, sin = ft.rope_tables(S)
        assert cos.shape == (1, 1, S * ft.P_TOKENS, 64) and sin.shape == cos.shape
        assert cos.dtype == torch.float32
        # frame tile: every frame reads the same positions
        assert torch.equal(cos[0, 0, : ft.P_TOKENS], cos[0, 0, -ft.P_TOKENS:])
        # special tokens (pos 0): cos 1, sin 0
        assert torch.equal(cos[0, 0, :5], torch.ones(5, 64)) and torch.equal(sin[0, 0, :5], torch.zeros(5, 64))
        ang = ft.rope_angles_merged(ft.frame_positions())
        assert torch.equal(ang[:, :16], ang[:, 32:48]) and torch.equal(ang[:, 16:32], ang[:, 48:64])
    # padded row count the device sees
    assert [ft.pad_to_tile(S * ft.P_TOKENS) for S in (1, 2, 3, 4)] == [1376, 2752, 4128, 5504]


def test_rope_permuted_standard_equals_vggt_2d_rope_bit_exact():
    _skip_without_vggt()
    from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter  # type: ignore
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding2D(frequency=ft.ROPE_BASE)
    perm = ft.rope_perm()
    for S in (1, 3):
        pos = PositionGetter()(S, ft.GRID, ft.GRID, torch.device("cpu")) + 1
        pos = torch.cat([torch.zeros(S, ft.N_SPECIAL, 2, dtype=pos.dtype), pos], dim=1)
        assert torch.equal(pos[0], ft.frame_positions())
        # frame attention: (S, H, P, 64) with the (1, 1, P, 64) table
        q = torch.randn(S, 16, ft.P_TOKENS, 64)
        ref = rope(q, pos)
        cos, sin = ft.rope_tables(1)
        got = ft.apply_rope_standard(q[..., perm], cos, sin)
        assert torch.equal(got, ref[..., perm])
        # global attention: (1, H, S*P, 64) with the (1, 1, S*P, 64) table
        qg = q.permute(1, 0, 2, 3).reshape(1, 16, S * ft.P_TOKENS, 64)
        refg = rope(qg, pos.reshape(1, S * ft.P_TOKENS, 2))
        cosS, sinS = ft.rope_tables(S)
        gotg = ft.apply_rope_standard(qg[..., perm], cosS, sinS)
        assert torch.equal(gotg, refg[..., perm])


def test_rope_permuted_scores_are_invariant_fp32_reassoc():
    _skip_without_vggt()
    from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter  # type: ignore
    torch.manual_seed(0)
    rope = RotaryPositionEmbedding2D(frequency=ft.ROPE_BASE)
    perm = ft.rope_perm()
    pos = PositionGetter()(1, ft.GRID, ft.GRID, torch.device("cpu")) + 1
    pos = torch.cat([torch.zeros(1, ft.N_SPECIAL, 2, dtype=pos.dtype), pos], dim=1)
    q = torch.randn(1, 16, ft.P_TOKENS, 64)
    k = torch.randn(1, 16, ft.P_TOKENS, 64)
    s_ref = rope(q, pos) @ rope(k, pos).transpose(-1, -2)
    cos, sin = ft.rope_tables(1)
    s_new = ft.apply_rope_standard(q[..., perm], cos, sin) @ ft.apply_rope_standard(k[..., perm], cos, sin).transpose(-1, -2)
    d = (s_ref - s_new).abs().max().item()
    assert d <= 1e-4, d                     # summation order only (|s| up to ~50)
    assert _pcc(s_ref, s_new) > 0.999999


def test_fused_aggregator_attention_matches_reference_module():
    """Whole attention branch in the fused formulation (folded qkv columns, folded q/k-norm affine,
    one standard RoPE per q and k, no score multiply, heads merged) vs the upstream ``Attention``."""
    _skip_without_vggt()
    from vggt.layers.attention import Attention  # type: ignore
    from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter  # type: ignore
    torch.manual_seed(0)
    C, H, Dh, S = 256, 4, 64, 2
    rope = RotaryPositionEmbedding2D(frequency=ft.ROPE_BASE)
    attn = Attention(C, num_heads=H, qkv_bias=True, proj_bias=True, qk_norm=True, rope=rope).eval()
    with torch.no_grad():
        attn.q_norm.weight.copy_(torch.rand(Dh) + 0.5); attn.q_norm.bias.copy_(torch.randn(Dh) * 0.1)
        attn.k_norm.weight.copy_(torch.rand(Dh) + 0.5); attn.k_norm.bias.copy_(torch.randn(Dh) * 0.1)
    grid = 6
    P = grid * grid + ft.N_SPECIAL
    pos = PositionGetter()(S, grid, grid, torch.device("cpu")) + 1
    pos = torch.cat([torch.zeros(S, ft.N_SPECIAL, 2, dtype=pos.dtype), pos], dim=1)
    x = torch.randn(S, P, C)
    with torch.no_grad():
        ref = attn(x, pos=pos)
        # fused formulation
        perm = ft.rope_perm(Dh)
        w_t, b = ft.fold_qkv(attn.qkv.weight.t().contiguous(), attn.qkv.bias, H, Dh, fold_scale=False, perm=perm)
        qg, qb = ft.fold_qk_norm(attn.q_norm.weight, attn.q_norm.bias, perm, scale=ft.attention_scale(Dh))
        kg, kb = ft.fold_qk_norm(attn.k_norm.weight, attn.k_norm.bias, perm)
        qkv = x @ w_t + b
        q, k, v = qkv.reshape(S, P, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q = F.layer_norm(q, (Dh,), qg, qb, attn.q_norm.eps)
        k = F.layer_norm(k, (Dh,), kg, kb, attn.k_norm.eps)
        pos1 = pos[0]                                             # identical in every frame
        ang = ft.rope_angles_merged(pos1, Dh)
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        q = ft.apply_rope_standard(q, cos, sin)
        k = ft.apply_rope_standard(k, cos, sin)
        scores = q @ k.transpose(-1, -2)                          # scale already in q
        ctx = scores.softmax(-1) @ v
        out = attn.proj(ctx.transpose(1, 2).reshape(S, P, C))
    d = (out - ref).abs().max().item()
    assert d <= 2e-5 * max(1.0, ref.abs().max().item()), d
    assert _pcc(out, ref) > 0.9999999


def test_fused_dinov2_attention_scores_bit_identical():
    """No qk-norm / no RoPE block (DINOv2): the 1/8 fold gives bit-identical fp32 scores."""
    _skip_without_vggt()
    from vggt.layers.attention import Attention  # type: ignore
    torch.manual_seed(0)
    C, H, Dh = 256, 4, 64
    attn = Attention(C, num_heads=H, qkv_bias=True, qk_norm=False, rope=None).eval()
    x = torch.randn(1, 40, C)
    with torch.no_grad():
        qkv = attn.qkv(x).reshape(1, 40, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        s_ref = (q * attn.scale) @ k.transpose(-1, -2)                     # reference: q * scale first
        s_legacy = (q @ k.transpose(-1, -2)) * attn.scale                  # legacy port: multiply after
        w_t, b = ft.fold_qkv(attn.qkv.weight.t().contiguous(), attn.qkv.bias, H, Dh, fold_scale=True, perm=None)
        qkv2 = F.linear(x, w_t.t().contiguous(), b).reshape(1, 40, 3, H, Dh).permute(2, 0, 3, 1, 4)  # same primitive as attn.qkv
        q2, k2, _ = qkv2.unbind(0)
        s_new = q2 @ k2.transpose(-1, -2)
    assert torch.equal(s_new, s_ref)
    assert torch.equal(s_new, s_legacy)


# =============================================================================== token assembly

def test_token_assembly_bit_equal_to_reference_cat():
    _skip_without_vggt()
    from vggt.models.aggregator import slice_expand_and_flatten  # type: ignore
    torch.manual_seed(0)
    C = 64
    camera_token = torch.randn(1, 2, 1, C)
    register_token = torch.randn(1, 2, 4, C)
    for S in (1, 2, 3, 4):
        x_norm = torch.randn(S, ft.P_TOKENS, C)            # DINOv2 output after the final LN (rows 0..4 = cls/reg)
        patches = x_norm[:, ft.N_SPECIAL:]
        ref = torch.cat([slice_expand_and_flatten(camera_token, 1, S), slice_expand_and_flatten(register_token, 1, S), patches], dim=1)
        got = ft.assemble_tokens_torch(x_norm, camera_token, register_token)
        assert torch.equal(got, ref)
        spec = ft.special_token_table(camera_token, register_token, S)
        assert spec.shape == (S, 5, C) and torch.equal(spec, ref[:, :5])
        T = ft.assembly_add_table(camera_token, register_token, S)
        assert T.shape == (S, ft.P_TOKENS, C) and torch.equal(T[:, 5:], torch.zeros(S, ft.N_PATCHES, C))
    m = ft.assembly_row_mask()
    assert m.shape == (1, ft.P_TOKENS, 1) and m.sum().item() == ft.N_PATCHES
    G = ft.patch_gather_matrix()
    assert G.shape == (ft.N_PATCHES, ft.P_TOKENS)
    y = torch.randn(2, ft.P_TOKENS, 16).to(torch.bfloat16).float()
    assert torch.equal(G @ y, y[:, ft.N_SPECIAL:])       # exact row gather (one 1.0 per row)
    assert torch.equal((G.to(torch.bfloat16).float() @ y), y[:, ft.N_SPECIAL:])


# =============================================================================== B6 bilinear

def test_interp_matrix_matches_f_interpolate_align_corners_true():
    torch.manual_seed(0)
    for n_in, n_out in ft.DPT_UPSAMPLES:
        A = ft.interp_matrix(n_in, n_out)
        assert A.shape == (n_out, n_in)
        assert torch.allclose(A.sum(1), torch.ones(n_out, dtype=A.dtype))
        assert torch.equal(A[0], F.one_hot(torch.tensor(0), n_in).double())
        assert torch.equal(A[-1], F.one_hot(torch.tensor(n_in - 1), n_in).double())
        x = torch.randn(1, 3, n_in, n_in)
        ref64 = F.interpolate(x.double(), size=(n_out, n_out), mode="bilinear", align_corners=True)
        got64 = ft.bilinear_via_matmul(x.double(), (n_out, n_out))
        assert (ref64 - got64).abs().max().item() <= 1e-12
        ref32 = F.interpolate(x, size=(n_out, n_out), mode="bilinear", align_corners=True)
        got32 = ft.bilinear_via_matmul(x, (n_out, n_out))
        assert (ref32 - got32).abs().max().item() <= 1e-4          # fp32 rounding of the weights only
    assert ft.interp_weights_bf16_exact(19, 37) is True            # only this one is bf16-exact ...
    for n_in, n_out in ft.DPT_UPSAMPLES[1:]:
        assert ft.interp_weights_bf16_exact(n_in, n_out) is False  # ... so the tables must be fp32


def test_align_corners_false_is_not_a_substitute():
    """Documents the root cause of the port's 'device bilinear drops conf PCC': the ttnn kernel
    is half-pixel (align_corners=False)."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 19, 19)
    t = F.interpolate(x, size=(38, 38), mode="bilinear", align_corners=True)
    f = F.interpolate(x, size=(38, 38), mode="bilinear", align_corners=False)
    assert _pcc(t, f) < 0.96
    assert _pcc(t, F.interpolate(x.to(torch.bfloat16).float(), size=(38, 38), mode="bilinear", align_corners=True)) > 0.9999


# =============================================================================== B7 DPT constants + chain

def test_dpt_pos_embed_tables_match_upstream():
    _skip_without_vggt()
    from vggt.heads.utils import create_uv_grid, position_grid_to_embed  # type: ignore
    for C in (256, 512, 1024):
        pe = position_grid_to_embed(create_uv_grid(ft.GRID, ft.GRID, aspect_ratio=1.0, dtype=torch.float32), C) * 0.1
        assert torch.equal(ft.dpt_pos_embed_tokens(C), pe.reshape(1, ft.N_PATCHES, C))
    pe = position_grid_to_embed(create_uv_grid(ft.IMG_SIZE, ft.IMG_SIZE, aspect_ratio=1.0, dtype=torch.float32), 128) * 0.1
    assert torch.equal(ft.dpt_pos_embed_rows(ft.IMG_SIZE, ft.IMG_SIZE, 128), pe.reshape(1, ft.IMG_SIZE, ft.IMG_SIZE * 128))
    # and against the head's own _apply_pos_embed on a zero map (NCHW -> our channels-last rows)
    from vggt.heads.dpt_head import DPTHead  # type: ignore
    head = DPTHead(dim_in=32, output_dim=2, activation="exp", conf_activation="expp1").eval()
    z = torch.zeros(1, 128, 40, 40)
    ref = head._apply_pos_embed(z, 518, 518)[0].permute(1, 2, 0)      # (H, W, C)
    assert torch.equal(ft.dpt_pos_embed_hwc(40, 40, 128), ref)


def _dpt_fused_torch(head, tokens_list, S: int, patch_h: int, img: int):
    """Torch mirror of ``TtVggt._dpt_head``: the device graph's op order and semantics
    (row gather after the 1x1 proj, in-place-ReLU residual, matmul upsampling, pos-embed add in
    the (S, Ho, Wo*C) layout, ReLU fused into conv1 / output_conv2[0])."""
    outs = []
    for dpt_idx, layer_idx in enumerate(head.intermediate_layer_idx):
        x = tokens_list[layer_idx].reshape(S, -1, tokens_list[layer_idx].shape[-1])   # (S, P, 2C) incl. specials
        x = head.norm(x)
        proj = head.projects[dpt_idx]
        y = x @ proj.weight.reshape(proj.out_channels, -1).t() + proj.bias                # 1x1 conv as linear
        G = ft.patch_gather_matrix(x.shape[1], ft.N_SPECIAL)
        y = G @ y                                                                          # (S, N, C)
        C = y.shape[-1]
        if head.pos_embed:
            y = y + ft.dpt_pos_embed_tokens(C, patch_h)
        fmap = y.reshape(S, patch_h, patch_h, C).permute(0, 3, 1, 2)                       # NCHW
        outs.append(head.resize_layers[dpt_idx](fmap))

    def resconv(unit, x):
        r = F.relu(x)
        c1 = F.relu(unit.conv1(r))
        return unit.conv2(c1) + r

    def up(x, size):
        return ft.bilinear_via_matmul(x, size)

    l1, l2, l3, l4 = [getattr(head.scratch, f"layer{i}_rn")(o) for i, o in enumerate(outs, 1)]
    sizes = {4: l3.shape[-2:], 3: l2.shape[-2:], 2: l1.shape[-2:], 1: (2 * l1.shape[-2], 2 * l1.shape[-1])}
    skips = {4: None, 3: l3, 2: l2, 1: l1}
    prev = l4
    for r in (4, 3, 2, 1):
        ref = getattr(head.scratch, f"refinenet{r}")
        if ref.has_residual:
            prev = prev + resconv(ref.resConfUnit1, skips[r])
        out = resconv(ref.resConfUnit2, prev)
        prev = ref.out_conv(up(out, sizes[r]))
    oc1 = head.scratch.output_conv1(prev)
    full = up(oc1, (img, img))
    if head.pos_embed:
        full = full + ft.dpt_pos_embed_hwc(img, img, full.shape[1], img_w=img, img_h=img).permute(2, 0, 1)[None]
    a = F.relu(head.scratch.output_conv2[0](full))
    return head.scratch.output_conv2[2](a)


def test_dpt_chain_reformulation_matches_reference_head():
    """Random-weight DPTHead at a reduced geometry (4x4 patches, 56x56 image) -- the chain is
    shape-agnostic, so the same code path as 37x37 / 518x518 is exercised."""
    _skip_without_vggt()
    from vggt.heads.dpt_head import DPTHead  # type: ignore
    from vggt.heads.head_act import activate_head  # type: ignore
    torch.manual_seed(0)
    dim_in, S, patch_h, img = 32, 2, 4, 56
    head = DPTHead(dim_in=dim_in, output_dim=4, activation="inv_log", conf_activation="expp1").eval()
    with torch.no_grad():
        for p in head.parameters():          # keep values O(1) so the activations do not blow up
            p.mul_(0.2)
    P = patch_h * patch_h + ft.N_SPECIAL
    tokens = [torch.randn(1, S, P, dim_in) for _ in range(24)]
    images = torch.zeros(1, S, 3, img, img)
    with torch.no_grad():
        ref_pred, ref_conf = head._forward_impl(tokens, images, ft.N_SPECIAL)
        raw = _dpt_fused_torch(head, [t.clone() for t in tokens], S, patch_h, img)
        pred, conf = activate_head(raw, activation=head.activation, conf_activation=head.conf_activation)
        pred, conf = pred.view(1, S, *pred.shape[1:]), conf.view(1, S, *conf.shape[1:])
    for a, b in ((pred, ref_pred), (conf, ref_conf)):
        d = (a - b).abs().max().item()
        assert d <= 1e-4 * max(1.0, b.abs().max().item()), d
        assert _pcc(a, b) > 0.999999


# =============================================================================== camera head glue

def _camera_fused_torch(ch, pose_tokens, iters: int):
    """Torch mirror of ``TtVggt._camera_device``: embed_pose on a 32-wide zero-padded pose vector,
    pose_branch.fc2 zero-padded to 32 outputs, modulate expanded as add/mul ops."""
    B, S, Cc = pose_tokens.shape
    target, pad = ch.target_dim, ft.pad_to_tile(ch.target_dim)
    ep_w = torch.zeros(pad, Cc); ep_w[:target] = ch.embed_pose.weight.t()
    w2 = torch.zeros(ch.pose_branch.fc2.weight.shape[1], pad); w2[:, :target] = ch.pose_branch.fc2.weight.t()
    b2 = torch.zeros(pad); b2[:target] = ch.pose_branch.fc2.bias
    iter0 = ch.embed_pose(ch.empty_pose_tokens).reshape(1, 1, Cc).expand(1, S, Cc)
    mod_lin = ch.poseLN_modulation[1]
    pred, preds = None, []
    for _ in range(iters):
        mod_in = iter0 if pred is None else pred @ ep_w + ch.embed_pose.bias
        mod = F.silu(mod_in) @ mod_lin.weight.t() + mod_lin.bias
        shift, scale, gate = mod[..., :Cc], mod[..., Cc:2 * Cc], mod[..., 2 * Cc:]
        normed = F.layer_norm(pose_tokens, (Cc,), eps=ch.adaln_norm.eps)
        x = gate * (normed * (scale + 1.0) + shift) + pose_tokens
        x = ch.trunk(x)
        h = F.gelu(ch.pose_branch.fc1(ch.trunk_norm(x)))
        delta = h @ w2 + b2
        pred = delta if pred is None else pred + delta
        preds.append(pred)
    return [p[..., :target] for p in preds]


def test_camera_head_glue_reformulation_matches_reference():
    _skip_without_vggt()
    from vggt.heads.camera_head import CameraHead  # type: ignore
    from vggt.heads.head_act import activate_pose  # type: ignore
    torch.manual_seed(0)
    ch = CameraHead(dim_in=64, trunk_depth=2).eval()
    with torch.no_grad():
        ch.empty_pose_tokens.copy_(torch.randn(1, 1, 9) * 0.1)
    S = 3
    pose_tokens = torch.randn(1, S, 64)
    with torch.no_grad():
        ref = ch.trunk_fn(pose_tokens, 4)
        got = [activate_pose(p, trans_act=ch.trans_act, quat_act=ch.quat_act, fl_act=ch.fl_act)
               for p in _camera_fused_torch(ch, pose_tokens, 4)]
    assert len(got) == len(ref) == 4
    for a, b in zip(got, ref):
        assert a.shape == b.shape == (1, S, 9)
        assert (a - b).abs().max().item() <= 1e-4 * max(1.0, b.abs().max().item())


# =============================================================================== op-count arithmetic

def test_op_count_bookkeeping():
    # legacy (evaluation section 2): DINOv2 22, aggregator 62 (69 with the decomposed softmax)
    assert ft.legacy_block_ops(False, False) == 22
    assert ft.legacy_block_ops(True, True) == 62
    assert ft.legacy_block_ops(True, True, decomposed_softmax=True) == 69
    # fused (TtVggt._block): 19 / 24 / 31
    assert ft.fused_block_ops(False, False) == 19
    assert ft.fused_block_ops(True, True) == 24
    assert ft.fused_block_ops(True, True, decomposed_softmax=True) == 31
    assert ft.fused_block_ops(False, False, scale_folded=False) == 20          # camera trunk keeps the multiply
    assert ft.fused_block_ops(True, True, attn="sdpa") == 21
    # per forward (S=1): 24 DINOv2 + 48 aggregator + 16 camera-trunk block calls
    legacy = 24 * 22 + 48 * 62 + 16 * 22
    fused = 24 * 19 + 48 * 24 + 16 * 20
    assert legacy == 3856 and fused == 1928


if __name__ == "__main__":  # plain-script mode: run everything with asserts
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                if "not importable" in str(e):
                    print(f"SKIP {name}: {e}")
                else:
                    failures += 1
                    print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)


def test_interp_gather_tables_reproduce_the_interpolation_matrix():
    """A == G0 * w0 + G1 * w1 (fp32), G0/G1 are 0/1 with one 1 per row (bf16-exact), and the
    gather formulation equals F.interpolate(align_corners=True) to fp32 rounding for every DPT size."""
    import torch.nn.functional as F
    for n_in, n_out in ft.DPT_UPSAMPLES:
        A = ft.interp_matrix(n_in, n_out, dtype=torch.float32)
        G0, G1, w0, w1 = ft.interp_gather_tables(n_in, n_out)
        assert torch.equal(G0.sum(1), torch.ones(n_out)) and torch.equal(G1.sum(1), torch.ones(n_out))
        assert torch.equal(G0.to(torch.bfloat16).float(), G0) and torch.equal(G1.to(torch.bfloat16).float(), G1)
        assert torch.allclose(G0 * w0 + G1 * w1, A, atol=1e-7)
        x = torch.randn(1, 3, n_in, n_in)
        ref = F.interpolate(x, size=(n_out, n_out), mode="bilinear", align_corners=True)
        # W-pass then H-pass with the gathers (X (H, W, C) row gathers along W, then along H)
        xc = x[0].permute(1, 2, 0)                                        # (H, W, C)
        yw = (G0 @ xc) * w0 + (G1 @ xc) * w1                              # (H, Wo, C)
        yh = torch.einsum("oh,hwc->owc", G0, yw) * w0[:, :, None] + torch.einsum("oh,hwc->owc", G1, yw) * w1[:, :, None]
        assert torch.allclose(yh.permute(2, 0, 1)[None], ref, atol=1e-4, rtol=1e-5), (n_in, n_out, (yh.permute(2, 0, 1)[None] - ref).abs().max())
