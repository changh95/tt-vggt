"""Device-resident VGGT-1B for one Blackhole p150a -- the ``TT_FUSED=1`` path.

Selected by ``models.demos.vggt.tt.ttnn_vggt`` when ``TT_FUSED=1`` is set at install time
(``_ensure_installed``); with the knob off the legacy class-level monkey patches in
``ttnn_vggt.py`` run unchanged.  Nothing in this module runs unless the knob is on.

Pipeline (per forward, B = 1, S = 1..4 views of 518x518; P = 1374 tokens per frame):

  host   ImageNet-normalise -> DINOv2 patch conv + cls + pos_embed + 4 registers (fp32 torch,
         the upstream ``prepare_tokens_with_masks``) -> ``(S, P, 1024)`` fp32 ROW_MAJOR upload
         into the persistent trace input (``copy_host_to_device_tensor``)
  device tilize -> 24 DINOv2 blocks -> final LN (fp32) -> token assembly (row mask x + const
         table, exact) -> 24 x (frame block (S, P, C) | reshape | global block (1, S*P, C) |
         reshape) with the 4 consumed (frame_i || global_i) concats {4, 11, 17, 23} on device
         -> camera head (4 refinement iterations, in-graph, or read back for the host)
         -> 2 DPT heads, one frame at a time (S=1 shapes; the batched 518^2 conv needs DRAM
            op-slicing at S >= 2, which is not trace-capturable): LN -> 1x1 proj -> +pos_embed ->
            resize (convT/conv) -> layer_rn -> refinenets with bilinear-as-matmul upsampling ->
            output_conv1 -> 296->518 upsample -> +pos_embed -> output_conv2 (ReLU fused)
            -> raw ``(1, 1, 518*518, {2,4})`` per frame
  host   ``activate_head`` (exp / expm1 / sign, precision-critical), ``activate_pose``.

Every ttnn call between the upload and the readbacks is captured into ONE metal trace per S
at warm-up (eager warm run first, then ``begin/end_trace_capture``; the server captures all
``VGGT_PREWARM_SEQS`` before READY) and replayed with ``execute_trace``; readbacks are the two
raw head outputs (+ 4 tiny pose accumulators or the camera tokens).  Compared with the legacy
path this removes the 144 fp32 residual round trips (0.8-3.1 GB of PCIe traffic), the 26 DPT
mid-chain transfers, the 24 host concats, all host glue and the eager dispatch of ~4k ops.

Exact reformulations (all proven in ``models/tests/test_fused_host.py``, torch only):
  B2  1/sqrt(64) = 1/8 folded into the Q projection (DINOv2) or the q-norm affine
      (aggregator) -> the ``multiply(scores, 0.125)`` on the fp32 NxN tensor disappears
      (bit-identical: power-of-two scaling commutes with every rounding).  The camera trunk
      (Dh = 128, scale 2^-3.5) is NOT a power of two and keeps its multiply.
  B3  VGGT's 2-D RoPE (36 eltwise launches per block) -> one ``ttnn.experimental.rotary_embedding``
      per q and per k with a fixed head-column permutation folded into the qkv weights and the
      q/k-norm affine, and a merged ``[1, 1, N, 64]`` cos/sin table.  Bit-equal in torch; the
      device kernel rounds differently from the legacy 7-op bf16 chain (bf16-round level).
  B4  ``nlp_create_qkv_heads(transpose_k_heads=False)`` + one k permute + ``nlp_concat_heads``
      replace the 4 permutes + permute/reshape head merge (pure data movement).
  B5  ReLU fused into ``conv2d`` (``Conv2dConfig(activation=RELU)``) in the DPT chain.
  B6  ``F.interpolate(align_corners=True)`` as two fp32 constant matmuls ``A_h @ X @ A_w^T``
      (the ttnn bilinear kernel is half-pixel/align_corners=False -- that, not bf16, was the
      port's "device bilinear drops conf PCC to 0.955").
  B7  the whole DPT chain on device, zero mid-chain transfers, S > 1 included (the legacy
      ``Bs > 1`` host fallback is gone).

Knob-gated device A/Bs (default = the port's validated numerics, see ``FusedConfig``):
  VGGT_FUSED_ATTN=sdpa      B10 flash attention (bf16 probabilities) -- precision-gated
  VGGT_FUSED_SOFTMAX=...    B11 fused fp32 softmax variants for the N >= 4000 global blocks
  VGGT_FUSED_MATMUL=minimal B8  ``minimal_matmul`` kernels for qkv/fc1 (HiFi2) and proj/fc2 (fp32 out)
  VGGT_FUSED_CAMERA=host    camera head on the host from one (S, 1, 2048) readback (exact torch)
  VGGT_FUSED_TRACE=ondemand|off   one trace at a time / eager device graph (debug, PCC compare)

Precision profile is the port's: bf16 weights and matmul inputs, fp32 residual stream, fp32
scores + softmax + context, HiFi4 + fp32 accumulation on proj / fc2 / DPT convs.

Tile-padding rule (review fix): in this tree a tiled ``ttnn.reshape`` that changes the padded
row structure (``reshape_tiled``) leaves the implicit tile-padding lanes UNFILLED unless the
caller passes ``pad_value`` explicitly (``reshape.cpp`` "Fill rules": ``should_fill =
is_block_float || pad_value_explicit``); ``ttnn.slice`` documents its padding as undefined by
default.  The legacy path never saw this because every block re-uploaded its input with a
host ``from_torch`` (zero padding) -- here the padded rows stay resident across all 72 blocks
and a stale NaN/Inf would reach logical outputs through K-contractions (interpolation and
gather matmuls) and through k's padded rows -> k^T's padded columns -> ``probs @ v``.  Every
reshape therefore goes through ``_reshape`` (``pad_value=0.0``, one ``fill_pad`` launch when
the op is not a view) and the two slices that create fresh padded rows pass ``pad_value=0.0``
too, so padding lanes are deterministic zeros exactly as with the legacy host tilize.
"""
from __future__ import annotations

import gc
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from models.demos.vggt.tt import fused_tables as ft


def _log(msg: str) -> None:
    print(f"[vggt-fused] {msg}", flush=True)


# ============================================================================ configuration

class FusedConfig:
    """All ``TT_FUSED`` sub-knobs, read ONCE when the wrapper is built (never per forward)."""

    TRACE_MODES = ("multi", "ondemand", "off")
    ATTN = ("matmul", "sdpa")
    MATMUL = ("linear", "minimal")
    SOFTMAX = ("legacy", "inplace", "scale_mask")
    CAMERA = ("device", "host")
    INPUT = ("rowmajor", "tile")
    GATHER = ("matmul", "slice")
    INTERP = ("exact", "rep", "flat", "bcast")
    LN32 = ("kernel", "eltwise")
    RESHAPE = ("rm", "tiled")
    PRELUDE = ("bf16", "fp32")

    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.trace_mode = ft.fused_option("TRACE", "multi", self.TRACE_MODES, env)
        self.attn = ft.fused_option("ATTN", "matmul", self.ATTN, env)
        self.matmul = ft.fused_option("MATMUL", "linear", self.MATMUL, env)
        self.softmax = ft.fused_option("SOFTMAX", "legacy", self.SOFTMAX, env)
        self.camera = ft.fused_option("CAMERA", "device", self.CAMERA, env)
        self.input = ft.fused_option("INPUT", "rowmajor", self.INPUT, env)
        self.gather = ft.fused_option("GATHER", "matmul", self.GATHER, env)
        # B6 bilinear upsampling.  `exact` (default) = the two source rows of every output row
        # gathered with 0/1 bf16 matmuls of a bf16 hi/lo split of the fp32 map, then fp32 eltwise
        # `y0*w0 + y1*w1` -- reproduces the host fp32 F.interpolate; it is the only device
        # formulation that keeps the synthetic test_vggt.py PCC within 0.001 of legacy at every S
        # (0.9952 / 0.9981 / 0.9984 / 0.9979 vs legacy 0.9950 / 0.9991 / 0.9978 / 0.9975).
        # `rep` = the interpolation matrix as ONE fp32 x fp32 matmul per pass with A replicated over
        # the bmm batch: the FPU rounds fp32 inputs tf32-class (rel 1e-3) -> synthetic PCC 0.0015-0.0027
        # below legacy, but on the 7-scene real-image set within 0.0002 of exact and legacy, and 21 %
        # faster (372 vs 448 ms at S=1) -- opt-in.  `flat` = transposed 2-D matmuls (4 transposes,
        # 2.4x slower than rep).  `bcast` = A (1, Ho, H) broadcast over X's batch as first written --
        # HANGS the p150a on this tree (matmul in0_reuse path, 2026-09-13; `tt-smi -r 0`).
        self.interp = ft.fused_option("INTERP", "exact", self.INTERP, env)
        # fp32 LayerNorm of the DINOv2 output (feeds all 48 aggregator blocks): `kernel` =
        # ttnn.layer_norm on fp32 (measured bf16-class output: maxabs 2.9e-2 = one bf16 rounding,
        # probe_ln.log), `eltwise` = fp32 eltwise decomposition with hi/lo split sums (maxabs
        # 1.6e-3 with plain sums; the reduce rounds its inputs tf32-class, probe_ln2.log).
        self.ln32 = ft.fused_option("LN32", "kernel", self.LN32, env)
        # The two layout-switching reshapes of every upsample ((H,Wo,C) <-> (1,H,Wo*C) and
        # (1,Ho,Wo*C) -> (1,1,Ho*Wo,C)): `rm` = untilize -> free ROW_MAJOR view -> tilize (the
        # tilize zero-pads), `tiled` = `reshape_tiled` + `fill_pad` (6-7 ms each at 518^2).
        self.reshape = ft.fused_option("RESHAPE", "rm", self.RESHAPE, env)
        # DPT prelude (LN -> 1x1 proj -> +pos_embed -> resize conv): `bf16` = device bf16 as the
        # port's VGGT_TT_PRELUDE=1 path, `fp32` = fp32 LN (per LN32) + fp32 x fp32 1x1 matmul +
        # fp32 pos-embed, cast to bf16 only at the resize conv (legacy default ran it on the host in fp32).
        self.prelude = ft.fused_option("PRELUDE", "bf16", self.PRELUDE, env)
        self.oc2_fp32 = ft.fused_int("OC2_FP32", 0, env) != 0
        self.log_level = ft.fused_int("LOG", 1, env)     # 0 quiet, 1 warm-up/capture lines, 2 + per-stage sync'd timing (eager only)
        self.verbose = self.log_level != 0
        # fp32 ttnn.softmax hangs on Blackhole at N >= ~4100 (port's BF0); the decomposed path
        # (or the B11 knob variants) is used at and above this row count.
        self.large_softmax_n = ft.fused_int("LARGE_SOFTMAX_N", 4000, env)
        blocks = str(env.get("VGGT_FUSED_MM_BLOCKS", "8,4,4,2,2")).split(",")
        if len(blocks) != 5:
            raise RuntimeError("VGGT_FUSED_MM_BLOCKS must be 'M,K,N,sub_h,sub_w' in tiles")
        self.mm_blocks = tuple(int(b) for b in blocks)
        chunks = str(env.get("VGGT_FUSED_SDPA_CHUNKS", "128,256")).split(",")
        if len(chunks) != 2:
            raise RuntimeError("VGGT_FUSED_SDPA_CHUNKS must be 'q_chunk,k_chunk'")
        self.sdpa_chunks = (int(chunks[0]), int(chunks[1]))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace": self.trace_mode, "attn": self.attn, "matmul": self.matmul, "softmax": self.softmax,
            "camera": self.camera, "input": self.input, "gather": self.gather, "interp": self.interp,
            "ln32": self.ln32, "prelude": self.prelude, "reshape": self.reshape, "oc2_fp32": self.oc2_fp32,
            "mm_blocks": list(self.mm_blocks), "sdpa_chunks": list(self.sdpa_chunks),
            "large_softmax_n": self.large_softmax_n,
        }


# ============================================================================ the wrapper

class TtVggt:
    """Owns the device weights / constant tables / per-S traces of one VGGT-1B."""

    def __init__(self, ref_model, device, cfg: Optional[FusedConfig] = None):
        import ttnn

        self.ttnn = ttnn
        self.model = ref_model.eval()
        self.device = device
        self.cfg = cfg or FusedConfig()
        self.grid = device.compute_with_storage_grid_size()
        arch = device.arch()
        # HiFi4 + fp32 dest: proj / fc2 / attention matmuls / DPT convs / fp32 glue (the port's recipe).
        self.kcfg_hifi4 = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        # ttnn.linear's default for bf16 inputs is HiFi2 without fp32 accumulation; the
        # minimal_matmul A/B pins the same fidelity explicitly so the A/B is kernel-only.
        self.kcfg_hifi2 = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=False, packer_l1_acc=True)
        if self.cfg.matmul == "minimal":
            M, K, N, sh, sw = self.cfg.mm_blocks
            self.mm_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=M, K_block_size=K, N_block_size=N, subblock_h=sh, subblock_w=sw,
                compute_with_storage_grid_size=self.grid)
        if self.cfg.attn == "sdpa":
            qc, kc = self.cfg.sdpa_chunks
            self.sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid, q_chunk_size=qc, k_chunk_size=kc,
                exp_approx_mode=False)
        self.relu_cfg = ttnn.Conv2dConfig(activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))

        agg = self.model.aggregator
        self.P = ft.P_TOKENS
        self.C = agg.frame_blocks[0].norm1.weight.shape[0]
        assert self.P == (ft.IMG_SIZE // agg.patch_size) ** 2 + agg.patch_start_idx
        self.inter_idx = tuple(self.model.depth_head.intermediate_layer_idx)  # (4, 11, 17, 23)

        t0 = time.perf_counter()
        self._build_backbone()
        self._build_aggregator()
        self._build_camera()
        self._build_dpt()
        self._per_s: Dict[int, Dict[str, Any]] = {}
        self._traces: Dict[int, Dict[str, Any]] = {}
        self._capturing = False
        self._stage_t0 = 0.0
        if self.cfg.verbose:
            _log(f"device weights + tables uploaded in {time.perf_counter() - t0:.1f}s; knobs {self.cfg.as_dict()}")

    # ------------------------------------------------------------------ upload helpers
    def _bf16(self, t: torch.Tensor, shape=None):
        t = t.detach().to(torch.bfloat16)
        if shape is not None:
            t = t.reshape(shape)
        return self.ttnn.from_torch(t.contiguous(), dtype=self.ttnn.bfloat16, layout=self.ttnn.TILE_LAYOUT,
                                    device=self.device)

    def _f32(self, t: torch.Tensor, shape=None):
        t = t.detach().to(torch.float32)
        if shape is not None:
            t = t.reshape(shape)
        return self.ttnn.from_torch(t.contiguous(), dtype=self.ttnn.float32, layout=self.ttnn.TILE_LAYOUT,
                                    device=self.device)

    def _host_conv_w(self, w: torch.Tensor):
        # conv2d / conv_transpose2d weights stay on host in PyTorch layout; the op prepares them on
        # the first (eager, per-S) call and we keep the prepared device tensors (return_weights_and_bias).
        return self.ttnn.from_torch(w.detach().to(torch.bfloat16), dtype=self.ttnn.bfloat16)

    def _host_conv_b(self, b: Optional[torch.Tensor]):
        if b is None:
            return None
        return self.ttnn.from_torch(b.detach().reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=self.ttnn.bfloat16)

    @staticmethod
    def _free(*ts):
        import ttnn
        for t in ts:
            if t is not None:
                ttnn.deallocate(t)

    def _stage(self, name: str) -> None:
        """``VGGT_FUSED_LOG=2`` debugging aid: synchronize and print the elapsed time of the graph
        stage that just ended (eager runs only -- never inside a trace capture)."""
        if self.cfg.log_level < 2 or self._capturing:
            return
        self.ttnn.synchronize_device(self.device)
        now = time.perf_counter()
        _log(f"  stage {name}: {(now - self._stage_t0) * 1000:.1f} ms")
        self._stage_t0 = now

    def _reshape(self, x, shape):
        """``ttnn.reshape`` with the implicit tile padding of the result filled with zeros.

        ``reshape_tiled`` (taken whenever the second-last dim changes to a non-tile-multiple or
        the last dim changes) does NOT fill its padding lanes in this tree unless ``pad_value``
        is explicit; a view (same last dim, tile-aligned or unchanged rows, ROW_MAJOR) keeps the
        input buffer and ignores the argument, so this is safe -- and free -- on every site."""
        return self.ttnn.reshape(x, shape, pad_value=0.0)

    def _relayout(self, x, shape):
        """A reshape that changes the tile structure (last dim changes): with ``VGGT_FUSED_RESHAPE=rm``
        an untilize, a free ROW_MAJOR view and a tilize (zero padding) -- measured 0.4 ms per
        layout switch at 518^2 vs 6-7 ms for ``reshape_tiled`` + ``fill_pad``; ``tiled`` = ``_reshape``.
        Consumes nothing (the caller frees ``x``)."""
        if self.cfg.reshape == "tiled":
            return self._reshape(x, shape)
        ttnn = self.ttnn
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        v = ttnn.reshape(rm, shape)                       # ROW_MAJOR: always a view
        out = ttnn.to_layout(v, ttnn.TILE_LAYOUT)         # tilize with zero padding
        self._free(rm)
        return out

    def _reshape_owned(self, x, shape):
        """``_reshape`` that consumes ``x``: the source is released only when the op produced a
        new buffer (a view -- same shape, or a tile-aligned/RM view -- shares it)."""
        out = self._reshape(x, shape)
        if out.buffer_address() != x.buffer_address():
            self._free(x)
        return out

    def _unpad_rows(self, x, shape):
        """Restore the logical shape of an op whose output reports its tile-padded rows as
        logical.  ``rotary_embedding``'s ``compute_output_specs`` returns ``padded_shape`` with
        ``seq_len`` rounded up to 32 (``rotary_embedding_device_operation.cpp:99-105``), so a
        ``(B, H, 1374, 64)`` input comes back as ``(B, H, 1376, 64)`` and the ``probs @ v``
        matmul then rejects ``1376 != 1374`` (first device run, ``s3_eager_s1.log``).  The
        two-shape ``ttnn.reshape(x, logical, padded)`` with the tensor's own padded shape is a
        metadata-only view (same buffer address, no launch; ``probe_ops.log``)."""
        shape = tuple(int(s) for s in shape)
        if tuple(x.shape) == shape:
            return x
        return self.ttnn.reshape(x, self.ttnn.Shape(shape), self.ttnn.Shape(tuple(x.padded_shape)))

    def _ln32(self, x, g, b, eps: float):
        """fp32 LayerNorm over the last dim with fp32 affine.  ``kernel``: ``ttnn.layer_norm``
        (HiFi4, fp32 acc) -- its output is bf16-class on this device.  ``eltwise``: mean and
        variance from ``ttnn.sum`` of a bf16 hi / fp32 lo split (the reduce rounds its inputs
        tf32-class; the split keeps ~2^-18 relative), everything else fp32 eltwise (exact).
        Consumes nothing; the caller frees ``x``."""
        ttnn = self.ttnn
        if self.cfg.ln32 == "kernel":
            return ttnn.layer_norm(x, weight=g, bias=b, epsilon=eps, compute_kernel_config=self.kcfg_hifi4)
        shape = tuple(x.shape)
        C = shape[-1]
        col = shape[:-1] + (1,)

        def rowsum(t):
            hi = ttnn.typecast(t, ttnn.bfloat16)
            hi32 = ttnn.typecast(hi, ttnn.float32)
            lo = ttnn.subtract(t, hi32)
            self._free(hi32)
            s_hi = ttnn.sum(hi, dim=-1, compute_kernel_config=self.kcfg_hifi4)
            s_lo = ttnn.sum(lo, dim=-1, compute_kernel_config=self.kcfg_hifi4)
            self._free(hi, lo)
            s_hi32 = ttnn.typecast(s_hi, ttnn.float32)
            self._free(s_hi)
            tot = ttnn.add(s_hi32, s_lo)
            self._free(s_hi32, s_lo)
            return self._reshape_owned(tot, col)   # keepdim reduce -> usually already (.., N, 1)

        s1 = rowsum(x)
        mean = ttnn.multiply(s1, 1.0 / C)
        self._free(s1)
        xc = ttnn.subtract(x, mean)
        self._free(mean)
        sq = ttnn.multiply(xc, xc)
        s2 = rowsum(sq)
        self._free(sq)
        var = ttnn.multiply(s2, 1.0 / C)
        self._free(s2)
        ve = ttnn.add(var, eps)
        self._free(var)
        rstd = ttnn.rsqrt(ve)
        self._free(ve)
        xn = ttnn.multiply(xc, rstd)
        self._free(xc, rstd)
        y = ttnn.multiply(xn, g)
        self._free(xn)
        out = ttnn.add(y, b)
        self._free(y)
        return out

    # ------------------------------------------------------------------ weights: blocks
    def _block_params(self, blk, *, rope: bool) -> Dict[str, Any]:
        """Device weights of one upstream ``Block`` with B2/B3 folded in.

        ``rope`` blocks (aggregator) get the RoPE column permutation on the Q/K columns of qkv and
        on the q/k-norm affine, and the 1/8 scale on the q-norm affine.  Blocks without qk-norm
        (DINOv2, camera trunk) get the scale on the Q columns of qkv when it is a power of two.
        """
        import torch.nn as nn
        from vggt.layers.layer_scale import LayerScale  # type: ignore

        attn, mlp = blk.attn, blk.mlp
        H, Dh = attn.num_heads, attn.head_dim
        qk_norm = isinstance(attn.q_norm, nn.LayerNorm)
        exact = ft.scale_fold_is_exact(Dh)
        perm = ft.rope_perm(Dh) if rope else None
        # qkv: torch Linear weight (3C, C) -> transposed (C, 3C) so columns are output features.
        w_t = attn.qkv.weight.detach().t().contiguous().float()
        b = attn.qkv.bias.detach().float() if attn.qkv.bias is not None else None
        fold_in_qkv = exact and not qk_norm
        w_t, b = ft.fold_qkv(w_t, b, H, Dh, fold_scale=fold_in_qkv, perm=perm)
        p: Dict[str, Any] = {
            "heads": H, "head_dim": Dh, "qk_norm": qk_norm, "rope": rope,
            "scale_folded": exact, "scale": ft.attention_scale(Dh),
            "ln1_g": self._bf16(blk.norm1.weight, (1, 1, -1)), "ln1_b": self._bf16(blk.norm1.bias, (1, 1, -1)),
            "ln1_eps": float(blk.norm1.eps),
            "ln2_g": self._bf16(blk.norm2.weight, (1, 1, -1)), "ln2_b": self._bf16(blk.norm2.bias, (1, 1, -1)),
            "ln2_eps": float(blk.norm2.eps),
            "qkv_w": self._bf16(w_t), "qkv_b": self._bf16(b, (1, 1, -1)) if b is not None else None,
            "proj_w": self._bf16(attn.proj.weight.detach().t().contiguous()),
            "proj_b": self._bf16(attn.proj.bias, (1, 1, -1)) if attn.proj.bias is not None else None,
            "fc1_w": self._bf16(mlp.fc1.weight.detach().t().contiguous()), "fc1_b": self._bf16(mlp.fc1.bias, (1, 1, -1)),
            "fc2_w": self._bf16(mlp.fc2.weight.detach().t().contiguous()), "fc2_b": self._bf16(mlp.fc2.bias, (1, 1, -1)),
            "ls1": self._bf16(blk.ls1.gamma, (1, 1, -1)) if isinstance(blk.ls1, LayerScale) else None,
            "ls2": self._bf16(blk.ls2.gamma, (1, 1, -1)) if isinstance(blk.ls2, LayerScale) else None,
        }
        if qk_norm:
            qg, qb = ft.fold_qk_norm(attn.q_norm.weight.detach().float(), attn.q_norm.bias.detach().float(),
                                     perm, scale=ft.attention_scale(Dh) if exact else 1.0)
            kg, kb = ft.fold_qk_norm(attn.k_norm.weight.detach().float(), attn.k_norm.bias.detach().float(), perm)
            p.update({
                "qn_g": self._bf16(qg, (1, 1, 1, -1)), "qn_b": self._bf16(qb, (1, 1, 1, -1)),
                "kn_g": self._bf16(kg, (1, 1, 1, -1)), "kn_b": self._bf16(kb, (1, 1, 1, -1)),
                "qn_eps": float(attn.q_norm.eps),
            })
        return p

    def _build_backbone(self):
        pe = self.model.aggregator.patch_embed  # DinoVisionTransformer
        self.dino_blocks = [self._block_params(b, rope=False) for b in pe.blocks]
        self.dino_norm = {"g": self._f32(pe.norm.weight, (1, 1, -1)), "b": self._f32(pe.norm.bias, (1, 1, -1)),
                          "eps": float(pe.norm.eps)}

    def _build_aggregator(self):
        agg = self.model.aggregator
        assert list(agg.aa_order) == ["frame", "global"] and agg.aa_block_size == 1, agg.aa_order
        assert agg.rope is not None and agg.position_getter is not None
        self.rope_base = float(agg.rope.base_frequency)
        self.frame_blocks = [self._block_params(b, rope=True) for b in agg.frame_blocks]
        self.global_blocks = [self._block_params(b, rope=True) for b in agg.global_blocks]
        self.row_mask = self._f32(ft.assembly_row_mask(self.P, agg.patch_start_idx))
        self.camera_token = agg.camera_token.detach().float()
        self.register_token = agg.register_token.detach().float()

    def _build_camera(self):
        ch = self.model.camera_head
        Cc = ch.token_norm.weight.shape[0]
        self.cam_dim, self.cam_target = Cc, int(ch.target_dim)
        self.cam_pad = ft.pad_to_tile(self.cam_target)  # 9 -> 32: explicit zero columns, no reliance on tile padding
        self.cam_iters = 4
        self.cam_blocks = [self._block_params(b, rope=False) for b in ch.trunk]
        self.cam = {
            "token_norm_g": self._f32(ch.token_norm.weight, (1, 1, -1)),
            "token_norm_b": self._f32(ch.token_norm.bias, (1, 1, -1)), "token_norm_eps": float(ch.token_norm.eps),
            "trunk_norm_g": self._f32(ch.trunk_norm.weight, (1, 1, -1)),
            "trunk_norm_b": self._f32(ch.trunk_norm.bias, (1, 1, -1)), "trunk_norm_eps": float(ch.trunk_norm.eps),
            "adaln_eps": float(ch.adaln_norm.eps),
        }
        if self.cfg.camera == "device":
            ep_w = torch.zeros(self.cam_pad, Cc)
            ep_w[: self.cam_target] = ch.embed_pose.weight.detach().t().float()
            self.cam["embed_w"] = self._f32(ep_w)
            self.cam["embed_b"] = self._f32(ch.embed_pose.bias, (1, 1, -1))
            with torch.no_grad():
                self.cam_iter0_host = ch.embed_pose(ch.empty_pose_tokens).float().reshape(1, 1, Cc)  # exact host constant
            mod = ch.poseLN_modulation[1]
            self.cam["mod_w"] = self._f32(mod.weight.detach().t().contiguous())
            self.cam["mod_b"] = self._f32(mod.bias, (1, 1, -1))
            pb = ch.pose_branch
            self.cam["pb1_w"] = self._bf16(pb.fc1.weight.detach().t().contiguous())
            self.cam["pb1_b"] = self._bf16(pb.fc1.bias, (1, 1, -1))
            w2 = torch.zeros(pb.fc2.weight.shape[1], self.cam_pad)
            w2[:, : self.cam_target] = pb.fc2.weight.detach().t().float()
            b2 = torch.zeros(self.cam_pad)
            b2[: self.cam_target] = pb.fc2.bias.detach().float()
            self.cam["pb2_w"] = self._bf16(w2)
            self.cam["pb2_b"] = self._bf16(b2, (1, 1, -1))

    def _build_dpt(self):
        import torch.nn as nn

        self.heads: Dict[str, Dict[str, Any]] = {}
        for name in ("depth_head", "point_head"):
            head = getattr(self.model, name)
            assert not head.feature_only and hasattr(head.scratch, "output_conv2")
            hp: Dict[str, Any] = {
                "activation": head.activation, "conf_activation": head.conf_activation,
                "norm_g": self._bf16(head.norm.weight, (1, 1, -1)), "norm_b": self._bf16(head.norm.bias, (1, 1, -1)),
                "norm_g32": self._f32(head.norm.weight, (1, 1, -1)) if self.cfg.prelude == "fp32" else None,
                "norm_b32": self._f32(head.norm.bias, (1, 1, -1)) if self.cfg.prelude == "fp32" else None,
                "norm_eps": float(head.norm.eps), "pos_embed": bool(head.pos_embed),
                "layers": [],
            }
            for i, (proj, rl) in enumerate(zip(head.projects, head.resize_layers)):
                out_c = proj.weight.shape[0]
                layer = {
                    "out_c": out_c,
                    "proj_w": self._bf16(proj.weight.detach().reshape(out_c, -1).t().contiguous()),
                    "proj_b": self._bf16(proj.bias, (1, 1, -1)) if proj.bias is not None else None,
                    "pe": self._bf16(ft.dpt_pos_embed_tokens(out_c)) if head.pos_embed else None,
                }
                if self.cfg.prelude == "fp32":
                    layer["proj_w32"] = self._f32(proj.weight.detach().reshape(out_c, -1).t().contiguous())
                    layer["proj_b32"] = self._f32(proj.bias, (1, 1, -1)) if proj.bias is not None else None
                    layer["pe32"] = self._f32(ft.dpt_pos_embed_tokens(out_c)) if head.pos_embed else None
                if isinstance(rl, nn.ConvTranspose2d):
                    layer.update(kind="convT", w=self._host_conv_w(rl.weight), b=self._host_conv_b(rl.bias),
                                 k=tuple(rl.kernel_size), s=tuple(rl.stride), pad=tuple(rl.padding), prepared={})
                    layer["hw"] = (ft.GRID - 1) * rl.stride[0] - 2 * rl.padding[0] + rl.kernel_size[0]
                elif isinstance(rl, nn.Conv2d):
                    layer.update(kind="conv", w=self._host_conv_w(rl.weight), b=self._host_conv_b(rl.bias),
                                 k=tuple(rl.kernel_size), s=tuple(rl.stride), pad=tuple(rl.padding), prepared={})
                    layer["hw"] = (ft.GRID + 2 * rl.padding[0] - rl.kernel_size[0]) // rl.stride[0] + 1
                else:
                    layer.update(kind="identity")
                    layer["hw"] = ft.GRID
                conv = getattr(head.scratch, f"layer{i + 1}_rn")
                layer["rn"] = {"w": self._host_conv_w(conv.weight), "b": None, "in_c": conv.in_channels,
                               "out_c": conv.out_channels, "prepared": {}}
                hp["layers"].append(layer)
            hp["refinenets"] = []
            for r in (4, 3, 2, 1):
                ref = getattr(head.scratch, f"refinenet{r}")
                rp = {"idx": r, "has_res": bool(ref.has_residual), "ch": ref.out_conv.in_channels}
                for uname, unit in (("u1", getattr(ref, "resConfUnit1", None)), ("u2", ref.resConfUnit2)):
                    if unit is None:
                        continue
                    rp[uname] = {
                        "c1": {"w": self._host_conv_w(unit.conv1.weight), "b": self._host_conv_b(unit.conv1.bias), "prepared": {}},
                        "c2": {"w": self._host_conv_w(unit.conv2.weight), "b": self._host_conv_b(unit.conv2.bias), "prepared": {}},
                    }
                oc = ref.out_conv
                rp["out_w"] = self._bf16(oc.weight.detach().reshape(oc.out_channels, oc.in_channels).t().contiguous())
                rp["out_b"] = self._bf16(oc.bias, (1, 1, -1))
                hp["refinenets"].append(rp)
            oc1 = head.scratch.output_conv1
            hp["oc1"] = {"w": self._host_conv_w(oc1.weight), "b": self._host_conv_b(oc1.bias),
                         "in_c": oc1.in_channels, "out_c": oc1.out_channels, "prepared": {}}
            c3, c1 = head.scratch.output_conv2[0], head.scratch.output_conv2[2]
            hp["oc2a"] = {"w": self._host_conv_w(c3.weight), "b": self._host_conv_b(c3.bias),
                          "in_c": c3.in_channels, "out_c": c3.out_channels, "prepared": {}}
            hp["oc2b"] = {"w": self._host_conv_w(c1.weight), "b": self._host_conv_b(c1.bias),
                          "in_c": c1.in_channels, "out_c": c1.out_channels, "prepared": {}}
            hp["out_c"] = c1.out_channels
            self.heads[name] = hp
        # Bilinear (align_corners=True) interpolation matrices, fp32 (only 19->37 is bf16-exact).
        # The DPT chain runs PER FRAME (see `_dpt_heads`), so every table is the S = 1 one:
        # `rep`: A replicated over the bmm batch, (H, Wo, W) for the W-pass on X viewed (H, W, C)
        # and (1, Ho, H) for the H-pass (standard bmm / plain matmul, no first-operand batch
        # broadcast); `bcast`: A (1, n_out, n_in); `flat`: A^T (n_in, n_out).
        self.interp: Dict[Tuple[int, int], Any] = {}
        self.interp_t: Dict[Tuple[int, int], Any] = {}
        self.interp_w: Dict[Tuple[int, int], Any] = {}
        self.interp_h: Dict[Tuple[int, int], Any] = {}
        self.interp_g: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for n_in, n_out in ft.DPT_UPSAMPLES:
            A = ft.interp_matrix(n_in, n_out, dtype=torch.float32)
            if self.cfg.interp == "exact":
                G0, G1, w0, w1 = ft.interp_gather_tables(n_in, n_out)
                # W-pass: X viewed (H, W, C) -> gathers replicated over the H batch (bf16 0/1);
                # H-pass: Y viewed (1, H, Wo*C) -> plain (1, Ho, H).  Weights (1, n_out, 1) fp32.
                self.interp_g[(n_in, n_out, "w")] = {
                    "g0": self._bf16(G0.reshape(1, n_out, n_in).expand(n_in, n_out, n_in)),
                    "g1": self._bf16(G1.reshape(1, n_out, n_in).expand(n_in, n_out, n_in)),
                    "w0": self._f32(w0.reshape(1, n_out, 1)), "w1": self._f32(w1.reshape(1, n_out, 1))}
                self.interp_g[(n_in, n_out, "h")] = {
                    "g0": self._bf16(G0.reshape(1, n_out, n_in)), "g1": self._bf16(G1.reshape(1, n_out, n_in)),
                    "w0": self._f32(w0.reshape(1, n_out, 1)), "w1": self._f32(w1.reshape(1, n_out, 1))}
            elif self.cfg.interp == "bcast":
                self.interp[(n_in, n_out)] = self._f32(A.reshape(1, n_out, n_in))
            elif self.cfg.interp == "flat":
                self.interp_t[(n_in, n_out)] = self._f32(A.t().contiguous())
            else:
                self.interp_w[(n_in, n_out)] = self._f32(A.reshape(1, n_out, n_in).expand(n_in, n_out, n_in))
                self.interp_h[(n_in, n_out)] = self._f32(A.reshape(1, n_out, n_in))
        # pos_embed of the 518x518 output plane in the (1, Ho, Wo*C) layout the upsample emits.
        self.pe_full = self._f32(ft.dpt_pos_embed_rows(ft.IMG_SIZE, ft.IMG_SIZE, self.heads["depth_head"]["oc1"]["out_c"]))
        # 0/1 bf16 gather matmuls (patch rows; exact upsampling): HiFi4, one 1.0 term per output.
        self.kcfg_gather = self.ttnn.init_device_compute_kernel_config(
            self.device.arch(), math_fidelity=self.ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=False, packer_l1_acc=False)
        if self.cfg.gather == "matmul":
            # (1, 1369, 1374) bf16 0/1 row-selection matrix for the per-frame (1, 1374, C) tokens
            # (plain matmul -- a batch-1 first operand broadcast over a token BATCH takes the
            # matmul in0_reuse path that hangs the p150a, see `_upsample`); 3.8 MB.
            self.gather_m = self._bf16(ft.patch_gather_matrix(self.P, self.model.aggregator.patch_start_idx).reshape(1, -1, self.P))

    # ------------------------------------------------------------------ per-S constants
    def _consts(self, S: int) -> Dict[str, Any]:
        """Constant tables that depend on S (built on first use, before any capture of that S)."""
        c = self._per_s.get(S)
        if c is None:
            cos1, sin1 = ft.rope_tables(1, base=self.rope_base)
            cosS, sinS = ft.rope_tables(S, base=self.rope_base)
            c = {
                "assembly": self._f32(ft.assembly_add_table(self.camera_token, self.register_token, S, self.P)),
                "rope_frame": (self._bf16(cos1), self._bf16(sin1)),       # (1,1,P,64), B = S rows wrap per frame
                "rope_global": (self._bf16(cosS), self._bf16(sinS)),      # (1,1,S*P,64)
            }
            if self.cfg.camera == "device":
                c["cam_iter0"] = self._f32(self.cam_iter0_host.expand(1, S, self.cam_dim))
            self._per_s[S] = c
        return c

    # ------------------------------------------------------------------ matmul helpers
    def _linear(self, h, w, b, *, fp32_out: bool = False, hifi4: Optional[bool] = None):
        """``h @ w + b``.  ``fp32_out`` = the port's proj/fc2 recipe (HiFi4 + fp32 dest, fp32 output);
        otherwise ttnn.linear's bf16 default (HiFi2, no fp32 acc) or the same fidelity pinned on
        ``minimal_matmul`` for the B8 A/B."""
        ttnn = self.ttnn
        if hifi4 is None:
            hifi4 = fp32_out
        if self.cfg.matmul == "minimal":
            kw: Dict[str, Any] = {"bias_tensor": b, "config": self.mm_cfg,
                                  "compute_kernel_config": self.kcfg_hifi4 if hifi4 else self.kcfg_hifi2}
            if fp32_out:
                kw["dtype"] = ttnn.float32
            return ttnn.experimental.minimal_matmul(h, w, **kw)
        kw = {"bias": b}
        if hifi4:
            kw["compute_kernel_config"] = self.kcfg_hifi4
        if fp32_out:
            kw["dtype"] = ttnn.float32
        return ttnn.linear(h, w, **kw)

    # ------------------------------------------------------------------ attention
    def _softmax_rows(self, scores, B: int, H: int, N: int):
        """fp32 softmax over the last dim of ``(B, H, N, N)``; the fused kernel below the hang
        threshold, the port's 7-op decomposition (or a B11 variant) at and above it."""
        ttnn = self.ttnn
        if N < self.cfg.large_softmax_n:
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.kcfg_hifi4)
            self._free(scores)
            return probs
        if self.cfg.softmax == "inplace":
            return ttnn.softmax_in_place(scores, compute_kernel_config=self.kcfg_hifi4, numeric_stable=True)
        if self.cfg.softmax == "scale_mask":
            return ttnn.scale_mask_softmax_in_place(scores, None, None, compute_kernel_config=self.kcfg_hifi4,
                                                    numeric_stable=True)
        # padded rows of m / s must be finite: they become padded rows of probs -> ctx -> the
        # residual stream -> next block's k -> k^T's padded COLUMNS -> probs @ v of every row
        # (the legacy path re-zeroed its padding every block with a host from_torch; we do not).
        m = self._reshape(ttnn.max(scores, dim=-1), (B, H, N, 1))
        shifted = ttnn.subtract(scores, m)
        self._free(m, scores)
        e = ttnn.exp(shifted)
        self._free(shifted)
        s = self._reshape(ttnn.sum(e, dim=-1), (B, H, N, 1))
        r = ttnn.reciprocal(s)
        self._free(s)
        probs = ttnn.multiply(e, r)
        self._free(e, r)
        return probs

    def _attention(self, n, p, B: int, N: int, rope):
        """LN1 output ``n (B, N, C)`` bf16 -> attention context ``(B, N, C)`` bf16 (heads merged)."""
        ttnn = self.ttnn
        H, Dh = p["heads"], p["head_dim"]
        qkv = self._linear(n, p["qkv_w"], p["qkv_b"])
        qkv4 = self._reshape(qkv, (B, 1, N, H * Dh * 3))    # view (same rows / last dim)
        need_k = p["qk_norm"] or p["rope"]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv4, num_heads=H, num_kv_heads=H, transpose_k_heads=not need_k and self.cfg.attn == "matmul")
        self._free(qkv4)
        if p["qk_norm"]:
            q2 = ttnn.layer_norm(q, weight=p["qn_g"], bias=p["qn_b"], epsilon=p["qn_eps"])
            k2 = ttnn.layer_norm(k, weight=p["kn_g"], bias=p["kn_b"], epsilon=p["qn_eps"])
            self._free(q, k)
            q, k = q2, k2
        if p["rope"]:
            cos, sin = rope
            q2 = self._unpad_rows(ttnn.experimental.rotary_embedding(q, cos, sin), q.shape)
            k2 = self._unpad_rows(ttnn.experimental.rotary_embedding(k, cos, sin), k.shape)
            self._free(q, k)
            q, k = q2, k2
        if self.cfg.attn == "sdpa":
            ctx = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=1.0 if p["scale_folded"] else p["scale"],
                program_config=self.sdpa_cfg, compute_kernel_config=self.kcfg_hifi4)
            self._free(q, k, v)
            merged = ttnn.experimental.nlp_concat_heads(ctx)  # (B, 1, N, C) bf16
            self._free(ctx)
            return self._reshape(merged, (B, N, H * Dh))      # view
        kt = ttnn.permute(k, (0, 1, 3, 2)) if need_k else k   # (B, H, Dh, N)
        if need_k:
            self._free(k)
        scores = ttnn.matmul(q, kt, compute_kernel_config=self.kcfg_hifi4, dtype=ttnn.float32)
        self._free(q, kt)
        if not p["scale_folded"]:
            scores2 = ttnn.multiply(scores, p["scale"])
            self._free(scores)
            scores = scores2
        probs = self._softmax_rows(scores, B, H, N)   # frees `scores` on the decomposed path
        ctx = ttnn.matmul(probs, v, compute_kernel_config=self.kcfg_hifi4, dtype=ttnn.float32)
        self._free(probs, v)
        merged = ttnn.experimental.nlp_concat_heads(ctx)     # (B, 1, N, C) fp32
        self._free(ctx)
        merged = self._reshape(merged, (B, N, H * Dh))       # view
        out = ttnn.typecast(merged, ttnn.bfloat16)
        self._free(merged)
        return out

    def _block(self, x, p, *, rope=None, free_input: bool = True):
        """One transformer block on the fp32 residual ``x (B, N, C)`` (device resident)."""
        ttnn = self.ttnn
        B, N = x.shape[0], x.shape[1]
        n = ttnn.typecast(x, ttnn.bfloat16)
        n2 = ttnn.layer_norm(n, weight=p["ln1_g"], bias=p["ln1_b"], epsilon=p["ln1_eps"])
        self._free(n)
        ctx = self._attention(n2, p, B, N, rope)
        self._free(n2)
        y = self._linear(ctx, p["proj_w"], p["proj_b"], fp32_out=True)
        self._free(ctx)
        if p["ls1"] is not None:
            y2 = ttnn.multiply(y, p["ls1"])
            self._free(y)
            y = y2
        x1 = ttnn.add(x, y)
        self._free(y)
        if free_input:
            self._free(x)
        n = ttnn.typecast(x1, ttnn.bfloat16)
        n2 = ttnn.layer_norm(n, weight=p["ln2_g"], bias=p["ln2_b"], epsilon=p["ln2_eps"])
        self._free(n)
        h = self._linear(n2, p["fc1_w"], p["fc1_b"])
        self._free(n2)
        h2 = ttnn.gelu(h)
        self._free(h)
        y = self._linear(h2, p["fc2_w"], p["fc2_b"], fp32_out=True)
        self._free(h2)
        if p["ls2"] is not None:
            y2 = ttnn.multiply(y, p["ls2"])
            self._free(y)
            y = y2
        x2 = ttnn.add(x1, y)
        self._free(y, x1)
        return x2

    # ------------------------------------------------------------------ graph: backbone + aggregator
    def _dinov2(self, x):
        for p in self.dino_blocks:
            x = self._block(x, p)
        return x

    def _assemble(self, x, S: int):
        """DINOv2 stream ``(S, P, C)`` fp32 -> aggregator tokens: final LN (fp32), then
        ``x * row_mask + T`` (drop the 5 DINOv2 special rows, put camera + register constants)."""
        ttnn = self.ttnn
        xn = self._ln32(x, self.dino_norm["g"], self.dino_norm["b"], self.dino_norm["eps"])
        self._free(x)
        xm = ttnn.multiply(xn, self.row_mask)
        self._free(xn)
        tokens = ttnn.add(xm, self._consts(S)["assembly"])
        self._free(xm)
        return tokens

    def _aggregator(self, tokens, S: int) -> List[Any]:
        """24 x (frame, global) blocks; returns the 4 consumed ``(S, P, 2C)`` fp32 concats."""
        ttnn = self.ttnn
        c = self._consts(S)
        merged_is_view = S == 1
        keep = set(self.inter_idx)
        inters: List[Any] = []
        x = tokens  # (S, P, C)
        for i, (fp, gp) in enumerate(zip(self.frame_blocks, self.global_blocks)):
            kept = i in keep
            f = self._block(x, fp, rope=c["rope_frame"], free_input=True)            # (S, P, C)
            fm = self._reshape(f, (1, S * self.P, self.C)) if S > 1 else f         # (1, S*P, C), zero-padded rows
            gm = self._block(fm, gp, rope=c["rope_global"], free_input=not merged_is_view)
            g = self._reshape(gm, (S, self.P, self.C)) if S > 1 else gm           # (S, P, C), zero-padded rows
            if S > 1:
                self._free(gm)
            if kept:
                inters.append(ttnn.concat([f, g], dim=-1))                          # (S, P, 2C) fp32
            self._free(f)
            x = g
        self._free(x)
        assert len(inters) == len(self.inter_idx)
        return inters

    # ------------------------------------------------------------------ graph: camera head
    def _camera_tokens(self, x_last, S: int):
        """Row 0 of every frame of the last ``(S, P, 2C)`` concat -> ``(S, 1, 2C)`` fp32
        (rows 1..31 of the result are fresh tile padding: filled with zeros explicitly)."""
        return self.ttnn.slice(x_last, [0, 0, 0], [S, 1, self.cam_dim], pad_value=0.0)

    def _camera_device(self, cam_tokens, S: int) -> List[Any]:
        """CameraHead.trunk_fn in-graph: returns the 4 un-activated pose accumulators ``(1, S, 32)``."""
        ttnn = self.ttnn
        cam = self.cam
        kc = self.kcfg_hifi4
        toks = self._reshape(cam_tokens, (1, S, self.cam_dim))                       # (1, S, 2C) fp32 (view at S=1)
        pose_tokens = ttnn.layer_norm(toks, weight=cam["token_norm_g"], bias=cam["token_norm_b"],
                                      epsilon=cam["token_norm_eps"], compute_kernel_config=kc)
        self._free(toks)
        pred = None
        preds: List[Any] = []
        for it in range(self.cam_iters):
            if pred is None:
                mod_in = self._consts(S)["cam_iter0"]
            else:
                mod_in = ttnn.linear(pred, cam["embed_w"], bias=cam["embed_b"], compute_kernel_config=kc,
                                     dtype=ttnn.float32)
            act = ttnn.silu(mod_in)
            if pred is not None:
                self._free(mod_in)
            mod = ttnn.linear(act, cam["mod_w"], bias=cam["mod_b"], compute_kernel_config=kc, dtype=ttnn.float32)
            self._free(act)
            Cc = self.cam_dim
            shift = ttnn.slice(mod, [0, 0, 0], [1, S, Cc])
            scale = ttnn.slice(mod, [0, 0, Cc], [1, S, 2 * Cc])
            gate = ttnn.slice(mod, [0, 0, 2 * Cc], [1, S, 3 * Cc])
            self._free(mod)
            normed = ttnn.layer_norm(pose_tokens, epsilon=cam["adaln_eps"], compute_kernel_config=kc)
            scale1 = ttnn.add(scale, 1.0)
            t = ttnn.multiply(normed, scale1)
            self._free(normed, scale1, scale)
            t2 = ttnn.add(t, shift)
            self._free(t, shift)
            t3 = ttnn.multiply(gate, t2)
            self._free(gate, t2)
            x = ttnn.add(t3, pose_tokens)                                              # (1, S, 2C) fp32
            self._free(t3)
            for p in self.cam_blocks:
                x = self._block(x, p)
            tn = ttnn.layer_norm(x, weight=cam["trunk_norm_g"], bias=cam["trunk_norm_b"],
                                 epsilon=cam["trunk_norm_eps"], compute_kernel_config=kc)
            self._free(x)
            tnb = ttnn.typecast(tn, ttnn.bfloat16)
            self._free(tn)
            h = ttnn.linear(tnb, cam["pb1_w"], bias=cam["pb1_b"])
            self._free(tnb)
            h2 = ttnn.gelu(h)
            self._free(h)
            delta = ttnn.linear(h2, cam["pb2_w"], bias=cam["pb2_b"], dtype=ttnn.float32)   # (1, S, 32) fp32
            self._free(h2)
            if pred is None:
                pred = delta
            else:
                pred2 = ttnn.add(pred, delta)
                self._free(delta)
                pred = pred2
            preds.append(pred)
        self._free(pose_tokens)
        return preds

    # ------------------------------------------------------------------ graph: DPT heads
    def _conv(self, x, cw: Dict[str, Any], S: int, H: int, W: int, in_c: int, out_c: int, *, k=(3, 3),
              stride=(1, 1), pad=(1, 1), dtype=None, relu: bool = False, transpose: bool = False):
        """conv2d / conv_transpose2d with per-S cached prepared weights.  The first (eager) call
        at a given S hands the op host weights and keeps the prepared device tensors it returns,
        so the captured trace never uploads weights (a host->device copy cannot be replayed)."""
        ttnn = self.ttnn
        prep = cw["prepared"].get(S)
        w, b = (prep if prep is not None else (cw["w"], cw["b"]))
        kw: Dict[str, Any] = dict(
            input_tensor=x, weight_tensor=w, bias_tensor=b, device=self.device, in_channels=in_c,
            out_channels=out_c, batch_size=S, input_height=H, input_width=W, kernel_size=tuple(k),
            stride=tuple(stride), padding=tuple(pad), return_weights_and_bias=True,
        )
        if dtype is not None:
            kw["dtype"] = dtype
        if transpose:
            out, (pw, pb) = ttnn.conv_transpose2d(**kw)
        else:
            kw["compute_config"] = self.kcfg_hifi4
            if relu:
                kw["conv_config"] = self.relu_cfg
            out, (pw, pb) = ttnn.conv2d(**kw)
        if prep is None:
            cw["prepared"][S] = (pw, pb)
        return out

    def _interleaved(self, x, *, free_input: bool = False):
        """DRAM-interleaved copy of a sharded conv output (identity otherwise).  With
        ``free_input`` the sharded original is released once the copy exists, so the relayout
        does not leak one buffer per call (eager mode) or per trace."""
        if x.is_sharded():
            y = self.ttnn.sharded_to_interleaved(x, self.ttnn.DRAM_MEMORY_CONFIG)
            if free_input:
                self._free(x)
            return y
        return x

    def _upsample(self, x, S: int, H: int, W: int, C: int, Ho: int, Wo: int, *, pe=None, out_bf16: bool = True):
        """Bilinear ``align_corners=True`` ``(H, W) -> (Ho, Wo)`` on the flat ``(1, 1, S*H*W, C)`` fp32
        map as two constant matmuls (W-pass then H-pass), optional pos_embed add in the
        ``(S, Ho, Wo*C)`` layout, back to flat ``(1, 1, S*Ho*Wo, C)``.  Exact to the FPU's fp32
        matmul rounding (measured max|d| ~1e-2 on unit-scale inputs at K=296, i.e. tf32-class
        input rounding, not bf16).  Consumes ``x``.  R1/R2 pad rows W..32 / H..32 that the
        matmuls contract over (K): they are zero-filled by ``_reshape`` (A's padded columns are 0
        but 0 * stale NaN = NaN).

        ``VGGT_FUSED_INTERP``: ``rep`` (default) = A replicated over the bmm batch (per-S tables,
        standard bmm; 296->518 at S=1 measured 3.4 + 3.8 ms for the two matmuls, 20.6 ms for the
        whole upsample eager); ``flat`` = transposed formulation ``X^T @ A^T`` with plain 2-D
        matmuls and 4 transposes (49 ms, no tables); ``bcast`` = ``A (1, Ho, H) @ X`` with A
        broadcast over X's batch (the tree's in0_reuse matmul path) -- HANGS on the p150a
        (`probe_ops.log`, 2026-09-13), kept only as the record of the original formulation."""
        ttnn = self.ttnn
        x = self._interleaved(x, free_input=True)
        if self.cfg.interp == "exact":
            return self._upsample_exact(x, S, H, W, C, Ho, Wo, pe=pe, out_bf16=out_bf16)
        if self.cfg.interp == "flat":
            return self._upsample_flat(x, S, H, W, C, Ho, Wo, pe=pe, out_bf16=out_bf16)
        if self.cfg.interp == "rep":
            assert S == 1, "the rep tables are the per-frame (S = 1) ones; the DPT chain runs per frame"
            A_w, A_h = self.interp_w[(W, Wo)], self.interp_h[(H, Ho)]
        else:
            A_w, A_h = self.interp[(W, Wo)], self.interp[(H, Ho)]
        x1 = self._reshape(x, (S * H, W, C))                                                      # R1
        self._free(x)
        y1 = ttnn.matmul(A_w, x1, compute_kernel_config=self.kcfg_hifi4, dtype=ttnn.float32)     # (S*H, Wo, C)
        self._free(x1)
        y2 = self._relayout(y1, (S, H, Wo * C))                                                   # R2
        self._free(y1)
        final_bf16 = out_bf16 and pe is None
        y3 = ttnn.matmul(A_h, y2, compute_kernel_config=self.kcfg_hifi4,
                         dtype=ttnn.bfloat16 if final_bf16 else ttnn.float32)                    # (S, Ho, Wo*C)
        self._free(y2)
        return self._upsample_tail(y3, S, C, Ho, Wo, pe=pe, out_bf16=out_bf16)

    def _interp_pass_exact(self, X, t: Dict[str, Any]):
        """Exact 1-D linear interpolation along the ROW dim of the fp32 ``X (B, n_in, N)``:
        ``Y = (G0 @ X) * w0 + (G1 @ X) * w1`` with 0/1 bf16 ``G0/G1`` applied to a bf16 hi/lo split
        of ``X`` (``hi = bf16(X)``, ``lo = bf16(X - hi)``: every gather is a single exact bf16
        product, the fp32 sum ``hi + lo`` restores ``X`` to ~2^-16) and fp32 eltwise weights.
        Consumes ``X``."""
        ttnn = self.ttnn
        hi = ttnn.typecast(X, ttnn.bfloat16)
        hi32 = ttnn.typecast(hi, ttnn.float32)
        lo32 = ttnn.subtract(X, hi32)
        self._free(X, hi32)
        lo = ttnn.typecast(lo32, ttnn.bfloat16)
        self._free(lo32)
        outs = []
        for g, w in ((t["g0"], t["w0"]), (t["g1"], t["w1"])):
            a = ttnn.matmul(g, hi, compute_kernel_config=self.kcfg_gather, dtype=ttnn.float32)
            b = ttnn.matmul(g, lo, compute_kernel_config=self.kcfg_gather, dtype=ttnn.float32)
            y = ttnn.add(a, b)
            self._free(a, b)
            yw = ttnn.multiply(y, w)
            self._free(y)
            outs.append(yw)
        self._free(hi, lo)
        out = ttnn.add(outs[0], outs[1])
        self._free(*outs)
        return out

    def _upsample_exact(self, x, S: int, H: int, W: int, C: int, Ho: int, Wo: int, *, pe, out_bf16: bool):
        """``VGGT_FUSED_INTERP=exact``: W-pass then H-pass with `_interp_pass_exact` (per frame, S = 1)."""
        assert S == 1, "the exact tables are the per-frame (S = 1) ones; the DPT chain runs per frame"
        x1 = self._reshape(x, (H, W, C))                                                          # R1 (zero-filled pad rows)
        self._free(x)
        y1 = self._interp_pass_exact(x1, self.interp_g[(W, Wo, "w")])                             # (H, Wo, C) fp32
        y2 = self._relayout(y1, (1, H, Wo * C))                                                   # R2
        self._free(y1)
        y3 = self._interp_pass_exact(y2, self.interp_g[(H, Ho, "h")])                             # (1, Ho, Wo*C) fp32
        if pe is None and out_bf16:
            y4 = self.ttnn.typecast(y3, self.ttnn.bfloat16)
            self._free(y3)
            y3 = y4
        return self._upsample_tail(y3, S, C, Ho, Wo, pe=pe, out_bf16=out_bf16)

    def _upsample_tail(self, y3, S: int, C: int, Ho: int, Wo: int, *, pe, out_bf16: bool):
        """``(S, Ho, Wo*C)`` (+ pos_embed, bf16 cast) -> flat ``(1, 1, S*Ho*Wo, C)``.  Consumes ``y3``."""
        ttnn = self.ttnn
        if pe is not None:
            y4 = ttnn.add(y3, pe)                                                                 # fp32 + fp32
            self._free(y3)
            y3 = y4
            if out_bf16:
                y4 = ttnn.typecast(y3, ttnn.bfloat16)
                self._free(y3)
                y3 = y4
        out = self._relayout(y3, (1, 1, S * Ho * Wo, C))                                           # R3
        self._free(y3)
        return out

    def _upsample_flat(self, x, S: int, H: int, W: int, C: int, Ho: int, Wo: int, *, pe, out_bf16: bool):
        """``VGGT_FUSED_INTERP=flat``: the interpolated axis is moved last with ``transpose`` so both
        passes are plain 2-D ``X^T @ A^T`` matmuls (no batch dims at all).  Consumes ``x``."""
        ttnn = self.ttnn
        kc = self.kcfg_hifi4
        A_wT, A_hT = self.interp_t[(W, Wo)], self.interp_t[(H, Ho)]
        x1 = self._reshape(x, (S * H, W, C))                                                      # R1
        self._free(x)
        xt = ttnn.transpose(x1, -2, -1)                                                           # (S*H, C, W)
        self._free(x1)
        xf = self._reshape(xt, (S * H * C, W))                                                    # view of xt
        y = ttnn.matmul(xf, A_wT, compute_kernel_config=kc, dtype=ttnn.float32)                  # (S*H*C, Wo)
        self._free(xt)
        y3 = self._reshape(y, (S, H, C * Wo))                                                     # rows h, cols (c, wo)
        self._free(y)
        yt = ttnn.transpose(y3, -2, -1)                                                           # (S, C*Wo, H)
        self._free(y3)
        yf = self._reshape(yt, (S * C * Wo, H))                                                   # view of yt
        z = ttnn.matmul(yf, A_hT, compute_kernel_config=kc, dtype=ttnn.float32)                  # (S*C*Wo, Ho)
        self._free(yt)
        z3 = self._reshape(z, (S, C * Wo, Ho))                                                    # view of z
        zt = ttnn.transpose(z3, -2, -1)                                                           # (S, Ho, C*Wo)
        self._free(z3)
        z4 = self._reshape(zt, (S * Ho, C, Wo))
        self._free(zt)
        zp = ttnn.transpose(z4, -2, -1)                                                           # (S*Ho, Wo, C)
        self._free(z4)
        y5 = self._reshape(zp, (S, Ho, Wo * C))                                                   # same layout as the rep H-pass output
        self._free(zp)
        if pe is None and out_bf16:
            y6 = ttnn.typecast(y5, ttnn.bfloat16)
            self._free(y5)
            y5 = y6
        return self._upsample_tail(y5, S, C, Ho, Wo, pe=pe, out_bf16=out_bf16)

    def _resconv(self, x, unit: Dict[str, Any], S: int, H: int, W: int, ch: int):
        """ResidualConvUnit with the reference's in-place ReLU semantics: ``conv2(relu(conv1(relu(x))))
        + relu(x)``; both convs emit fp32, the first has its ReLU fused (B5)."""
        ttnn = self.ttnn
        r = ttnn.relu(x)
        c1 = self._conv(r, unit["c1"], S, H, W, ch, ch, dtype=ttnn.float32, relu=True)
        c2 = self._conv(c1, unit["c2"], S, H, W, ch, ch, dtype=ttnn.float32)
        self._free(c1)
        if r.dtype != ttnn.float32:
            r2 = ttnn.typecast(r, ttnn.float32)
            self._free(r)
            r = r2
        out = ttnn.add(r, c2)
        self._free(r, c2)
        return out

    def _dpt_prelude(self, hp: Dict[str, Any], layer: Dict[str, Any], tokens, S: int):
        """``(S, P, 2C)`` fp32 tokens -> LN -> 1x1 proj -> +pos_embed -> resize; returns the
        ``(1, 1, S*h*w, out_c)`` bf16 feature map and its side ``h``."""
        ttnn = self.ttnn
        out_c = layer["out_c"]
        fp32 = self.cfg.prelude == "fp32"
        if fp32:
            n = self._ln32(tokens, hp["norm_g32"], hp["norm_b32"], hp["norm_eps"])                    # fp32 (S, P, 2C)
            y = ttnn.linear(n, layer["proj_w32"], bias=layer["proj_b32"], compute_kernel_config=self.kcfg_hifi4,
                            dtype=ttnn.float32)                                                     # fp32 x fp32 (tf32-class)
            self._free(n)
            # the 0/1 gather matmul would round y tf32-class: slice the patch rows instead (exact copy)
            yp = ttnn.slice(y, [0, self.P - ft.N_PATCHES, 0], [S, self.P, out_c], pad_value=0.0)
            self._free(y)
            if layer["pe32"] is not None:
                yp2 = ttnn.add(yp, layer["pe32"])
                self._free(yp)
                yp = yp2
            yb = ttnn.typecast(yp, ttnn.bfloat16)                                                   # conv input dtype as the bf16 path
            self._free(yp)
            yp = yb
        else:
            tb = ttnn.typecast(tokens, ttnn.bfloat16)
            n = ttnn.layer_norm(tb, weight=hp["norm_g"], bias=hp["norm_b"], epsilon=hp["norm_eps"])
            self._free(tb)
            y = ttnn.linear(n, layer["proj_w"], bias=layer["proj_b"], compute_kernel_config=self.kcfg_hifi4)  # (S, P, out_c)
            self._free(n)
            if self.cfg.gather == "matmul":
                # exact bf16 row gather (rows 5..): G (1, 1369, 1374) @ y (1, 1374, out_c), HiFi4, one 1.0 per row
                assert S == 1, "the gather matrix is the per-frame one; the DPT chain runs per frame"
                yp = ttnn.matmul(self.gather_m, y, compute_kernel_config=self.kcfg_gather)
            else:
                yp = ttnn.slice(y, [0, self.P - ft.N_PATCHES, 0], [S, self.P, out_c], pad_value=0.0)
            self._free(y)
            if layer["pe"] is not None:
                yp2 = ttnn.add(yp, layer["pe"])                                                     # (S, 1369, out_c)
                self._free(yp)
                yp = yp2
        if layer["kind"] == "identity":
            flat = self._reshape(yp, (1, 1, S * ft.N_PATCHES, out_c))      # view at S=1, zero-padded copy at S>1
            if S > 1:
                self._free(yp)
            return flat, ft.GRID
        rm = ttnn.to_layout(yp, ttnn.ROW_MAJOR_LAYOUT)
        self._free(yp)
        flat = self._reshape(rm, (1, 1, S * ft.N_PATCHES, out_c))         # ROW_MAJOR: always a view
        out = self._conv(flat, layer, S, ft.GRID, ft.GRID, out_c, out_c, k=layer["k"], stride=layer["s"],
                         pad=layer["pad"], transpose=layer["kind"] == "convT")
        self._free(flat)
        return out, layer["hw"]

    def _dpt_head(self, hp: Dict[str, Any], inters: List[Any], S: int):
        """One DPTHead (depth or point): device-resident from the aggregator concats to the raw
        ``(1, 1, S*518*518, out_c)`` output_conv2 result (activate_head runs on the host)."""
        ttnn = self.ttnn
        feats = []
        for layer, tokens in zip(hp["layers"], inters):
            fmap, hw = self._dpt_prelude(hp, layer, tokens, S)
            rn = self._conv(fmap, layer["rn"], S, hw, hw, layer["rn"]["in_c"], layer["rn"]["out_c"])
            self._free(fmap)
            feats.append((rn, hw))
        (l1, h1), (l2, h2), (l3, h3), (l4, h4) = feats                     # 148, 74, 37, 19
        sizes = {4: (h4, h3), 3: (h3, h2), 2: (h2, h1), 1: (h1, 2 * h1)}
        skips = {4: None, 3: l3, 2: l2, 1: l1}
        prev = l4
        for rp in hp["refinenets"]:
            r = rp["idx"]
            Hi, Ho = sizes[r]
            ch = rp["ch"]
            if rp["has_res"]:
                sk = self._resconv(skips[r], rp["u1"], S, Hi, Hi, ch)
                self._free(skips[r])
                if prev.dtype != ttnn.float32:
                    p2 = ttnn.typecast(prev, ttnn.float32)
                    self._free(prev)
                    prev = p2
                s2 = ttnn.add(prev, sk)
                self._free(prev, sk)
                prev = s2
            out = self._resconv(prev, rp["u2"], S, Hi, Hi, ch)
            self._free(prev)
            up = self._upsample(out, S, Hi, Hi, ch, Ho, Ho, out_bf16=True)         # bf16 flat (1,1,S*Ho*Ho,ch)
            prev = ttnn.linear(up, rp["out_w"], bias=rp["out_b"], compute_kernel_config=self.kcfg_hifi4,
                               dtype=ttnn.float32)
            self._free(up)
        Hf = 2 * h1                                                            # 296
        oc1 = self._conv(prev, hp["oc1"], S, Hf, Hf, hp["oc1"]["in_c"], hp["oc1"]["out_c"], dtype=ttnn.float32)
        self._free(prev)
        C1 = hp["oc1"]["out_c"]
        up = self._upsample(oc1, S, Hf, Hf, C1, ft.IMG_SIZE, ft.IMG_SIZE,
                            pe=self.pe_full if hp["pos_embed"] else None, out_bf16=not self.cfg.oc2_fp32)
        a = self._conv(up, hp["oc2a"], S, ft.IMG_SIZE, ft.IMG_SIZE, hp["oc2a"]["in_c"], hp["oc2a"]["out_c"],
                       relu=True, dtype=ttnn.float32 if self.cfg.oc2_fp32 else None)
        self._free(up)
        raw = self._conv(a, hp["oc2b"], S, ft.IMG_SIZE, ft.IMG_SIZE, hp["oc2b"]["in_c"], hp["oc2b"]["out_c"],
                         k=(1, 1), pad=(0, 0), dtype=ttnn.float32 if self.cfg.oc2_fp32 else None)
        self._free(a)
        return self._interleaved(raw, free_input=True)

    def _dpt_heads(self, inters: List[Any], S: int) -> Dict[str, List[Any]]:
        """Both DPT heads, ONE FRAME AT A TIME: the 4 ``(S, P, 2C)`` concats are sliced per frame
        (batch-dim slices: whole tiles, exact, padding inherited) and ``_dpt_head`` runs at S = 1
        shapes for every frame.  Returns per-head lists of ``(1, 1, 518*518, out_c)`` raw outputs
        (one readback per frame; ``_finish`` concatenates).

        Why per frame (measured, p150a 2026-09-13): the batched ``(1, 1, S*518*518, 128)`` bf16
        output_conv2 input does not fit L1 at S >= 2 (1.25 MB per core at S = 1 already), so
        conv2d falls back to DRAM op-slicing (``op_slicing.cpp`` "Failed to find valid config with
        width-slicing. Attempting fallback to height-slicing"), and the sliced conv does host
        reads / event synchronisation -- ``TT_FATAL: Reads are not supported during trace capture``
        -- and, run eagerly while another trace is live, hung the chip (two resets).  Per frame the
        chain is exactly the S = 1 chain that captures and replays; the interpolation / gather
        tables and the prepared conv weights are the S = 1 ones (211 MB of fp32 tables instead of
        211 MB * S per pre-warmed S), and the work is the same (the convs are linear in S)."""
        ttnn = self.ttnn
        outs: Dict[str, List[Any]] = {"depth_raw": [], "point_raw": []}
        for s in range(S):
            if S == 1:
                ins = inters
            else:
                ins = [ttnn.slice(x, [s, 0, 0], [s + 1, self.P, x.shape[-1]]) for x in inters]   # (1, P, 2C) fp32 each
            outs["depth_raw"].append(self._dpt_head(self.heads["depth_head"], ins, 1))
            outs["point_raw"].append(self._dpt_head(self.heads["point_head"], ins, 1))
            if S > 1:
                self._free(*ins)
        return outs

    # ------------------------------------------------------------------ the whole device graph
    def _graph(self, x_in, S: int) -> Dict[str, Any]:
        """Persistent device input (``(S, P, C)`` fp32 ROW_MAJOR or TILE) -> dict of device outputs.
        Everything here is captured into the trace for this S."""
        ttnn = self.ttnn
        self._stage_t0 = time.perf_counter()
        if x_in.layout == ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.tilize_with_zero_padding(x_in, use_multicore=True)
        else:
            x = ttnn.clone(x_in)  # the trace must not free its persistent input
        self._stage("input tilize")
        x = self._dinov2(x)
        self._stage("dinov2 24 blocks")
        tokens = self._assemble(x, S)
        self._stage("assemble")
        inters = self._aggregator(tokens, S)
        self._stage("aggregator 48 blocks")
        outs: Dict[str, Any] = {}
        cam_tokens = self._camera_tokens(inters[-1], S)
        if self.cfg.camera == "device":
            outs["pose_preds"] = self._camera_device(cam_tokens, S)
            self._free(cam_tokens)
        else:
            outs["cam_tokens"] = cam_tokens
        self._stage("camera head")
        outs.update(self._dpt_heads(inters, S))          # per-frame lists of (1, 1, 518*518, out_c)
        self._stage("dpt heads (per frame)")
        self._free(*inters)
        return outs

    @staticmethod
    def _flat_outs(outs: Dict[str, Any]) -> List[Any]:
        flat = []
        for v in outs.values():
            flat.extend(v if isinstance(v, list) else [v])
        return flat

    # ------------------------------------------------------------------ host side
    def host_embed(self, images: torch.Tensor) -> torch.Tensor:
        """``(1, S, 3, 518, 518)`` in [0, 1] -> DINOv2 input tokens ``(S, P, C)`` fp32 (the upstream
        ``Aggregator.forward`` normalisation + ``DinoVisionTransformer.prepare_tokens_with_masks``)."""
        agg = self.model.aggregator
        B, S, Ci, Hh, Ww = images.shape
        if B != 1:
            raise ValueError("the device graph is built for batch 1")
        if (Hh, Ww) != (ft.IMG_SIZE, ft.IMG_SIZE):
            raise ValueError(f"expected {ft.IMG_SIZE}x{ft.IMG_SIZE} views, got {Hh}x{Ww}")
        with torch.no_grad():
            x = ((images.float() - agg._resnet_mean) / agg._resnet_std).reshape(S, Ci, Hh, Ww)
            tok = agg.patch_embed.prepare_tokens_with_masks(x)
        assert tuple(tok.shape) == (S, self.P, self.C), tok.shape
        return tok.contiguous()

    def host_input(self, tokens: torch.Tensor):
        """Host ttnn tensor with the spec of the persistent trace input."""
        ttnn = self.ttnn
        if self.cfg.input == "tile":
            return ttnn.from_torch(tokens, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        return ttnn.from_torch(tokens, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _read(self, outs: Dict[str, Any]) -> Dict[str, Any]:
        ttnn = self.ttnn
        got: Dict[str, Any] = {}
        for k, v in outs.items():
            if isinstance(v, list):
                got[k] = [ttnn.to_torch(t).float() for t in v]
            else:
                got[k] = ttnn.to_torch(v).float()
        return got

    def _finish(self, got: Dict[str, Any], images: torch.Tensor, S: int) -> Dict[str, torch.Tensor]:
        """Raw readbacks -> the dict ``VGGT.forward`` returns (same keys / shapes as the legacy path)."""
        from vggt.heads.head_act import activate_head, activate_pose  # type: ignore

        ch = self.model.camera_head
        with torch.no_grad():
            if "pose_preds" in got:
                pose_list = [activate_pose(p[..., : self.cam_target], trans_act=ch.trans_act, quat_act=ch.quat_act,
                                           fl_act=ch.fl_act) for p in got["pose_preds"]]
            else:
                toks = got["cam_tokens"].reshape(1, S, self.cam_dim)
                pose_list = ch.trunk_fn(ch.token_norm(toks), self.cam_iters)
            preds = {"pose_enc": pose_list[-1], "pose_enc_list": pose_list}
            for key, hname in (("depth", "depth_head"), ("world_points", "point_head")):
                hp = self.heads[hname]
                raw = got[f"{'depth' if key == 'depth' else 'point'}_raw"]
                if isinstance(raw, list):                      # per-frame readbacks (1, 1, 518*518, oc)
                    raw = torch.cat([r.reshape(1, ft.IMG_SIZE, ft.IMG_SIZE, -1) for r in raw], dim=0)
                oc = hp["out_c"]
                raw = raw.reshape(S, ft.IMG_SIZE, ft.IMG_SIZE, -1)[..., :oc].permute(0, 3, 1, 2).contiguous()
                val, conf = activate_head(raw, activation=hp["activation"], conf_activation=hp["conf_activation"])
                preds[key] = val.view(1, S, *val.shape[1:])
                preds[f"{key}_conf"] = conf.view(1, S, *conf.shape[1:])
            preds["images"] = images
        return preds

    # ------------------------------------------------------------------ eager / trace plumbing
    def _zero_tokens(self, S: int) -> torch.Tensor:
        return torch.zeros(S, self.P, self.C, dtype=torch.float32)

    def _eager_warm(self, S: int) -> None:
        """Run the device graph eagerly once at S: compiles every program, builds ttnn's reshape
        lookup tables and the per-S prepared conv weights -- all allocations that must exist
        BEFORE any trace is captured (tt-metal TraceCorrectness: later allocations are unsafe)."""
        ttnn = self.ttnn
        t0 = time.perf_counter()
        self._consts(S)
        x_in = ttnn.to_device(self.host_input(self._zero_tokens(S)), self.device)
        outs = self._graph(x_in, S)
        got = self._read(outs)
        self._finish(got, torch.zeros(1, S, 3, ft.IMG_SIZE, ft.IMG_SIZE), S)  # exercises the host tail too
        self._free(*self._flat_outs(outs), x_in)
        ttnn.synchronize_device(self.device)
        if self.cfg.verbose:
            _log(f"eager warm S={S}: {time.perf_counter() - t0:.1f}s")

    def _capture(self, S: int) -> None:
        ttnn = self.ttnn
        t0 = time.perf_counter()
        dev_in = ttnn.to_device(self.host_input(self._zero_tokens(S)), self.device)  # persistent trace input
        if not self._traces:
            # Warm run on the very buffer the trace reads from (program cache hits, prepared
            # weights).  Only while no trace is live: `warm` has already run every S eagerly,
            # and eager execution of the graph with a live trace hung the p150a twice
            # (2026-09-13, batched DPT conv; kept conservative -- a capture records without executing).
            if self.cfg.verbose:
                _log(f"capture S={S}: warm run on the persistent input (no live trace)")
            outs = self._graph(dev_in, S)
            ttnn.synchronize_device(self.device)
            self._free(*self._flat_outs(outs))
        elif self.cfg.verbose:
            _log(f"capture S={S}: {len(self._traces)} live trace(s), capturing without a warm run")
        if self.cfg.verbose:
            _log(f"capture S={S}: begin_trace_capture")
        self._capturing = True
        try:
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            outs = self._graph(dev_in, S)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
        finally:
            self._capturing = False
        if self.cfg.verbose:
            _log(f"capture S={S}: end_trace_capture done, synchronizing")
        ttnn.synchronize_device(self.device)
        if hasattr(ttnn, "mark_corruptible"):
            # Other traces may overwrite these; we always write the input before use and read
            # the outputs right after execute_trace (tt-metal TraceCorrectness pattern).
            for t in [dev_in] + self._flat_outs(outs):
                ttnn.mark_corruptible(t)
        self._traces[S] = {"tid": tid, "in": dev_in, "outs": outs}
        if self.cfg.verbose:
            _log(f"trace captured S={S}: {time.perf_counter() - t0:.1f}s ({len(self._traces)} live trace(s))")

    def _release_trace(self, S: int) -> None:
        tr = self._traces.pop(S, None)
        if tr is None:
            return
        self.ttnn.release_trace(self.device, tr["tid"])
        self._free(*self._flat_outs(tr["outs"]), tr["in"])

    def warm(self, seqs: Sequence[int]) -> None:
        """Warm-up contract: eager run at EVERY S first (compilation + implicit allocations), then
        capture one trace per S (``multi``) or none (``ondemand`` captures on first use, ``off``
        never).  Called by ``_ensure_installed`` before the server reports READY."""
        seqs = tuple(sorted(set(int(s) for s in seqs)))
        for S in seqs:
            self._eager_warm(S)
        if self.cfg.trace_mode == "multi":
            for S in seqs:
                self._capture(S)

    def __call__(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        ttnn = self.ttnn
        S = int(images.shape[1])
        host_in = self.host_input(self.host_embed(images))
        if self.cfg.trace_mode == "off":
            self._consts(S)
            x_in = ttnn.to_device(host_in, self.device)
            outs = self._graph(x_in, S)
            got = self._read(outs)
            self._free(*self._flat_outs(outs), x_in)
            return self._finish(got, images, S)
        if S not in self._traces:
            if self.cfg.trace_mode == "ondemand":
                for other in list(self._traces):
                    self._release_trace(other)
            if S not in self._per_s:
                # An eager warm allocates program-cache entries, reshape tables, prepared conv
                # weights and per-S constants.  Allocating those while other traces are live is
                # unsafe (their replay could overwrite them), so every live trace is released
                # and re-captured afterwards.  Cheap to avoid: list S in VGGT_PREWARM_SEQS.
                live = sorted(self._traces)
                _log(f"WARNING: S={S} was not pre-warmed; releasing {live}, compiling S={S}, re-capturing")
                for other in live:
                    self._release_trace(other)
                self._eager_warm(S)
                for other in live:
                    self._capture(other)
            self._capture(S)
        tr = self._traces[S]
        ttnn.copy_host_to_device_tensor(host_in, tr["in"], cq_id=0)
        ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
        got = self._read(tr["outs"])  # blocking readbacks; outputs copied to host before any other trace runs
        return self._finish(got, images, S)

    def release(self) -> int:
        """Release traces and drop every device tensor (call before ``ttnn.close_device``)."""
        n = 0
        for S in list(self._traces):
            self._release_trace(S)
            n += 1
        for attr in ("dino_blocks", "frame_blocks", "global_blocks", "cam_blocks", "heads", "interp", "cam",
                     "dino_norm", "_per_s", "row_mask", "pe_full", "gather_m", "interp_t", "interp_w", "interp_h", "interp_g"):
            if hasattr(self, attr):
                delattr(self, attr)
                n += 1
        gc.collect()
        return n
