"""A shape-checking stand-in for ``ttnn`` used by ``test_fused_graph_dryrun.py``.

It lets the whole ``TtVggt`` device graph run on a host WITHOUT ttnn and without a
device: every op returns a ``FakeTensor`` carrying logical shape / dtype / layout /
buffer identity, checks the argument rules the real op validates (inner dims, dtype
equality for concat and copy, tile alignment where ttnn requires it) and raises on any
use of a deallocated buffer (views share their source's buffer exactly like ttnn's
zero-cost reshapes).  Numerics are NOT modelled: ``to_torch`` returns zeros of the
logical shape.  Install with ``sys.modules["ttnn"] = fake_ttnn`` before importing the
wrapper -- never alongside the real ttnn.
"""
from __future__ import annotations

import builtins
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

TILE = 32
_counter = itertools.count()


class _DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"ttnn.{self.name}"


class _Layout:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"ttnn.{self.name}"


bfloat16 = _DType("bfloat16")
float32 = _DType("float32")
TILE_LAYOUT = _Layout("TILE_LAYOUT")
ROW_MAJOR_LAYOUT = _Layout("ROW_MAJOR_LAYOUT")
DRAM_MEMORY_CONFIG = "DRAM_MEMORY_CONFIG"
L1_MEMORY_CONFIG = "L1_MEMORY_CONFIG"


class _Buffer:
    def __init__(self):
        self.id = next(_counter)
        self.freed = False


class Shape(tuple):
    pass


class FakeTensor:
    def __init__(self, shape, dtype, layout, on_device, buf=None, sharded=False):
        self.shape = Shape(int(s) for s in shape)
        self.dtype = dtype
        self.layout = layout
        self.on_device = on_device
        self.buf = buf or _Buffer()
        self._sharded = sharded

    def is_sharded(self):
        return self._sharded

    def device(self):
        return "fake-device"

    def numel(self):
        return math.prod(self.shape)

    def padded(self):
        s = list(self.shape)
        if self.layout is TILE_LAYOUT and len(s) >= 2:
            s[-1] = -(-s[-1] // TILE) * TILE
            s[-2] = -(-s[-2] // TILE) * TILE
        return tuple(s)

    @property
    def padded_shape(self):
        return Shape(self.padded())

    def buffer_address(self):
        return self.buf.id

    def __repr__(self):
        return f"FakeTensor({tuple(self.shape)}, {self.dtype}, {self.layout.name}, {'dev' if self.on_device else 'host'})"


# ---------------------------------------------------------------------------- bookkeeping

class Stats:
    launches: List[str] = []
    capturing = False
    trace_ops: Dict[int, int] = {}

    @classmethod
    def reset(cls):
        cls.launches = []
        cls.capturing = False
        cls.trace_ops = {}


def _launch(name):
    Stats.launches.append(name)


def _alive(*ts):
    for t in ts:
        if isinstance(t, FakeTensor):
            if t.buf.freed:
                raise RuntimeError(f"use of deallocated tensor {t}")
            if not t.on_device:
                raise RuntimeError(f"host tensor {t} passed to a device op")


def _new(shape, dtype, layout=TILE_LAYOUT, sharded=False):
    return FakeTensor(shape, dtype, layout, on_device=True, sharded=sharded)


def deallocate(t, force=True):
    if isinstance(t, FakeTensor):
        t.buf.freed = True  # idempotent like Buffer::deallocate


def mark_corruptible(t):
    _alive(t)


def synchronize_device(device):
    pass


# ---------------------------------------------------------------------------- host <-> device

def from_torch(t: torch.Tensor, dtype=None, layout=ROW_MAJOR_LAYOUT, device=None, memory_config=None):
    dt = dtype or (float32 if t.dtype == torch.float32 else bfloat16)
    return FakeTensor(t.shape, dt, layout, on_device=device is not None)


def to_device(t: FakeTensor, device, memory_config=None):
    assert not t.on_device
    return FakeTensor(t.shape, t.dtype, t.layout, on_device=True)


def to_torch(t: FakeTensor):
    _alive(t)
    return torch.zeros(tuple(t.shape), dtype=torch.float32 if t.dtype is float32 else torch.bfloat16)


def copy_host_to_device_tensor(host: FakeTensor, dev: FakeTensor, cq_id=None):
    assert not host.on_device and dev.on_device
    _alive(dev)
    if (tuple(host.shape), host.dtype, host.layout) != (tuple(dev.shape), dev.dtype, dev.layout):
        raise RuntimeError(f"copy_host_to_device_tensor spec mismatch {host} vs {dev}")


def clone(t, **kw):
    _alive(t); _launch("clone")
    return _new(t.shape, t.dtype, t.layout)


# ---------------------------------------------------------------------------- data movement

def tilize_with_zero_padding(t, memory_config=None, use_multicore=True, dtype=None):
    _alive(t); _launch("tilize")
    if t.layout is not ROW_MAJOR_LAYOUT:
        raise RuntimeError("tilize_with_zero_padding needs ROW_MAJOR input")
    return _new(t.shape, dtype or t.dtype, TILE_LAYOUT)


def to_layout(t, layout, **kw):
    _alive(t); _launch("to_layout")
    return _new(t.shape, t.dtype, layout)


def typecast(t, dtype, **kw):
    _alive(t); _launch("typecast")
    return _new(t.shape, dtype, t.layout)


def sharded_to_interleaved(t, memory_config=None, **kw):
    _alive(t); _launch("sharded_to_interleaved")
    return _new(t.shape, t.dtype, t.layout, sharded=False)


def _is_view(old: FakeTensor, new_shape) -> bool:
    o, n = tuple(old.shape), tuple(new_shape)
    if o[-1] != n[-1]:
        return False
    if old.layout is ROW_MAJOR_LAYOUT:
        return True
    if len(o) < 2 or len(n) < 2:
        return True
    return o[-2] == n[-2] or (o[-2] % TILE == 0 and n[-2] % TILE == 0)


def _has_tile_padding(shape) -> bool:
    return len(shape) >= 2 and (shape[-1] % TILE != 0 or shape[-2] % TILE != 0)


def reshape(t, shape, pad_value=None, **kw):
    """Mirrors ``reshape.cpp`` in the gbp-tt tree: a view keeps the buffer; otherwise
    ``reshape_tiled`` runs and fills the implicit tile padding of the result ONLY when the
    caller passed ``pad_value`` (``should_fill = is_block_float || pad_value_explicit``) -- one
    extra ``fill_pad`` launch.  A tiled non-view reshape that leaves padding unfilled is an
    error here, because the fused graph keeps padded rows resident across blocks."""
    _alive(t)
    shape = tuple(int(s) for s in shape)
    if isinstance(pad_value, (tuple, list, Shape)):
        # two-shape overload ``reshape(t, logical_shape, padded_shape)``: modelled only as the
        # metadata view the wrapper uses to undo an op that reports padded rows as logical
        # (``TtVggt._unpad_rows``); the tensor's own padded shape must be passed.
        padded = tuple(int(s) for s in pad_value)
        if padded != t.padded():
            raise RuntimeError(f"two-shape reshape {tuple(t.shape)} -> {shape}/{padded}: only the tensor's own "
                               f"padded shape {t.padded()} is modelled")
        probe = FakeTensor(shape, t.dtype, t.layout, True)
        if probe.padded() != padded:
            raise RuntimeError(f"two-shape reshape: {shape} does not pad to {padded}")
        return FakeTensor(shape, t.dtype, t.layout, True, buf=t.buf, sharded=t._sharded)
    if math.prod(shape) != t.numel():
        raise RuntimeError(f"reshape {tuple(t.shape)} -> {shape}: numel mismatch")
    if _is_view(t, shape):
        return FakeTensor(shape, t.dtype, t.layout, True, buf=t.buf, sharded=t._sharded)
    _launch("reshape")
    if t.layout is TILE_LAYOUT and _has_tile_padding(shape):
        if pad_value is None:
            raise RuntimeError(f"reshape_tiled {tuple(t.shape)} -> {shape} leaves tile padding "
                               f"UNFILLED (stale DRAM): pass pad_value explicitly")
        _launch("fill_pad")
    return _new(shape, t.dtype, t.layout)


def transpose(t, d0, d1, **kw):
    _alive(t); _launch("transpose")
    dims = list(range(len(t.shape)))
    dims[d0], dims[d1] = dims[d1], dims[d0]
    return _new([t.shape[d] for d in dims], t.dtype, t.layout)


def permute(t, dims, **kw):
    _alive(t); _launch("permute")
    return _new([t.shape[d] for d in dims], t.dtype, t.layout)


def slice(t, starts, ends, step=None, pad_value=None, **kw):
    """``slice.cpp``: padding of a tiled result is undefined unless ``pad_value`` is passed, in
    which case ``fill_implicit_tile_padding`` runs when the last two dims need it."""
    _alive(t); _launch("slice")
    if len(starts) != len(t.shape) or len(ends) != len(t.shape):
        raise RuntimeError("slice rank mismatch")
    for s, e, n in zip(starts, ends, t.shape):
        if not (0 <= s < e <= n):
            raise RuntimeError(f"slice bounds {starts}:{ends} for {tuple(t.shape)}")
    shape = [e - s for s, e in zip(starts, ends)]
    if pad_value is not None and t.layout is TILE_LAYOUT and _has_tile_padding(shape):
        _launch("fill_pad")
    return _new(shape, t.dtype, t.layout)


def concat(ts, dim, **kw):
    _alive(*ts); _launch("concat")
    ref = ts[0]
    for t in ts[1:]:
        if t.dtype is not ref.dtype or t.layout is not ref.layout:
            raise RuntimeError("concat: dtype/layout mismatch")
        for i, (a, b) in enumerate(zip(t.shape, ref.shape)):
            if i != dim % len(ref.shape) and a != b:
                raise RuntimeError(f"concat: shapes {tuple(t.shape)} vs {tuple(ref.shape)}")
    shape = list(ref.shape)
    shape[dim] = builtins.sum(t.shape[dim] for t in ts)
    return _new(shape, ref.dtype, ref.layout)


# ---------------------------------------------------------------------------- eltwise

def _bcast(a, b):
    if not isinstance(b, FakeTensor):
        return tuple(a.shape)
    sa, sb = tuple(a.shape), tuple(b.shape)
    n = builtins.max(len(sa), len(sb))
    sa = (1,) * (n - len(sa)) + sa
    sb = (1,) * (n - len(sb)) + sb
    out = []
    for x, y in zip(sa, sb):
        if x != y and 1 not in (x, y):
            raise RuntimeError(f"broadcast {sa} vs {sb}")
        out.append(builtins.max(x, y))
    return tuple(out)


def _binary(name):
    def op(a, b, dtype=None, **kw):
        _alive(a, b); _launch(name)
        # mixed dtypes are allowed (the legacy port multiplies the fp32 branch by bf16 layerscale)
        return _new(_bcast(a, b), dtype or a.dtype, a.layout)
    return op


add = _binary("add")
subtract = _binary("subtract")
multiply = _binary("multiply")


def _unary(name, dtypes=None):
    def op(t, *a, **kw):
        _alive(t); _launch(name)
        if dtypes is not None and t.dtype not in dtypes:
            raise RuntimeError(f"{name}: dtype {t.dtype} unsupported")
        return _new(t.shape, t.dtype, t.layout)
    return op


exp = _unary("exp")
reciprocal = _unary("reciprocal")
relu = _unary("relu")
silu = _unary("silu")
gelu = _unary("gelu", dtypes=(bfloat16,))   # docstring: BFLOAT16 / BFLOAT8_B only


def max(t, dim=-1, **kw):  # noqa: A001
    _alive(t); _launch("max")
    return _new(list(t.shape[:-1]) + [1], t.dtype, t.layout)


def rsqrt(t, **kw):
    _alive(t); _launch("rsqrt")
    return _new(t.shape, t.dtype, t.layout)


def sum(t, dim=-1, **kw):  # noqa: A001
    _alive(t); _launch("sum")
    return _new(list(t.shape[:-1]) + [1], t.dtype, t.layout)


# ---------------------------------------------------------------------------- normalisation / softmax

def layer_norm(t, weight=None, bias=None, epsilon=1e-5, residual_input_tensor=None, compute_kernel_config=None, **kw):
    _alive(t); _launch("layer_norm")
    for g in (weight, bias):
        if g is not None:
            _alive(g)
            if g.shape[-1] != t.shape[-1]:
                raise RuntimeError(f"layer_norm gamma {tuple(g.shape)} vs input {tuple(t.shape)}")
            if g.layout is TILE_LAYOUT and g.padded()[-2] != TILE:
                raise RuntimeError("layer_norm TILE gamma must be one tile high")
            if g.dtype not in (bfloat16, float32):
                raise RuntimeError("layer_norm gamma dtype")
    return _new(t.shape, t.dtype, t.layout)


def softmax(t, dim=-1, compute_kernel_config=None, numeric_stable=True, **kw):
    _alive(t); _launch("softmax")
    return _new(t.shape, t.dtype, t.layout)


def softmax_in_place(t, compute_kernel_config=None, numeric_stable=True, **kw):
    _alive(t); _launch("softmax_in_place")
    return t


def scale_mask_softmax_in_place(t, scale=None, mask=None, compute_kernel_config=None, numeric_stable=False, **kw):
    _alive(t); _launch("scale_mask_softmax_in_place")
    return t


# ---------------------------------------------------------------------------- matmul family

def _mm_shape(a, b, name):
    sa, sb = tuple(a.shape), tuple(b.shape)
    if sa[-1] != sb[-2]:
        raise RuntimeError(f"{name}: inner dims {sa} @ {sb}")
    ba, bb = sa[:-2], sb[:-2]
    if len(ba) != len(bb):
        if all(x == 1 for x in bb) or not bb:
            batch = ba
        elif all(x == 1 for x in ba) or not ba:
            batch = bb
        else:
            raise RuntimeError(f"{name}: batch rank mismatch {sa} @ {sb}")
    elif all(x == 1 for x in ba):
        if any(x > 1 for x in bb):
            # A (batch 1) broadcast over B's batch selects the matmul in0_reuse path, which HUNG
            # the p150a on this tree (2026-09-13, probe_ops.log: A (1,37,19) fp32 @ X (38,19,256)).
            raise RuntimeError(f"{name}: first-operand batch broadcast {sa} @ {sb} hangs on the p150a "
                               "(in0_reuse matmul path); replicate A over the batch or transpose")
        batch = bb
    elif all(x == 1 for x in bb):
        batch = ba
    elif ba == bb:
        batch = ba
    else:
        raise RuntimeError(f"{name}: batch dims {sa} @ {sb}")
    return tuple(batch) + (sa[-2], sb[-1])


def matmul(a, b, compute_kernel_config=None, dtype=None, memory_config=None, **kw):
    _alive(a, b); _launch("matmul")
    if a.layout is not TILE_LAYOUT or b.layout is not TILE_LAYOUT:
        raise RuntimeError("matmul needs TILE inputs")
    if a._sharded or b._sharded:
        raise RuntimeError("fake matmul: sharded inputs not modelled")
    return _new(_mm_shape(a, b, "matmul"), dtype or a.dtype)


def linear(x, w, bias=None, compute_kernel_config=None, dtype=None, memory_config=None, **kw):
    _alive(x, w); _launch("linear")
    if x.layout is not TILE_LAYOUT:
        raise RuntimeError("linear needs a TILE input")
    if x.shape[-1] != w.shape[-2]:
        raise RuntimeError(f"linear inner dims {tuple(x.shape)} @ {tuple(w.shape)}")
    if bias is not None:
        _alive(bias)
        if bias.shape[-1] != w.shape[-1]:
            raise RuntimeError("linear bias width")
    return _new(list(x.shape[:-1]) + [w.shape[-1]], dtype or x.dtype)


class MathFidelity:
    LoFi, HiFi2, HiFi3, HiFi4 = "LoFi", "HiFi2", "HiFi3", "HiFi4"


def init_device_compute_kernel_config(arch, math_fidelity=None, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False, **kw):
    return {"fidelity": math_fidelity, "fp32": fp32_dest_acc_en}


class MinimalMatmulConfig:
    def __init__(self, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, compute_with_storage_grid_size=None):
        if subblock_h * subblock_w > 8:
            raise RuntimeError("subblock too large")
        self.blocks = (M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)


class SDPAProgramConfig:
    def __init__(self, compute_with_storage_grid_size, q_chunk_size, k_chunk_size, exp_approx_mode=None, **kw):
        if q_chunk_size % 32 or k_chunk_size % 32:
            raise RuntimeError("SDPA chunk sizes must be multiples of 32")


class UnaryOpType:
    RELU = "RELU"


class UnaryWithParam:
    def __init__(self, op, param=None):
        self.op = op


class Conv2dConfig:
    def __init__(self, activation=None, weights_dtype=None, shard_layout=None, **kw):
        self.activation = activation
        self.weights_dtype = weights_dtype


class _Experimental:
    @staticmethod
    def nlp_create_qkv_heads(x, num_heads, num_kv_heads=None, transpose_k_heads=True, **kw):
        _alive(x); _launch("nlp_create_qkv_heads")
        B, one, S, W = x.shape
        if one != 1:
            raise RuntimeError("nlp_create_qkv_heads: shape[1] must be 1")
        if x.layout is not TILE_LAYOUT:
            raise RuntimeError("nlp_create_qkv_heads: TILE input")
        kv = num_kv_heads or num_heads
        dh = W // (num_heads + 2 * kv)
        if dh * (num_heads + 2 * kv) != W:
            raise RuntimeError("nlp_create_qkv_heads: width")
        q = _new((B, num_heads, S, dh), x.dtype)
        k = _new((B, kv, dh, S) if transpose_k_heads else (B, kv, S, dh), x.dtype)
        v = _new((B, kv, S, dh), x.dtype)
        return q, k, v

    @staticmethod
    def nlp_concat_heads(x, **kw):
        _alive(x); _launch("nlp_concat_heads")
        if x.dtype not in (bfloat16, float32):
            raise RuntimeError("nlp_concat_heads dtype")
        B, H, S, dh = x.shape
        return _new((B, 1, S, H * dh), x.dtype)

    @staticmethod
    def rotary_embedding(x, cos, sin, token_index=None, **kw):
        _alive(x, cos, sin); _launch("rotary_embedding")
        X = x.padded()[-1]
        if not (X == TILE or X % (2 * TILE) == 0):
            raise RuntimeError("rotary_embedding: X must be 32 or a multiple of 64")
        if cos.dtype is not sin.dtype or tuple(cos.shape) != tuple(sin.shape):
            raise RuntimeError("rotary_embedding: cos/sin mismatch")
        if cos.shape[0] != 1 or cos.shape[1] != 1 or cos.shape[-1] != x.shape[-1]:
            raise RuntimeError(f"rotary_embedding: cos shape {tuple(cos.shape)} for input {tuple(x.shape)}")
        if cos.padded()[-2] < x.padded()[-2]:
            raise RuntimeError("rotary_embedding: cos seq shorter than input seq")
        if x.layout is not TILE_LAYOUT or cos.layout is not TILE_LAYOUT:
            raise RuntimeError("rotary_embedding: TILE only")
        # compute_output_specs returns the PADDED shape (seq_len rounded up to 32) as the
        # logical shape (measured: (2,16,1374,64) -> (2,16,1376,64), probe_ops.log).
        return _new(list(x.shape[:-2]) + [x.padded()[-2], x.shape[-1]], x.dtype)

    @staticmethod
    def minimal_matmul(a, w, bias_tensor=None, config=None, compute_kernel_config=None, dtype=None, **kw):
        _alive(a, w); _launch("minimal_matmul")
        if a.dtype is not w.dtype:
            raise RuntimeError("minimal_matmul: activation/weight dtype must match")
        if len(w.shape) > 2 and any(d != 1 for d in w.shape[:-2]):
            raise RuntimeError("minimal_matmul: weight leading dims must be 1")
        if bias_tensor is not None:
            _alive(bias_tensor)
            if any(d != 1 for d in bias_tensor.shape[:-1]) or bias_tensor.shape[-1] != w.shape[-1]:
                raise RuntimeError("minimal_matmul: bias shape")
        return _new(list(a.shape[:-1]) + [w.shape[-1]], dtype or a.dtype)


experimental = _Experimental()


class _Transformer:
    @staticmethod
    def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, scale=None, program_config=None, compute_kernel_config=None, **kw):
        _alive(q, k, v); _launch("sdpa")
        for t in (q, k, v):
            if t.dtype is not bfloat16:
                raise RuntimeError("SDPA: bf16/bf8/bf4 inputs only")
            if t.layout is not TILE_LAYOUT or t._sharded:
                raise RuntimeError("SDPA: TILE interleaved")
        if tuple(k.shape) != tuple(v.shape) or tuple(q.shape)[:2] != tuple(k.shape)[:2] or q.shape[-1] != k.shape[-1]:
            raise RuntimeError("SDPA shapes")
        return _new(q.shape, q.dtype)


transformer = _Transformer()


# ---------------------------------------------------------------------------- conv

def _conv_common(x, weight_tensor, bias_tensor, in_channels, out_channels, batch_size, input_height, input_width):
    _alive(x)
    if isinstance(weight_tensor, FakeTensor) and weight_tensor.buf.freed:
        raise RuntimeError("conv: freed weight")
    if tuple(x.shape)[-1] != in_channels or x.numel() != batch_size * input_height * input_width * in_channels:
        raise RuntimeError(f"conv input {tuple(x.shape)} vs N={batch_size} H={input_height} W={input_width} C={in_channels}")
    if x.dtype not in (bfloat16, float32):
        raise RuntimeError("conv input dtype")
    if Stats.capturing and not weight_tensor.on_device:
        raise RuntimeError("conv2d given a HOST weight inside trace capture (upload would not be replayed)")
    pw = _new(weight_tensor.shape, bfloat16) if not weight_tensor.on_device else weight_tensor
    pb = None
    if bias_tensor is not None:
        pb = _new(bias_tensor.shape, bfloat16) if not bias_tensor.on_device else bias_tensor
    return pw, pb


def conv2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width,
           kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias_tensor=None, conv_config=None,
           compute_config=None, dtype=None, memory_config=None, return_output_dim=False, return_weights_and_bias=False, **kw):
    _launch("conv2d")
    pw, pb = _conv_common(input_tensor, weight_tensor, bias_tensor, in_channels, out_channels, batch_size, input_height, input_width)
    Ho = (input_height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    Wo = (input_width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    out = _new((1, 1, batch_size * Ho * Wo, out_channels), dtype or input_tensor.dtype, TILE_LAYOUT, sharded=True)
    res = (out,)
    if return_output_dim:
        res += ((Ho, Wo),)
    if return_weights_and_bias:
        res += ((pw, pb),)
    return res if len(res) > 1 else out


def conv_transpose2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width,
                     kernel_size, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1,
                     bias_tensor=None, conv_config=None, compute_config=None, dtype=None, memory_config=None,
                     mirror_kernel=False, return_output_dim=False, return_weights_and_bias=False, **kw):
    _launch("conv_transpose2d")
    pw, pb = _conv_common(input_tensor, weight_tensor, bias_tensor, in_channels, out_channels, batch_size, input_height, input_width)
    Ho = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    Wo = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
    out = _new((1, 1, batch_size * Ho * Wo, out_channels), dtype or input_tensor.dtype, TILE_LAYOUT, sharded=True)
    res = (out,)
    if return_output_dim:
        res += ((Ho, Wo),)
    if return_weights_and_bias:
        res += ((pw, pb),)
    return res if len(res) > 1 else out


# ---------------------------------------------------------------------------- trace

_trace_ids = itertools.count(100)


def begin_trace_capture(device, cq_id=None):
    if Stats.capturing:
        raise RuntimeError("nested trace capture")
    Stats.capturing = True
    Stats._trace_start = len(Stats.launches)
    return next(_trace_ids)


def end_trace_capture(device, trace_id, cq_id=None):
    Stats.capturing = False
    Stats.trace_ops[trace_id] = len(Stats.launches) - Stats._trace_start


def execute_trace(device, trace_id, cq_id=None, blocking=True):
    if trace_id not in Stats.trace_ops:
        raise RuntimeError("execute_trace: unknown trace id")
    _launch("execute_trace")


def release_trace(device, trace_id):
    Stats.trace_ops.pop(trace_id)


class FakeDevice:
    def arch(self):
        return "blackhole"

    def compute_with_storage_grid_size(self):
        return (13, 10)
