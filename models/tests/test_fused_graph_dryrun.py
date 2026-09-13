"""Dry-run of the whole ``TtVggt`` device graph on a fake ``ttnn`` (shape rules + use-after-free
tracking, see ``fake_ttnn.py``).  No ttnn, no device, no numerics: it proves the Python control
flow, every op's shape/dtype plumbing, the per-S trace bookkeeping and the host tail
(``_finish``) for S = 1..3 and every knob combination -- the class of bug a device pass would
otherwise hit first.  Needs the upstream ``vggt`` package (random-initialised VGGT-1B, ~1.2 GB
of host RAM, a few seconds to build); skips when it is not importable or when the real ttnn is
already imported in this process.
"""
from __future__ import annotations

import os
import sys

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_CODE_ROOT, _TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
_VGGT_REF = os.environ.get("VGGT_REF") or os.path.abspath(
    os.path.join(_CODE_ROOT, "..", "..", "..", "ports", "vggt-upstream"))
if os.path.isdir(os.path.join(_VGGT_REF, "vggt")) and _VGGT_REF not in sys.path:
    sys.path.insert(0, _VGGT_REF)

import torch  # noqa: E402

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise RuntimeError("SKIP: " + msg)


def _install_fake():
    real = sys.modules.get("ttnn")
    if real is not None and not hasattr(real, "FakeDevice"):
        _skip("real ttnn already imported in this process; the dry run needs the fake")
    import fake_ttnn  # noqa: E402
    sys.modules["ttnn"] = fake_ttnn
    return fake_ttnn


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        try:
            from vggt.models.vggt import VGGT  # type: ignore
        except Exception:
            _skip("upstream vggt package not importable (set VGGT_REF)")
        torch.manual_seed(0)
        _MODEL = VGGT(enable_track=False).eval()
    return _MODEL


def _build(env):
    fake = _install_fake()
    from models.demos.vggt.tt.ttnn_vggt_fused import FusedConfig, TtVggt
    fake.Stats.reset()
    wrapper = TtVggt(_model(), fake.FakeDevice(), FusedConfig(env=dict(env, VGGT_FUSED_LOG="0")))
    return fake, wrapper


def _check_outputs(out, S):
    assert set(out) >= {"pose_enc", "pose_enc_list", "depth", "depth_conf", "world_points", "world_points_conf", "images"}
    assert tuple(out["pose_enc"].shape) == (1, S, 9)
    assert len(out["pose_enc_list"]) == 4
    assert tuple(out["depth"].shape) == (1, S, 518, 518, 1)
    assert tuple(out["depth_conf"].shape) == (1, S, 518, 518)
    assert tuple(out["world_points"].shape) == (1, S, 518, 518, 3)
    assert tuple(out["world_points_conf"].shape) == (1, S, 518, 518)
    assert torch.isfinite(out["pose_enc"]).all()


def test_dryrun_default_multi_trace_s1_s2():
    fake, w = _build({})
    w.warm((1, 2))
    assert sorted(w._traces) == [1, 2]
    ops = {S: fake.Stats.trace_ops[w._traces[S]["tid"]] for S in (1, 2)}
    # S=2 adds the 48 frame<->global relayouts and the camera-token reshape (free views only at
    # S=1), each followed by its zero fill of the tile padding (pad_value=0.0 -> one fill_pad
    # launch), plus a SECOND copy of the whole DPT chain (both heads run per frame at S=1 shapes)
    # and the 4 per-frame slices of the (S, P, 2C) concats.
    per_frame_dpt = ops_dpt = None
    launches = fake.Stats.launches
    conv = {S: 0 for S in (1, 2)}
    assert launches.count("conv2d") > 0
    assert ops[1] < ops[2] < 2 * ops[1], ops
    assert ops[2] - ops[1] > (48 + 1) * 2, ops
    assert 1800 < ops[1] < 2600, ops
    # every non-view tiled reshape is followed by its padding fill (the fake raises otherwise)
    seg = fake.Stats.launches
    assert seg.count("fill_pad") > 0 and "reshape" in seg
    for S in (1, 2):
        n0 = len(fake.Stats.launches)
        out = w(torch.rand(1, S, 3, 518, 518))
        _check_outputs(out, S)
        # a warmed S replays its trace: exactly one launch (execute_trace), no eager ops
        assert fake.Stats.launches[n0:] == ["execute_trace"], fake.Stats.launches[n0:]
    print(f"[dryrun] launches per traced forward: S=1 {ops[1]}, S=2 {ops[2]}")
    assert w.release() > 0 and not w._traces


def test_dryrun_eager_off_and_host_camera():
    fake, w = _build({"VGGT_FUSED_TRACE": "off", "VGGT_FUSED_CAMERA": "host", "VGGT_FUSED_INPUT": "tile"})
    w.warm((1,))
    assert not w._traces
    n0 = len(fake.Stats.launches)
    out = w(torch.rand(1, 1, 3, 518, 518))
    _check_outputs(out, 1)
    assert "execute_trace" not in fake.Stats.launches[n0:]
    assert "tilize" not in fake.Stats.launches[n0:] and "clone" in fake.Stats.launches[n0:]


def test_dryrun_ab_knobs_sdpa_minimal_slice():
    fake, w = _build({"VGGT_FUSED_ATTN": "sdpa", "VGGT_FUSED_MATMUL": "minimal", "VGGT_FUSED_GATHER": "slice",
                      "VGGT_FUSED_OC2_FP32": "1"})
    w.warm((1,))
    launches = fake.Stats.launches
    assert "sdpa" in launches and "minimal_matmul" in launches and "softmax" not in launches
    out = w(torch.rand(1, 1, 3, 518, 518))
    _check_outputs(out, 1)


def test_dryrun_interp_flat_and_bcast_hang_modelled():
    # `rep`: one fp32 x fp32 matmul per pass (tf32-class), `flat`: transposed formulation (plain
    # 2-D matmuls + transposes) run through the graph.
    fake, w = _build({"VGGT_FUSED_INTERP": "exact", "VGGT_FUSED_TRACE": "off"})
    w._eager_warm(1)
    assert fake.Stats.launches.count("transpose") == 0
    assert fake.Stats.launches.count("matmul") > 2 * 5 * 4 * 2   # 4 gather matmuls per pass, 2 passes, 5 upsamples, 2 heads
    fake, w = _build({"VGGT_FUSED_INTERP": "flat", "VGGT_FUSED_TRACE": "off"})
    w._eager_warm(1)
    assert fake.Stats.launches.count("transpose") == 2 * 5 * 4  # 4 transposes per upsample, 5 per head, 2 heads
    # `bcast` (the original A (1, Ho, H) @ X batch-broadcast) hangs on the p150a: the fake refuses it.
    fake, w = _build({"VGGT_FUSED_INTERP": "bcast", "VGGT_FUSED_TRACE": "off"})
    with pytest.raises(RuntimeError, match="hangs on the p150a"):
        w._eager_warm(1)


def test_dryrun_ln32_eltwise_and_prelude_fp32():
    fake, w = _build({"VGGT_FUSED_LN32": "eltwise", "VGGT_FUSED_PRELUDE": "fp32", "VGGT_FUSED_TRACE": "off"})
    w._eager_warm(2)
    assert fake.Stats.launches.count("rsqrt") == 1 + 2 * 4 * 2   # dino norm + 8 prelude LNs per frame (S=2)
    assert fake.Stats.launches.count("slice") >= 8           # fp32 prelude gathers by slice


def test_dryrun_large_n_softmax_variants_s3():
    for variant, expect in (("legacy", "exp"), ("inplace", "softmax_in_place"), ("scale_mask", "scale_mask_softmax_in_place")):
        fake, w = _build({"VGGT_FUSED_SOFTMAX": variant, "VGGT_FUSED_TRACE": "ondemand"})
        w.warm((3,))
        assert not w._traces           # ondemand captures on first use
        out = w(torch.rand(1, 3, 3, 518, 518))
        _check_outputs(out, 3)
        assert sorted(w._traces) == [3]
        assert expect in fake.Stats.launches, variant
        # global blocks at S=3 have N = 4122 >= 4000: the fused fp32 softmax must not be used there;
        # it serves the 24 DINOv2 + 24 frame + 16 camera-trunk block calls only; the graph ran 3x
        # (eager warm, the warm run on the persistent input inside _capture, the captured run)
        assert fake.Stats.launches.count("softmax") == 3 * (24 + 24 + 16), variant
        assert fake.Stats.launches.count(expect) >= 3 * 24, variant


def test_dryrun_unwarmed_s_releases_and_recaptures():
    fake, w = _build({})
    w.warm((1,))
    assert sorted(w._traces) == [1]
    out = w(torch.rand(1, 2, 3, 518, 518))   # S=2 was never warmed: release, warm, re-capture both
    _check_outputs(out, 2)
    assert sorted(w._traces) == [1, 2]
    tid1 = w._traces[1]["tid"]
    n0 = len(fake.Stats.launches)
    w(torch.rand(1, 1, 3, 518, 518))
    assert fake.Stats.launches[n0:] == ["execute_trace"] and w._traces[1]["tid"] == tid1


def test_dryrun_ondemand_switches_traces():
    fake, w = _build({"VGGT_FUSED_TRACE": "ondemand"})
    w.warm((1, 2))
    w(torch.rand(1, 1, 3, 518, 518))
    assert sorted(w._traces) == [1]
    w(torch.rand(1, 2, 3, 518, 518))
    assert sorted(w._traces) == [2]        # the S=1 trace was released before capturing S=2
    assert len(fake.Stats.trace_ops) == 1


def test_fake_reshape_requires_pad_value_on_tiled_copy():
    """The fake mirrors the tree's default-off padding fill: a tiled reshape that is not a view
    must be told to fill its padding (``pad_value``), a view needs nothing, ROW_MAJOR is free."""
    import pytest
    fake = _install_fake()
    fake.Stats.reset()
    x = fake._new((2, 1374, 1024), fake.float32)                     # (S, P, C) TILE, P % 32 != 0
    with pytest.raises(RuntimeError, match="UNFILLED"):
        fake.reshape(x, (1, 2 * 1374, 1024))
    n0 = len(fake.Stats.launches)
    fake.reshape(x, (1, 2 * 1374, 1024), pad_value=0.0)
    assert fake.Stats.launches[n0:] == ["reshape", "fill_pad"]
    v = fake.reshape(x, (2, 1, 1374, 1024), pad_value=0.0)           # same rows / last dim -> view
    assert v.buf is x.buf and len(fake.Stats.launches) == n0 + 2
    rm = fake._new((2, 1374, 1024), fake.float32, fake.ROW_MAJOR_LAYOUT)
    fake.reshape(rm, (1, 2 * 1374, 1024))                            # ROW_MAJOR: always a view
    assert len(fake.Stats.launches) == n0 + 2


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                if str(e).startswith("SKIP"):
                    print(f"{e}")
                else:
                    failures += 1
                    import traceback; traceback.print_exc()
                    print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
