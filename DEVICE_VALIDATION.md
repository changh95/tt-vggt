# vggt-1b-p150 — device validation plan for the `TT_FUSED=1` path (branch `opt/vggt-1b-p150-megakernel`)

Sections 0-8 were written **before the device pass** (host torch proofs only,
`reports/megakernel/BRIEF.md` §0) and are kept as the plan; the section **Results (device,
2026-09-13)** at the end records what actually happened on the p150a. The plan was the
exact recipe for the pass: which commands, which numbers to expect, which gates,
what to A/B first, and the op constraints that could not be checked on the host.

The legacy path (`TT_FUSED` unset) is untouched: every change to `ttnn_vggt.py`, `app.py`,
`test_vggt.py`, `eval_vggt.py`, `make_demo.py` is additive and inert with the knob off
(`git diff tt-model-package -- code/models/demos/vggt/tt/ttnn_vggt.py` shows no removed lines).

## 0. What is on the branch

| Piece | File | Status |
|---|---|---|
| Torch-only tables + exact reformulations (B2 scale fold, B3 RoPE permutation + merged tables, token-assembly constants, B6 bilinear matrices, B7 pos-embed constants) | `code/models/demos/vggt/tt/fused_tables.py` | host-proven (18 tests) |
| Device-resident `TtVggt` wrapper: device weights, persistent per-S input, whole graph (DINOv2 → assembly → 48 blocks → 4 concats → camera head → 2 DPT heads) as ttnn calls, one metal trace per S, eager fallback, knob-gated A/Bs | `code/models/demos/vggt/tt/ttnn_vggt_fused.py` | import-checked with the host ttnn only |
| Knob dispatch (`_ensure_installed`, `vggt_forward`, `release_device_state`), `trace_region_bytes()` | `code/models/demos/vggt/tt/ttnn_vggt.py` | host-tested plumbing |
| Server: `_open_device(trace_region_size=…)` when fused, `/info` echoes the knobs, warm-up captures before READY | `code/models/server/app.py` | import surface unchanged |
| Harnesses open the device with the trace region when fused | `code/test_vggt.py`, `code/eval_vggt.py`, `code/make_demo.py` | additive |
| Host tests | `code/models/tests/test_fused_host.py` | 18 passed (see §1) |

Knobs (all read once when the wrapper is built; defaults = the port's validated numerics):

| Env | Default | Meaning |
|---|---|---|
| `TT_FUSED` | **`1` (default since the device pass)** | `0` selects the legacy monkey-patch path (byte-for-byte the 2026-09-12 shipped one) |
| `VGGT_TRACE_REGION_MB` | 1536 | `trace_region_size` passed to `ttnn.open_device` when fused (measured use: 15.8 MiB/bank for the four traces) |
| `VGGT_L1_SMALL_KB` | 64 | `l1_small_size` when fused (32 KiB legacy); conv2d keeps a sliding-window config per conv shape there |
| `VGGT_FUSED_TRACE` | `multi` | `multi` one trace per pre-warmed S · `ondemand` one live trace, re-captured when S changes · `off` eager device graph (debug / PCC compare) |
| `VGGT_FUSED_ATTN` | `matmul` | `sdpa` = B10 `ttnn.transformer.scaled_dot_product_attention` (bf16 probabilities; precision-gated) |
| `VGGT_FUSED_SOFTMAX` | `legacy` | for the N ≥ 4000 global blocks only: `legacy` 7-op fp32 decomposition · `inplace` `softmax_in_place` · `scale_mask` `scale_mask_softmax_in_place` (B11, hang risk) |
| `VGGT_FUSED_MATMUL` | `linear` | `minimal` = B8 `minimal_matmul` for qkv/fc1 (HiFi2, no fp32 acc — pinned to `ttnn.linear`'s bf16 default) and proj/fc2 (HiFi4 + fp32 acc, fp32 out) |
| `VGGT_FUSED_MM_BLOCKS` | `8,4,4,2,2` | `MinimalMatmulConfig(M,K,N,subblock_h,subblock_w)` in tiles |
| `VGGT_FUSED_SDPA_CHUNKS` | `128,256` | `SDPAProgramConfig(q_chunk_size, k_chunk_size)` |
| `VGGT_FUSED_CAMERA` | `device` | camera head in the trace (fp32 glue on device) · `host` = one `(S,1,2048)` readback, upstream torch camera head (exact) |
| `VGGT_FUSED_GATHER` | `matmul` | DPT patch-row gather: exact 0/1 bf16 HiFi4 matmul (rf-detr-proven) · `slice` = `ttnn.slice` rows 5: |
| `VGGT_FUSED_INTERP` | `exact` | B6 bilinear upsampling: `exact` = 0/1 bf16 gathers of a bf16 hi/lo split + fp32 eltwise weights (= host fp32 `F.interpolate`) · `rep` = one fp32 x fp32 matmul per pass (tf32-class input rounding; 21 % faster, synthetic PCC 0.0015-0.0027 below legacy, real images equal) · `flat` = transposed 2-D matmuls · `bcast` = the original first-operand batch broadcast -- **hangs the p150a**, documented only |
| `VGGT_FUSED_RESHAPE` | `rm` | the two layout-switching reshapes of every upsample: untilize -> RM view -> tilize (1.5 ms) · `tiled` = `reshape_tiled` + `fill_pad` (6-7 ms at 518^2), bit-identical |
| `VGGT_FUSED_LN32` | `kernel` | fp32 LayerNorm of the DINOv2 output: `kernel` = `ttnn.layer_norm` (bf16-class output on this device) · `eltwise` = fp32 eltwise with hi/lo split sums (+0.0002 PCC, +1 ms) |
| `VGGT_FUSED_PRELUDE` | `bf16` | DPT prelude dtype; `fp32` measured worse (0.9916 vs 0.9935) |
| `VGGT_FUSED_INPUT` | `rowmajor` | fp32 ROW_MAJOR upload + in-trace `tilize_with_zero_padding` · `tile` host tilize |
| `VGGT_FUSED_OC2_FP32` | `0` | `1` feeds output_conv2 fp32 instead of the legacy bf16 cast |
| `VGGT_FUSED_LARGE_SOFTMAX_N` | `4000` | row count from which the fp32 fused softmax is avoided (port's BF0) |
| `VGGT_FUSED_LOG` | `1` | `0` quiet · `1` warm-up / capture-phase prints · `2` + per-stage synchronized timings of eager runs |

`VGGT_S_CANON=N` with the knob on means "warm + capture S = 1..N" (no padding path is needed;
each S has its own trace).

## 1. Host evidence already collected (no device)

```
TREE=/home/deepgadget/experiments/gbp-tt/tt-metal
cd /home/deepgadget/experiments/tt-models/models/vggt-1b-p150/code
VGGT_REF=/home/deepgadget/experiments/tt-models/ports/vggt-upstream \
  $TREE/python_env/bin/python -m pytest models/tests/test_fused_host.py -q -p no:cacheprovider
# 18 passed in 1.68s
```

What the tests prove (torch, vs the upstream `vggt` modules): scale fold bit-identical (fp32
scores, bf16 weights and bf16 q-norm affine); permuted 64-wide standard RoPE `torch.equal` to
`RotaryPositionEmbedding2D` (frame and global tables); score invariance to the shared
permutation (max|d| 1.5e-5, |s| ≤ 49); a full aggregator attention branch in the fused
formulation vs `Attention.forward` (max|d| ≤ 2e-5·max|ref|); DINOv2 attention scores
bit-identical; token assembly `torch.equal` to `slice_expand_and_flatten` + `cat` for S=1..4;
patch-gather matrix exact; interpolation matrices ≤ 1e-12 (fp64) / ≤ 1e-4 (fp32) vs
`F.interpolate(align_corners=True)` for all five sizes, only 19→37 bf16-exact; pos-embed
tables `torch.equal` to `create_uv_grid`/`position_grid_to_embed`/`_apply_pos_embed`; a
random-weight `DPTHead._forward_impl` vs the torch mirror of the device DPT chain
(max|d| ≤ 1e-4·max|ref|, PCC > 0.999999); camera-head glue mirror (32-wide zero-padded pose
vector, expanded `modulate`) vs `CameraHead.trunk_fn`; knob plumbing (`TT_FUSED` unset → legacy
selection, `trace_region_bytes() == 0`).

Dry run of the WHOLE device graph on a fake `ttnn` (`code/models/tests/fake_ttnn.py`: shape /
dtype / layout rules of every op used, batch-broadcast matmul rule, tile-alignment checks,
view-vs-copy reshape rule with the tree's default-off padding fill (a tiled non-view reshape
without `pad_value` is an error in the fake; `pad_value` adds one `fill_pad` launch), use-after-free
tracking, trace capture bookkeeping, host-weight-inside-capture detection; no numerics):
```
VGGT_REF=... $TREE/python_env/bin/python -m pytest models/tests/test_fused_graph_dryrun.py -q -p no:cacheprovider -s
# 6 passed: S=1/2 multi-trace (replay = one execute_trace, 2239 / 2339 launches per traced forward,
#   of which 29 / 78 are the zero fills of tile padding after reshape/slice — review fix, §8),
# eager+host-camera+tile input, sdpa+minimal_matmul+slice gather+fp32 oc2, the three S=3 softmax
# variants (ondemand trace), un-warmed S (release / warm / re-capture), ondemand trace switching
```

Import check with the host ttnn (no device):
```
TT_METAL_HOME=$TREE PYTHONPATH=.:$VGGT_REF $TREE/python_env/bin/python -c \
  "import models.demos.vggt.tt.fused_tables, models.demos.vggt.tt.ttnn_vggt, models.demos.vggt.tt.ttnn_vggt_fused"
```

## 2. Reference numbers to compare against (legacy path, same weights)

* Served, `tt-model serve` of the packaged image (2026-09-12, `SERVING.md` §"Hardware-validated",
  `reports/publish-p150/vggt-1b-p150.json`): warm **S=1 forward 1661–1793 ms** (median ≈ 1720),
  **S=2 forward 2843–3198 ms**; ready in 2 m 0 s (warm-up S=1..4 1 m 47 s; per-S 49 / 15 / 20–26 / 20–27 s).
* Port author (`code/results.tsv`, `code/TODO.md`, `README.md`): `test_vggt.py` S=1 best-of-3
  1294–1640 ms, **min-channel PCC 0.9959** (synthetic), 0.9947 on real CO3D input; S=3 ≈ 4.4 s,
  S=4 ≈ 6.2 s; CO3Dv2 apple/hydrant/teddybear (+bottle/chair/laptop) **AUC@30° 86.1 vs 87.2
  reference**, Chamfer within +0.003 (`code/co3d_eval_results.md`).
* Gate (unchanged from the port): `test_vggt.py` prints `status: PASS` iff min PCC ≥ 0.99 over
  `depth, depth_conf, world_points, world_points_conf, pose_enc`; the conf channels are the
  precision-critical ones. Any change that is not bit-identical must additionally keep
  AUC@30° within ±1 of the reference on the CO3D scenes.

## 3. Order of operations on the device

Setup (host run, not the container; `SERVING.md` §1 has the venv notes):
```
ROOT=/home/deepgadget/experiments/tt-models; REPO=$ROOT/models/vggt-1b-p150
TREE=/home/deepgadget/experiments/gbp-tt/tt-metal; VGGT_REF=$ROOT/ports/vggt-upstream
export PYTHONPATH=$REPO/code:$VGGT_REF:$TREE:$TREE/ttnn:$TREE/tools
export TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE
export HF_MODEL=facebook/VGGT-1B TT_WEIGHTS_REVISION=860abec7937da0a4c03c41d3c269c366e82abdf9
export TT_METAL_TRACE_ALLOC_TRACKING=1   # step 3.3 only (trace-safety checker; read before `import ttnn`)
cd $REPO/code
```

### 3.1 Baseline (legacy, knob off) — must reproduce the port's numbers first
```
$TREE/python_env/bin/python test_vggt.py --seq 1 --runs 3            # expect PASS, pcc ≈ 0.9959, latency ≈ 1.3–1.7 s
$TREE/python_env/bin/python test_vggt.py --seq 2 --runs 3            # expect PASS
```

### 3.2 Fused, eager device graph (no trace) — isolates op errors from trace errors
```
TT_FUSED=1 VGGT_FUSED_TRACE=off $TREE/python_env/bin/python test_vggt.py --seq 1 --runs 2
```
Expected: `status: PASS`, min PCC within ±0.001 of the legacy 0.9959 (B2 is bit-identical;
B3 changes rounding at the bf16 level; the DPT chain moves from bf16-in/host-fp32-interp to
fp32 matmul upsampling — expected equal or slightly better). Latency is NOT the point here
(eager dispatch of ~2k ops plus the DPT convs); anything that fails is an op-constraint
error and lands in the table of §5. First things to try if an op rejects:
`VGGT_FUSED_GATHER=slice`, `VGGT_FUSED_CAMERA=host`, `VGGT_FUSED_INPUT=tile`.

Per-key PCC lines (`pcc_depth_conf`, `pcc_world_points_conf`, `pcc_pose_enc`) tell which
sub-graph misbehaves: pose_enc → camera head glue (`VGGT_FUSED_CAMERA=host` is exact);
conf channels → DPT chain (`VGGT_FUSED_OC2_FP32=1` to test the fp32 output_conv2 input).

### 3.3 Fused, traced (the production path)
```
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_FUSED=1 \
  $TREE/python_env/bin/python test_vggt.py --seq 1 --runs 5 --prewarm-seqs 1,2,3,4
TT_FUSED=1 $TREE/python_env/bin/python test_vggt.py --seq 2 --runs 3 --prewarm-seqs 1,2,3,4
TT_FUSED=1 $TREE/python_env/bin/python test_vggt.py --seq 4 --runs 2 --prewarm-seqs 1,2,3,4
```
Expected: PCC identical to §3.2 at the same S (a trace replays the same programs). With
`TT_METAL_TRACE_ALLOC_TRACKING=1` the checker raises on `execute_trace` if any buffer allocated
after a capture is still alive and unsafe — the wrapper follows the tt-metal FunGen pattern
(eager warm of every S first, then one capture per S, persistent inputs/outputs
`mark_corruptible`, outputs read back right after `execute_trace`), so this should stay silent.
If it raises: run with `VGGT_FUSED_TRACE=ondemand` (one live trace) to confirm the graph, then
report the traceback (`TT_METAL_TRACE_ALLOC_TRACEBACKS=1`).

Cross-S consistency check (multi-trace corruption would show up here): with all four traces
live, run S=1, then S=4, then S=1 again and compare the two S=1 outputs bit-for-bit (a small
script calling `ttnn_vggt.vggt_forward` twice with the same `images`).

Latency expectation (ESTIMATES from `reports/megakernel/vggt-1b-p150.md` §7, low–medium
confidence): S=1 forward **≈ 500–700 ms** (was ≈ 1700 served), S=4 **≈ 3.0 s** (was ≈ 6.2). The
two largest unknowns are the eager-dispatch residual (inferred ≈ 850–900 ms at S=1) and the
device cost of the 518² DPT convs (the author's "break-even vs ~454 ms CPU" was never split) —
the Tracy run in §3.5 replaces both.

Trace region: if `end_trace_capture` throws out-of-memory for the trace region, raise
`VGGT_TRACE_REGION_MB` (2048, 3072); record the smallest value that holds all four traces.

### 3.4 Server contract
```
TT_FUSED=1 VGGT_PREWARM_SEQS=1,2,3,4 $TREE/python_env/bin/python -m uvicorn --host 127.0.0.1 --port 20000 --lifespan on models.server.app:app
# expect: "[vggt] Opening device ... trace_region_size=1610612736 [TT_FUSED=1]", "[vggt-fused] eager warm S=n", "[vggt-fused] trace captured S=n" x4, "Warmup complete", "Application startup complete"
python3 $REPO/code/models/server/smoke_test.py --url http://127.0.0.1:20000
python3 $REPO/code/models/server/smoke_test.py --url http://127.0.0.1:20000 --views 2
curl -s localhost:20000/info | python3 -m json.tool | grep -A3 '"fused"'     # fused: true + VGGT_FUSED_* echo
```
Expect smoke PASS at S=1 and S=2 with the same depth statistics as the legacy run
(depth p50 ≈ 0.965, z~depth rho ≈ 0.979, fov0 ≈ 0.866); boot must stay < 30 min (tt-model
readiness timeout; legacy warm-up was 1 m 47 s, the fused one adds one capture per S plus
more conv kernels — expect a few minutes, record it). Then repeat with the packaged image
(`tt-model package` / `tt-model serve` with `TT_FUSED=1` added to `serve.env` in a scratch copy
of `tt-model.yaml` — do NOT flip the default in the repo until §4 passes).

### 3.5 Tracy profile (replaces the two estimates)
```
PROFILE_SEQ=1 TT_FUSED=1 VGGT_FUSED_TRACE=off bash $REPO/code/profile_tracy.sh   # per-op device time, eager
PROFILE_SEQ=4 TT_FUSED=1 VGGT_FUSED_TRACE=off bash $REPO/code/profile_tracy.sh
```
(`profile_tracy.sh` needs a Tracy-enabled build; the op report `ops_perf_results_*.csv` gives the
device-time split: attention matmuls/softmax vs DPT convs vs the 518² output_conv2 vs
interpolation matmuls vs reshapes.) Read off: total device time per forward (the traced
forward can not be faster than this), the cost of the 10 upsample reshape/matmul chains, and
whether the decomposed softmax at S=4 is where the eval predicted (~90 ms per global block).

Look specifically at the two interpolation matmuls per upsample (`A (1,Wo,W) @ X (S·H,W,C)` and
`A (1,Ho,H) @ Y (S,H,Wo·C)`): the tree's batch-broadcast rule (A batch = 1, B batch > 1 →
`mcast_in0=false` with in0 reuse, `matmul_program_config.cpp` ~L1213) assigns
`per_core_M = ceil(M / num_cores)` tiles with **M = Wo / Ho rows**, so these ops run on only
2–17 of the 130 cores (M = 37…518 → 2…17 row tiles) and the S>1 H-pass has N = Wo·C up to 75 776
columns per core. The §7 latency estimate did not account for this. Not a correctness issue
(the exact-fp32 proof in `test_fused_host.py` stands); if Tracy shows the 10 chains costing more
than ≈ 30 ms per forward, the fallbacks are (a) a transposed formulation that puts `S·H` /
`Wo·C` on M (`X^T @ A^T` via `permute` or a `transpose_b` matmul), or (b) bf16 `A` matrices
where bf16-exact (only 19→37 is; the others round). Also read the cost of the `fill_pad`
launches (29 per S=1 forward, ≈ 1 ms at the 35 µs floor) — they are part of the correctness
fix in §8, not an optimisation to remove.

## 4. Accuracy gates (apply to every configuration that is switched on)

1. `test_vggt.py --seq {1,2,4}`: `status: PASS`, and min PCC not more than 0.001 below the
   legacy value at the same S (legacy: 0.9959 / 0.9967 / 0.9981 per `TODO.md`).
2. Real data, `eval_vggt.py` on the CO3Dv2 scenes of `code/co3d_eval_results.md`
   (`--co3d-root` / `VGGT_CO3D_ROOT`; `VGGT_S_CANON=3` for S=3 exactly as the author ran it):
   ```
   TT_FUSED=1 VGGT_S_CANON=3 $TREE/python_env/bin/python eval_vggt.py --num-views 3 \
     --categories apple,bottle,chair,laptop,hydrant,teddybear --co3d-root $VGGT_CO3D_ROOT
   ```
   AUC@30° within ±1.0 of the reference column, mean relative PCC ≥ 0.99 on every key,
   Chamfer within +0.003 (the author's own thresholds).
3. `smoke_test.py` PASS at S=1 and S=2.

B2 / B4 / B5 / token assembly / concats are bit-identical by construction; B3 (RoPE kernel
rounding) and B6 (fp32 matmul upsampling instead of bf16 in/out host interpolation) are
bf16-round-level or better. The following moves ARE precision-affecting (they change where a
computation runs and in which dtype) and must be isolated by gate 1 explicitly — run
`test_vggt.py --seq 1` per-key PCC with the default configuration, then with
`VGGT_FUSED_CAMERA=host` to split them:

| Move | Legacy default (`TT_FUSED` unset) | Fused | Split by |
|---|---|---|---|
| DPT prelude (LN → 1x1 proj → +pos_embed → resize conv/convT) | host torch fp32 (`VGGT_TT_PRELUDE=0`, `ttnn_vggt.py:1157-1162`; the device prelude was opt-in) | device bf16: `typecast` → bf16 `layer_norm` (bf16 gamma) → bf16 `linear` → bf16 add → conv on bf16 (`_dpt_prelude`) | conf-channel PCC at S=1; the author measured `VGGT_TT_PRELUDE=1` at S=1 PCC 0.9957 = baseline (`ttnn_vggt.py:1377-1383`), so this is expected to hold, but it is a measured claim, not a proof — the host mirror in `test_fused_host.py` is fp32 torch and proves semantics only |
| DINOv2 final LN | torch fp32 on the host | device fp32 `layer_norm` with fp32 TILE gamma/beta (`_assemble`) | all keys (it feeds every aggregator block) |
| Camera-head glue (token LN, adaLN, silu, modulation linears, 4 iterations) | torch fp32 | device fp32 eltwise/linear (`_camera_device`) | `pcc_pose_enc` with `VGGT_FUSED_CAMERA=device` vs `host` (host is exact torch) |
| B7 mid-chain DPT transfers removed | bf16 round trips between device stages | fp32 stays on device | bf16-round-level or better |

Anything in §6 is precision-affecting as well and needs gates 1–3.

## 5. Op constraints NOT verifiable on the host (check in §3.2 first)

| # | Assumption in the code | Where | How to check / fallback |
|---|---|---|---|
| 1 | `ttnn.experimental.rotary_embedding` with a 4-D `[S,16,1374,64]` (frame) / `[1,16,S·1374,64]` (global) bf16 input and a `[1,1,N,64]` table; the reader wraps `cos_sin_curr_id` every `Ht` tile rows (read in `reader_rotary_embedding_interleaved_start_id.cpp`), so batch/heads re-use the table | `_attention` | §3.2 PCC; a 1-block probe vs torch if pose_enc/conf PCC drops |
| 2 | `nlp_create_qkv_heads(transpose_k_heads=False)` output feeds `layer_norm` over the last dim (64) and `rotary_embedding` directly; `permute(0,1,3,2)` on the bf16 k afterwards (legacy op) | `_attention` | §3.2 |
| 3 | `nlp_concat_heads` on the fp32 `(B,16,N,64)` context (validate allows FLOAT32) | `_attention` | §3.2 |
| 4 | fp32 `layer_norm` with fp32 TILE gamma/beta `(1,1,C)` (DINOv2 final norm, camera token/trunk norm, affine-less adaln) | `_assemble`, `_camera_device` | §3.2; fallback `VGGT_FUSED_CAMERA=host` for the camera part |
| 5 | fp32 eltwise glue on `(1,S,2048)`: `silu`, `add(scalar)`, `multiply`, `add`; fp32×fp32 `ttnn.linear` (embed_pose 32→2048, modulation 2048→6144) — HiFi4 auto-selected for fp32 inputs | `_camera_device` | pose_enc PCC; fallback `VGGT_FUSED_CAMERA=host` (exact torch) |
| 6 | `multiply(x_fp32 (S,1374,1024), mask (1,1374,1))` width+batch broadcast in binary_ng | `_assemble` | §3.2 (the legacy softmax decomposition uses the same `(B,H,N,1)` broadcast) |
| 7 | `ttnn.reshape(..., pad_value=0.0)` `(S,1374,1024) ↔ (1,S·1374,1024)` fp32 for S>1 (tile-padded → `reshape_tiled` + `fill_pad`); `(S,1,2048) → (1,S,2048)`; the upsample reshapes `(1,1,S·H·W,C) → (S·H,W,C) → (S,H,Wo·C) → (1,1,S·Ho·Wo,C)`; the softmax `max`/`sum` `(B,H,N) → (B,H,N,1)` reshapes. `fill_implicit_tile_padding` accepts FLOAT32/BFLOAT16 TILE (`fill_pad_program_factory.hpp` `data_type_to_size`) and is a device op (trace-safe) | `_reshape` (all sites) | §3.2 at S=2; cost via Tracy (29 fills per S=1 forward); fallback: `to_layout(ROW_MAJOR)` → reshape → `to_layout(TILE)` in `_upsample` (one place; a tilize always zero-pads) |
| 8 | `ttnn.concat([f, g], dim=-1)` on two fp32 `(S,1374,1024)` TILE tensors (concat dim tile-aligned) | `_aggregator` | §3.2 |
| 9 | `ttnn.slice(x, [0,0,0], [S,1,2048])` on the fp32 `(S,1374,2048)` TILE tensor (aligned begin) | `_camera_tokens` | §3.2 |
| 10 | batched fp32 matmul with the first operand broadcast over the second's batch: `A (1,Wo,W) @ X (S·H,W,C)` and `A (1,Ho,H) @ Y (S,H,Wo·C)` (matmul docstring rule 4: both rank ≥3, interleaved, non-sharded) — output dtype bf16 on the second pass; interp weights are fp32 (only 19→37 is bf16-exact) and fp32 matmul inputs may be TF32-class on the FPU | `_upsample` | §3.2 conf PCC; per-stage PCC via `VGGT_FUSED_TRACE=off` + `ttnn.to_torch` probes if needed |
| 11 | `conv2d(..., return_weights_and_bias=True)` prepared weights are valid for re-use at the same S (kept per S); `conv_transpose2d` likewise; `Conv2dConfig(activation=UnaryWithParam(RELU))` with fp32 output dtype; conv2d accepts TILE/sharded inputs from the previous conv and the fp32 activations of `_resconv` (legacy-proven) | `_conv`, `_resconv`, `_dpt_prelude` | §3.2; if a prepared weight is rejected at another S the per-S cache already isolates it |
| 12 | conv2d autoshard / DRAM slicing of the `(1,1,S·518²,128)` bf16 output_conv2 input INSIDE a trace (legacy did it eagerly from an uploaded tensor) | `_dpt_head` | §3.3; fallback: keep `VGGT_FUSED_TRACE=off` for the DPT tail only (would need a split trace) |
| 13 | `tilize_with_zero_padding` on an fp32 ROW_MAJOR `(S,1374,1024)` input (validate accepts FLOAT32; rf-detr used it on fp32) | `_graph` | fallback `VGGT_FUSED_INPUT=tile` |
| 14 | Trace memory: 4 coexisting traces of ~2k ops each in a 1536 MiB region; DRAM peak at S=4 with the decomposed softmax (two–three `(1,16,5504,5504)` fp32 tensors ≈ 1.9 GB each live at once) | `_capture`, `_softmax_rows` | §3.3; `VGGT_TRACE_REGION_MB`; `ttnn.dump_device_memory_state` |
| 15 | Warm-up time (eager warm ×4 + capture ×4, more conv kernels than legacy) < 30 min readiness timeout | `warm` | §3.4 boot log |
| 16 | Padded key columns (1374→1376, 4122→4128, 5496→5504) carry non-zero FINITE values in the fused/decomposed softmax (`ttnn.softmax` masks padding via `mask_padded_data`; the 7-op decomposition does not): the padded token rows are zeros at every relayout (tilize / `fill_pad`, see §8), so the padded k/v rows are `LN(0) = β`-derived and deterministic — the same finite contamination the legacy path had (its per-block `from_torch` zero-padded the same way; S=3/4 PCC 0.9967/0.9981 include it). What WAS new on this branch — stale, possibly NaN padding after `reshape_tiled`, persisting across all 72 blocks — is fixed by the explicit fills (§8); the `_softmax_rows` `max`/`sum` reshapes are filled too because a NaN padded row of `m` would propagate to the residual stream's padded rows → next block's `k` padded rows → `k^T` padded COLUMNS → every logical row of `probs @ v` | `_softmax_rows`, `_reshape` | §3.2 at S=3 (first S with the decomposed path); SDPA (B10) masks padded keys itself |
| 17 | Batch-broadcast interpolation matmuls `A (1,Wo,W) @ X (S·H,W,C)` / `A (1,Ho,H) @ Y (S,H,Wo·C)` auto-select `Mcast1D` with `mcast_in0=false` (in0 reuse, `matmul_program_config.cpp` ~L1213, `matmul_device_operation.cpp` L276-299) → `per_core_M = ceil(M/num_cores)` with M = Wo / Ho → only 2–17 active cores, `per_core_N` = full width (up to 75 776 columns at S=4). Valid per the tree; L1 fit of the wide `per_core_N` output block and the actual cost are unknown | `_upsample` | §3.5 Tracy; fallbacks in §3.5 (transposed formulation or bf16 A where exact) |

## 6. Knob-gated A/Bs (default off) — order and gates

Run each on top of the passing default fused configuration; keep a knob only if §4 holds.

1. **B10 SDPA**: `TT_FUSED=1 VGGT_FUSED_ATTN=sdpa test_vggt.py --seq {1,4}`. Precision-affecting
   (probabilities fed to P·V stay bf16 even with fp32_dest_acc; the port measured conf PCC 0.89
   on an older kernel). Gate: conf-channel PCC ≥ 0.99 AND CO3D AUC@30° within ±1. Expected gain
   (est.): −0.1 s at S=1, −2 s at S=4 (it removes the materialised N×N fp32 scores and the
   N ≥ 4100 fp32-softmax hang class). Try `VGGT_FUSED_SDPA_CHUNKS=96,256` if 128 is slower.
2. **B11 softmax variants** (S ≥ 3 only): `VGGT_FUSED_SOFTMAX=inplace`, then `scale_mask`, with
   `--seq 3` and `--seq 4`. Hang risk (same kernel family as the fp32 `ttnn.softmax` that hangs
   at N ≥ ~4100; a hang inside a capture wedges the chip — BF1 in `TODO.md`: recovery is
   `tt-smi -r` of the whole mesh, so run these with `VGGT_FUSED_TRACE=off` first). fp32 math,
   expected fp32-reassoc only; est. −1.2 s at S=4 if it works.
3. **B8 minimal_matmul**: `VGGT_FUSED_MATMUL=minimal` (`VGGT_FUSED_MM_BLOCKS` to tune; L1 CB fit
   for K=4096 / M up to 5504 unverified). Numerics bf16-round-level (K-blocking order; fidelity
   is pinned to `ttnn.linear`'s HiFi2/no-fp32-acc for qkv/fc1 and HiFi4/fp32 for proj/fc2).
   Speed gain unknown at these shapes (rf-detr's 3–5× was on N=384-class shapes).
4. **B9(a) `dit_minimal_matmul_addcmul_fused` with fp32 proj/fc2 weights** — NOT implemented on
   this branch (blocked as-is by `TT_FATAL(ternary_a_data_format == in1_data_format)`: the fp32
   residual would force bf16 weights → bf16 residual, the port's measured 0.978 failure). If B8
   shows the minimal kernel family pays off, prototype with fp32 `in1` weights (+20 MB DRAM per
   block) and gate as in §4.
5. `VGGT_FUSED_OC2_FP32=1` (fp32 into output_conv2): accuracy-only A/B for the conf channels;
   watch the per-core L1 of the 518² conv (2× the bf16 activation).
6. `VGGT_FUSED_CAMERA=host` vs `device`: keep whichever is faster if both pass pose_enc PCC.

## 7. If the default fused path passes §4

* Flip `serve.env` in `tt-model.yaml`: add `TT_FUSED: "1"` (and the trace region if it had to
  change), re-package, re-run §3.4 with the image, record forward/total ms at S=1..4 and the
  boot time in `SERVING.md` and the card (`README.md` "Accuracy and speed").
* Update `code/results.tsv` with one row per kept lever (commit, fps, accuracy, keep/discard),
  the port's own protocol.
* Leave the legacy path in place as the fallback (`TT_FUSED=0`) until a full CO3D re-run is on
  file.

## 8. Review fixes (host-only, 2026-09-13) — what changed after the first review and why

All findings were confirmed against the tree sources; none was disputed.

1. **Major — unfilled tile padding after `ttnn.reshape` (fixed).** In this tree
   `reshape_tiled` fills the implicit tile-padding lanes of its result only when the caller
   passes `pad_value` explicitly (`ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp`
   "Fill rules": `should_fill = is_block_float_output || (pad_value_explicit && !skip_padding_fill)`;
   the tiled path is taken whenever the last dim changes or the second-last dim changes to a
   non-tile-multiple, `this_is_view` rule ~L613). `ttnn.slice` documents its padding as
   "undefined by default" (`slice_nanobind.cpp` L62) and fills only with `pad_value`
   (`slice.cpp` L391). The legacy path never depended on this: every block re-uploaded its
   input with a host `from_torch` (zero padding, `ttnn_vggt.py:358`), so padded rows were
   re-zeroed 72 times per forward. The fused graph keeps them resident on device, and a stale
   NaN/Inf in a padded row would reach logical outputs through
   (a) K-contractions — the interpolation matmuls contract over the padded W/H rows of R1/R2
   (A's padded columns are 0, but 0·NaN = NaN in the FPU) and the 0/1 gather matmul over the
   1374→1376 padded token rows;
   (b) attention — padded token rows → `k` padded rows → `k^T` padded columns → `scores`
   padded columns → in the decomposed softmax `e = exp(scores − m)` is NaN there (the
   additive −inf mask does not neutralise NaN) → `probs @ v` contracts over them;
   (c) the `_softmax_rows` `max`/`sum` reshapes `(B,H,N) → (B,H,N,1)`: a NaN padded row of `m`
   makes `probs`/`ctx` padded rows NaN → residual stream → next block's `k` → path (b). This
   site is identical to the legacy code, but legacy re-zeroed between blocks and we do not.
   Fix: every reshape goes through `TtVggt._reshape` = `ttnn.reshape(x, shape, pad_value=0.0)`
   (a view ignores the argument; a `reshape_tiled` gets one `fill_pad` launch —
   `fill_implicit_tile_padding` supports FLOAT32/BFLOAT16 TILE and is a device op, so it is
   trace-safe), and the two slices that create fresh padded rows (`_camera_tokens`
   `(S,1,2048)`, the `VGGT_FUSED_GATHER=slice` path) pass `pad_value=0.0`. The three
   `shift/scale/gate` slices on `mod` keep their inherited padded rows (aligned begins copy
   whole tiles of a tensor whose padding is already deterministic; M-rows only). Padding lanes
   are therefore deterministic zeros at every relayout, exactly the legacy host-tilize state,
   and §5 #16 is back to "inherited" (finite `LN(0)=β`-derived padded keys, same as legacy).
   Cost: 29 `fill_pad` launches per S=1 forward (≈ 1 ms at the 35 µs floor), 78 at S=2
   (2239 / 2339 launches per traced forward in the fake-ttnn dry run). The fake `ttnn` now
   raises on any tiled non-view reshape without `pad_value`, so a regression fails
   `test_fused_graph_dryrun.py` on the host.
2. **Minor — §4 precision-affecting moves not named (fixed, doc).** The DPT prelude moves from
   the legacy host fp32 default to device bf16, the DINOv2 final LN and the camera-head glue
   from torch fp32 to device kernels; §4 now lists them with the legacy/fused dtype and how
   gate 1 isolates each (`VGGT_FUSED_CAMERA=host` splits the camera part).
3. **Minor — batch-broadcast interpolation matmuls on 2–17 cores (doc).** Recorded as §5 #17
   and as an explicit Tracy read-off in §3.5 with the two fallbacks; no code change (correct
   per the tree, perf unknown until measured).
4. **Minor — `_interleaved()` leaked the sharded original (fixed).** `_interleaved(x,
   free_input=True)` releases the sharded conv output once the DRAM copy exists; `_upsample`
   now documents that it consumes its input, and `_dpt_head` frees `raw` after the final
   relayout. Verified by the fake-ttnn use-after-free tracker (no double free: the callers
   never freed those tensors).
5. **Minor — `tt-model.yaml` verify asserted `fused_enabled({})` (fixed).** The line now calls
   `ft.fused_enabled()` so the build-time check reads the real environment of the image.


## Results (device, 2026-09-13)

One p150a, tree `/home/deepgadget/experiments/gbp-tt/tt-metal` = `v0.78.0-dev20260820-25-g8b98410e730`,
host python_env for the harness runs, the shipped image `tt-model/vggt-1b-p150:9ce60f4e98cc` with the
working tree bind-mounted for the served A/B. Every log, probe script and command is under
`/home/deepgadget/experiments/tt-models/logs/megakernel-validate/vggt-1b/`; one row per experiment in
`reports/megakernel/VALIDATION.md`. Commits on this branch: `5d6d729` (rotary shape, interp/gather
broadcast), `f37f89f` (per-frame DPT, l1_small, capture without warm run, LN32/PRELUDE knobs),
`0690ca1` (exact interpolation), `3a4bfa3` (exact default), `2c31bd3` (RM relayouts), `3b98d5c`
(`TT_FUSED` default ON, serve.env, /info hints) and the docs/card commit. Nothing merged or pushed.

### What the hardware rejected (and the fixes)

| # | Symptom on the p150a | Cause | Fix |
|---|---|---|---|
| 1 | first block: `matmul ... width=1376 height=1374` at `probs @ v` | `ttnn.experimental.rotary_embedding` returns its tile-PADDED rows as the logical shape (`compute_output_specs`: `round_up(seq_len, 32)`) | `_unpad_rows`: two-shape `ttnn.reshape(x, logical, padded)` = zero-cost view (same buffer) |
| 2 | `A (1,37,19) fp32 @ X (38,19,256)` never returned; the next `open_device` hung; `tt-smi -r 0` | a matmul whose FIRST operand has batch 1 and whose second is batched takes the tree's in0_reuse path (`matmul_program_config.cpp:1213`) and hangs; both B6 interpolation matmuls and the S>1 gather used it | replicate the constant over the bmm batch (`rep`), later `exact`; gather matrix per frame |
| 3 | `Out of Memory: ... L1_SMALL ... allocated 32704 B, free 64 B` in the S=2 eager warm after S=1 | conv2d keeps a sliding-window config per conv shape in l1_small; the on-device DPT at every S overflowed 32 KiB | `VGGT_L1_SMALL_KB` (64 when fused; 32 legacy) |
| 4 | S=2 capture: `TT_FATAL: Reads are not supported during trace capture`; the same S=2 DPT run eagerly with the S=1 trace live hung the chip twice (`tt-smi -r 0` x2) | the batched `(1,1,S*518^2,128)` output_conv2 input does not fit L1 at S>=2 (1.25 MB/core at S=1) -> conv2d DRAM op-slicing (`op_slicing.cpp` fallback to height-slicing) with host reads | the DPT heads run **per frame** at S=1 shapes (`_dpt_heads`: batch-dim slices of the 4 concats, one readback per frame); `_capture` skips its eager warm run when a trace is live |
| 5 | wp_conf PCC 0.9935 vs legacy 0.9950 at S=1 (0.0015-0.0027 below at every S) | golden injection (`probe_attrib.py`): only the upsampling matters -- the fp32 x fp32 HiFi4 matmul rounds its inputs tf32-class (rel 1e-3, `probe_ln2.log`); RoPE kernel and device resize convs neutral | `VGGT_FUSED_INTERP=exact`: 0/1 bf16 gathers of a bf16 hi/lo split + fp32 eltwise weights (= host fp32 interpolation: 0.9952) |
| 6 | the two layout-switching reshapes of the 518^2 upsample cost 6-7 ms each | `reshape_tiled` + `fill_pad` | `VGGT_FUSED_RESHAPE=rm`: untilize -> free RM view -> tilize (1.5 ms), bit-identical |

Device facts measured on the way (all in `VALIDATION.md`): the fp32 `ttnn.layer_norm` output is
bf16-class (maxabs 2.9e-2 = one bf16 rounding); fp32 eltwise mul/add/sub/rsqrt are exact; `ttnn.sum`
rounds its inputs tf32-class; bf16 x bf16 HiFi4 with fp32 output accumulates to ~2.8e-4 relative,
so a single-term 0/1 gather is exact but a two-term interpolation is not; the fused fp32 softmax
variants (`softmax_in_place`, `scale_mask_softmax_in_place`) do NOT hang at N=4122 on this tree;
DRAM 3882 MiB/bank x 8, weights+tables 332 MiB/bank, 450 MiB/bank with all four S warmed, the four
traces take 15.8 MiB/bank of the 192 MiB/bank trace region; the traced replay equals the eager device
time (355 ms at S=1: the graph is device-bound, the trace buys nothing beyond eager async dispatch).

### Gates

Legacy on this tree (`TT_FUSED` unset before the flip, `test_vggt.py`, synthetic input): S=1 min PCC
0.9950 / best-of-3 1322 ms, S=2 0.9991 / 2314, S=3 0.9978 / 4075, S=4 0.9975 / 5748 (author: 0.9959 /
1294-1640, 0.9967, 0.9981 / 6.2 s). Host tests: 18 -> 28 passed (host proofs + fake-ttnn dry run).

| Configuration (traced, all 4 traces live) | S=1 PCC / ms | S=2 | S=3 | S=4 | real-image set (mean / worst of 7 scenes) | verdict |
|---|---|---|---|---|---|---|
| legacy (`TT_FUSED=0`) | 0.9950 / 1322 | 0.9991 / 2314 | 0.9978 / 4075 | 0.9975 / 5748 | 0.99903 / 0.99791 (S=1 1529, S=2 2857, S=3 4741 ms) | reference |
| **default: exact interp, rm relayouts, matmul attention, device camera** | **0.9952 / 406** | **0.9981 / 874** | **0.9984 / 1945** | **0.9979 / 2759** | **0.99887 / 0.99772** (410 / 879 / 1832 ms) | **kept** (within 0.001 of legacy at every S; eager == traced; cross-S bit-identical) |
| `INTERP=rep` (fp32 matmul interpolation) | 0.9935 / 369 | 0.9964 / 803 | 0.9952 / 1719 | 0.9957 / 2618 | 0.99885 / 0.99754 (372 / 802 / 1711) | opt-in (fails the synthetic -0.001 clause, equal on real images) |
| `ATTN=sdpa` (B10) | 0.9957 / 243 | 0.9976 / 480 | 0.9986 / 723 | 0.9973 / 980 | 0.99793 / 0.99488 (320 / 630 / 940; depth_conf -0.0027 on the KITTI S=3 scene) | **dropped from the default** (real-data gate), knob kept |
| `MATMUL=minimal` (B8) | 0.9984 / 372 | -- | -- | 0.9986 / 2592 | 0.99897 / 0.99708 (457 / 960 / 1934, with exact) | knob only (no speed gain) |
| `SOFTMAX=inplace` / `scale_mask` (B11, S=3 eager) | -- | -- | 0.9987 / 1553 (no hang) | -- | -- | superseded by SDPA; knobs kept |
| `CAMERA=host` | 0.9952 / 584 | -- | -- | -- | -- | dropped (+214 ms) |
| `OC2_FP32=1`, `PRELUDE=fp32`, `LN32=eltwise` | 0.9935 / 388, 0.9916 / 374, 0.9937 / 372 (eager) | -- | -- | -- | -- | dropped / dropped / opt-in |

Cross-S consistency (§3.3): with four traces live every S replayed before and after the others is
bit-identical, back-to-back replays are bit-identical. The precision attribution showed that the
synthetic wp_conf proxy moves by +-0.002 with unrelated perturbations (1-ULP bf16 flips on a
noise image), so the real-image set (7 scenes, S=1..3: vggt `media/input.png`, the CO3D apple pair,
KITTI frames, moge/rf-detr/gaze-lle/hamer media) is the accuracy evidence that decided the levers.

### Served A/B (shipped image, `tt-model serve --print` flags, working tree bind-mounted)

| run | boot | 30 warm S=1 `timing_ms.forward` median / min / max | `total` | S=2 x10 forward | smoke | stop |
|---|---|---|---|---|---|---|
| legacy (`TT_FUSED` unset before the flip) | 31.9 s | 1602.1 / 1517.3 / 1673.7 | 1716.8 / 1631.7 / 1789.9 | 2887.4 / 2793.0 / 2955.5 | PASS S=1 (1548.8, depth p50 0.965, rho 0.979, fov 0.866) + S=2 (2862.5) | 1 s, "device closed", tt-smi OK |
| fused, exact interp, tiled reshapes (`--env TT_FUSED=1`, HEAD 3a4bfa3) | 70.9 s (fused kernels JIT: eager warm S=1 45.7 s) | 447.2 / 446.2 / 450.7 | 561.2 / 559.3 / 564.5 | 957.5 / 954.6 / 963.4 | PASS S=1 (449.8, p50 0.964, rho 0.979, fov 0.868) + S=2 (957.0) | 1 s clean |
| **final default, no env (HEAD 3b98d5c)** | **29.9 s** (warm cache) | **411.4 / 407.9 / 435.0** | **525.5 / 521.3 / 547.6** | **880.6 / 874.9 / 886.6** | PASS S=1 (425.7) + S=2 (882.0) | 2 s clean |
| final code, `--env TT_FUSED=0` (knob) | 29.4 s | 1581.5 / 1546.2 / 1640.5 | 1696.2 / 1660.6 / 1754.0 | 2869.1 / 2793.8 / 2961.3 | PASS S=1 (1506.7, p50 0.965) + S=2 (2937.1); `/info fused=false`; outputs **byte-identical** to the pre-flip legacy run | 2 s clean |

Malformed requests answered 400 (bad base64, 5 views, empty) / 422 (json_stride 999) in every
run; the served fused outputs agree with the legacy server's at PCC >= 0.9997 on every key
(`compare_served.py`). `encode` (host npz) is 105-125 ms in both paths.

### Unverified / open

* CO3Dv2 AUC@30 (the port's own real-data metric): the CO3D data is not on the validation host;
  the real-image set above is the substitute. Recommend a CO3D re-run before publishing.
* SDPA: fails the real-data gate by 0.0011 mean / 0.0027 worst on depth_conf while being 1.8-2.8x
  faster than the kept path at S>=2 (S=4 980 ms). A CO3D AUC evaluation could still admit it.
* Memory headroom at S=4 with SDPA or with `exact` was not measured beyond the allocator stats
  after the four captures (DRAM 450 MiB/bank of 3882).
* Served load: 30 (S=1) + 10 (S=2) + 5 warm requests per run, not >= 100.
* The `bcast` interpolation and the batched DPT at S>=2 are documented hangs on this tree; not retried.
* Three `tt-smi -r 0` resets were needed during the pass (all after hangs, none speculative).
