# VGGT on p150a — Remaining Optimization Plan

Current state: 1294 ms / frame, 0.773 fps, PCC 0.9957 (B=1 S=1 518×518, best-of-3).
Baseline (pure torch CPU reference): 5037 ms / frame, 0.1985 fps.
Total gain so far: +289% (~3.9× speedup).

Ports already on p150a (commit range `c7d238e..2b2bf2d` on `changh95/vggt`):
- Full transformer Block: norm1, qkv, q_norm/k_norm, Q·Kᵀ, softmax, ·V, merge_heads, proj, ls1, residual_add, norm2, fc1, gelu, fc2, ls2, residual_add. fp32 residual accumulator, fp32 softmax intermediate, HiFi4 + fp32 dest on proj/fc2/output_conv2.
- 2D RoPE with precomputed cos/sin tables.
- DPTHead.scratch.output_conv2 (3×3 128→32 + relu + 1×1 32→output at 518×518).

Still on host (in priority order by remaining CPU time).

---

## Bug fixes (correctness — block further eval)

### ~~BF0 — S>2 first-forward compile stall~~ ✅ RESOLVED

**Root cause (verified):** `ttnn.softmax(fp32_tensor, dim=-1)` hangs
indefinitely on Blackhole when the sequence length N ≥ ~4122. This is
**not** a general compile-time explosion — it is a single-op hang that
blocked every S≥3 forward (global-attention N = S × 1374; S=3 → N=4122,
S=4 → N=5496). Earlier hypotheses about cumulative matmul compile time
were incorrect.

**Probing (conducted with `VGGT_BLOCK_TRACE=1`):**
- Frame attention (N=1374): 0.05s per block ✓
- Global attention (N=5496): hung indefinitely at the softmax line.
- Standalone probe confirmed: `ttnn.softmax(fp32, (1,16,5496,5496))` hangs.
  `ttnn.softmax(bf16, ...)` without kcfg works (0.009s), but gives PCC
  0.9597 per-step → end-to-end S=4 PCC 0.9867 (below 0.99 threshold).
- Manual decomposition `max / subtract / exp / sum / reciprocal` all work
  in fp32 at N=5496; combined they give PCC 1.000 vs fp32 reference.

**Fix (landed):** In `tt_block_forward`, when N ≥ 4000 (global-attn only),
replace the fused `ttnn.softmax` with the stable manual decomposition:
```
_sm    = ttnn.reshape(ttnn.max(tt_scores, dim=-1), (B,H,N,1))
_e     = ttnn.exp(ttnn.subtract(tt_scores, _sm))
_es    = ttnn.reshape(ttnn.sum(_e, dim=-1), (B,H,N,1))
tt_attn = ttnn.multiply(_e, ttnn.reciprocal(_es))
```
The fp32 broadcast mask-add that was previously reported to hang was also
re-probed on a clean chip and works fine (chip-state artifact, not an op
limitation). Mask is now applied in fp32 before the decomposed softmax.

**Results after fix:**

| S | VGGT_S_CANON | PCC    | Status | Throughput |
|---|-------------|--------|--------|-----------|
| 1 | (not needed) | 0.9997 | PASS | 0.32 fps |
| 2 | (not needed) | 0.9967 | PASS | 0.41 fps |
| 3 | 3            | 0.9989 | PASS | 0.68 fps |
| 4 | 4            | 0.9981 | PASS | 0.64 fps |

For S≥3 set `VGGT_S_CANON=S` (e.g. `VGGT_S_CANON=4 python3 test_vggt.py
--seq 4`). First-forward at the new S prewarms in ~8s (was 20+ min hang).
The `_install_ttnn_aggregator_padding` + `_ACTIVE_GLOBAL_MASK` infrastructure
remains but is only needed when S_real < S_canon (padding path).

### BF1 — Hard-kill corrupts the device mesh

Any ungraceful exit of a ttnn process (SIGKILL, OOM, stack overflow)
leaves the target ASIC with ETH heartbeat stuck at `post code: c0de0000`,
cascading to topology discovery failures across all 4 chips on the host.
Recovery requires `tt-smi -r 0 1 2 3` which bounces the whole mesh and
disturbs other experiments' sessions (pi0, medgemma, mast3r all run on
chips 0-3 on this box). **Per-chip `tt-smi -r N` is not sufficient** —
verified on the post-BF0-kill wedge: `tt-smi -r 2` re-ran re-init but
the heartbeat check kept failing on the same ASIC.

Python-level `SIGINT`/`SIGTERM` handlers (test_vggt.py + eval_vggt.py)
are wired but **do not fire during the exact scenario we need**, because
the hang is inside a deep C++ ttnn-compile call and Python's signal
dispatcher can't run until C returns. This means ordinary benign exits
(Ctrl-C during Python code) close cleanly, but the BF0-style compile
hang still forces SIGKILL → chip wedge. Handler plumbing is still worth
having for the benign paths.

### Plan (remaining)

1. ~~Register a `signal.SIGTERM` / `SIGINT` handler in `test_vggt.py` and
   `eval_vggt.py`.~~ **Done** — closes the device on benign exits; does
   not help compile-stall kills (see above).
2. ~~Add a `try: ... finally: ttnn.close_device(device)` at the topmost
   script level.~~ **Done** — `_close_once()` helper shared between
   finally-block and signal handler.
3. **Still open:** prevent the BF0 compile stall in the first place
   (see option 2 in BF0) so operators never need SIGKILL. This is the
   real fix; the cosmetic shutdown handlers are a secondary measure.
4. **Still open:** document the recovery command in
   `co3d_eval_results.md` / `README` so operators don't escalate to a
   full host reboot, AND warn that per-chip `-r N` doesn't recover —
   only the full 4-chip reset does.
5. **Still open:** investigate whether a ttnn compile timeout exists
   that could watchdog the forward and abort before the wedge. If not,
   consider a Python-side watchdog thread that sends SIGKILL to self if
   a single op takes >N seconds, so the abort happens before we reach
   the stuck state (rather than after).

---

## P0 — Device-native DPT `scratch_forward` refinenets ✅ DONE

Landed on main with `VGGT_TT_SCRATCH=1` as default. Measured S=1
best-of-3: **1343 ms vs baseline 1466 ms = 8.4 % wall-clock win**.
PCC 0.9957 (baseline 0.9959; Δ = 0.0002 on min-channel PCC, well above
the 0.99 threshold). S=2 regresses by ~7 % because host-upsample
round-trip volume scales linearly and the per-frame conv compute win
doesn't fully absorb it — tracked as an optimisation follow-up below.

Install + patch infrastructure lives in `_install_ttnn_dpt_scratch()`
with `VGGT_TT_SCRATCH_COMPARE=1` as a live PCC harness. Helpers follow
mast3r's layout pattern from
`/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r/tt/ttnn_dust3r.py:813+`
(`_tokens_to_nhwc`, `_flat_to_nhwc`, `_nhwc_to_flat`, `_linear_1x1`,
`_conv2d`, `_resconv`).

### The bug that ate 3 hours

`_resconv`'s residual add looked like
`return ttnn.add(tt_x, tt_c2)` — mirroring a textbook ResNet block.
But VGGT's `ResidualConvUnit` uses `nn.ReLU(inplace=True)`:

```python
def forward(self, x):
    out = self.activation(x)   # inplace: x is now relu(x)
    out = self.conv1(out); ...
    out = self.conv2(out)
    return self.skip_add.add(out, x)  # x here is RELU(x_original)
```

So the math the reference host forward is computing is
`conv2(relu(conv1(relu(x)))) + relu(x)`, not `... + x`. My device
`_resconv` needed `ttnn.add(tt_relu, tt_c2)` where `tt_relu` is the
*first* relu's output. This masqueraded as an alias/corruption bug
because the compare harness I wrote *also* relied on the in-place-
mutated `layer_4_rn_host` as the reference, which made the device
output look catastrophically wrong. Once the compare harness was
rewritten to use a `.clone()` + `F.relu` (out-of-place), per-op PCCs
went to 1.0 and only the end-to-end number stayed bad — at which
point the missing `relu(x)` in the device residual became obvious.

### Precision knobs used

Conf channels feed `expp1` in `activate_head` and are unusually
precision-sensitive. The settled configuration:

- `_conv2d` calls inside `_resconv` request `dtype=ttnn.float32` so
  their outputs stay in fp32.
- `tt_relu` is cast to fp32 before `ttnn.add(tt_relu, tt_c2)` so the
  residual add is fp32 end-to-end.
- `_linear_1x1` out_conv inside `_refinenet_device` requests fp32
  output so the next refinenet reads an fp32 tensor.
- `output_conv1` (final scratch conv) emits fp32 too — feeds host
  `custom_interpolate` → existing output_conv2 port → `expp1`.
- ttnn.upsample bilinear requires bf16 input on Blackhole, so
  `_refinenet_device` casts back to bf16 just before upsample.
- **Host upsample** (torch `F.interpolate` fp32 bilinear) is used for
  every refinenet, not just the non-integer 19→37 one. Device bf16
  bilinear drops conf PCC to ~0.955 — good for depth / world_points
  but below the 0.99 threshold. Host upsample brings it back to 0.9957.

### Known-good / remaining knobs

- Current S=1 win: **+8.1 %** (1466 → 1348 ms best-of-3). Not yet
  the P0-expected ~150 ms because host upsample round-trips eat
  most of the saved device compute.
- At Bs > 1 the port regresses (host-upsample round-trip volume
  scales linearly, per-refinenet ttnn Python overhead compounds).
  **Gated off automatically for Bs > 1** — falls back to the
  original host `scratch_forward`, so S=2 matches baseline
  (2399 ms ≈ 2392 ms noise) at PCC 0.9981. Opt back in at Bs > 1
  via `VGGT_TT_SCRATCH_ALL_BS=1` if testing a fix.
- Future S>1 speedup paths (in order of ease):
  1. **fp32 device bilinear upsample.** The current 0.955 PCC
     collapse is from the bf16 input the Blackhole bilinear kernel
     demands. If ttnn gains a fp32 bilinear path on Blackhole, we
     can drop the host round-trip entirely and both S=1 and S=2
     should win.
  2. **Stream the host upsample.** The current code does the
     download → torch F.interpolate → upload serially per
     refinenet. Overlapping the next refinenet's compute with the
     previous upsample's upload would hide the transfer. Needs
     async ttnn, not just PCIe.
  3. **Pre-sharded matmul/upsample program_config** (the same
     mast3r-style trick listed in the precision-follow-ups
     section). Likely removes most of the per-call Python
     overhead that compounds at Bs > 1.

**What's verified correct:**
- `layer{1..4}_rn`: PCC = 1.0000 at all 4 spatial sizes (148/74/37/19)
  immediately after the conv2d.
- Each primitive in `_resconv` in isolation: `relu(x)`, `conv1`, `relu`,
  `conv2` — PCC = 1.0000 when measured right after each op against
  the host-rebuilt reference.

**What's broken (root cause still open):**

Tracing with per-op `ttnn.to_torch` probes revealed that `tt_x` (the
residual tensor, also the first arg to `ttnn.add(tt_x, tt_c2)`) is
**destructively aliased** somewhere between `layer_rn` output and the
final add:

```
[scratch compare] layer4_rn:                     PCC=1.0000   (pristine)
[scratch compare] step1 relu:                    PCC=1.0000
[scratch compare] step2 conv1:                   PCC=1.0000
[scratch compare] step3 relu2:                   PCC=1.0000
[scratch compare] step4 conv2:                   PCC=1.0000
[scratch compare] tt_l4 pre-add:                 PCC=0.3365   (corrupted!)
[scratch compare] tt_c2 pre-add:                 PCC=1.0000
[scratch compare] step5 add (raw):               PCC=0.3475   (bad input → bad output)
[scratch compare] step5 add (fresh tt_l4 upload): PCC=1.0000   (proves corruption)
```

- `tt_l4`'s Python ref is live across the chain; nothing visible mutates it.
- Forcing TILE↔ROW_MAJOR layouts, `ttnn.sharded_to_interleaved`,
  `ttnn.to_memory_config(DRAM_MEMORY_CONFIG)`, `ttnn.synchronize_device`
  between ops, and `ttnn.clone(tt_x)` **all fail** to preserve `tt_l4`.
- The **only** workaround that gives PCC 1.0 on the add in the probe
  harness is re-uploading from a host-side copy of the *host-computed*
  value (`layer_4_rn_host` from torch, not `ttnn.to_torch(tt_l4)`),
  which proves the corruption is in the original device buffer itself,
  not in Python bookkeeping.
- Disabling the program cache via `VGGT_NO_PROGRAM_CACHE=1` (calling
  `device.enable_program_cache()` is skipped) does **not** help —
  end-to-end PCC stays broken. Rules out program-cache-driven reuse.
- Pulling `tt_l4` to host inside `_resconv` via `ttnn.to_torch(tt_x)`
  at the top + `synchronize_device` + `from_torch` re-upload and doing
  the add on the fresh buffer **still** gives the same bad end-to-end
  PCC. Either the device→host copy itself is incomplete (stale read)
  or the fresh-upload buffer gets aliased too.
- Even a full host-side add (`host_x + host_c2` → re-upload) gives
  the same bad PCC, confirming the downstream pipeline is consuming
  a corrupt residual-path output regardless of how the residual is
  combined.

### Next debug steps (assume ttnn ≥ tt-metal-level understanding)

1. **Repro in a minimal standalone script.** Strip out everything except:
   upload a (1, 256, 19, 19) tensor → `ttnn.conv2d` (just the 3×3 with bias) →
   `ttnn.relu` → `ttnn.conv2d` → `ttnn.add(original, last)`. If this
   minimal repro fails the same way, file it upstream.
2. **Investigate program-cache aliasing.** ttnn caches programs on
   `(shape, layout, memory_config)`. Two ops with the same cache entry
   may re-use device scratch regions. Try disabling the program cache
   (`device.disable_and_clear_program_cache()`) and see if
   `_resconv` starts producing correct output.
3. **Check whether conv2d's prepared weight tensor aliases the input.**
   ttnn.conv2d creates an internal "prepared weights" buffer the first
   time a new (H, W, C, stride) is seen. Maybe that allocation
   intersects `tt_l4`'s buffer on Blackhole's allocator strategy.
4. **Cross-reference with a Blackhole-native mast3r run.** mast3r's
   `_resconv` is believed to work on Blackhole — verify, then diff
   allocation / memory-config flows.
5. **Fall back to host add if all else fails.** Running the residual
   on host costs ~3 µs of L2/DRAM bandwidth per refinenet (184 KB at
   (19,19); larger but still small at 148). A host-add `_resconv` gives
   up some device efficiency but lets us ship a PCC-clean port in the
   meantime while upstream investigates the alias.

### Harness available

- `VGGT_TT_SCRATCH=1` enables the port at `_ensure_installed` time.
- `VGGT_TT_SCRATCH_COMPARE=1` prints per-op PCC probes against a
  synchronously-computed host reference; any regression (or recurrence
  of this alias bug) shows up immediately.

Reference working pattern at `/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r/tt/ttnn_dust3r.py:811+` (`_conv2d`, `_resconv`, `_tokens_to_nhwc`, `_flat_to_nhwc`, `_nhwc_to_flat`, `_linear_1x1`).

### Plan

1. Copy mast3r's layout helpers verbatim (`_tokens_to_nhwc`, `_flat_to_nhwc`, `_nhwc_to_flat`, `_linear_1x1`, `_conv2d`, `_resconv`). They explicitly switch layouts at the right places:
   - Before `ttnn.conv2d`: ROW_MAJOR flat `(1, 1, B·H·W, C)`.
   - Before `ttnn.linear`: TILE flat `(1, 1, B·H·W, C)`.
   - Before `ttnn.upsample`: ROW_MAJOR NHWC `(B, H, W, C)`.
   - Between convs: TILE flat is fine for `ttnn.relu` / `ttnn.add`.
2. Implement `scratch_forward` on device in this order:
   1. `layer{1,2,3,4}_rn` — 3×3 convs (bias=False, 256→256, 512→256, 1024→256, 1024→256) at sizes (148,74,37,19).
   2. refinenet4 (`has_residual=False`): ResCU2 → bilinear upsample (19→37, non-integer, host round-trip or use `ttnn` nearest-neighbour fallback + sharpness loss check) → 1×1 out_conv.
   3. refinenet3/2/1: ResCU1(`layer{3,2,1}_rn`) + skip add + ResCU2 → 2× bilinear upsample (integer, `ttnn.upsample` directly) → 1×1 out_conv.
   4. `output_conv1` (3×3 256→128).
3. **Non-integer upsample for refinenet4 (19→37):**
   - Option A: host round-trip on that one 92 KB tensor (cheap, already implemented in the discarded commit).
   - Option B: explore `ttnn.upsample` with float scale factor in nearest mode + corrective interpolation (will shift PCC, must verify).
   - Option A is safer first.
4. Gate the port behind `_tt_scratch_ready` and fall back to original `scratch_forward` if any assertion fails — lets us roll out incrementally.
5. Debugging aid: add a `_TT_SCRATCH_COMPARE=1` env var that runs both device and host `scratch_forward` and prints PCC per refinenet output. This is the fastest way to isolate the layout bug.

### Expected gain
- Wall-clock: ~150–300 ms if conv2d fidelity at 37–148 spatial is reasonable on p150a. Could be break-even if the chain of small convs is overhead-bound.
- Regardless of wall-clock, it moves ~30 convs per head off host.

---

## P1 — DPT per-layer prelude (`norm` + `projects` + `resize_layers`) ⚠️ IMPLEMENTED, NO GAIN

**Cold-bench CPU cost:** ~67 ms × 2 heads = 134 ms (measured with `torch.randn` + fresh alloc).
**Hot-path CPU cost in actual pipeline:** ~20 ms (L3-hot data, vectorized, pipeline-overlapped).

### What was built

`_install_ttnn_dpt_prelude` + `_dpt_prelude_on_device` live in `ttnn_vggt.py`.
For each `dpt_idx` (0–3), the device path runs:
1. Upload tokens `(Bs, 1369, 2048)` → bfloat16 TILE
2. `ttnn.layer_norm`
3. `ttnn.linear` (1×1 project, 2048→out_c)
4. `ttnn.add` pos_embed `(1, 1369, out_c)` broadcast (precomputed at install)
5. `ttnn.conv_transpose2d` (dpt_idx=0: k=4,s=4; dpt_idx=1: k=2,s=2) or Identity or `ttnn.conv2d` stride=2
6. Download resize output → CPU → `scratch_forward` re-uploads it

**Measured result at S=1:**

| Metric     | baseline | P1 (VGGT_TT_PRELUDE=1) | Δ |
|------------|---------|------------------------|---|
| latency_ms | 1389    | 1395                   | +6 ms |
| PCC        | 0.9957  | 0.9957                 | 0.0 |
| status     | PASS    | PASS                   | — |

**Why no gain:** Hot-path CPU prelude (~20 ms) ≈ device prelude overhead (8 uploads ×
5.6 MB + conv_transpose2d + 8 downloads × 5–11 MB = ~25 ms). The PCIe round-trips
cancel the compute savings. The cold-bench 134 ms estimate was unrepresentative
(cache-cold allocs, no PyTorch threadpool warmup).

**Disabled by default.** Enable via `VGGT_TT_PRELUDE=1` to test.

### How to actually fix it

The +6 ms regression comes from downloading resize outputs and then re-uploading them
in `scratch_forward`. Eliminating the download+re-upload would save ~80 MB PCIe per
forward (~16 ms at 5 GB/s), giving ~10 ms net win. Implement via a `_TTDeviceFeature`
wrapper: `_dpt_prelude_on_device` returns a device tensor; `_layer_rn` in
`tt_scratch_forward` detects it and skips the upload step. Not implemented because
the expected gain (~10 ms, <1%) doesn't justify the complexity.

---

## On-device residual stream (`_TTPassed`) ✅ DONE

**Measured S=1:** 1294 ms vs 1389 ms baseline — **−95 ms (−6.8%)**, PCC 0.9957 unchanged.

**Root cause addressed:** Each of the 264+ Block.forward calls (DINOv2 24 + frame 24 + global 24
+ camera head 16 + …) previously did a full PCIe upload + download round-trip regardless of
whether the result was immediately consumed by the next block. For S=1 the frame↔global reshape
between blocks is a NOP `(1,1374,1024)→(1,1374,1024)`, so adjacent blocks see the same shape
and the residual accumulator can stay on device.

**Implementation:** `_TTPassed` proxy class wraps a device `ttnn.float32` tensor and exposes
`.shape`, `.dtype`, `.view()`, `.reshape()`, and `__torch_function__` for `torch.cat` and
`F.layer_norm`. In `tt_block_forward`:
- If `_tt_can_pass=True` on the block AND `isinstance(x, _TTPassed)` AND
  `x._logical_shape == x._shape_3d` (shape unchanged since last block) → skip the
  `ttnn.from_torch` upload; re-use the previous block's output tensor directly.
- Return `_TTPassed(tt_x, orig_dtype, (B,N,C))` instead of downloading.

**Blocks marked `_tt_can_pass=True`:** aggregator `frame_blocks` + `global_blocks` (48 blocks),
DINOv2 `patch_embed.blocks` (24 blocks). Camera head trunk blocks are not marked (small N=5,
negligible PCIe).

**DINOv2 fix:** `NestedTensorBlock.forward(x_or_x_list)` has `isinstance(x, Tensor)` guard that
raises `AssertionError` for non-Tensor inputs. Patched at class level in `_install_ttnn_block`
to short-circuit to `tt_block_forward(self, x)` when `isinstance(x, _TTPassed)`.

**S>1 behaviour:** For S>1 (canonical padding mode) the frame→global reshape changes shape
`(B*S,P,C)→(B,S*P,C)`. In that case `_logical_shape != _shape_3d` so `_use_pass=False` and the
block materializes the `_TTPassed` (download + re-upload). Only DINOv2 blocks save PCIe at S>1.

**PCIe savings (S=1):**
- Before: 96 frame/global transfers + 48 DINOv2 transfers = 144 × 5.6 MB = 806 MB
- After: 1 agg upload + 48 cat downloads + 1 DINOv2 upload + 1 norm download = 51 × 5.6 MB = 286 MB
- Saved: ~520 MB → ~104 ms at 5 GB/s (measured: ~95 ms wall-clock, consistent with
  competing device compute and PCIe contention).

---

## P2 — `custom_interpolate` bilinear 296→518

Remaining CPU cost: ~13 ms × 2 heads = **26 ms**.

Non-integer scale (518/296 ≈ 1.75), so `ttnn.upsample` bilinear won't do it directly (requires integer).

### Plan
- Try `ttnn.upsample` with nearest mode at scale 2 (296→592), then crop/slice to 518 — changes numerics, needs PCC verification.
- Or keep on host — 26 ms is a small residual, probably not worth the precision risk given the downstream `output_conv2` + `expp1` sensitivity.

---

## P3 — `activate_head` (exp / expm1 / sign / norm)

Remaining CPU cost: **small** (<10 ms).

Precision-sensitive: the `expp1` on confidence channels is what breaks first under bf16 (saw PCC collapse to 0 in earlier autocast experiments). Port would need fp32 ttnn `exp`/`expm1` — possible but the gain is tiny.

### Plan
Low priority. Only do this after P0 is solid. Keep `activate_head` in fp32 via `ttnn.typecast` to fp32 before the `exp` family.

---

## P4 — Aggregator glue ops

Remaining CPU cost: a few ms total.

- Image normalization `(images - mean) / std`: trivial, on host.
- `slice_expand_and_flatten` for camera/register tokens: a few tensor slice/cat ops.
- `position_getter` + `pos + 1` shift + prepend zeros for special tokens: already precomputed for RoPE tables; could be unified.

### Plan
Batch all of these into a single install-time preprocessing step. Upload the normalized images + concatenated token positions in one go. Only useful once every other CPU op is gone.

---

## Precision-related follow-ups

### `ttnn.transformer.scaled_dot_product_attention` fused kernel
Retried (April 2026) with HiFi4 `compute_kernel_config` and explicit `scale=1/√Dh`.
Layer-level probe at (1,16,1374,64): PCC vs manual = **0.999958** — excellent.
End-to-end test result: **FAIL** — `pcc_world_points_conf` collapsed to **0.89**,
`pcc_depth_conf` = 0.98 (both below the 0.99 threshold).

Root cause: FlashAttention-2 computes softmax in bf16 internally even with
`fp32_dest_acc_en=True`. VGGT's conf channels pass through `expp1` in
`activate_head` which is highly sensitive to small precision errors accumulated
over 24 blocks. The fp32 Q@Kt + fp32 softmax + fp32 @V path is mandatory.

Upside: SDPA run was 1214ms vs 1389ms baseline (**−175ms, −12.6%**). That
savings exists if a fp32-accurate SDPA path ever lands on Blackhole. Track as a
long-term item; do not re-try without a fp32-softmax SDPA API change.

### `matmul.program_config` tuning
The 1024→3072 qkv and 1024→4096 fc1 matmuls are generic DRAM-interleaved runs. Pre-sharding weights and using a block-matmul program_config could improve Tensix utilisation. Worth an hour of experimentation per big matmul shape.

### L1-acc / dst_full_sync_en
Currently `packer_l1_acc=True` on HiFi4 configs. Haven't tried `dst_full_sync_en=True`. May close ~1% on each residual-path matmul.

---

## Harness / benchmark TODOs

- `test_vggt.py` reports `inference_speed = 1000 / latency_ms`, i.e. calls/second. For S>1 this understates frames/second by S. Fine for the current S=1 benchmark; fix before comparing different seq lengths.
- Results sheet (`results.tsv`) is untracked by design — keep it that way; only the commit log is canonical.
- `eval_vggt.py` single-pair-per-scene stats (S=2 → 1 pair, 3 scenes → 3 pairs) are too coarse: RRA@5° flips 0 → 100% per outlier sample. Once BF0 is fixed, bump `--num-views` to 8 for 28 pairs/scene and run over 5+ categories (each category ~190 MB, cheap to pull from the CO3Dv2 single-sequence subset). Ten categories × 8 views × 28 pairs ≈ 2240 pairs → AUC@30° has ~2% sampling uncertainty, enough to resolve the currently-observed 1.1-point port-vs-ref gap as signal rather than noise.
- `eval_vggt.py` hardcodes the CO3D→OpenCV viewpoint conversion to the PyTorch3D convention stored in `frame_annotations.jgz`. Verify on a second dataset (e.g., Re10K or ScanNet) before trusting the absolute AUC numbers — the conversion is a 3-minute hand-derivation and is easy to get subtly wrong.
- Add a Chamfer distance metric on `world_points` vs CO3D's `pointcloud.ply`. Needs median-depth rescaling since VGGT outputs are scale-ambiguous.

---

## Code hygiene backlog

- `_install_ttnn_block` currently preloads q_norm/k_norm weights even for DINOv2 blocks where they're Identity. Wastes ~2 MB. Gate on `isinstance(attn.q_norm, nn.LayerNorm)`.
- RoPE precompute now handles variable S via per-pos-shape cache (`_tt_lookup_cache`). But the cos/sin **base** tables grow on demand from `pos.max()` — for configs with larger image size this will rebuild the host-side table each time. Parameterise from `model.aggregator.{patch_size, img_size}` at install to size tables correctly up front.
- Consider splitting `ttnn_vggt.py` into `block.py`, `rope.py`, `dpt.py` once the file gets much larger — it's ~600 lines now, still manageable.
- `eval_vggt.py` loads a second full VGGT instance for the reference path (~3.6 GB weights duplicated on host). Memory-fine but slow at startup (~20 s). Could share weights between ported + reference models via a shallow wrapper since the reference just needs the unpatched `Block.forward` path, and `_tt_block_ready` gates on the instance anyway.

---

## Tracy profiling

Wrapper script at `profile_tracy.sh`. Pins to chip 2, resolves to
`medgemma/tt-metal` (Tracy-enabled build — the venv's default `.pth`
points at `pi0_5/tt-metal` which has `ENABLE_TRACY:BOOL=OFF`).

Works today:
- `tracy_profile_log_host.tracy` (~53 MB per S=1 run) — open in the
  Tracy GUI for a full flame / op-by-op / zone breakdown.
- `tracy_ops_data.csv` (~2 MB) — per-op JSON metadata, joinable via
  `GLOBAL CALL COUNT` with the device-side CSV.
- `cpp_device_perf_report.csv` — device-side kernel durations
  (column values need verification; the units / cycle-count mapping on
  Blackhole look off vs the wall-clock; usable for relative ordering
  but not absolute latency).

Known Blackhole post-processor issues (worth a tt-metal PR upstream):
- `--collect-noc-traces` triggers `TT_FATAL: Invalid NoC transfer type
  on device: 2` — `noc_xfer_type` validator at
  `tt_metal/impl/profiler/profiler.cpp:600` doesn't cover Blackhole
  event IDs. Dropped from the wrapper until fixed.
- `--profiler-capture-perf-counters=all` causes a pandas dtype error
  in `tracy/process_ops_logs.py` (tries to set int64 values on a
  "str" trace-id column). Dropped for now.
- Python post-processor `_enrich_ops_from_perf_csv` asserts
  `Device data missing: Op N not present in cpp_device_perf_report.csv
  for device 2` when joining host ops to device perf rows. Raw data
  is fine — a custom parser would round-trip it, but the first-party
  `ops_perf_results_*.csv` summary doesn't get generated. Usable as
  GUI session only.

Recommended usage:
- `bash profile_tracy.sh` → load `tracy_profile_log_host.tracy` in the
  Tracy GUI for a live, interactive device profile. Use that to pick
  which matmul / LN / softmax to target next.
- For automated op ranking, parse `cpp_device_perf_report.csv` +
  `tracy_ops_data.csv` directly (GLOBAL CALL COUNT is the join key).
