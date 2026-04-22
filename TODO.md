# VGGT on p150a — Remaining Optimization Plan

Current state: 1694 ms / frame, 0.59 fps, PCC 0.9959 (B=1 S=1 518×518, best-of-3).
Baseline (pure torch CPU reference): 5037 ms / frame, 0.1985 fps.
Total gain so far: +196% (~3× speedup).

Ports already on p150a (commit range `c7d238e..2b2bf2d` on `changh95/vggt`):
- Full transformer Block: norm1, qkv, q_norm/k_norm, Q·Kᵀ, softmax, ·V, merge_heads, proj, ls1, residual_add, norm2, fc1, gelu, fc2, ls2, residual_add. fp32 residual accumulator, fp32 softmax intermediate, HiFi4 + fp32 dest on proj/fc2/output_conv2.
- 2D RoPE with precomputed cos/sin tables.
- DPTHead.scratch.output_conv2 (3×3 128→32 + relu + 1×1 32→output at 518×518).

Still on host (in priority order by remaining CPU time).

---

## Bug fixes (correctness — block further eval)

### BF0 — S>2 first-forward compile stall

Blocks CO3Dv2 evaluation with >2 views per scene and any real multi-image
inference use case. Discovered during `eval_vggt.py` runs — S=2 completes
in ~5 min for 3 scenes, S=3 and S=4 hang for 20+ min on the **first**
forward of scene 1 and never produce output. Killing the process (`kill -9`)
leaves the p150a chip in a bad firmware state; recovery requires
`tt-smi -r 0 1 2 3` which also bounces the other 3 chips on the host.
(Per-chip `tt-smi -r 2` does **not** recover — the ETH heartbeat on the
wedged ASIC stays at `post code: c0de0000` through the re-init.)

Root cause (hypothesised, unverified): each `ttnn` op compiles a new
program on first encounter of a (shape, layout, memory_config) tuple. The
on-device Block forward hits many primitives (matmul, softmax, layer_norm,
linear, multiply, add, slice, concat, permute, reshape, nlp_create_qkv_heads).
S=1 primed the program cache for ~1374-token shapes; S=3 introduces ~4122
(global-attn) + 1374 (frame-attn) × per-head and per-split variants, all
first-time. Compile per new shape is seconds-to-tens-of-seconds; cumulative
compile time blows up.

Candidate fixes — **option 1 has been attempted and ruled out:**

1. ~~**Pre-warm at install time.**~~ **Tried and does not work.**
   `_prewarm_seqs(model, device, (1,2,4,8))` was added to
   `_ensure_installed` and verified: S=1 prewarm 2.0 s / PCC 0.9959;
   S=2 prewarm 3.3 s / PCC 0.9981; **S=4 prewarm hangs 40+ min at 99%
   CPU** with zero log progress after device open. Python `SIGTERM`
   never fires (handler can't run while ttnn is in a deep C++ compile
   call), so recovery requires `SIGKILL`, which wedges the chip. The
   "prewarm" call is still exercised by test_vggt.py / eval_vggt.py but
   is clamped to `S ≤ 2` by default; pass `--prewarm-seqs` to opt in.
2. **Pad to a canonical S.** Always pass `S=S_max` (say 8) to the
   aggregator and mask out unused frames. Only one shape set ever hits
   the program cache. Needs attention-mask plumbing in
   `Aggregator.forward` (global attention: prevent padding frames from
   contributing to real ones) and result-slice to trim the output back
   to the requested S. **Current recommendation — implement this next.**
3. **mast3r-style pre-computed `MatmulMultiCoreReuseMultiCastProgramConfig`.**
   Pin matmul shard strategies manually so the auto-discovery compile path
   (which is what's actually slow) is bypassed. Most invasive. Needed
   anyway for further perf tuning; can be staged after option 2.

Until BF0 is fixed, evaluation is stuck at S=2 / 1 pair per scene (coarse
but functional — see `co3d_eval_results.md`).

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

## P0 — Device-native DPT `scratch_forward` refinenets

Remaining CPU cost: ~194 ms × 2 heads = **388 ms** (biggest single chunk left).

Attempted twice. First pass (`2b2bf2d` discard row) broke to PCC -0.08.
Second pass copied mast3r's layout helpers verbatim (`_tokens_to_nhwc`,
`_flat_to_nhwc`, `_nhwc_to_flat`, `_linear_1x1`, `_conv2d`, `_resconv`
from `/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r/tt/ttnn_dust3r.py:813+`)
and split the port into per-refinenet steps with a PCC probe at every
boundary (env `VGGT_TT_SCRATCH_COMPARE=1`). Install + patch infrastructure
lives in `_install_ttnn_dpt_scratch()`, currently gated behind
`VGGT_TT_SCRATCH=1` (default OFF) so the rest of the model still runs
while this is in debug.

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

## P1 — DPT per-layer prelude (`norm` + `projects` + `resize_layers`)

Remaining CPU cost: ~54 ms × 2 heads = **108 ms**.

A previous port (`2b2bf2d` discard row) did `norm` + 1×1 `projects` as standalone per-op ttnn calls and netted +0 ms because each op paid its own up/down round-trip.

### Plan

Fold the prelude into the device `scratch_forward` entrypoint instead of running it as isolated ops:

1. At install: preload `h.norm` LN weights, each `h.projects[i]` 1×1 conv weight (as linear matmul weight), each `h.resize_layers[i]` weight (ConvTranspose2d for `[0,1]`, Identity for `[2]`, Conv2d 3×3 s=2 for `[3]`).
2. Rewrite the prelude loop to stay on device:
   - Upload the 4 selected aggregated tokens once (each `(B, N, 2048)`).
   - LN + permute/reshape + 1×1 linear + pos_embed add + resize_layer — all stay in ttnn tensors.
   - Feed directly into `scratch_forward_device` without downloading.
3. `pos_embed` add: the positional embedding comes from `create_uv_grid` + `position_grid_to_embed` on host. Precompute once at install for the fixed 518×518 geometry and upload as a ttnn tensor.

### Expected gain
- ~100 ms saved, but more importantly eliminates the host↔device round-trip between prelude and `scratch_forward`.

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
`53d46c1` uses a manual Q·Kᵀ → softmax → ·V chain in fp32 because the fused FlashAttention-2 kernel dropped world_points_conf PCC to 0.57 for 1374-token non-causal attention with `is_causal=False`. This is either a ttnn bug at this shape or a config I missed. Worth a retry with:
- Newer ttnn (check if Blackhole SDPA path has matured since April 2026).
- Explicit `scale` argument instead of implicit 1/√Dh.
- `SDPAProgramConfig` tuned for (B=1, H=16, N=1374, Dh=64).

A working fused SDPA could replace the manual matmul+softmax+matmul chain and likely save 50–150 ms.

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
