# tt-vggt

VGGT-1B (Meta, `facebookresearch/vggt`) running on a single Tenstorrent
Blackhole p150a via `tt-nn` / `tt-metallium`. This repo holds the port
code, benchmark + evaluation harnesses, and the full optimization
trajectory — not the upstream model code and not the tt-metal SDK.

## Results at a glance

| | torch CPU reference | ttnn port on p150a | ratio |
|---|---|---|---|
| latency / frame (B=1 S=1 518×518) | 5037 ms | **~1694 ms** | **2.97×** |
| throughput | 0.1985 fps | **0.59 fps** | **+196 %** |
| min-PCC (port vs torch ref, synthetic rand input) | — | 0.9959 | — |
| min-PCC (port vs torch ref, real CO3Dv2 apple images) | — | **0.9947** | — |
| AUC@30° (pair-wise pose, CO3Dv2 apple S=2, 3 scenes) | 87.2 | **86.1** | Δ −1.1 |

The port moves every transformer block, RoPE, and the final DPT
`output_conv2` conv stack to the p150a. The remaining CPU work (DPT
refinenets, DPT prelude, `custom_interpolate`, `activate_head`, image
normalization) is why this is 3× instead of fully device-bound.

## Repository layout

```
tt-vggt/
├── README.md                # this file
├── PROGRAM.md               # original problem brief
├── TODO.md                  # future bug-fix + optimization plan
├── co3d_eval_results.md     # detailed CO3Dv2 eval write-up
├── results.tsv              # one row per experiment (keep/discard)
├── test_vggt.py             # perf benchmark harness (B=1 S=1)
├── eval_vggt.py             # CO3Dv2 correctness harness (PCC + GT pose)
└── models/
    └── demos/
        └── vggt/
            ├── reference/
            │   └── torch_vggt.py    # thin loader over facebookresearch/vggt
            └── tt/
                └── ttnn_vggt.py     # monkey-patches Block / Mlp / DPT on device
```

## Hardware / software environment

- **Hardware**: Tenstorrent Blackhole p150a (single chip on a 2× p300c
  host, 4 chips total, chip 0 used).
- **Tenstorrent SDK**: `tt-metal` (Tracy profiler enabled) + `tt-nn`
  Python bindings. Both taken from the sibling `medgemma/tt-metal` build
  via `sys.path` shim at the top of `test_vggt.py` / `eval_vggt.py` —
  the machine's venv-pinned `ttnn` points at a scrubbed pi0_5 checkout
  without matching kernel sources, so we reuse the medgemma tree.
- **Reference model**: `facebookresearch/vggt`, cloned into
  `vggt_ref/` (outside this repo, see `.gitignore`). Weights from HF
  `facebook/VGGT-1B` (≈3.6 GB safetensors, in HF cache).
- **Venv**: `~/.tenstorrent-venv`.

## How to reproduce

### 1. Clone and set up paths

```bash
git clone https://github.com/changh95/tt-vggt.git
cd tt-vggt

# Clone the VGGT reference repo alongside (for torch CPU reference).
git clone --depth 1 https://github.com/facebookresearch/vggt.git vggt_ref

# Download the 1B weights into the HF cache.
source ~/.tenstorrent-venv/bin/activate
python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.safetensors')"
```

The harnesses import `ttnn` via `sys.path` from
`/home/ttuser/experiments/medgemma/tt-metal` — adjust `_TT_METAL_ROOT`
at the top of each harness to point at your own tt-metal checkout.

### 2. Benchmark

```bash
python3 test_vggt.py --runs 3 --seq 1 --device-id 0
```

Expected (this commit): `latency_ms: ~1700`, `pcc: ~0.996`, `status: PASS`.

### 3. Correctness on CO3Dv2

```bash
# Download one single-sequence apple chunk (~190 MB of images + 120 KB annotations).
mkdir -p co3d_data && cd co3d_data
curl -sSL https://dl.fbaipublicfiles.com/co3dv2_231130/apple_000_singlesequence.zip -o apple_000.zip
curl -sSL https://dl.fbaipublicfiles.com/co3dv2_231130/apple_001_singlesequence.zip -o apple_001.zip
unzip -q apple_000.zip && unzip -q apple_001.zip
cd ..

python3 eval_vggt.py --num-views 2 --category apple --co3d-root co3d_data
```

See `co3d_eval_results.md` for the full write-up.

## Performance benchmark

Best-of-3 `latency_ms` at B=1 S=1 518×518. Full trajectory in
`results.tsv`; this table shows the keep-commits only.

| step | commit | op(s) on device | latency | fps | vs baseline | min PCC |
|---|---|---|---:|---:|---:|---:|
| initial port scaffolding | c7d238e | — (CPU passthrough) | — | — | — | — |
| CPU reference run | f718b74 | — | **5037 ms** | 0.1985 | — | 1.000 |
| MLP port (72 instances) | 185868f | fc1 + gelu + fc2 bf16 LoFi | 3122 ms | 0.3203 | +61 % | 0.9948 |
| attn qkv bf16 | b311533 | + attn qkv | 2669 ms | 0.3746 | +89 % | 0.9930 |
| attn proj HiFi4 + fp32 dest | 0a04e6f | + attn proj (precision fix) | 2495 ms | 0.4008 | +102 % | 0.9955 |
| attn scores/softmax on device | 53d46c1 | + Q·Kᵀ, softmax, ·V (fp32) | 2232 ms | 0.4481 | +126 % | 0.9943 |
| fuse norm1 into attn | 9c7d71a | + Block.norm1 | 2193 ms | 0.4559 | +130 % | 0.9955 |
| keep qkv in bf16 through CPU | a86d4aa | no new op, no fp32 roundtrip | 2067 ms | 0.4837 | +144 % | 0.9946 |
| full on-device Block | 8969cef | + q_norm, k_norm, ls1, ls2, norm2, residual adds | 1712 ms | 0.5841 | +194 % | 0.9961 |
| 2D RoPE on device | bae1d60 | + cos/sin + rotate_half + mul/add | 1640 ms | 0.6097 | +207 % | 0.9957 |
| DPT output_conv2 on device | db0ff6a | + 3×3 128→32 + relu + 1×1 at 518×518 | ~1694 ms | 0.59 | +196 % | 0.9959 |

Precision profile across the port (preserved through every commit):

- bf16 weights, bf16 matmul inputs.
- fp32 residual accumulator inside each Block (proj/fc2 output
  `dtype=float32`). bf16 residuals over 48 aggregator blocks collapsed
  `world_points_conf` PCC to 0.978.
- fp32 attention scores + softmax + context via HiFi4 + `dtype=float32`.
  bf16 softmax over 1374-long rows dropped conf PCC below 0.99.
- HiFi4 + `fp32_dest_acc_en=True` on proj / fc2 / DPT `output_conv2`.

## PCC tests

"Relative" PCC: Pearson correlation between the torch CPU reference
output and the ttnn port output, per output channel. Threshold for
`status: PASS` is min-channel PCC ≥ 0.99.

### Synthetic input (`torch.rand(1, 1, 3, 518, 518)`)

```
pcc_pose_enc          = 1.0000
pcc_depth             = 1.0000
pcc_depth_conf        = 0.9997
pcc_world_points      = 1.0000
pcc_world_points_conf = 0.9959    <- min, above 0.99 floor
```

### Real CO3Dv2 images (apple / 3 scenes / S=2)

```
mean pcc_pose_enc          = 0.9999
mean pcc_depth             = 0.9947
mean pcc_depth_conf        = 0.9985
mean pcc_world_points      = 0.9999
mean pcc_world_points_conf = 0.9996
```

Per-scene breakdown is in `co3d_eval_results.md`. The conf channels —
which were the bf16-stress bottleneck during porting — stay well above
0.99 on natural-image inputs.

## CO3Dv2 ground-truth evaluation

Pair-wise relative poses, 3 single-sequence apple scenes, S=2 (1 pair
each, 3 pairs total). GT viewpoints converted from CO3D PyTorch3D
convention to OpenCV via `diag(-1, -1, 1) @ R.T` on the rotation.

|  | RRA@5° | RRA@15° | RTA@5° | RTA@15° | **AUC@30°** |
|---|---:|---:|---:|---:|---:|
| torch reference | 0.0 % | 100.0 % | 66.7 % | 100.0 % | **87.2** |
| ttnn port | 33.3 % | 100.0 % | 66.7 % | 100.0 % | **86.1** |
| Δ (port − ref) | +33.3 | +0.0 | +0.0 | +0.0 | **−1.1** |

Port costs ≈1.1 AUC@30° points against the torch reference on real data.
That's the honest quantization bill for the 3× speedup. The +33.3 % on
RRA@5° is a 3-pair-sample artefact, not signal (one extra pair below
5° flips the fraction by 33 %).

Scaling up to more views per scene and more categories is **blocked** by
a ttnn kernel-compile-on-first-new-shape stall at S>2 (documented in
`TODO.md` as BF0). S=2 / 3 pairs is statistically coarse but was enough
to measure the port's bf16+HiFi4 cost against the reference.

## Optimization trajectory — what worked and what didn't

The `results.tsv` log has a row for every experiment (31 total).
Condensed themes:

**Worked (kept):** op-by-op device ports where every new device op was
paired with a correctness check (`status: PASS` ≡ min PCC ≥ 0.99). The
wins that actually moved latency were the big matmul-heavy ops: MLP,
attn qkv, attn proj, the Q·Kᵀ → softmax → ·V chain, and finally the
full-block fusion that kept the residual stream on chip. DPT
`output_conv2` moved CPU work to device at ~break-even wall-clock.

**Didn't work, discarded:** CPU tricks masquerading as device work
(generic bf16 autocast) — broke the conf heads' PCC. Individual
LayerNorm ports — overhead exceeded compute. `ttnn.transformer.
scaled_dot_product_attention` fused kernel — PCC collapse to 0.57 on
this model's 1374-token non-causal attention (possible ttnn bug at this
shape). Per-conv wrapper on the DPT scratch_forward — 120 individual
up/down round-trips added 255 ms. Device-native scratch_forward —
partially implemented but broke PCC to -0.08 due to ttnn layout-chaining
between conv2d / linear / upsample / add; blocked until I replicate
mast3r's proven layout helpers.

**Bigger principles observed, in hindsight:**

- **Always validate against a real reference, not the previous port.** The
  `eval_vggt.py` harness loads a *fresh un-patched* VGGT instance
  alongside the ported one; otherwise the "reference" is just the port
  comparing against itself, and every experiment looks like PCC 1.0.
- **Precision budget is per-port, not per-model.** Adding one more bf16
  op is OK until it isn't; I blew past 0.99 twice by moving ops whose
  individual error was small but cumulative damage over 48 aggregator
  blocks collapsed the confidence head. Fix was targeted — fp32
  intermediates only where numerically hot (residual accumulator,
  softmax), not everywhere.
- **Host↔device round-trip cost dominates small ops.** Most "this
  should be faster on the chip" experiments that targeted tiny ops
  (Block.norm1 alone, tiny head linears, single 1×1 convs) came out
  break-even or slower because 72 × (upload + download) per forward
  ate the device compute gain. The wins came from batching device ops
  end-to-end in fused functions that upload once, compute many ops on
  chip, and download once.
- **ttnn precision knobs matter more than which op you port.** HiFi4 +
  `fp32_dest_acc_en` + `packer_l1_acc` on precision-hot matmuls
  recovered PCC budget that pure bf16 had spent. Without it, the attn
  proj port was FAIL (0.988); with it, the same port was PASS (0.996).

## Known limitations

Full backlog is in `TODO.md`. Highlights:

- **S > 2 compile stall** (BF0): first forward at a new sequence length
  hangs for 20+ min due to ttnn kernel-compile-on-first-new-shape.
  Blocks deeper CO3D eval. Fix candidates: pre-warm cache at install,
  pad to canonical S, or pre-specify matmul program_config.
- **Device wedges on hard-kill** (BF1): `kill -9` of a ttnn process
  leaves chip 0 with ETH heartbeat stuck, requires `tt-smi -r 0 1 2 3`
  to recover. Need a SIGTERM/SIGINT handler in the harnesses.
- **3×3 convs in the DPT refinenets** are still on CPU (~388 ms of the
  1700 ms total). The port attempt broke on ttnn layout chaining; fix
  is to copy mast3r's `_conv2d`/`_resconv`/`_tokens_to_nhwc` helpers
  verbatim (see P0 in TODO).
- **CPU-pinned glue** (image normalization, token concatenation,
  activate_head) remains, mostly for numerical reasons
  (`activate_head`'s `expp1` is precision-sensitive). Small wins
  individually.

## Credits

- **VGGT model**: Meta AI, `facebookresearch/vggt`, Apache 2.0.
- **Tenstorrent SDK** (`tt-metal`, `tt-nn`): Tenstorrent, Apache 2.0.
- **Sibling `mast3r` port** on the same hardware provided the ttnn
  layout-handling pattern reference
  (`/home/ttuser/experiments/mast3r/tt-metal/models/demos/mast3r/`).

## License

Apache 2.0 — same as upstream VGGT and tt-metal.
