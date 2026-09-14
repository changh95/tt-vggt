# vggt-1b-p150 (VGGT-1B, 518x518, S=1 and S=2) -- Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied (with its source line) from the validation / publish reports. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | The port's own torch reference: `models/vggt-1b-p150/code/models/demos/vggt/reference/torch_vggt.py::load_vggt(eval_mode=True) -> upstream vggt.models.vggt.VGGT(enable_track=False) @ 44b3afbd1869d8bde4894dd8ea1e293112dd5eba (checkout /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/vggt-1b/vggt_upstream, byte-identical to the image's /opt/tt-metal/vggt): Aggregator (DINOv2 ViT-L/14 + 24 frame/global alternating blocks, dim 1024, 2D RoPE, F.scaled_dot_product_attention) + CameraHead + DPTHead x2; forward(images) -> dict`. 1190.6 M parameters (track head dropped, as the port and the server load it). This is the network the p150a port is PCC-gated against (`test_vggt.py`, the real-image set, the served A/B, the point-map study) |
| Weights | `facebook/VGGT-1B` @ `860abec7937da0a4c03c41d3c269c366e82abdf9` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`), `model.safetensors` (5.03 GB fp32) from the HF cache `/home/deepgadget/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.safetensors`, resolved by the port's `torch_vggt.resolve_weights_path`, `HF_HUB_OFFLINE=1` |
| Input | S=1: media/input.png (518x518 RGB PNG); S=2: media/source_1.png + media/source_2.png (kitchen 00/03, 779x520 RGB PNG) -> models.server.app.preprocess_pad (imported): longer side 518, shorter round(x/14)*14, PIL BICUBIC, /255, white pad to 518x518 -> (1, S, 3, 518, 518) fp32 in [0,1], batch 1 (== server/app.py::predict; the Aggregator normalises with ImageNet mean/std internally) |
| Per-view preprocess records | S1: ['media/input.png'] -> [{'orig_w': 518, 'orig_h': 518, 'resized_w': 518, 'resized_h': 518, 'scale_x': 1.0, 'scale_y': 1.0, 'pad_left': 0, 'pad_top': 0}]; S2: ['media/source_1.png', 'media/source_2.png'] -> [{'orig_w': 779, 'orig_h': 520, 'resized_w': 518, 'resized_h': 350, 'scale_x': 0.6649550706033376, 'scale_y': 0.6730769230769231, 'pad_left': 0, 'pad_top': 84}, {'orig_w': 779, 'orig_h': 520, 'resized_w': 518, 'resized_h': 350, 'scale_x': 0.6649550706033376, 'scale_y': 0.6730769230769231, 'pad_left': 0, 'pad_top': 84}] |
| Output | `pose_enc (1,S,9)`, `depth (1,S,518,518,1)`, `depth_conf (1,S,518,518)`, `world_points (1,S,518,518,3)`, `world_points_conf (1,S,518,518)` -- the five tensors `server/app.py` reads back from the port; readback 6.4 MB at S=1, 12.9 MB at S=2 |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600.00 W, 32607 MiB; idle 31.2 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/main` -- Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0, torchvision 0.26.0+cu128, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 1.31.0, pillow 12.3.0, einops 0.8.2, triton 3.6.0; fastapi 0.141.1 / pydantic 2.13.5 added in this pass so `models.server.app` imports unchanged |
| Repo state | `models/vggt-1b-p150` @ `9586c93` (branch `tt-model-package`), read-only; `reference/torch_vggt.py` and `server/app.py` (`preprocess_pad`, `_decode_image`, `_b64_npz`) imported unchanged; upstream `vggt` package cloned to `logs/gpu-vs-p150/vggt-1b/vggt_upstream` @ `44b3afbd1869d8bde4894dd8ea1e293112dd5eba` (tt-model.yaml `source.extra_code` ref; `diff -rq` against the image's `/opt/tt-metal/vggt` copy: identical) |
| Script | `logs/gpu-vs-p150/vggt-1b/bench_vggt_gpu.py` (uses `logs/gpu-vs-p150/bench_common.py`); logs `full_run.log` (the run below), `smoke.log` (3-iteration dry run, incl. the CPU reference pass), `smoke2_lowprec.log` (dry run of the converted-module path after the fix below), `full_run_attempt1_bf16weights_dtype_error.log` (first full run: reference rows complete and within 1 ms of the run below, then aborted at the `bf16_weights` variant because upstream `DPTHead._apply_pos_embed` promotes a bf16 activation to fp32 -- fixed by casting that constant to `x.dtype` for the converted copies only, see `bench_vggt_gpu.py`); raw JSON `reports/gpu-vs-p150/vggt-1b.json` (= `logs/gpu-vs-p150/vggt-1b/result.json`, incl. every raw wall-clock sample); CPU reference tensors `cpu_fp32_reference.pt`; `render_report.py` renders this file from the JSON |
| Command | `cd logs/gpu-vs-p150/vggt-1b && HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_vggt_gpu.py --iters 50 --warmup 10 --skip-cpu > full_run.log 2>&1` (one process; exit 0) |
| Loop | per S, precision and variant 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before and after each; wall-clock (`perf_counter`) is the primary number, CUDA-event time recorded alongside |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/POINTMAP_SUMMARY.md:89` (Hub `tt serve` after the demo republish: S=1 x30 forward **420.1 ms** median (414.1 min / 437.7 max), total **528.5**; S=2 x20 forward **887.3** (873.7 / 1021.8), total **1104.6**); `models/vggt-1b-p150/DEVICE_VALIDATION.md:411` (pre-publish served A/B, final default: S=1 411.4 / 407.9 / 435.0 forward, 525.5 total; S=2 880.6), `DEVICE_VALIDATION.md:416` (`encode` host npz 105-125 ms), `VALIDATION_SUMMARY.md:40`, `PUBLISH_SUMMARY.md:20` (419.9 / 881.0) |

Timing definitions (they match the p150a `timing_ms` keys of `server/app.py`):

- **incl_h2d** = `images.to("cuda")` (one `(1,S,3,518,518)` fp32 pageable host tensor, 3.2 MB per view, as the server preprocess produces) + forward + the five outputs `.float().cpu()`. Compare with p150a `timing_ms.forward` = `vggt_forward(images)`: host ImageNet-normalise + DINOv2 patch-embed prelude (fp32 torch), upload of the `(S, 1374, 1024)` token tensor, one whole-graph metal-trace replay (aggregator + camera head + per-frame DPT heads), readback of the raw head outputs, host `activate_head` / `activate_pose` (**420.1 ms** S=1, **887.3** S=2).
- **excl_h2d** = forward only, input already resident, outputs left on the device.
- **served-like** = base64 decode + PNG decode + `preprocess_pad` of every view + `torch.stack` (`preprocess`) + incl_h2d forward (`forward`) + `pose_encoding_to_extri_intri` + float16/float32 casts + `np.savez_compressed` + base64 of the npz (`encode`) = the interval `server/app.py` reports as `timing_ms.total` (HTTP/JSON framing is outside on both sides). Compare with p150a `timing_ms.total` (**528.5 ms** S=1, **1104.6** S=2, of which the `encode` npz stage is 105-125 ms of identical host work on the p150a host).

Precision rows: the upstream `VGGT.forward` wraps the camera head and both DPT heads in `torch.cuda.amp.autocast(enabled=False)`, so in the bf16 / fp16 autocast rows only the aggregator (DINOv2 + 48 alternating-attention blocks) runs in low precision and the heads run fp32 (with TF32 allowed). The `bf16_weights` / `fp16_weights` rows convert the whole module once (heads included, closer to the p150a's bf16-on-device class) and are informational: not the reference as shipped.

## Accuracy check (GPU vs the CPU fp32 reference)

CPU fp32 reference: the same module on the host (16 torch threads, per smoke.log): forward 5540 ms at S=1, 8988 ms at S=2 (`smoke.log`; the p150a point-map study quotes 7.9 s for torch CPU S=2 on the same host under load). PCC is Pearson correlation over the flattened tensor; the port's gate is the minimum over the five output keys.

| S | GPU precision / variant | pose_enc | depth | depth_conf | world_points | world_points_conf | min PCC | max abs diff depth | depth median (ref) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | GPU fp32 strict (PCC check) | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | **1.0000000** | 4.09e-05 | 0.96509 (0.96509) |
| 1 | fp32 + TF32 ('high') | 1.0000000 | 0.9999991 | 0.9999995 | 0.9999968 | 0.9999997 | **0.9999968** | 1.16e-02 | 0.96508 (0.96509) |
| 1 | bf16 autocast (aggregator; heads fp32/TF32) | 0.9999999 | 0.9992216 | 0.9999176 | 0.9992729 | 0.9999892 | **0.9992216** | 2.54e-01 | 0.9636 (0.96509) |
| 1 | fp16 autocast (aggregator; heads fp32/TF32) | 1.0000000 | 0.9999975 | 0.9999985 | 0.9999957 | 0.9999994 | **0.9999957** | 1.52e-02 | 0.96493 (0.96509) |
| 1 | bf16_weights (module converted) | 0.9999980 | 0.9992063 | 0.9998382 | 0.9991676 | 0.9998973 | **0.9991676** | 2.79e-01 | 0.96484 (0.96509) |
| 1 | fp16_weights (module converted) | 0.9999998 | 0.9999607 | 0.9999485 | 0.9999338 | 0.9999979 | **0.9999338** | 8.07e-02 | 0.96533 (0.96509) |
| 1 | torch.compile fp16_autocast+default | 1.0000000 | 0.9999938 | 0.9999991 | 0.9999885 | 0.9999994 | **0.9999885** | 2.78e-02 | 0.96497 (0.96509) |
| 1 | torch.compile fp16_autocast+reduce-overhead | 1.0000000 | 0.9999938 | 0.9999991 | 0.9999885 | 0.9999994 | **0.9999885** | 2.78e-02 | 0.96497 (0.96509) |
| 1 | torch.compile bf16_weights+reduce-overhead | 0.9999954 | 0.9996907 | 0.9998696 | 0.9995378 | 0.9999276 | **0.9995378** | 1.95e-01 | 0.96094 (0.96509) |
| 2 | GPU fp32 strict (PCC check) | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | **1.0000000** | 4.49e-05 | 1.0232 (1.0232) |
| 2 | fp32 + TF32 ('high') | 1.0000000 | 0.9999990 | 0.9999997 | 0.9999963 | 0.9999997 | **0.9999963** | 1.62e-02 | 1.02322 (1.0232) |
| 2 | bf16 autocast (aggregator; heads fp32/TF32) | 0.9999999 | 0.9994821 | 0.9999570 | 0.9993312 | 0.9999758 | **0.9993312** | 2.83e-01 | 1.02244 (1.0232) |
| 2 | fp16 autocast (aggregator; heads fp32/TF32) | 1.0000000 | 0.9999936 | 0.9999981 | 0.9999855 | 0.9999993 | **0.9999855** | 3.77e-02 | 1.02336 (1.0232) |
| 2 | bf16_weights (module converted) | 0.9999892 | 0.9992768 | 0.9996268 | 0.9992485 | 0.9998776 | **0.9992485** | 2.53e-01 | 1.02344 (1.0232) |
| 2 | fp16_weights (module converted) | 0.9999999 | 0.9999718 | 0.9999807 | 0.9999426 | 0.9999974 | **0.9999426** | 8.18e-02 | 1.02344 (1.0232) |
| 2 | torch.compile fp16_autocast+default | 1.0000000 | 0.9999945 | 0.9999986 | 0.9999878 | 0.9999994 | **0.9999878** | 3.48e-02 | 1.02302 (1.0232) |
| 2 | torch.compile fp16_autocast+reduce-overhead | 1.0000000 | 0.9999945 | 0.9999986 | 0.9999878 | 0.9999994 | **0.9999878** | 3.48e-02 | 1.02302 (1.0232) |
| 2 | torch.compile bf16_weights+reduce-overhead | 0.9999959 | 0.9996145 | 0.9997712 | 0.9995052 | 0.9998895 | **0.9995052** | 1.67e-01 | 1.02344 (1.0232) |

PCC gate (GPU fp32 strict vs CPU fp32, > 0.999): **PASS** -- min PCC 1.0000000 (S=1) / 1.0000000 (S=2); max |d| pose_enc 4.8e-07 / 4.9e-07, world_points 6.4e-04 / 3.2e-04. For scale: the p150a served S=2 output vs the same torch fp32 reference is depth 0.99927 / world_points 0.99911 / pose_enc 0.9999994 (POINTMAP_SUMMARY.md:69), synthetic min-PCC 0.9952 (S=1) / 0.9981 (S=2) (VALIDATION_SUMMARY.md:40).

## GPU timing (median / min / p90 of 50 iterations after 10 warm-ups, ms)

Model load (`load_vggt` incl. the 5.0 GB safetensors read from the page cache + `.cuda()`): 5.312 s. First call (fp32 strict, incl. cuDNN/SDPA autotune): 390.3 ms (S=1) / 256.8 ms (S=2).

### Reference as shipped

| S | precision | incl_h2d wall med / min / p90 | excl_h2d wall med / min / p90 | excl_h2d CUDA-event med | first call under this precision | GPU power mean (max) excl loop | peak mem alloc (reserved) MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fp32 strict (no TF32) | 134.07 / 133.78 / 140.52 | 133.24 / 132.90 / 139.45 | 133.23 | 132.66 | 524.3 (526.0) | 5420 (6354) |
| 1 | fp32 + TF32 ('high') | 93.04 / 92.90 / 98.83 | 92.22 / 92.07 / 97.49 | 92.20 | 98.49 | 402.8 (407.6) | 5420 (6354) |
| 1 | bf16 autocast (aggregator; heads fp32/TF32) | 65.58 / 65.44 / 69.13 | 64.64 / 64.50 / 69.36 | 64.63 | 64.88 | 378.6 (387.1) | 5420 (6354) |
| 1 | fp16 autocast (aggregator; heads fp32/TF32) | 63.32 / 63.15 / 67.65 | 62.43 / 62.23 / 67.74 | 62.41 | 66.63 | 422.0 (425.7) | 5420 (6354) |
| 2 | fp32 strict (no TF32) | 244.97 / 243.95 / 251.15 | 246.11 / 242.75 / 249.88 | 245.51 | 247.86 | 517.4 (520.4) | 5815 (6682) |
| 2 | fp32 + TF32 ('high') | 166.20 / 165.96 / 172.72 | 170.75 / 164.70 / 179.88 | 170.72 | 170.88 | 453.5 (461.3) | 5815 (6682) |
| 2 | bf16 autocast (aggregator; heads fp32/TF32) | 103.37 / 102.94 / 109.45 | 101.66 / 101.44 / 107.60 | 101.64 | 107.3 | 434.4 (436.1) | 5816 (6682) |
| 2 | fp16 autocast (aggregator; heads fp32/TF32) | 95.72 / 95.55 / 101.50 | 94.21 / 94.01 / 100.41 | 94.20 | 99.73 | 489.9 (496.6) | 5816 (6682) |

Pinned (page-locked) host input, bf16 autocast, incl_h2d median: S=1 65.49 ms, S=2 102.91 ms (vs pageable above; informational).

### Module converted once to bf16 / fp16 (informational; heads included)

| S | variant | incl_h2d wall med / min / p90 | excl_h2d wall med / min / p90 | min PCC vs fp32 ref | power mean W | peak mem MiB |
|---|---|---:|---:|---:|---:|---:|
| 1 | bf16_weights | 54.47 / 54.30 / 55.67 | 53.69 / 53.59 / 56.66 | 0.9991676 | 379.0 | 7497 |
| 2 | bf16_weights | 84.59 / 84.43 / 90.62 | 83.10 / 82.95 / 89.06 | 0.9992485 | 415.0 | 7698 |
| 1 | fp16_weights | 52.32 / 52.16 / 53.15 | 51.46 / 51.24 / 56.89 | 0.9999338 | 409.7 | 9779 |
| 2 | fp16_weights | 76.80 / 76.60 / 82.50 | 75.42 / 75.14 / 81.09 | 0.9999426 | 482.3 | 9981 |

### torch.compile (informational)

torch.compile(dynamic=False) of the as-shipped module; best autocast precision by S=1 excl_h2d median with min PCC >= 0.99 -> fp16_autocast; one compile per (S, mode); budget 300 s per compile (a variant whose compile exceeds it is recorded and the remaining variants are skipped)

| S | compiled variant | compile (first call) s | incl_h2d wall med / min / p90 | excl_h2d wall med / min / p90 | min PCC | power mean W | peak mem MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fp16_autocast+default | 26.5 | 83.63 / 83.32 / 88.85 | 82.63 / 82.47 / 88.44 | 0.9999885 | 339.2 | 9735 |
| 2 | fp16_autocast+default | 23.1 | 111.58 / 111.40 / 117.77 | 109.92 / 109.66 / 116.32 | 0.9999878 | 400.4 | 10466 |
| 1 | fp16_autocast+reduce-overhead | 17.9 | 82.58 / 82.15 / 87.70 | 81.61 / 81.15 / 87.49 | 0.9999885 | 341.3 | 9373 |
| 2 | fp16_autocast+reduce-overhead | 18.2 | 111.56 / 110.51 / 117.38 | 109.00 / 108.74 / 115.80 | 0.9999878 | 401.1 | 9636 |
| 1 | bf16_weights+reduce-overhead | 22.1 | 74.23 / 73.29 / 79.15 | 73.12 / 72.50 / 78.68 | 0.9995378 | 317.0 | 9243 |
| 2 | bf16_weights+reduce-overhead | 20.7 | 103.49 / 102.77 / 109.45 | 101.34 / 100.96 / 107.55 | 0.9995052 | 333.0 | 9379 |

### Served-like loop (the `server/app.py::predict` host stages around the GPU forward, npz float16, all four dense keys)

| S | precision | preprocess med | forward (incl_h2d) med | encode med | **total med / min / p90** | power mean W | npz base64 bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fp32 strict (no TF32) | 4.31 | 133.59 | 109.68 | **247.65 / 246.71 / 253.79** | 322.6 | 4578776 |
| 1 | fp32 + TF32 ('high') | 4.30 | 93.48 | 110.09 | **208.00 / 206.88 / 213.78** | 233.5 | 4581076 |
| 1 | bf16 autocast (aggregator; heads fp32/TF32) | 4.32 | 65.89 | 109.98 | **180.34 / 179.32 / 182.55** | 192.4 | 4580536 |
| 1 | fp16 autocast (aggregator; heads fp32/TF32) | 4.30 | 63.47 | 110.20 | **178.04 / 177.25 / 180.62** | 197.2 | 4579212 |
| 2 | fp32 strict (no TF32) | 20.94 | 248.68 | 223.44 | **493.07 / 486.43 / 495.52** | 306.3 | 9303936 |
| 2 | fp32 + TF32 ('high') | 21.44 | 166.48 | 226.01 | **414.50 / 412.68 / 420.39** | 234.6 | 9310592 |
| 2 | bf16 autocast (aggregator; heads fp32/TF32) | 20.95 | 103.86 | 224.14 | **349.11 / 348.21 / 355.11** | 190.1 | 9309164 |
| 2 | fp16 autocast (aggregator; heads fp32/TF32) | 20.97 | 96.49 | 224.77 | **342.27 / 341.07 / 349.02** | 201.2 | 9310104 |

`preprocess` here is PNG decode + PIL bicubic resize + pad; `encode` is the camera conversion + `np.savez_compressed` (zlib) of ~6.4 MB (S=1) / 12.9 MB (S=2) of dense arrays + base64 -- pure host work, single-threaded numpy/zlib, identical in kind to the p150a server's `encode` (105-125 ms at S=1 per DEVICE_VALIDATION.md:416).

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 weights and matmul inputs on device, fp32 residual stream / scores / softmax, HiFi4 + fp32 accumulation on proj / fc2 / DPT convs, exact fp32 eltwise bilinear interpolation (`VGGT_FUSED_INTERP=exact`), matmul attention (SDPA not default), one whole-graph metal trace per S (`TT_FUSED=1`, tt-metal 0.65.2.dev9100 @ 8b98410e7). GPU precision per row as stated.

| S | row | p150a ms (definition) | GPU variant / precision | GPU ms | ratio p150a/GPU |
|---|---|---:|---|---:|---:|
| 1 | device forward (p150a `timing_ms.forward` vs GPU incl_h2d) | 420.1 (POINTMAP_SUMMARY.md:89 Hub tt serve, DEVICE_VALIDATION.md:411 411.4) | ref, fp32 strict (no TF32) | 134.07 | **3.13** |
| 1 |  |  | ref, fp32 + TF32 ('high') | 93.04 | **4.52** |
| 1 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 65.58 | **6.41** |
| 1 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 63.32 | **6.63** |
| 1 | | | bf16_weights (module converted, eager; informational) | 54.47 | 7.71 |
| 1 | | | fp16_weights (module converted, eager; informational) | 52.32 | 8.03 |
| 1 | | | torch.compile fp16_autocast+default (informational) | 83.63 | 5.02 |
| 1 | | | torch.compile fp16_autocast+reduce-overhead (informational) | 82.58 | 5.09 |
| 1 | | | torch.compile bf16_weights+reduce-overhead (informational) | 74.23 | 5.66 |
| 1 | GPU forward only (excl_h2d, no PCIe) vs the same p150a number (which cannot exclude its transfers) | 420.1 | ref, fp32 strict (no TF32) | 133.24 | 3.15 |
| 1 |  |  | ref, fp32 + TF32 ('high') | 92.22 | 4.56 |
| 1 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 64.64 | 6.50 |
| 1 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 62.43 | 6.73 |
| 1 | served e2e (p150a `timing_ms.total` vs GPU served-like total, same host stages, npz float16) | 528.5 (POINTMAP_SUMMARY.md:89) | ref, fp32 strict (no TF32) | 247.65 | **2.13** |
| 1 |  |  | ref, fp32 + TF32 ('high') | 208.00 | **2.54** |
| 1 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 180.34 | **2.93** |
| 1 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 178.04 | **2.97** |
| 2 | device forward (p150a `timing_ms.forward` vs GPU incl_h2d) | 887.3 (POINTMAP_SUMMARY.md:89 Hub tt serve, DEVICE_VALIDATION.md:411 880.6) | ref, fp32 strict (no TF32) | 244.97 | **3.62** |
| 2 |  |  | ref, fp32 + TF32 ('high') | 166.20 | **5.34** |
| 2 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 103.37 | **8.58** |
| 2 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 95.72 | **9.27** |
| 2 | | | bf16_weights (module converted, eager; informational) | 84.59 | 10.49 |
| 2 | | | fp16_weights (module converted, eager; informational) | 76.80 | 11.55 |
| 2 | | | torch.compile fp16_autocast+default (informational) | 111.58 | 7.95 |
| 2 | | | torch.compile fp16_autocast+reduce-overhead (informational) | 111.56 | 7.95 |
| 2 | | | torch.compile bf16_weights+reduce-overhead (informational) | 103.49 | 8.57 |
| 2 | GPU forward only (excl_h2d, no PCIe) vs the same p150a number (which cannot exclude its transfers) | 887.3 | ref, fp32 strict (no TF32) | 246.11 | 3.61 |
| 2 |  |  | ref, fp32 + TF32 ('high') | 170.75 | 5.20 |
| 2 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 101.66 | 8.73 |
| 2 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 94.21 | 9.42 |
| 2 | served e2e (p150a `timing_ms.total` vs GPU served-like total, same host stages, npz float16) | 1104.6 (POINTMAP_SUMMARY.md:89) | ref, fp32 strict (no TF32) | 493.07 | **2.24** |
| 2 |  |  | ref, fp32 + TF32 ('high') | 414.50 | **2.66** |
| 2 |  |  | ref, bf16 autocast (aggregator; heads fp32/TF32) | 349.11 | **3.16** |
| 2 |  |  | ref, fp16 autocast (aggregator; heads fp32/TF32) | 342.27 | **3.23** |

Reading: on the device-forward definition (upload + network + readback of the five outputs) the RTX 5090 running the port's fp32 reference eagerly in strict fp32 is **3.13x faster than the p150a's fused bf16 trace at S=1 (134.1 vs 420.1 ms) and 3.62x at S=2 (245.0 vs 887.3 ms)**; with TF32 4.52x / 5.34x; in the p150a's own precision class (bf16 autocast) **6.41x / 8.58x** (65.6 / 103.4 ms); fp16 autocast 6.63x / 9.27x. The gap widens with S because the p150a scales almost linearly in S (420 -> 887 ms) while the GPU's S=2 forward costs only 1.58x its S=1 forward under bf16 autocast (the 1374-token frame blocks fill the GPU better at S=2 and the per-frame DPT heads are cheap). PCIe transfers are a small part of the GPU number (incl vs excl within ~1-2 ms; input 3.2 MB per view, readback 6.4 / 12.9 MB). Converting the whole module to fp16 once (informational) gives 52.3 / 76.8 ms (8.03x / 11.55x) at min PCC 0.99993 / 0.99994. `torch.compile` (inductor, torch 2.11.0+cu128, triton 3.6.0, sm_120) was **slower** than eager for this module in every variant tried (83.6 ms compiled vs 63.3 ms eager at S=1 fp16 autocast; compile 18-27 s per (S, mode), peak memory ~9.2-10.5 GB vs 5.4-5.8 GB eager): the compiled rows are reported as measured, not as a tuned deployment.

On accuracy, every GPU precision tracks the fp32 reference more closely (min PCC 0.9992 bf16 autocast S=1, >= 0.99999 fp16 autocast / TF32) than the p150a's bf16 graph does (served S=2 depth 0.99927 / world_points 0.99911 vs the same reference; synthetic min-PCC 0.9952 / 0.9981). bf16 is the least accurate GPU option here (min PCC 0.9992 / 0.9993, depth max |d| 0.25-0.28 on a ~1.0 median depth): the 8-bit bf16 mantissa costs more than fp16's range restriction on this network, whose exp/expm1 head activations stay fp32 under autocast.

End-to-end (the `server/app.py` interval): **180-248 ms vs 528.5 ms at S=1 (2.13-2.97x) and 342-493 ms vs 1104.6 ms at S=2 (2.24-3.23x)**. The ratio compresses because ~110 ms (S=1) / ~224 ms (S=2) of every request is the same single-threaded host `np.savez_compressed` of the float16/float32 dense arrays on both sides (the p150a server logs 105-125 ms of `encode` at S=1 on this host), plus 4 / 21 ms of PNG decode + bicubic pad: on the GPU the response encoding is now the largest stage of a served request at every precision.

Not measured / not claimed: p150a power (not measured in any pass -> no power or efficiency comparison; the GPU power figures above are nvidia-smi means over each timed loop, 31.2 W idle). p150a numbers were not re-measured. The GPU numbers exclude HTTP/JSON framing, as do the p150a `timing_ms` keys. S=3 / S=4 were not run on the GPU (the p150a card has only best-of-N `test_vggt.py` numbers for them, no served medians). The p150a's `timing_ms.forward` includes its host prelude (ImageNet-normalise + DINOv2 patch-embed on the host, fp32 torch) and the host `activate_head`; the GPU incl_h2d runs the same stages on the GPU inside the module, so both sides cover the whole `images -> outputs` path.

## Reproduce

```bash
cd /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/vggt-1b
# upstream package (pinned, no __init__.py -> namespace package): git clone https://github.com/facebookresearch/vggt vggt_upstream && git -C vggt_upstream checkout 44b3afbd1869d8bde4894dd8ea1e293112dd5eba
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_vggt_gpu.py --iters 50 --warmup 10 --skip-cpu | tee full_run.log
/home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python render_report.py
# outputs: reports/gpu-vs-p150/vggt-1b.json + .md, ./result.json, ./cpu_fp32_reference.pt (written by the first run without --skip-cpu, see smoke.log)
# --no-compile skips the torch.compile rows; --no-served / --no-lowprec-weights skip those sections; --seqs 1 runs S=1 only
```

GPU released after the run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` -> empty.
