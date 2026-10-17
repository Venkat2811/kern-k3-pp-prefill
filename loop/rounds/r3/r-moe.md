Run `r-moe` · **target: D's routed MoE at its floor: `k3_moe_front` (routing, latent quantization, shared SiTU, the routing tables), FC1 / FC2 (TRT-LLM gen mxfp8 × mxfp4 batched GEMMs) and the launches around them.** Notes go in `notes/r-moe.md`.

Read [`loop/rounds/r3/ROOF.md`](ROOF.md) first: done-means, references, numerics, GPUs.

Where it stands (`roofline/decode.md`, "MoE"): a touched expert costs 1.63-1.67 µs (FC1) and 0.81 µs (FC2), 98% of HBM; the gap is a fixed intercept per launch, FC1 6-12 µs and FC2 3-5 µs, which makes FC1 2.95× and FC2 3.62× their floor at 1 row, 1.49× / 1.44× at 8, ~1.2× at 32-64. `moe_front` is 6.5 µs at every row count (2.0× its estimated floor of two dependent passes).

Directions, measured first:

1. The intercept. At 1 row only ~90 (FC1) and ~53 (FC2) CTAs have work on 152 SMs, and the first and last waves run partly empty. The kernel index has other TRT-LLM gen variants of both GEMMs (tile N, split-K, cluster shape, `$KERN_INDEX_DIR`); route per row bucket with `when` (`--moe-variants` takes one pair today). Find each bucket's best variant by measurement.
2. `moe_front`: one block a row, the last block builds the tables. FlashInfer / TRT-LLM's routing (`trtllm_fused_moe_routing_*`, `routingIndicesClusterKernel`: one cluster does top-k and the tables) and SGLang's `route_quant_fused_kernel` are the references; aim at one pass.
3. FC2 → finalize → all-reduce: FlashInfer's `trtllm_moe_allreduce_fusion` and `cake_moe_finalize_allreduce_fusion` fuse the finalize into the all-reduce, as `ar_finalize_rms` does; check what they do that ours does not (the `r-ar` run owns that kernel: coordinate through notes, do not edit it).
4. Overlap of the shared expert's `sh_down` with FC1 / FC2, or FC2 behind FC1's tiles, hides the intercept (`roofline/decode.md` headroom item 2, ≤ 1400 / 900 µs a step at 64 / 8 rows): say whether the runtime can express it (read `$K3_KERN/docs/runtime.md`); if not, a proposal with a measured estimate.

The bench's pseudo-random tokens touch two thirds of the experts real text does: judge with `k3_step` teacher-forced on real text too (`loop/d/check`'s token file) where touched experts matter.

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
