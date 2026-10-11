Run `g-moe` · **target: D's MoE GEMMs, `moe_fc1` / `moe_fc2` (TRT-LLM gen mxfp8 x mxfp4 grouped GEMMs, 112 local experts), and the routing / finalize launches around them.** Notes go in `notes/g-moe.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

The MoE GEMMs are D's largest GEMM time (5.6 / 8.4 ms of a 93-layer step at 24 / 48 rows on the bench's routing) and already stream at 6.2-7.0 TB/s, so a better kernel alone has little to give; real text routes wider (28.8 / 42.7 active experts a member) and the busiest member sets the step. Directions:

1. Measure first: per-call time against the bytes of the experts actually active (bench routing and real-text routing, `notes/d-mlp.md`), per row count; where the gap is (ramp, tail, the 2-CTA cluster's waves over few experts).
2. A grouped kernel of ours only where (1) shows a gap worth it, in the same numerics (mxfp4 weights with their scales, mxfp8 activations, f32 accumulation): bit-identical to TRT-LLM's or proved per GEMM.md.
3. Fusion around them: `moe_front`'s quantize into fc1's prologue, fc1's SwiGLU + requantize already in its epilogue, fc2's output with `moe_finalize` / the shared expert (coordinate `sh_down` with `g-small`).

Expert-parallel skew and expert replicas are layout: not this run's. Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
