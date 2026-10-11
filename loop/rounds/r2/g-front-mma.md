Run `g-front-mma` · **target: D's K=7168 f32-out GEMMs, `front` (6016), `qkvg` (6400), `wfu` (3648); design: a weight-streaming kernel (`cp.async.bulk` / TMA ring, `mma.sync` or whatever wins), split-K with an in-kernel deterministic reduction.** Notes go in `notes/g-front-mma.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

The same shapes as `g-front`, on purpose with a different design; the orchestrator keeps the better. At 8-48 rows these GEMMs need ~150-300 TFLOP/s against the weight stream, so the MMA family is free to choose; what decides the time is the stream (the plain `cp.async.bulk` read of `front`'s 86 MB is 13.6 us alone), filling all 152 SMs (split K), and reducing without a second launch. Directions:

1. Get the stream to the read floor at every row count 1-64, reduction included, deterministic.
2. Then the bits: if the kernel does not match cuBLAS bit for bit, either change the order until it does (see GEMM.md) or prove the difference (GEMM.md's three steps). A faster kernel with a proved, smaller-or-equal error is a valid result; say which you have.
3. Then the prologue: fold the RMSNorm before `front` / `qkvg` into the GEMM if the bits allow.

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
