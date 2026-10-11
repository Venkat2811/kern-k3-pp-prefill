Run `g-front` · **target: D's K=7168 f32-out GEMMs, `front` (6016), `qkvg` (6400), `wfu` (3648), as one kernel family; design: tcgen05 (UMMA, TMEM), aiming at bit-identity with cuBLAS.** Notes go in `notes/g-front.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

These three are ~60% of what D's dense GEMMs have to give (front 0.32, qkvg 0.23, wfu 0.07 ms a step alone at 24 rows, more in the step). cuBLAS runs them split-K with a separate `splitKreduce_kernel` (185 launches a step across the three). Directions:

1. Identify cuBLAS's kernel per row count (main's D step under nsys), then build a tcgen05 kernel that reproduces its accumulation order with the split-K reduction inside the kernel (fixed order: last-arriving CTA or cluster / DSMEM reduction), so the launch count drops by 185 a step and the bits stay main's.
2. Then the input side: `front` and `qkvg` read the normed residual that the kernel before them writes (`ar_attnres_rms`, `land_add2_attnres_rms_snap`); folding the norm into the GEMM's prologue removes a pass, if it keeps the bits.
3. `front`'s output feeds `moe_front` (router top-k on the first 896 columns): f32 stays.

`g-front-mma` takes the same shapes with a different design; you do not share code. Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
