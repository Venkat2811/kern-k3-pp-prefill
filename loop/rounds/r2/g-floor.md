Run `g-floor` · **target: why D's GEMMs run slower in the step than alone, and the cross-GEMM levers that follow (scheduling, L2, overlap); measurement first, then kernels.** Notes go in `notes/g-floor.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

Alone, `front` takes 17.0 us at 24 rows (split-K reduce included); in the step nsys measured 19.5 us plus a ~2.9 us reduce. Directions:

1. Profile main's D step (`loop/d/bench`'s manifest, real TP8) per GEMM call: start-to-end, clocks, HBM throughput, what overlaps it (PDL early blocks of the next kernel, the all-reduce's L2 prefetch, the other ranks' NVLink traffic), against the same call alone (`harness/d_gemm_floor.cu`). Split the gap into causes with numbers. Write the per-shape table (alone / in step / floor) into your notes early; the other runs use it.
2. Then act on what (1) finds, in kernels or the generator: e.g. prefetching weights into L2 during communication (round 1 did it for `front` and `lat_up` from the all-reduces' idle blocks), ordering, launch shape. Do not rewrite a GEMM another run owns; give them the finding.

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
