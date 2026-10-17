Run `r-gemm` · **target: D's dense projections at their weight-stream floor: `qkvg`, `front`, `wfu`, `q_b`, `lm_head` (f32 out) and `o_proj`, `sh_down`, `lat_up`, `wgu`, `w_dn` (bf16 out).** Notes go in `notes/r-gemm.md`.

Read [`loop/rounds/r3/ROOF.md`](ROOF.md) first, then [`loop/rounds/r2/GEMM.md`](../r2/GEMM.md) (its bit-identity rule applies) and round 2's notes on the branches `agent/g-front`, `agent/g-small`, `agent/g-floor` (in `$K3_LOOP_DATA/../repo`): our own GEMMs did not beat cuBLAS there; do not repeat that without a new reason.

Where they stand (`roofline/decode.md`, "Dense projections"): every weight byte is read once, but only the three projections whose weight the all-reduce before them prefetches into L2 run at 1.08-1.17×; the rest sit at 1.38-2.5×.

Measured levers:

1. Prefetch every projection's weight into L2 in the launch before it (`k3_ar_fused` does it for `front`, `wgu`, `lat_up` with `prefetch.global.L2` from its idle blocks). Headroom item 3: 740 / 680 / 660 / 520 µs a step at 1 / 8 / 32 / 64 rows. The predecessors are other runs' kernels (`r-ar`, `r-attn`, `r-moe`): add the prefetch arguments in the generator and the prefetch loop in a shared helper; tell the owner in your notes and keep the change minimal in their file.
2. The n = 7168 bf16 projections launch 112 CTAs on 152 SMs (64 × 8 tile, no split-K): pin an algorithm that fills the SMs (cuBLASLt algo / tile / split-K in the manifest), bit-identical or proved.
3. The f32-out path pays a `splitKreduce_kernel` (2.5-3.6 µs, 186 extra launches at 1 row): an algorithm or kernel without it.
4. SGLang runs FlashInfer's TGV GEMM and its CuTe-DSL bf16 GEMM at 1 and 8 sequences: measure them against cuBLAS on our shapes before deciding anything.

Check: `loop/d/check loop/out/d-l4-tp8.json` (bit-identical first). Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
