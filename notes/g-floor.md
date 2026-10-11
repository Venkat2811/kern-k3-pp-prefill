# g-floor: D's GEMMs in the step against alone, and the cross-GEMM levers

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
main 4e59b91: 12.9 launches/layer, 3.486 ms/step (orchestrator's number).

## The table (2026-10-11, main 4e59b91, real TP8, rank 0 on its own GPU)

"Alone" is `work/h/gemm_alone` (the call exactly as kern issues it: `cublasGemmEx` f32 out with a 32 MB
workspace for `gemm_f32`, cuBLASLt heuristic #0 bf16 out for `gemm_bf16`; a graph of 32 calls, weight rotating
past L2, per call), run on rank 0's GPU in the same lease right before the step (`ALONE=1 work/dprof.sh`).
"In step" is the **critical path** of the call in the 16-layer step under nsys (`work/table.py`): its end
(split-K reduce included) minus the later of its start and the previous kernel's end, median over the calls
of the step and three replays. "s-e" is start-to-end (GEMM start to reduce end), what nsys shows as the
kernel's span. Floor: the plain read of the weight (GEMM.md). us per call.

| op | floor | 8: alone / step / s-e | 16 | 24 | 36 | 48 |
|---|---|---|---|---|---|---|
| front | 13.6 | 16.4 / **13.4** / 19.8 | 16.6 / 14.1 / 21.7 | 17.0 / **15.0** / 22.9 | 17.8 / 15.5 / 22.4 | 18.0 / 16.4 / 24.0 |
| qkvg | 14.4 | 17.1 / 18.1 / 22.2 | 17.4 / 18.0 / 22.1 | 17.8 / 18.6 / 22.8 | 18.0 / 19.1 / 24.2 | 18.2 / 19.2 / 23.5 |
| wfu | 9.0 | 11.6 / 12.2 / 16.1 | 11.8 / 12.3 / 16.4 | 11.9 / 12.6 / 16.7 | 12.1 / 13.2 / 18.5 | 12.1 / 13.0 / 17.1 |
| q_b | 9.5 | 10.3 / 10.9 / 13.0 | 10.4 / 10.8 / 12.6 | 10.5 / 11.0 / 13.1 | 10.9 / 11.7 / 14.6 | 10.7 / 11.3 / 13.3 |
| o_proj | 5.0 | 5.0 / 5.1 / 13.5 | 5.0 / 5.1 / 7.0 | 5.7 / 5.7 / 7.3 | 5.9 / 6.1 / 7.4 | 5.9 / 5.9 / 7.4 |
| sh_down | 2.2 | 3.8 / 3.7 / 6.0 | 3.8 / 3.7 / 6.0 | 4.1 / 3.9 / 6.1 | 4.1 / 4.2 / 6.6 | 4.1 / 4.0 / 6.2 |
| lat_up | 8.9 | 9.7 / **8.4** / 25.3 | 9.8 / 8.4 / 28.4 | 9.8 / **8.5** / 31.2 | 9.8 / 8.8 / 33.4 | 9.8 / 8.7 / 38.8 |
| lm_head | 41.4 | 51.1 / 50.7 / 54.9 | 50.9 / 50.7 / 55.1 | 45.9 / 47.6 / 51.7 | 46.1 / 49.9 / 54.8 | 46.4 / 47.7 / 51.7 |

Sum over the 16-layer step's dense GEMMs (us): in step 823 / 835 / 866 / 909 / 905 at 8 / 16 / 24 / 36 / 48
rows, alone 873 / 882 / 904 / 924 / 929, read floor 739; the step is 2586 / 3042 / 3534 / 4186 / 4704.

Same kernels in both: nvjet `tss` (f32) / `tst` (bf16) tiles per row count match name for name (e.g. front
64x8 / 64x16 / 64x32 / 128x48 / 128x48, grids 288 / 282 / 282 / 144 / 144). The 8 GPUs of a lease agree
within ~0.5% alone.

## Why the step looked slower: the split of the gap

1. **Start-to-end is not the cost under PDL.** Our kernels trigger their dependents at entry, so the cuBLAS
   kernel becomes resident as soon as the predecessor's blocks free SMs and waits in `griddepcontrol.wait`.
   Its nsys span starts 4.2 us (qkvg), 7.6 (front), 18-30 (lat_up, behind `ar_finalize_rms`'s early-exiting
   idle blocks), 1.3-8 (o_proj) before the predecessor ends. The 19.5 + 2.9 us of `front` is that span; its
   critical path is 13.4-16.4 us.
2. **nvjet does nothing useful before its wait** (`work/h/pdl_pre.cu`: a predecessor spinning X us with a PDL
   trigger at entry, then the GEMM): front's critical path 16.58 us when its CTAs sat resident beside a
   24-block spinner for 5-20 us, 16.83 when they could not get on (152 blocks), 16.74 back to back. Early
   residency hides ~0.25 us of launch, no weight. Also: a 0-smem, 256-thread spinner on all 152 SMs keeps the
   GEMM off every SM (the shared-memory carveout cannot change under a resident block), so co-residency
   needs free SMs, not small blocks.
3. **The all-reduces' L2 prefetch makes the prefetched GEMMs faster than alone**: front -1.6..-3.0 us,
   lat_up -1.0..-1.3, and `l1.qkvg` (after layer 1's `ar_attnres_rms`, which prefetches it) 13.9-14.1 us
   against 18.3-19.4 for every other qkvg (after `land_add2_attnres_rms_snap`, which does not).
4. **The cold f32 split-K calls run ~0.5-1.1 us slower than alone** (qkvg, wfu, q_b; o_proj / sh_down
   within 0.1-0.2; lm_head +1.5-3.8 at 24-36 rows). Not the TLB (alone is flat from 0.6 to 30 GB of weight
   span), not the predecessor's footprint (spin test above), not a different kernel. Open.

So the GEMMs are not slower in the step; what the step leaves is (a) every GEMM's distance from the read
floor (the kernel runs' business: g-front, g-front-mma, g-small, g-wide) and (b) the cold GEMMs that sit
after an HBM-idle kernel and are not prefetched: qkvg (69 a step, ~4.5 us each when prefetched), wfu (24),
q_b, o_proj, sh_down. (b) is this run's lever.

## Log
