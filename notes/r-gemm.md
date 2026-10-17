# r-gemm: D's dense projections at their weight-stream floor (round 3)

Target: `qkvg`, `front`, `wfu`, `q_b`, `lm_head` (f32 out, `cublas_bf16_tn_f32`) and `o_proj`, `sh_down`,
`lat_up`, `wgu`, `w_dn` (bf16 out, `cublaslt_bf16_tn`). Check `loop/d/check loop/out/d-l4-tp8.json`, bench
`loop/d/bench loop/out/d-l16-tp8.json`. Main 01d6dc5: 12.9 launches/layer, 3.488 ms/step (base-r3).

## State (rebased on main d85613c)

Merged on main (95bc0b0): the closing add's hints, the all-reduces' loads, the o_proj evict_last pin (3.488 ->
3.405). On this branch, for the orchestrator: dbedd6a, our own split-K GEMM for qkvg / front at 1-32 rows.
Against main d85613c, same lease, 3 rounds each: **3.3873 -> 3.3690 ms/step (-0.54%)**; 8 / 16 / 24 rows -12..-33
us a step, 36 / 48 within +-5. loop/d/check PASS, bit-identical.
Launches run per step: main's minus one per qkvg / front call at 1-32 rows (cuBLAS's splitKreduce_kernel gone);
loop/count reads 20.4 a layer against main's 13.7 (it counts the four `when` buckets of gemm_t6400 / gemm_t6016,
of which one runs; a single bucket loses the whole gain, see below).
Step p50 at the round's A/B shapes, main 01d6dc5 -> this branch before the rebase (prof8, same lease): 1 x 8k
2.02 -> 1.95 ms, 8 x 8k 2.52 -> 2.43, 32 x 8k 3.70 -> 3.61, 64 x 8k 4.76 -> 4.65, 32 x 32k 3.78 -> 3.72.

## The table (rank 0, nsys critical path, us a call, main -> head, ratio to the HBM floor of decode.md)

A ratio under 1 means the predecessor pulled the weight into L2 during its own wait: the call's floor is then
L2's, and the HBM bytes moved earlier. 16-layer step, ctx as named.

| call | n x k | 1 x 8k | 8 x 8k | 32 x 8k | 64 x 8k | 32 x 32k |
|---|---|---:|---:|---:|---:|---:|
| `qkvg` | 6400 x 7168 f32 | 17.9 -> **12.2** (0.98x) | 17.6 -> **12.4** (0.99x) | 18.9 -> **14.7** (1.16x) | 18.0 -> **14.5** (1.12x) | 19.2 -> **14.6** (1.15x) |
| `front` | 6016 x 7168 f32 | 14.9 -> **10.0** (0.85x) | 13.4 -> **10.5** (0.89x) | 15.8 -> **13.0** (1.08x) | 18.4 -> **16.7** (1.38x) | 15.9 -> **13.5** (1.13x) |
| `lat_up` | 7168 x 3584 | 8.3 -> **8.6** (1.23x) | 8.4 -> **5.6** (0.80x) | 8.5 -> **6.0** (0.85x) | 8.1 -> **6.6** (0.92x) | 8.6 -> **6.0** (0.85x) |
| `o_proj` | 7168 x 1536 | 5.1 -> **4.3** (1.43x) | 5.0 -> **4.4** (1.45x) | 5.8 -> **4.4** (1.40x) | 5.6 -> **4.5** (1.41x) | 5.7 -> **4.5** (1.47x) |
| `sh_down` | 7168 x 768 | 3.7 -> **3.7** (2.50x) | 3.7 -> **3.6** (2.41x) | 4.0 -> **4.0** (2.56x) | 4.0 -> **4.0** (2.48x) | 4.0 -> **4.0** (2.58x) |
| `wfu` | 3648 x 7168 f32 | 11.6 -> **10.3** (1.45x) | 11.9 -> **9.7** (1.36x) | 13.0 -> **10.9** (1.50x) | 13.3 -> **10.4** (1.41x) | 12.9 -> **10.9** (1.51x) |
| `q_b` | 18432 x 1536 f32 | 10.6 -> **10.6** (1.38x) | 10.4 -> **10.4** (1.34x) | 11.1 -> **11.0** (1.36x) | 11.2 -> **11.3** (1.35x) | 11.1 -> **11.0** (1.37x) |
| `lm_head` | 20480 x 7168 f32 | 48.8 -> **46.3** (1.15x) | 50.6 -> **47.8** (1.19x) | 48.3 -> **45.3** (1.12x) | 48.7 -> **44.6** (1.09x) | 47.6 -> **44.8** (1.11x) |
| `wgu` | 8448 x 7168 | 18.7 -> **17.3** (1.05x) | 17.2 -> **15.9** (0.96x) | 20.3 -> **19.5** (1.16x) | 22.9 -> **23.1** (1.36x) | 20.5 -> **19.8** (1.18x) |
| `w_dn` | 7168 x 4224 | 11.5 -> **11.3** (1.36x) | 11.7 -> **11.8** (1.41x) | 13.6 -> **12.9** (1.53x) | 13.8 -> **14.0** (1.63x) | 13.2 -> **13.1** (1.55x) |

Paid elsewhere for it (same profile): the closing add +27..+48 us a step (its hint blocks, 1.7-3 us a call), the
attention all-reduce +40..+58 at 1-8 rows (the loads' grace), within +-20 at 32-64.

The named limits of what is not within ~1.15x:
- `sh_down` (10.5 MB): it runs right after FC2, whose experts stream 100s of MB through L2, and nothing before FC2
  can hold it: pinned at evict_last from the attention all-reduce it costs more than it saves (3.4185 vs 3.4025);
  normal loads do not survive the stream (work/h/el_gemm.cu). Cold, cuBLAS's 112-CTA 64x8 tile reaches 2.8-3.0
  TB/s; no split-K-1 cuBLASLt config is faster at 1-16 rows (19202 tried), 0.7-0.9 us at 17-48 alone but nothing
  in the step. What would move it: computing it inside ar_finalize_rms (its A, shared_act, is ready before FC1;
  a K-ascending mma chain is bit-identical to cuBLAS's split-K 1, g-small): -1 launch a layer and its 3.7 us,
  but that kernel is r-ar's and being rewritten: a proposal for r-ar.
- `o_proj` (21 MB): read from L2 (pinned by the MoE all-reduce one layer earlier), it is at cuBLAS's own L2 time
  (3.2-4.3 us alone, hot): 112 CTAs (n / 64, one K chain an output, no split to add without changing the bits).
- `wfu` / `q_b` (4 calls a step each): wfu gets only the closing add's hints; our split-K kernel at 2 splits is
  slower than cuBLAS (114 CTAs, 12.6-14.4 vs 11.7-12.1 cold). q_b follows mla_prep (r-attn's, 3 us): no host.
- `front` at 49-64 rows: cuBLAS runs one K chain an output there (no split): our kernel would need a 448-MMA
  chain a 64-row tile (23 us vs 16.7); kept on cuBLAS.
- `w_dn` (layer 0 only, after situ), `lm_head` (1 call, 280 MB: the closing add hints the first 96 MB).


## Tools (work/, not committed)

- `work/dprof.sh PROFILE MANIFEST:OUT...`: one 8-GPU lease, k3_step `--profile` per manifest, nsys on rank 0
  (`ROWS=8,24,48`, one sqlite per shape). `work/cmp.py MANIFEST DIR...`: per call class, summed critical path
  (end - max(start, previous end), split-K reduce folded in) per 16-layer step, and the delta of the last DIR.
- `work/h/lt_algos.cu N K bf16|f32 [m...]`: kern's call and every cuBLASLt heuristic candidate, cold (8 weight
  copies) and hot (one copy, L2), output bits against kern's. `work/h/lt_enum.cu N K [m...]`: every split-K-1
  config (tile x custom x cluster) cuBLASLt accepts for bf16 out. Run with `work/h/run1.sh` (the hosts'
  cuBLAS 13.7, the one k3_step loads).
- `work/h/pf_pair.cu N K f32|bf16 m T MODE CAP [CHUNK]`: a predecessor (152 x 1024; B row blocks spin T us,
  the rest pull the next weight into L2 by MODE) then kern's GEMM call; pair time per iteration.

## Findings

### L2 prefetch hints fill L2 about half as well as loads (work/h/pf_pair.cu, qkvg 6400 x 7168 f32, 8 rows)

GEMM time after a predecessor that pulled the whole weight with ... (us; cold 17.2, hot 9.5-10.1):

| predecessor pulls the weight by | T = 4.5 us | T = 16 us |
|---|---|---|
| nothing | 17.1 | 17.3 |
| `prefetch.global.L2` a line (main's `l2_prefetch`) | 13.9 | 13.5 |
| `prefetch.global.L2::evict_last` | 13.1 | 13.5 |
| `cp.async.bulk.prefetch.L2`, 4-64 KB pieces, paced at 6 TB/s or not | 13.7-14.0 | 13.4-13.7 |
| `ld.global.L2::128B` a line (real loads) | 10.2 (pred +9 us) | 9.8 |

A hint never holds the block, but the GEMM after it runs as if about half the bytes were in L2, however much
time the hints had; pacing does not change it, so it is not queue overflow. Real loads leave the GEMM at its
hot time but hold the predecessor until the data is in. Loads while the row blocks still work, stopping when
they are done ("load-while"): pair (pred + front 6016 x 7168, 24 rows) vs hints:

| T (row blocks' time) | 2 | 4.5 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| hints | 19.5 | 21.1 | 24.2 | 28.4 | 32.2 |
| load-while | 20.8 | 21.4 | 23.2 | 25.4 | 28.7 |

So: hints after a short kernel (`land_add2`, ~4.5 us), load-while in the all-reduces (12-34 us of waiting).

### cuBLASLt algorithms for the bf16 n = 7168 projections

Every split-K-1 nvjet candidate gives kern's bits (one K-ascending chain an output). The heuristic's 8
candidates and a full enumeration (19202 configs at o_proj 8 rows) find nothing faster than kern's pick at
1-16 rows (5.16 vs 5.27 us); at 24-48 rows candidate #1 (tile 64x32 / 64x48 / 48x64) is 0.6-0.7 us faster cold
for o_proj and sh_down, nothing for lat_up.

## Log

1. **ace8c55** `land_add2_attnres_rms` (grid rows -> 152) hints the next weight (qkvg, wfu, lm_head) from its idle
   blocks: 3.484 -> 3.4635 (A/B). Bit-identical. Its blocks are held ~2 us a call by the hints (+27 us a step
   at 8 rows) against qkvg -40, wfu -8, lm_head -3.
2. **073a343** the all-reduces load (not hint) the next weight while their rows wait, stopping a grace after the
   first row's data is in (`source/k3_l2_prefetch.cuh`): 3.4635 -> 3.416 (A/B; -1.95% vs main). Bit-identical x3.
   Grace sweep (A/B, same lease): 0 / 0 us 3.424, 2.5 / 0.5 3.416, 4 / 1.5 3.424, unbounded 3.509 (the last
   member to arrive is held: the loads must stop with the wait). A first version compiled the loads away
   (ptxas drops a load whose value nobody reads; an empty asm is no use): keep the `sink` store.
   **For r-ar** (owner of k3_ar_fused.cu): the change in your file is the include, the two `l2_load_while` /
   `l2_arrive` / `l2_close` call sites and state[1], [3], [7]; the loop is in the shared header.
3. **53f1442** ar_finalize_rms also loads the next layer's o_proj weight (21 MB) at evict_last (cache-policy loads);
   the ar_attnres_rms after that o_proj demotes it (applypriority evict_normal) after its own loads: 3.4175 ->
   3.4090 (A/B, -0.25%), o_proj -1.0..-1.3 us a call. Bit-identical x3. Demoting before the loads: front +0.7 us
   a call (3.4105); demoting in the next ar_finalize_rms (pinned through front and the MoE): no gain (3.4185).
   r-ar: two more params on each of your all-reduces (pf2 on finalize, dm on attnres).
4. **1ba6abe** qkvg / front at 1-32 rows: our own GEMM (`source/k3_gemm_f32.cu`), cuBLAS's split-K order with the
   reduce in the same launch (cluster of the three splits, DSMEM push, one barrier): 3.4120 -> 3.3995 (A/B,
   -0.37%). Bit-identical (harness every m 1..48, three seeds; d/check). See "Own f32 GEMM" below.

Tried and dropped (A/B in one lease each, ms/step):
- sh_down before FC1 with its weight appended to front's (one range for the AR to load): 3.429 vs 3.422. Likely
  fc1's PDL overlap with moe_front lost behind a cuBLAS launch.
- cp.async.bulk.prefetch.L2 (64 KB pieces) in the closing add instead of line hints: 3.434 vs 3.422.
- q_b's weight after wfu's in one buffer (the closing add hints both): ~-0.1%, inside noise; not kept.
- the closing add's hints capped at 32 / 48 / 64 MB: 3.4205 / 3.4220 / 3.4170 vs 3.4110 at 96 MB.
- ar_finalize_rms loading the head of the GEMM after lat_up too (qkvg / wfu): normal loads 48 MB 3.4365 vs
  3.4235 (lat_up +1.6 us a call: its own lines evicted, L2 holds well under 129 MB of weights in practice);
  evict_last 16 / 32 MB: 3.4180 / 3.4305 vs 3.4075.
- cuBLASLt algorithm pins for o_proj / sh_down at 17-64 rows (64x32 / 64x48 / 48x64 tiles, 0.7-0.9 us a call
  faster alone, bit-identical): 3.4205 vs 3.4230 in the step, inside noise; not worth four `when` buckets each.
- sh_down pinned at evict_last by the attention all-reduce, demoted by the MoE one: 3.4185 vs 3.4025.
- own GEMM: one build for 1-32 rows with the MMA's N set at run time (fewer `when` buckets): a stage less than
  the bucketed builds, 0.4-0.9 us a call slower cold; with the buckets, run-time N changes nothing (3.4030 vs 3.4025).
- no closing-add hints before qkvg where our GEMM runs it: neutral at 8-24 rows (3.4055 vs 3.4035, 36-48 lose
  them); our CTAs issuing cp.async.bulk.prefetch.tensor L2 prefetches of their whole slice before
  griddepcontrol.wait: +40 us a step (3.4375).
- wfu with our GEMM (2 splits, block 56): bit-identical, slower than cuBLAS.
- one build for 1-32 rows (run-time N, A box of 32 rows; loop/count 14.6 a layer instead of 19.7): 3.4185 vs
  3.4005 (worse than cuBLAS alone): the buckets' extra ring stages are the whole gain. So the four `when` buckets
  stay; if loop/count is fixed to its own docstring (a `when`-gated launch counts once) they count as one.
- front on our GEMM up to 48 rows (BN 40 / 48 at two CTAs an SM, cold -1.3 us alone): 3.4055 vs 3.4030, +10-15 us
  at 36 rows. qkvg / front from 33 rows stay on cuBLAS.
- our CTAs' idle warps 2-3 loading the rest of their weight slice before griddepcontrol.wait (real loads): +60 us
  at 8-24 rows (3.4415 vs 3.4000), with or without the closing add's hints: the loads race the TMA stream for the
  same bytes. Stopped when griddepcontrol.wait returns (an smem flag): neutral (3.3945 vs 3.3965), and worse
  without the closing add's hints (3.4065): our CTAs are not resident early enough for it to matter.
- prefetching K-major (line t of every row and split before line t + 1, so every CTA's first blocks warm evenly)
  instead of in address order: worse (hints 15.9 vs 14.3 us after, loads too slow to issue): DRAM page locality
  matters more than CTA balance.
- o_proj by a kernel holding its whole 192 KB weight slice in shared memory, requested before
  griddepcontrol.wait (work/h/k3_gemm_pre.cu, bit-identical 1..64 rows): 8.6 vs 5.2 us alone, 7.1 vs 4.7 after a
  9 us PDL predecessor: A's 64-row boxes through a 3-deep ring after the wait cost more than the preload saves.

## Lever 4: FlashInfer's TGV GEMM and SGLang's CuTe-DSL bf16 GEMM

`work/h/tgv.cu` builds FlashInfer's `tgv_gemm_host` (include/flashinfer/gemm/tgv_gemm.cuh, its 10 configs: 64x8 /
64x16 / 64x32 at 6-12 stages, 128x16) against kern's bf16 call (cuBLASLt heuristic #0), weight over 8 copies.
Every config is **bit-identical** to cuBLAS (one K-ascending tcgen05 chain an output, as nvjet's split-K 1), but the
best of them is slower cold on every shape, us a call at 1 / 8 / 16 / 24 rows:

| shape | cuBLAS cold (hot) | best TGV cold (hot) |
|---|---|---|
| o_proj 7168 x 1536 | 5.22 / 5.25 / 5.27 / 5.96 (4.66 / 3.23 / 3.30 / 4.13) | 6.04 / 6.03 / 6.07 / 6.20 (4.93 / 4.25 / 4.07 / 4.36) |
| sh_down 7168 x 768 | 3.04 / 2.80 / 2.81 / 3.54 (3.04 / 2.30 / 2.36 / 2.86) | 3.38 / 3.47 / 3.53 / 3.65 (3.37 / 3.19 / 3.22 / 3.48) |
| lat_up 7168 x 3584 | 9.81 / 9.78 / 9.81 / 9.85 (7.98 / 5.18 / 5.36 / 5.68) | 10.31 / 10.31 / 10.31 / 10.57 (6.27 / 5.92 / 5.96 / 6.20) |

Its one win is lat_up hot at 1 row (6.3 vs 8.0: cuBLAS's 1-row kernel is 2.8 us slower than its 8-row one from L2),
a shape the bench does not weight; not worth a CuTe kernel in the manifest. The CuTe-DSL version
(sglang/.../cutedsl_bf16_gemm.py) is the same design (64 x 8-256 tiles, 6-12 stages, plus a 2-CTA mode whose 128-row
pair tiles still cover n = 7168 with 56 pairs); it needs the CUTLASS Python DSL and torch, which these hosts do not
have, so it is not measured.

## Own f32 GEMM (source/k3_gemm_f32.cu)

Round 2's reason for not beating cuBLAS: the main loop. What made it competitive here, in order (harness
`work/h/gf32.cu`, qkvg 6400 x 7168, us a call, cuBLAS cold 17.3 / hot 9.5 incl. its reduce):
- clusters of three at one CTA an SM (200 KB ring): only ~48 clusters resident at once, qkvg's 50 run in two
  waves: cold 24, hot 20.
- two CTAs an SM (BM 128, 5 stages): cold 16.8 at 8 rows, but 3-4 stages at 32-48 rows (16 KB weight stages
  beside the A rows): 17.8 / 20.4.
- BM 64 (8 KB stages): three CTAs an SM fit every cluster (packing leaves some SMs idle: hot time bimodal
  10 / 14 us), two CTAs an SM only up to ~94-99 clusters (front's 94 fit, qkvg's 100 do not: a second wave).
- epilogue: pull (two barriers, scalar remote loads) 1.5-2 us -> push (16-byte DSMEM stores into the owner, one
  barrier) ~1 us.
- a pre-tiled weight (one cp.async.bulk a stage, made by a derive op) streams ~0.3 us faster than TMA boxes but
  cannot serve cuBLAS; the SPLITS 1 build cuBLAS uses from 49 rows was 23 us vs 16.7 (one 448-MMA chain a
  tile). TMA on the original weight keeps cuBLAS for 33+ rows.
- MMA issue is not the limit (a no-MMA build runs as long); hot, the L2 -> SM stream is (~6.5-12 TB/s by
  placement).
- per-call phases (hot, 8 rows): ~1.4-2.4 us to the first stage (PDL overlaps it in the step), 6.7 us loop,
  ~1 us epilogue.
