# r-gemm: D's dense projections at their weight-stream floor (round 3)

Target: `qkvg`, `front`, `wfu`, `q_b`, `lm_head` (f32 out, `cublas_bf16_tn_f32`) and `o_proj`, `sh_down`,
`lat_up`, `wgu`, `w_dn` (bf16 out, `cublaslt_bf16_tn`). Check `loop/d/check loop/out/d-l4-tp8.json`, bench
`loop/d/bench loop/out/d-l16-tp8.json`. Main 01d6dc5: 12.9 launches/layer, 3.488 ms/step (base-r3).

## State (rebased on main 21a9174)

Merged on main (95bc0b0): the closing add's hints, the all-reduces' loads, the o_proj evict_last pin (3.488 ->
3.405). Offered on this branch (71464fe own split-K GEMM for qkvg / front at 1-32 rows; c8d2d9f bf16 projections
over the rows rounded up to 8; 205a325 front GEMM pins sh_down in L2), all bit-identical (loop/d/check PASS):
**against main 21a9174, same lease, 3 rounds each: 3.2903 -> 3.2590 ms/step (-0.95%)**; 8 / 16 / 24 rows -30..-55
us a step, 36 / 48 even. Earlier, against main d85613c: own GEMM -0.54%, sh_down pin -0.11%, row rounding even
on the bench and -1.3% at 1 row.
Launches run per step: main's minus one per qkvg / front call at 1-32 rows (cuBLAS's splitKreduce_kernel gone);
loop/count reads 20.4 a layer against main's 13.7 (it counts the four `when` buckets of gemm_t6400 / gemm_t6016,
of which one runs; a single bucket loses the whole gain, see below; three buckets (16 / 24 / 32, loop/count 18.8)
cost +0.15%, 3.3657 vs 3.3607, the 8-row steps +10-15 us).

## The table (rank 0, nsys critical path, us a call, main d85613c -> this branch 499159d, ratio to decode.md's HBM floor)

A ratio under 1 means the weight came from L2 (a predecessor loaded it during its own wait): the call's floor is then
L2's, and the HBM bytes moved earlier. main d85613c already carries the L2 loads / hints / o_proj pin (95bc0b0); the
columns differ by the own GEMM (qkvg / front at 1-32 rows), the sh_down pin and the rows rounded up to 8. 16-layer
step, same lease (prof11). Step p50, main -> branch: 1 x 8k 1.95 -> 1.92 ms, 8 x 8k 2.40 -> 2.37, 32 x 8k 3.65 ->
3.57, 64 x 8k 4.62 -> 4.60, 32 x 32k 3.68 -> 3.65.

| call | n x k | 1 x 8k | 8 x 8k | 32 x 8k | 64 x 8k | 32 x 32k |
|---|---|---:|---:|---:|---:|---:|
| `qkvg` | 6400 x 7168 f32 | 15.4 -> **12.2** (0.98x) | 14.2 -> **12.4** (0.99x) | 14.9 -> **14.5** (1.15x) | 14.5 -> **14.5** (1.12x) | 14.9 -> **14.6** (1.15x) |
| `front` | 6016 x 7168 f32 | 12.9 -> **10.8** (0.91x) | 11.3 -> **10.9** (0.92x) | 14.2 -> **13.8** (1.16x) | 17.4 -> **16.8** (1.39x) | 14.2 -> **13.9** (1.16x) |
| `lat_up` | 7168 x 3584 | 7.5 -> **5.5** (0.79x) | 5.3 -> **5.6** (0.80x) | 5.9 -> **6.1** (0.87x) | 6.5 -> **6.6** (0.92x) | 5.9 -> **6.1** (0.87x) |
| `o_proj` | 7168 x 1536 | 4.2 -> **4.2** (1.39x) | 4.3 -> **4.4** (1.45x) | 4.4 -> **4.4** (1.41x) | 4.5 -> **4.5** (1.40x) | 4.6 -> **4.6** (1.48x) |
| `sh_down` | 7168 x 768 | 3.6 -> **2.7** (1.79x) | 3.7 -> **2.7** (1.80x) | 4.0 -> **3.0** (1.92x) | 4.0 -> **4.0** (2.48x) | 4.0 -> **3.0** (1.94x) |
| `wfu` | 3648 x 7168 f32 | 9.9 -> **9.5** (1.34x) | 9.7 -> **9.8** (1.37x) | 10.9 -> **10.8** (1.49x) | 10.6 -> **10.4** (1.41x) | 10.8 -> **10.9** (1.51x) |
| `q_b` | 18432 x 1536 f32 | 10.7 -> **10.2** (1.33x) | 10.6 -> **10.7** (1.37x) | 11.2 -> **11.2** (1.39x) | 11.6 -> **11.6** (1.38x) | 11.4 -> **11.1** (1.38x) |
| `lm_head` | 20480 x 7168 f32 | 46.1 -> **46.6** (1.16x) | 48.5 -> **48.3** (1.20x) | 45.5 -> **45.1** (1.11x) | 44.9 -> **44.5** (1.09x) | 45.3 -> **44.5** (1.10x) |
| `wgu` | 8448 x 7168 | 15.6 -> **14.6** (0.88x) | 16.3 -> **15.6** (0.95x) | 19.9 -> **19.6** (1.17x) | 24.4 -> **23.9** (1.41x) | 19.9 -> **19.4** (1.16x) |
| `w_dn` | 7168 x 4224 | 11.6 -> **11.3** (1.36x) | 11.5 -> **11.9** (1.43x) | 12.9 -> **13.0** (1.54x) | 13.8 -> **14.4** (1.68x) | 13.0 -> **12.9** (1.53x) |

Against main 01d6dc5 (before any of this run's work; prof8, rank 0): qkvg 17.6-19.2 -> 12.2-14.6 us, front 13.4-18.4
-> 10.0-16.7, lat_up 8.1-8.6 -> 5.5-6.6, o_proj 5.0-5.8 -> 4.2-4.6, sh_down 3.7-4.0 -> 2.7-4.0, wfu 11.6-13.3 ->
9.5-10.9.

Paid elsewhere for the L2 loads (now on main; prof8 against 01d6dc5): the closing add +27..+48 us a step (its hint
blocks, 1.7-3 us a call), the attention all-reduce +40..+58 at 1-8 rows (the loads' grace), within +-20 at 32-64.
r-ar has since reworked both (their branch: the closing add's hints capped at 16 MB, the loads' window).

The named limits of what is not within ~1.15x:
- `sh_down` (10.5 MB): pinned at evict_last by our front GEMM (1-32 rows) it reads from L2 at 2.7-3.0 us (1.8-1.9x;
  cuBLAS's hot time for it alone is 2.3-2.9): the 112-CTA 64 x 8 tile cuBLAS keeps at every row count (no
  split-K-1 cuBLASLt config is faster at 1-16 rows, 19202 tried; 0.7-0.9 us at 17-48 alone, nothing in the step).
  At 64 rows cuBLAS runs front and nothing pins it (4.0 us, 2.5x). What would move it further: computing it inside
  ar_finalize_rms (its A, shared_act, is ready before FC1; a K-ascending mma chain is bit-identical to cuBLAS's
  split-K 1, g-small): -1 launch a layer and its 2.7-4 us, but that kernel is r-ar's: a proposal for r-ar.
- `o_proj` (21 MB): read from L2 (pinned by the MoE all-reduce one layer earlier), it is at cuBLAS's own L2 time
  (3.2-4.3 us alone, hot): 112 CTAs (n / 64, one K chain an output, no split to add without changing the bits).
- `wfu` / `q_b` (4 calls a step each): wfu gets only the closing add's hints; our split-K kernel at 2 splits is
  slower than cuBLAS (114 CTAs, 12.6-14.4 vs 11.7-12.1 cold). q_b follows mla_prep (r-attn's, 3 us): no host.
- `front` at 49-64 rows: cuBLAS runs one K chain an output there (no split): our kernel would need a 448-MMA
  chain a 64-row tile (23 us vs 16.7); kept on cuBLAS.
- `w_dn` (layer 0 only, after situ), `lm_head` (1 call, 280 MB: the closing add hints the first 96 MB).

Not tried, for a next session: sh_down (and o_proj) computed inside the all-reduce after them (r-ar's kernels, a
proposal: -1 launch a layer each; for sh_down a 48-MMA mma.sync chain a 16 x 8 tile, ~2.5-3 us inside the AR vs
2.7 us + a launch now, so mostly a launch). qkvg / front at 33-48 rows: tried once (below, swapped operands);
what is left is a 2-3-CTA-an-SM kernel whose A stages do not crowd the weight ring.

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
4. **dbedd6a** (was 1ba6abe before the rebase) qkvg / front at 1-32 rows: our own GEMM (`source/k3_gemm_f32.cu`),
   cuBLAS's split-K order with the reduce in the same launch (cluster of the three splits, DSMEM push, one
   barrier): 3.4120 -> 3.3995 (A/B, -0.37%); rebased, 3.3873 -> 3.3690 vs main d85613c. Bit-identical (harness
   every m 1..48, three seeds; d/check). See "Own f32 GEMM" below.
5. **2603ca7** D's bf16 projections (o_proj, sh_down, lat_up, wgu, w_dn) pass m = ceil(tokens / 8) * 8: cuBLAS's
   1-7-row kernels read L2 at 0.6-0.7x its 8-row one, same bits (work/h/m1.cu). 1 row 1.828 -> 1.804 ms.
6. **499159d** our front GEMM ends with one cp.async.bulk.prefetch.L2 a CTA at evict_last over its share of the
   layer's sh_down weight; the next of our GEMMs demotes it before its wait: -0.11%; sh_down 3.7 -> 2.7 us a call
   at 8 / 24 rows. Issued from warps 2-3 before the epilogue instead (hints a line), front paid 1 us a call (no net
   gain).

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
- the f32 projections over the rows rounded up to 8 too (wfu / q_b 1-row cuBLAS kernels are 0.75x their 8-row
  rate from L2, bit-identical; our GEMM's buckets unchanged): 1 row +5 us, the bench even (3.3690 vs 3.3680).
- o_proj kernel holding its whole slice, second try with A boxes sized to the rows and a 12-deep A ring: still
  6.6 vs 4.6 us after a 9 us PDL predecessor (the dependent 96-MMA chain and the post-wait work, not A).
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
- qkvg / front at 33-48 rows with the operands swapped (rows of A the M = 64 operand, 128 weight rows N: a 16 KB
  weight box beside an 8 KB A box at any m, 8 stages, one CTA an SM, the splits added in order through global
  scratch by the last of three; work/h/k3_gemm_f32w.cu): bit-identical 1..48, but 26 vs 18 us cold, 20 vs 10 hot.
  Its loop is fine (7.0 us) and the epilogue 1.7; a 194 KB CTA cannot sit beside the kernel before it, so every
  call pays its whole start (~2.5 us) after the predecessor, and it is worse back to back.
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
