# r-gemm: D's dense projections at their weight-stream floor (round 3)

Target: `qkvg`, `front`, `wfu`, `q_b`, `lm_head` (f32 out, `cublas_bf16_tn_f32`) and `o_proj`, `sh_down`,
`lat_up`, `wgu`, `w_dn` (bf16 out, `cublaslt_bf16_tn`). Check `loop/d/check loop/out/d-l4-tp8.json`, bench
`loop/d/bench loop/out/d-l16-tp8.json`. Main 01d6dc5: 12.9 launches/layer, 3.488 ms/step (base-r3).

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
