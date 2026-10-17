# r-moe: D's routed MoE at its floor (round 3)

Target: `k3_moe_front`, FC1 / FC2 (TRT-LLM gen mxfp8 x mxfp4 batched GEMMs) and the launches around
them in the TP8 / EP8 / DCP8 decode step. Main at the start: 3.486 ms/step weighted, 12.9 launches a layer.

## Where it stands

- Committed: `c70361f` (moe_front top-k by warp bitonic sorts and a merge tree, bit-identical),
  `79e5da7` (moe_front sorts only the keys above the 16th largest warp maximum, bit-identical).
  moe_front 6.5-6.6 -> 5.5-5.6 us a call in the step at 8-64 rows; D 16 layers weighted
  3.4857 -> 3.4733 ms/step against main in one lease (3 interleaved rounds each). Launches 12.9/layer.
- FC1 / FC2: no variant, bucket or grid change beats main's pair (below); the remaining gap is the
  cubins' one CTA per SM (wave quantization, ~46 GB/s a CTA) and FC1's wait at entry.
- The largest MoE cost left is EP skew (ar_finalize_rms waits for the busiest rank), a placement
  question; measured estimate under Proposals.

## Tools (work/, not committed)

- `work/dprof.sh OUT MANIFEST [ROWS] [CTX]`: the D bench with rank 0 under nsys (graph nodes); 20 s a run
  when GPUs are free. `work/dprof2.sh OUT MANIFEST <k3_step args>`: the same for teacher-forced runs.
- `work/kt.py`, `work/sum.py DIR...` (FC1 / FC2 / moe_front / ar_finalize busy us per call by rows),
  `work/tl.py SQLITE ROWS` (one step's per-layer timeline around the MoE).
- `work/ab.sh OUT ROUNDS tag=manifest...`: one TP8 lease, manifests benched interleaved (loop/d/bench's
  shapes); the noise between leases is larger than most effects here.
- Real routing: `work/mkrows.py` (R rows of the check's real text, distinct tokens per step: row r
  reads the four rows concatenated from offset r * 2560 / R), `work/peridx.py` (each layer's top-k ids
  in its own buffer, for `--dump`), `work/touched.py` (touched experts per rank from the dumps).
- `work/mf/` (moe_front harness: any cubin, CUDA graph, outputs dumped for bit compares, %globaltimer
  phase stamps), `work/shd/` (sh_down harness vs cuBLASLt and fp64), `work/pdl/` (PDL ordering test).
- Main's stored manifests in `$K3_LOOP_DATA/base/` no longer verify with this k3_step (`span_at`):
  A/B against a manifest generated from HEAD's scripts (`git archive HEAD scripts kernels.toml`).

## Floors (real text, 16-layer bench, rank 0)

Touched experts per rank (of 112), layers 1-15, real text (`work/touched.py`; 48 rows x 160 steps):

| rows | 1 | 8 | 16 | 24 | 36 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| rank 0 | 2.03 | 12.75 | 22.56 | 29.97 | 41.28 | 50.27 |
| mean rank | 2.00 | 12.60 | 22.14 | 29.76 | 40.97 | 49.98 |
| busiest rank | 4.25 | 17.58 | 28.12 | 36.28 | 48.18 | 57.44 |

An expert is 11,698,176 B (FC1) + 5,849,088 B (FC2): 1.596 + 0.798 us at 7.33 TB/s.

| kernel (us/call, rank 0) | 1 row (bench) | 8 rows real | 24 rows real | 48 rows real |
|---|---|---|---|---|
| FC1 measured / floor | 8.8 / 3.2 (2.75x) | 28.1 / 20.4 (1.38x) | 60.6 / 47.8 (1.27x) | 91.8 / 80.2 (1.14x) |
| FC2 measured / floor | 5.4 / 1.6 (3.4x) | 13.7 / 10.2 (1.34x) | 29.1 / 23.9 (1.22x) | 43.8 / 40.1 (1.09x) |
| moe_front (main) | 6.9 | 6.2 | 6.5 | 6.5 |

Bench routing (pseudo-random tokens) at 8 / 24 / 48 rows: FC1 26.6 / 53.0 / 79.6, FC2 12.7 / 25.3 / 37.6,
moe_front 6.6 / 6.5 / 6.6, ar_finalize_rms 22.5 / 29.0 / 32.1 (most of it waiting for the busiest rank).

## Done table (rank 0, us per call; floors from real-text touched experts; bench = pseudo-random tokens)

Real text, teacher-forced (`work/dprof2.sh`, `work/mkrows.py` rows), 16-layer step, layers 1-15:

| kernel | 1 row | 8 rows | 24 rows | 32 rows | 48 rows | 64 rows | limit |
|---|---|---|---|---|---|---|---|
| FC1 meas / floor (x) | 8.8 / 3.2 (2.75) | 28.1 / 20.4 (1.38) | 60.6 / 47.8 (1.27) | 72.9 / 57.6 (1.27) | 91.8 / 80.2 (1.14) | 105.9 / 93.4 (1.13) | cubin waves + entry wait |
| FC2 meas / floor (x) | 5.4 / 1.6 (3.4) | 13.7 / 10.2 (1.34) | 29.1 / 23.9 (1.22) | 34.9 / 28.8 (1.21) | 43.8 / 40.1 (1.09) | 50.7 / 46.7 (1.09) | cubin waves |
| moe_front now / est. floor | 5.9 / 4.3 | 5.6 / 4.3 | 5.5 / 4.3 | 5.6 / 4.3 | 5.6 / 4.3 | 5.7 / 4.3 | latency chain (below) |

(1-row FC numbers are bench routing: 2.0 touched experts on rank 0 either way. 32 and 64 rows: touched
counts from the first rows of the 64-row dump, the profile from its own 32 / 64-row file.)
Bench (8k context) 1 / 8 / 32 / 64 rows after the commits: FC1 8.8 / 26.3 / 61.3 / 91.3, FC2 5.4 / 12.5 /
29.1 / 43.4, moe_front 5.9 / 5.6 / 5.6 / 5.7. At 64k / 128k the same within 1%.

moe_front's floor estimate is its chain of dependent steps, each measured in the harness: launch to
first instruction ~0.8 us, the front row from L2 ~0.6, the selection ~1.0 (warp max, tau, compaction,
one sort + merge), the cross-block hand-off (atomic acq_rel round trip) ~0.8, the ids from L2 ~0.6,
the tables ~0.5. Measured 5.5-5.7 in the step, 5.1-5.4 in the harness: 1.2-1.3x of that chain.

## Why FC1 / FC2 sit where they are (the named limit)

- Both cubins run one CTA an SM: 512 threads x 128 registers is the whole register file, and 223-225 KB
  of shared memory. A CTA streams one 128-row panel of one expert (FC1 244 KB, FC2 196 KB).
- FC1's first instruction is `griddepcontrol.wait` (SASS: ACQBULK at 0x140, before both PREEXITs):
  nothing of FC1 overlaps moe_front, and its HBM stream starts only after moe_front's grid completes.
  FC2's weight loader runs before its wait (no ACQBULK on that path): it streams during FC1's tail.
- A wave model fits every measurement: FC1 ~ ceil(48 E / 152) waves x 5.2-5.9 us + ~2 us
  (E = touched experts; 5.06 us is a full wave's 152 panels at 7.33 TB/s): 8 / 24 / 48 rows real
  5 / 10 / 16 waves -> 28.1 / 60.6 / 91.8 us measured. The intercept is the last wave's quantization
  (half a wave on average, partly refilled by FC2's early weight loads) plus the wait at entry.
- No other variant does better (measured in the step, nsys, rank 0; `work/list*.txt`):
  - FC1 tile 16, stages 3-6, 1- or 2-CTA MMA, u2 or not, Pd scheduler: main's (s6, 2-CTA) is best;
    s5 +0.3-0.6 us, s4 +2-4 us, 1-CTA +4-17 us. The static scheduler (schedS) is 2x slower everywhere.
  - FC2 likewise: main's u2_s6 2-CTA = non-u2 s6 (12.9 / 25.1 / 37.2); fewer stages or 1-CTA worse.
  - Tile 8 (needs moe_front's tables at TILE 8: built and checked): s6 = main at 8-48 rows
    (27.2 / 53.2 / 80.7), others worse; split-K 2 (tile 8 only) worse at every row count including
    1-4 rows (FC1 10.4-11.2 vs 8.8 at 1 row): its 2-deep cluster along K leaves 2 x 48 CTAs a touched
    expert, which spills into a second wave already at 2 experts.
  - At 1 / 2 / 4 rows main's pair is also the best (FC1 8.8 / 12.2 / 17.9, FC2 5.4 / 6.4 / 8.8).
  - Grid: the batch dim is launched at its bound (112 + T CTA rows, mostly exiting at once). Cutting the
    bound to 40 + T (a timing-only hack) changed FC1 / FC2 by <= 1%: exiting CTAs are cheap.
- So per-bucket routing (`when`) has nothing to route to; the generator change that would do it
  (`--moe-buckets`, moe_front's TILE define) was written, measured, and not committed.

## moe_front (one block a row, the last one builds the tables)

Harness (`work/mf`, CUDA graph of 20 calls, inputs from L2): 6.0-6.2 us at 8-64 rows. Phase stamps
(%globaltimer, 24 rows, per block): load + sigmoid + top-k 2.65 us, waiting for quant 0.4, the
last-block hand-off (atomic acq_rel round trip + barriers) 0.8, the tables (L2 read of the ids, smem
histogram, scan, writes) 1.3; launch and teardown ~0.8 more between back-to-back calls.

Tried:
- **Threshold, then sort** (kept, `79e5da7`): each warp's maximum 64-bit key (one shuffle butterfly), tau =
  the 16th largest of the 28 maxima (one warp sort), the keys >= tau compacted (ballot + one shared
  atomic a warp; 16-30 of them typically), one warp's running top 16 over them 32 at a time. Exact for
  any candidate count. 5.9 -> 5.1-5.4 us in the harness, 6.5 -> 5.6 in the step. Bit-identical.
- **Bitonic top-k** (kept, `c70361f`, superseded by the above): warp w of 28 sorts its 32 (score key, ~expert) 64-bit keys by a bitonic
  network, 14 warps merge the warps' top 16 pairwise in 5 levels. Bit-identical picks, order and
  weights (the same total order); 6.1 -> 5.9 us in the harness.
- One 8-CTA cluster doing every row's top-k and the tables from DSMEM after a cluster barrier
  (FlashInfer's routingIndicesClusterKernel shape), row blocks doing quant / SiTU: bit-identical but
  the top-k is throughput-bound on the SM, ~1.6 us per extra row a CTA: 5.7 (8 rows) / 7.3 (24) /
  10.6 (48) us. With 8 slice warps a row (<= 4 rows a CTA): 5.9 / 7.3 (16) / 8.9 (24). Dropped.
- Ballot-based rounds (cached lane best, the winner lane rescans): slower (6.6), the divergent winner
  branch costs more than the second REDUX. 16 slice warps: slower (merge over 256 candidates).
- Generation-tagged picks polled by an extra block instead of the atomic hand-off: no faster.
- Quant moved to 4 spare warps from the start, hand-off issued right after the picks: slower (6.5:
  four warps cannot hide the quant's L2 loads); hand-off early alone: no change.
- **L2 prefetch of FC1's first wave from moe_front** (blocks prefetch their picks' FC1 weights below a
  row-count-dependent expert id): FC1 got 1.6-3.5 us faster, as predicted, but moe_front took 50-58 us:
  `prefetch.global.L2` from a few SMs runs at ~0.25 TB/s each, and `cp.async.bulk.prefetch.L2` the
  same, because a grid does not complete before its prefetches land, and FC1 waits for the grid.
  Prefetching from the kernel before FC1 cannot overlap FC1's wait.

## The launches around: sh_down (r-gemm's GEMM)

- cuBLAS's sh_down already launches as a programmatic dependent (it starts ~2 us before FC2 ends) and
  ends 3.7-3.8 us after FC2.
- sh_down before FC2 (FC1 -> sh_down -> FC2): 3.489 -> 3.508 ms (worse).
- PDL ordering (work/pdl, measured): a programmatic dependent B that never executes
  griddepcontrol.wait runs fully overlapped with its primary A, and the launch after B (normal, or
  PDL with griddepcontrol.wait) still starts only after A completes. So a sh_down of ours launched
  `pdl: true` after FC2 that never waits is safe, and takes the SMs FC2's last wave leaves.
- `source/k3_sh_down.cu` (written for this, not committed): bf16 mma.sync, weights by one bulk copy a
  warp, the activation staged once a CTA. vs cuBLASLt alone: 3.4 / 3.4 / 3.9 / 4.2 / 7.0 us at
  1 / 8 / 16 / 24 / 48 rows against 3.0 / 2.9 / 2.8 / 3.7 / 3.8; 99.97% of outputs bit-identical to
  cuBLAS, the rest one bf16 ulp either way, same max / mean error vs fp64. mma.sync's latency on
  sm_103 (~140 ns a dependent mma) bounds it: it needs 8-16 independent accumulator chains a warp.
  In the step (three builds by `when`): 3.485 -> 3.482 ms, noise. Not committed.

## Proposals

- **EP balance (largest MoE lever, outside the fixed layout).** Simulated on the dumped real routing
  (`work/epsim.py`, 64 rows x 160 steps x 15 layers; a replicated expert goes to the holder with fewer
  touched experts so far, which every rank can compute from the same picks): busiest rank's touched
  experts at 8 / 24 / 64 rows

  | placement | 8 | 24 | 64 |
  |---|---:|---:|---:|
  | EP8 today (mean rank) | 17.32 (12.48) | 35.38 (29.07) | 65.59 (58.23) |
  | each layer's 8 / 16 / 32 hottest experts also on the next rank | 16.68 / 16.24 / 15.63 | 34.65 / 33.93 / 32.81 | 64.69 / 63.81 / 62.43 |
  | every expert on two ranks (r, r + 1) | 13.75 | 30.17 | 59.07 |

  At 2.4 us an expert (FC1 + FC2 at HBM rate), every expert on two ranks saves 8.6 / 12.5 / 15.6 us a
  layer on the critical rank, but doubles the expert memory (182 GB more a rank at 93 layers: does not
  fit). The 32 hottest a layer (+29%) save 4.1 / 6.2 / 7.6 us a layer.
  Why it matters: the busiest rank touches 17.6 / 36.3 /
  57.4 experts at 8 / 24 / 48 rows of real text against a mean of 12.6 / 29.8 / 50.0; every rank waits
  for it in ar_finalize_rms (22.5 / 29.0 / 32.1 us a call on rank 0 against ~10 of its own). The skew
  is binomial spread in the count of distinct experts touched, not a few hot experts, so it needs each
  expert reachable on two ranks; the batch is replicated, so every rank can compute the same balanced
  assignment from the picks with no exchange. Memory is the cost (a second copy of a share of the
  experts).
- **A persistent MoE kernel** (FC1 -> FC2 per expert panel, sh_down filling the drain): removes FC1's
  entry wait behind the tables only if the tables are built inside it, and FC1's / FC2's last-wave
  quantization (half a wave each, 2-3 us). Needs block-scaled tcgen05 MMA and, for numerics, the
  GEMM.md proof (FC1's mxfp8 output requantizes any change of summation order).
