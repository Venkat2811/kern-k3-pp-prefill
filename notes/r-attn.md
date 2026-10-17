# r-attn: D's attention kernels at their floor (round 3)

Kernels: `k3_kda_core`, MLA (`k3_mla_prep`, `k3_mla_absorb_tc` (`_mma` under `--kv fp8`), the DSL
`mla_decode_h96_p64` and its grid), the DCP exchange (`k3_dcp`).
Previous lane notes: `notes/d-attn.md` (round 1, every idea tried there).

## Status (04:15 UTC)

Branch rebased on main 21a9174 (r-ar merged): check PASS bit-identical, D
3.295 / 3.295 → 3.294 / 3.293. New on it: absorb on tcgen05 (a block a head,
W_UK by TMA; -0.3..-2.0 µs a call from 8 rows, +0.2 at 1 row) and the
fp8 builds' e4m3 cast in hardware (absorb fp8 16.4 → 8.1 µs at 48 rows;
`--kv fp8` only). Tried and dropped since: W_UK into L2 from mla_prep, the
next KDA layer's state from kda_core (both slower; numbers below), the
fp8 query on tcgen05.

## Status (03:40 UTC)

Merged as d85613c. The orchestrator's 02:52 task 1 (a new MLA kernel) is
answered by measurement in "MLA attention: what a new kernel could win":
the 96 → 128 head padding costs nothing at the bench's cluster counts (the
DSL is HBM-bound there, its MMAs 66% busy, so a positions-as-M tiling,
which fits the DSL's shared-memory budget, would save MMA time that is not
on the critical path), and a no-math kernel with the DSL's grid already takes
16.0 / 39.2 / 53.5 / 133 µs of the DSL's 21.9 / 46.7 / 65.0 / 160.9 at the
bench shapes. What a new kernel could take is the difference: 6-28 µs a
call, 0.8-2.4% of a 16-layer step, by streaming at the full HBM rate under
its math. That is a DSL-class kernel (2-CTA, TMEM, a sub-tile KV pipeline),
not started. kda_core: two more ideas measured and dropped (below).

## Status (02:57 UTC): lane closed

Branch vs main 95bc0b0: D 3.414 → 3.386 ms/step (-0.84%), bit-identical
(loop/d/check, and teacher-forced at 20 / 36 rows), launches 12.9 → 13.7 a
layer counted (kda_core's two `when` builds; one runs). What moved:
kda_core (9.2 → 5.7 µs at 8 rows, 14.0 → 13.0 at 48), absorb (6.05 → 4.8),
mla_prep (3.1 → 1.6). The MLA slot tiers were reverted (skewed batches).
Every kernel's remaining gap is named in "Done table" with its ablation /
ncu numbers; the DSL attention is a closed prebuilt; k3_dcp is r-comm's.
Every other D op belongs to another run and P is closed this round, so no
next op was taken.

## State

| | launches/layer (d-l16-tp8) | D weighted (A/B, same lease) |
|---|---|---|
| main 01d6dc5 | 12.9 | 3.487 |
| MLA split slots 16 → 4 from 16 rows (e33e3a1) | 13.2 | 3.474 |
| kda_core: loads at entry, state in smem ≤ 25 rows (4f31f73) | 13.9 | 3.453 (vs 3.473 same lease) |
| absorb: ldmatrix + mma.sync, one wave | 13.9 | 3.443 (vs 3.4485 same lease) |
| mla_prep fast path at TP8 (41d420d) | 13.9 | 3.436 (vs 3.441) |
| kda ROWS_PER_ITER 1 / 2 (f8bd7e6), copy issue from 16 threads | 13.9 | 3.431 (vs 3.433) |
| rebased on main 95bc0b0; slot tiers reverted (skewed batches) | 13.7 | **3.386 vs main 3.414** (-0.84%, same lease, 2 rounds) |
| absorb on tcgen05, a block a head (697a625; vs 68ebced) | 13.7 | 3.367 / 3.368 vs 3.369 / 3.369 (same lease, 2 rounds) |
| rebased on main 21a9174 (absorb on tcgen05 + the fp8 cast) | 13.7 | **3.294 / 3.293 vs main 3.295 / 3.295** (same lease, 2 rounds; check bit-identical) |

Launches counted 13.7 a layer against main's 12.9: kda_core's second `when`
launch (S / R builds); one of them runs, so executed launches are unchanged.

## Where the half stands at the bench's shapes (nsys, rank 0, 16 layers, main)

µs per step (calls x µs):

| | 8 x 64k | 24 x 64k | 48 x 128k |
|---|---|---|---|
| DSL attention | 4 x 21.9 | 4 x 52.5 | 4 x 168.9 |
| kda_core | 12 x 9.67 | 12 x 11.05 | 12 x 14.54 |
| dcp_exchange | 4 x 11.9 | 4 x 12.9 | 4 x 22.5 |
| absorb | 4 x 6.4 | 4 x 7.6 | 4 x 8.7 |
| mla_prep | 4 x 3.2 | 4 x 3.1 | 4 x 3.2 |
| step busy | 2836 | 3341 | 4618 |

Attention floor (the rank's latent rows once, 1152 B a position): 10.3 /
31 / 123 µs at 8 x 8k / 24 x 8k / 48 x 16k positions a rank.

## Tools (work/, not committed)

- `work/prof.sh MANIFEST OUT rows@ctx,...` (under `loop/withgpu -n 4 -H 2`):
  the bench with rank 0 under nsys; `nsys export -t sqlite` on a pool host,
  then `work/busy.py OUT/r0.N.sqlite` gives busy µs per kernel.
- `work/ab.sh OUT A.json B.json [rounds]`: A/B of two D manifests in one
  8-GPU lease; `work/shapes.py DIR...` per-shape p50s.
- `work/plan.py`: the MLA split plan replicated (splits per shape and cap).
- `work/t/kda.cu`: kda_core D build on one GPU, cold state and weights
  (256 MB read between calls, the partial warmed as the GEMM leaves it),
  any cubins side by side, out + state compared bit for bit with the first.

## Done

- **MLA split slots** (e33e3a1, **reverted** in 779342f): grid z = 16 below
  16 rows, 4 from 16 rows, the plan capped to match: -0.37% on the uniform
  bench. The orchestrator asked for skewed batches: measured in-graph (work/
  skew.py adds a test kernel that rewrites seq_lens into a workspace copy
  every call reads: row 0 at the bench's 256k, the rest at 8-16k or spread
  over 10k-250k), 16 slots vs the tiers, ms/step: one long row 16 rows
  2.885 → 3.080, 24 rows 3.280 → 3.455, 48 rows 4.138 → 4.271; spread 16
  rows 3.042 → 3.103, 36 rows 4.114 → 4.204. The capped long row (4 splits
  where the plan wants 16+) is every MLA layer's tail. The slots would have
  to follow the batch's longest row; no launch expression sees it and the
  DSL kernel is non-persistent. Back to 16. A cap of 32 (`--mla-split-max
  32`, same lease): uniform 3.379 → 3.419 (+1.2%), and slower on the skewed
  batch too (one 256k row: 16 rows 2.877 → 2.898, 48 rows 4.149 → 4.226):
  16 is the measured best on both.

- **kda_core** (4f31f73): ablation on one GPU (work/t/kda.cu under nsys,
  `work/t/kda.sh`) showed a call at 8 rows was 9.0 µs of which 7.0 with no
  state touched (an empty launch 0.5): ~7 dependent round trips. Now every
  (b, h)-only load goes at entry (w_f_b tile by cp.async into smem, scalars
  to registers), f_b per thread (same tree), one barrier for both l2 sums;
  KDA_SMEM_STATE=1 bulk-copies the 64 KB state slice into smem after the
  line index (2 blocks an SM, ≤ 25 rows at TP8), else the register build at
  128 regs (4/SM). 8 rows 9.2 → 5.9, 24 rows 10.6 → 8.8, 48 rows 14.0 → 13.4.
  Tried on the way: L2 prefetch of the weights / the state at entry
  (registers 154-160, slower from 16 rows); 256 threads (no gain at 8 rows,
  the delta loop is not the limit); bulk copy as 1 / 4 / 16 pieces (16 best
  by 0.2 µs); per-thread cp.async of the state (it waited with the tile).
  At 8 rows what is left (5.9): prologue alone 3.4, the state lands at ~4.2,
  the delta rule from smem + its 64 KB of stores 1.9.

- **absorb**: 96 registers allowed 2 blocks an SM, so the 384 blocks were a
  wave of 304 and a tail; and `wmma::load_matrix_sync` compiled to generic
  4-byte `LD.E` from shared memory (64 a warp, + MOVM transposes): with 24
  warps an SM that was ~1 µs a 16-row tile. Now `__launch_bounds__(256, 3)`
  (80 regs, one wave) and raw `ldmatrix` (.trans for W) + `mma.sync
  m16n8k16` (the same HMMA.16816 in the same k order: bit-identical), the
  accumulators stored straight to q_abs. One GPU (work/t/absorb.cu, W cold),
  µs: 8 rows 6.05 → 4.80, 24 rows 7.38 → 5.63, 48 rows 8.51 → 6.69.
  What is left: loads alone (W 12.6 MB + q into smem, return) 3.7 µs = the
  launch + one HBM round trip + 12.6 MB at 7.3 TB/s; the tiles ~0.4 µs each.
  Tried: every tile's accumulator at once (wmma and mma.sync): slower.

- **kda_core 26-38 rows with the state in smem, 3 blocks an SM** (the w_f_b
  tile in registers, a row a thread, to fit 70 KB): slower than the register
  build everywhere from 16 rows (36 rows 13.8 vs 11.1 µs). Dropped: the
  shared-memory build only pays where its blocks are one wave at 2 an SM.

- **mla_prep** (41d420d): the fast path was compiled for gridDim.y == 4
  (12288 gate columns); at TP8 (1536) GSEGU / NTF was zero passes, so the
  generator launched (B, 3, 1) onto the generic path. GYF = 1 +
  ceil(INNER / 4096), a bounded last gate pass, TP8 launches (B, 2, 1).
  In the step 3.1-3.5 → 1.6-1.9 µs a call. -0.16%.

- **kda_core ROWS_PER_ITER** (S build 1, R build 2): R at 4 rows a lane was
  128 registers with 24 B of spill; at 2 it is 111 registers, no spill, the
  same 4 blocks an SM: 26 rows 10.08 → 9.66, 48 rows 13.47 → 13.04, 64 rows
  20.18 → 18.75 µs. S from shared memory at 1 row a lane: 16 rows 8.16 →
  7.73. Also measured: R at 3 blocks an SM (166 regs) is 0.1-0.3 µs better
  at 26-38 rows but loses from 40: not worth a third `when` launch; R at 5-6
  blocks an SM (80-96 regs, spills) slower.

## Done table (rank 0 busy µs a call in the step, nsys; floors from roofline/decode.md)

The branch on main 21a9174 at 2f5eb5f (`work/prof.sh`, 16 layers):

| kernel | 1 | 8 | 32 | 64 | 32 × 32k | floor (1 / 8 / 32 / 64 / 32×32k) | named limit |
|---|---|---|---|---|---|---|---|
| kda_core | 5.79 | 6.14 | 11.15 | 19.41 | 10.90 | 0.29 / 1.87 / 7.29 / 14.52 / 7.29 | chain (≤ 25 rows), prologue beside the state (R build) |
| mla_attn (DSL, 16 slots) | 8.68 | 10.60 | 27.37 | 48.51 | 43.82 | 0.42 / 2.48 / 7.36 / 12.99 / 22.80 | a kernel with these items: ≥ the no-math bound below |
| mla_absorb (tcgen05) | 5.02 | 4.84 | 5.33 | 5.79 | 5.09 | 1.72 (12.6 MB) | launch + HBM round trip + 12.6 MB (loads alone 3.7) |
| mla_prep | 1.71 | 1.62 | 1.96 | 1.87 | 1.63 | launch | the launch |

The table below is the earlier one (the branch at 6029a75), kept for its
derivations.

A/B shapes, rows × 8k (and 32 × 32k), the branch at 6029a75:

| kernel | 1 | 8 | 32 | 64 | 32 × 32k | floor (1 / 8 / 32 / 64 / 32×32k) | main (roofline) |
|---|---|---|---|---|---|---|---|
| kda_core | 5.65 | 6.11 | 10.90 | 20.59 | 10.90 | 0.29 / 1.87 / 7.29 / 14.52 / 7.29 | 9.5 / 9.6 / 12.2 / 22.8 / 12.2 |
| mla_attn (DSL, main's grid) | 8.58 | 10.60 | 27.29 | 50.07 | 43.79 | 0.42 / 2.48 / 7.36 / 12.99 / 22.80 | 8.8 / 10.5 / 27.2 / 48.1 / 43.7 |
| mla_absorb | 4.79 | 4.89 | 6.03 | 8.37 | 5.99 | 1.72 (12.6 MB) | 6.5 / 6.3 / 7.9 / 10.1 / 7.8 |
| mla_prep | 1.63 | 1.59 | 1.57 | 2.91 | 1.58 | launch | 3.5 / 3.1 / 3.1 / 3.5 / 3.0 |

Bench shapes (8 / 24 / 36 rows × 64k, 48 × 128k): kda 6.45 / 9.15 / 11.65 /
13.68; DSL 21.9 / 46.7 / 65.0 / 160.9 (floors 10.3 / 31 / 46 / 123);
absorb 5.0 / 5.8 / 6.7 / 6.9; prep 1.6-1.9 after 41d420d. Absorb after
697a625 (tcgen05, a block a head), in the step: 4.98 at 8 × 64k, 5.31 at
48 × 128k (one GPU 4.64 / 4.80 / 5.09 / 5.12 at 8 / 24 / 36 / 48 rows).

Named limits for what is not within 1.15x:
- **kda_core, 1-24 rows** (shared-memory build): a latency chain, not bytes.
  Ablated at 8 rows (one GPU, cold): empty launch 0.5, prologue alone 3.4
  (line index → conv window, l2 sums, f_b projection from the smem tile, the
  decay, one block sum), the 64 KB state bulk copy lands at ~4.2 (issued
  right after the line index: line index RT + 64 KB into one SM), delta rule
  from smem + its 64 KB of stores 1.9 (stores alone 0.55; 1.4 at 16-24 rows,
  the L2 write rate). Its floor is the dependent chain launch → line index
  → state → one pass → write-back, ~4.5-5 µs; it measures 5.8.
- **kda_core, 26-64 rows** (register build): every block is resident at
  once and in the same phase, so the prologue does not overlap the state
  read. Ablated (one GPU, cold): without the state 6.2 / 6.6 / 7.6 µs at
  26 / 36 / 48 rows (4.6 at 8: four blocks an SM contend; the per-block
  32 KB tile alone is 0.5 / 0.65 / 0.9 µs of it); the state adds 3.6 / 4.5 /
  5.5 µs, its read time. Overlapping the two needs a block's 64 KB of state
  on chip during its prologue: at 4 blocks an SM that is the whole register
  file. Staged in shared memory it loses at 2 an SM (S build, 2 waves from
  26 rows) and at 3 (tile in registers, 69 KB: 36 rows 11.8 vs 11.0 µs);
  a quarter of the state loaded early into registers spills. At 64 rows
  768 blocks are 1.26 waves at 4 an SM. Bytes are the minimum (ncu above).
  The no-state prologue at 48 rows (7.50 µs), piece by piece: without the
  tile 6.34 (the per-block 32 KB tile is 18.9 MB through L2 for 384 KB of
  unique bytes), without f_b's projection 5.66, without the window 5.02,
  an empty launch 0.74. Two rows of a head a block (256 threads, one tile)
  would recover ~0.5 µs at 36-48 rows, ~2 µs a step: not done.
- **mla_attn**: FlashInfer's prebuilt CuTe-DSL kernel (no source build: the
  cutlass DSL package is absent; no PDL in the cubin). Its grid is now the
  plan's slots; the plan is near the kernel's best (uniform splits a row,
  76 clusters; round 1's makespan plan measured worse). Per launch ~7 µs
  fixed (q 110 KB a split from L2, 196 KB f32 partial a split written),
  ~1.7 µs a 128-position tile per cluster: 96 heads pad to 128 MMA rows
  (MMA-bound below ~53 clusters), ~6.2 TB/s at 72 clusters. Closing it needs
  a new kernel (a 96-row or positions-as-M tiling).
- **mla_absorb**: loads alone (W 12.6 MB + q into smem, then return) 3.7 µs
  = launch + one HBM round trip + 12.6 MB at 7.3 TB/s in one wave; the MMA
  ~0.4 µs a 16-row tile on top. Only overlapping it with q_b (PDL; round 1:
  no gain, cuBLAS triggers no early launch) could hide the stream.
  After 697a625 (tcgen05, below): 4.96 / 4.64 / 4.58 / 4.80 / 5.09 / 5.12 /
  5.44 µs at 1 / 8 / 16 / 24 / 36 / 48 / 64 rows, one GPU: the load plus
  ~0.9-1.7 µs of the last column group's MMA, TMEM read and stores.
- **mla_prep**: one launch, one L2 round trip, ≤ 0.2 µs of bytes: the launch floor.

## Tried after the done table

- **MLA slot caps 6 at 8-15 rows, 3 from 16** (fewer, longer splits at 8 / 16
  rows; fewer empty slots at 24-48): 3.431 → 3.439 ms/step, 16 × 128k +34
  µs, 8 rows +5-13 µs; only 48 rows gains (-5 µs). The plan's 8 / 4 splits
  hold; a (32, 3) tier alone is ~0.4 µs a step: not worth a launch.
- **kda_core: the decay before the conv** (the window loaded into registers
  at entry, f_b and the decay computed while it is in flight, the conv and
  l2 sums after): slower in both builds (S +0.2-0.4 µs, R +0.5-0.8).
  Reverted.
- Gaps between graph nodes around these kernels are 0.15 µs each.

## Numerics beyond the check's 4 rows

loop/d/check runs 4 rows, which never reaches the kda register build (≥ 26
rows), the 4-slot MLA tier (≥ 16) or larger mla_prep / absorb batches. The
check's 4 rows tiled to 20 and to 36 rows, teacher-forced 640 steps on TP8,
main's d-l4-tp8 against this branch's (after f8bd7e6): top-1 1.0000, relRMS
max 0.00000 over every kept row-step, 8 ranks agreeing, at both. (Run
loop/d/ranks with an absolute OUT: the ranks run on other hosts.) Again for
the tcgen05 absorb (N = 32 / 48 columns there), main 21a9174 against the
rebased branch (`work/tf.sh 20|36`): top-1 1.0000, relRMS max 0.00000 at 20
and 36 rows, 8 ranks.
- **kda_core KDA_SPLIT** (a (b, h) item on a 2-CTA cluster, each CTA half the
  state by bulk copy and half the dv rows, attn halves exchanged by DSMEM,
  the window shift deferred past the cluster barrier; bit-identical): the
  S build jumps 6.26 → 7.34 µs from 12 to 13 rows (SMs with two items), but
  the split is slower everywhere (8 rows 5.74 → 7.17, 16 rows 7.74 → 9.06):
  both CTAs pay the whole prologue (32 KB tile, window, scalars) and the
  cluster barriers. Dropped.
- **kda_core state copy issue**: one thread issuing the 16 bulk pieces vs 16
  threads one each: 1 row 5.60 → 5.31, 20-25 rows -0.3 µs, bit-identical;
  D 3.433 → 3.431 (kept). Per-thread cp.async of the state (its own commit
  group behind the tile's): no better. Bulk *stores* of the updated slice
  from smem (in place, fence.proxy.async, 16 × 4 KB): +0.5-1.3 µs, dropped.
  The 12 → 13-row step (6.2 → 7.4 µs, two items an SM) is the write-back:
  without stores 16 rows drops 1.44 µs vs 0.55 at 8 rows, ~46 B/clk of
  STG.128 an SM.
- One-GPU harnesses are in `harness/attn/` (`run.sh kda|absorb OUT ARGS`).

## Bytes (ncu, one GPU, cold, after f8bd7e6)

| kernel | rows | DRAM read | model (state + window + inputs + weights) |
|---|---|---|---|
| kda_core S | 8 | 7.21 MB | 6.29 + 0.22 + 0.20 + 0.47 = 7.2 MB |
| kda_core S | 24 | 20.64 MB | 18.9 + 0.66 + 0.61 + 0.47 = 20.6 MB |
| kda_core R | 36 | 30.71 MB | 28.3 + 1.0 + 0.9 + 0.47 = 30.7 MB |
| kda_core R | 48 | 40.78 MB | 37.7 + 1.3 + 1.2 + 0.47 = 40.7 MB |
| absorb | 8 | 13.19 MB | 12.58 W_UK + 0.59 q |
| absorb | 48 | 16.13 MB | 12.58 + 3.54 |

DRAM writes ~0: the state lands in L2 and is written back under the next
kernels. Every byte is read once; what is left over the floors above is
latency (kda's chain, absorb's one round trip) and kda's prologue not
overlapping its stream (register build), as the done table says.

## Runtime proposals

1. **Choose a launch by a device-written scalar** (CUDA conditional graph
   nodes, IF / SWITCH on a value a kernel wrote): the MLA plan knows the
   batch's largest split count; a switch on it would launch the DSL
   attention with exactly that many slots. Measured value: the uniform
   bench's -0.37% (16 → 4 slots from 16 rows) without the skewed batches'
   +30-195 µs a 16-layer step (both measured above).
2. **`shared_mem` as a launch expression**, or `when` launches counted once
   by loop/count: kda_core's two builds (S up to 25 rows, R above) are one
   op with two `when` launches today, +0.8 counted launches a layer for
   none executed. With an expression a single cubin could take both paths
   (branching on %dynamic_smem_size). kern verify rejects any expression in
   `shared_mem` (an untagged-enum mismatch).
3. **(idea, not a runtime feature) static grids that follow the longest
   row**: two DSL launches every step, A with 4 slots over every row, B with
   16 slots over a compacted list of the batch's long rows (at most a few:
   the plan writes their count, lengths, splits and copied page-table rows;
   absorb also writes their q into a small q_abs_B; A sees them as empty).
   The exchange (r-comm's k3_dcp) would read a long row's partials from B's
   workspace. Costs one more launch a MLA layer and a contract change in
   k3_dcp; worth it only if real batches' long rows are rare and short rows
   many. A variant that leaves k3_dcp alone: B over every row with 12
   slots writing at acc_o + 4 splits (a static offset: the split stride is
   512 floats), each long row's page table shifted by 4 x per tiles and its
   length cut by as much (rows of <= 4 splits get length 0 in B, and B's
   empty slots land at split indices the exchange never reads). It breaks
   where B's per (ceil of the remaining tiles over S - 4) comes out below
   the plan's per: B then writes fewer than S - 4 splits and the exchange,
   which derives S from (seq_lens, bsk), merges a stale one. It needs the
   plan to pick per so the remainder divides evenly, or the exchange to
   take B's split count.
- **absorb as a programmatic dependent** (pdl, W_UK's loads before
  griddepcontrol.wait, the query after): bit-identical, but absorb's busy
  time rises (8 rows 5.03 → 5.34 µs, 24 rows 5.80 → 6.21; its predecessor is
  q_b's splitKreduce) and D 3.379 → 3.377 / 3.389. Dropped, as in round 1.
  An L2 prefetch of W_UK from mla_prep has no idle block to run in (every
  block holds a row at TP8) and would cross q_b's 54 MB.
- harness/run.sh `conv` now also runs the D step's two kda builds against
  their SGLang variants (CONV_TAP_MAJOR + LINE_GUARD, padded rows) bit for
  bit (`--smem 65536` for the shared-memory one): PASS at 2 / 8 / 64 rows;
  `tp8` kda_core (non-conv build) PASS against its CPU reference. `sass`
  flags k3_dcp's exchange (STACK:16): main's, not this branch's (r-comm's
  kernel now).

## MLA decode: the engines at our shapes (orchestrator 02:52 task 1)

A member's view: 96 heads (every head, a DCP member's positions), latent 512
+ rope 64, bf16. µs a call; DSL in the step (nsys), the others one GPU.

| shape (rows × positions a member) | floor | DSL (ours, prebuilt) | CUTLASS sm100 MLA (example 77, best split) | trtllm-gen MLA | FlashMLA sm100 |
|---|---|---|---|---|---|
| 8 × 8k | 10.3 | 21.9 | 50.2 (split 8; auto 78) | n/a | n/a |
| 24 × 8k | 31 | 46.7 | 128 (auto) | n/a | n/a |
| 36 × 8k | 46 | 65.0 | 126.5 (auto; split 1-4 131-139) | n/a | n/a |
| 48 × 16k | 123 | 160.9 | 258 (split 1) | n/a | n/a |
| 1 × 1k | 0.42 | 8.6 | 45 | n/a | n/a |
| 64 × 1k | 13.0 | 50.1 | 84 | n/a | n/a |

- CUTLASS: `flashinfer/3rdparty/cutlass/examples/77_blackwell_fmha/77_blackwell_mla.cu`
  built for sm_103a (`-DFP16`, the same MMA rate as bf16; `work/cmla/`),
  128 heads (its fixed tile; our 96 pad to it as in the DSL), 128-token pages
  (its TMA path rejects 64; FlashInfer's `cutlass_mla` wrapper takes the
  cp.async variant for other sizes), its own split-KV reduction included.
  A 128 × 128 one-CTA tile: ~0.65-1.0 PF where the DSL's 2-CTA cluster runs
  ~1.3 PF. 1.6-5× slower everywhere.
- trtllm-gen MLA (`trtllm_batch_decode_with_kv_cache_mla`'s
  `FmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512...` cubins): the 432
  HQk576 cubins under `$K3_REF/trtllm/.../fmha/cubin` are git-LFS pointers
  (130 B each); no flashinfer-cubin wheel on the hosts, none in the kern
  index or blob cache (scanned every cached blob's entry names). Not
  loadable here: measure it where the wheel is.
- FlashMLA: only the sm90 (wgmma) kernel is in the refs (TRT-LLM's
  kernels/flashMLA); no sm100 decode.
- The DSL kernel stays the best available. A kernel that beats it must
  avoid the 96 → 128 head padding: tcgen05 MMAs take M = 64 / 128 (one CTA)
  or 128 / 256 (a pair), so heads as M always pads; positions as M (S^T =
  K Q^T with N = 96 heads, O^T = V^T P^T) does not, but its softmax runs over
  TMEM lanes (a cross-thread max and sum per head per tile). Its ceiling at
  36 × 8k: the tile from ~1.70 µs (DSL, MMA-bound at 128 rows) to the HBM
  rate ~1.45 µs, 65 → ~56 µs a call, ~0.9% of the 16-layer step at 36
  rows. A from-scratch tcgen05 split-KV kernel (paged TMA gather, TMEM,
  cross-lane softmax, the DSL's partial layout for k3_dcp) is days of work,
  not this session's remaining hours: not started.
- Where a new kernel would win, per SM: the DSL's 1.70 µs a 128-position
  tile a 2-SM cluster is 37.6 positions/µs/SM against HBM's fair share of
  41.9 (7.33 TB/s / 1152 B / 152 SMs): the 96 → 128 padding costs ~10% in
  the steady state. The larger part of the gap is fixed: ~7-10 µs a launch
  (each split loads its row's 110 KB of q from L2 and writes a 196 KB f32
  partial; 8 x 8k runs 8 splits a row of 8 tiles, 64 clusters: 21.9 µs
  where the tiles alone are ~12) and
  clusters left idle (72 of 76 at 24-36 rows, uniform splits). A kernel
  that merged its splits on chip (a persistent cluster per row group, bf16
  or in-cluster partials) and fed every cluster would target those first.

## MLA attention: what a new kernel could win (orchestrator 02:52 task 1, cont.)

**tcgen05 rates by shape** (`harness/attn/mma_rate.cu`, bf16 → f32, operands in
smem, 152 SMs, flop/clk/SM): one CTA M=64: 4096 at N=96/128/256 (half rate);
M=128: 7020 at N=96, 8190 at N=128/256; a CTA pair M=128 (64 rows an SM):
6776 / 8188 / 8190; pair M=256: 7968 / 8190 / 8190. Peak 8190 = 2.41 PF.
So heads-as-M pads 96 heads to 128 rows (64 an SM is the smallest full-rate
unit, as in the DSL). Positions-as-M avoids the padding in a pair: QK^T at
pair M=128 (64 positions a CTA, q split 48 heads a CTA as the B operand,
55 KB; 6776 flop/clk at N=96), PV at pair M=256 over the latent dims (P^T
split 48 heads a CTA, exchanged by DSMEM; each CTA also needs V of the other
CTA's 64 positions for its dims, 32 KB a tile), S^T 96 + O^T 192 TMEM
columns. A tile is then 106 KB of KV a CTA beside 55 KB of q, 12 KB of P:
the same 1.3-1.5 tiles of 8 KB chunks the DSL streams with its 74 KB of q,
not a double buffer. It fits; what it saves is MMA time, ~10% (96 rows at
83% rate against 128 at 100%), and MMA is not what limits at the bench
shapes (below). Its cost is a cross-lane softmax (max and sum per head over
TMEM lanes) and the P exchange. (A cta_group::1 PV beside a cta_group::2
QK^T is not an option: ptxas rejects a kernel that uses both granularities.)

**Padding is not what costs at the bench shapes**: the DSL's 1.70 µs a
128-position tile a pair is 89% of the pair's HBM share (1.53 µs at 7.33 TB/s
over 76 pairs, 6.2 TB/s measured at 72 clusters), while its 128-row MMAs
(QK^T 576 + PV 512 deep) take 1.13 µs at the measured peak: 66% busy. The
padding only shows where few clusters run (1-8 rows at short contexts).

**The item chain** (`harness/attn/mla_chain.cu`): a kernel with the DSL's
grid, cluster and shared memory (one CTA an SM) that does everything but the
math: page-table entry, the row's q half from L2, the item's KV pages by bulk
copy (two 73.7 KB pages in flight a CTA), a 96 × 512 f32 partial and lse
stored. Its time is the floor of any non-persistent kernel with these items.
µs a call, one GPU, KV cold:

| rows × positions a member | items × tiles | no-math kernel | DSL (step) | floor |
|---|---|---|---|---|
| 1 × 1k | 8 × 1 | 4.22 | 8.58 | 0.42 |
| 8 × 1k | 64 × 1 | 6.22 | 10.60 | 2.48 |
| 32 × 1k | 64 × 4 | 10.15 | 27.29 (16 slots) | 7.36 |
| 64 × 1k | 64 × 8 | 16.38 | 50.07 (16 slots) | 12.99 |
| 32 × 4k | 64 × 16 | 28.11 | 43.79 | 22.80 |
| 8 × 8k (bench) | 64 × 8 | 16.01 | 21.9 | 10.3 |
| 24 × 8k (bench) | 72 × 22 | 39.16 | 46.7 | 31 |
| 36 × 8k (bench) | 72 × 32 | 53.51 | 65.0 | 46 |
| 48 × 16k (bench) | 144 × 43 | 133.02 | 160.9 | 123 |

A lone item's chain is 4.2 µs (launch, table, q + first page, 98 KB out a
CTA); the DSL's one-tile item is 8.6 (its math, TMEM, softmax and epilogue
on top). At the bench shapes the no-math kernel is 1.08-1.55x the floor,
the DSL 1.21-1.37x the no-math kernel. The difference (5.9 / 7.5 / 11.5 /
27.9 µs a call; 24 / 30 / 46 / 112 µs a 16-layer step, 0.8-2.4% of it) is the
most a kernel with the same items could take: the DSL's tile rate below the
HBM share (~0.17 µs a tile) and its math's fill and drain. Most of the rest
of the gap to the floor is structural to any split-KV launch: the chain once
a wave (two waves at 48 rows), 64 of 76 pairs at 8 rows. A persistent kernel
over the same items would also drop the 16-slot grid's empty clusters
(the tier experiment's -0.37%) and skew's tail. Not started: it is the
DSL's own design (2-CTA heads-as-M, KV streamed by 64-dim chunks under the
math, TMEM O, bf16 P) to write and tune from PTX, more than this session.

- **kda_core S at 3 blocks an SM, ablated** (36 rows, one GPU): 11.81 µs;
  without the state stores 10.51; without the smem state reads 11.65. Even
  storeless it does not reach the register build with its stores (10.96).
  The persistent pipeline of the orchestrator's task 2 meets the same wall:
  at 26-48 rows the items (312-576) are about the resident slots (608 at 4
  an SM), so a persistent block has no next item whose prologue could hide
  under this one's stream; overlapping inside one item needs its state on
  chip during the prologue (register file full at 4 an SM; smem staging
  measured slower at 2 and 3).

## kda_core, tried after the merge (one GPU, harness/attn kda, state cold)

- **R build: one-thread bulk L2 prefetch of the head's state at entry**
  (`cp.async.bulk.prefetch.L2.global` of the 64 KB slice as soon as the line
  index is in: no registers, 111 as before), µs at 26 / 36 / 48 / 64 rows:
  9.94 / 10.96 / 12.99 / 18.79 → 9.76 / 11.01 / 13.30 / 18.37. The state's
  37.7 MB (48 rows) stream ahead of the prologue's own window and partial
  loads, which then wait behind them: the critical path moves, it does not
  shrink. Round 1's per-thread version said the same. Dropped.
- **S build: one mbarrier per 8 KB (2-8 barriers), each delta-rule pass
  waiting for its own 16 dv rows only**: slower at every row count (1 row
  5.09 → 5.76-6.27, 24 rows 8.50 → 8.93-9.28). Waiting for all of them
  before the loop is as slow, and pinning the line-index load first
  (`asm volatile` load) does not recover it: the extra barriers move ptxas's
  schedule (main's build issues `line_index[b]` as its first load; with a
  barrier array it sank below the barrier setup). Per-pass waits against
  wait-all differ by ≤ 0.13 µs: the 16 pieces land together, so there is no
  arrival to follow. Dropped.
- Paired f32x2 math (FFMA2 / FMUL2, bit-identical where the pairs are
  independent: dec scaling, the update, the f_b products and tree levels)
  would remove ~10% of the R build's ~1200 instructions a thread; the m / aq
  chains are sequential sums and do not pair without reordering them. At
  48 rows that is worth ≲ 0.3 µs a call: not attempted.

## mla_absorb on tcgen05 (697a625)

- **mma.sync on GB300** (`work/mma/hmma.cu`): m16n8k16 bf16 → f32 peaks at
  ~2030 flop/clk/SM (596 TF), a quarter of tcgen05's 8190; a dependent chain
  issues one every ~22 cycles. Absorb's 48-row tiles were ~1.2 µs of that
  throughput plus three serial tile chains: 6.66 µs against 3.7 of loads.
- **A kernel that uses TMEM is resident one block an SM** (`work/mma/occk.cu`:
  cudaOccupancy gives 1 for a 256-thread kernel with only a 64-column
  tcgen05.alloc, 8 without; EIATTR AT_ENTRY_FRAGMENT_TMEM). A hybrid that kept
  main's mma.sync path below 17 rows in the same kernel ran it at 8 µs (one
  block an SM instead of three). So the tcgen05 build is a block a head: 96
  blocks, W_UK's 128 KB a head by 8 TMA boxes (64 columns x 128 rows, 128 B
  swizzle: the MN-major SW128 atoms the A descriptor reads, LBO = the 16 KB
  box, SBO = 1 KB), q once a head (main loaded it 4x), D^T in TMEM (M = 128
  columns, N = rows to 16), four groups pipelined behind their boxes.
- Bit-identical to mma.sync at 1-64 rows (every output byte), and
  loop/d/check bit-identical vs main. One GPU, µs at 1 / 8 / 16 / 24 / 36 /
  48 / 64 rows: 4.64 4.77 4.96 5.63 6.34 6.66 7.49 → 4.96 4.64 4.58 4.80 5.09
  5.12 5.44. 1 row loses 0.3 µs: the last group's commit → TMEM read →
  stores is longer than one mma.sync tile, and 96 blocks pull 128 KB each.
- The fp8 query (`--kv fp8`) keeps k3_mla_absorb_mma. Its e4m3 casts were
  cuda_fp8's software __NV_NOSAT (~40 integer instructions a pair: 16.4 µs at
  48 rows). aac88c1: `cvt.rn.satfinite.e4m3x2` with |x| > 464 put back to
  NaN with x's sign (`source/k3_e4m3.cuh`) is the same byte for all 2^32
  floats (`work/mma/e4m3.cu`, exhaustive; 464 itself rounds to 448, every
  NaN gives 0x7f); the absorb's and mla_prep's fp8 builds use it. Absorb
  fp8, µs at 1 / 8 / 24 / 48 / 64 rows: 6.50 6.62 10.46 16.38 20.38 → 4.70
  4.86 6.14 8.06 9.09, byte-identical. The tcgen05 build with the same cast
  (a byte a column, then lane pairs by shuffle, then the fix-up skipped by a
  warp vote): 5.18-5.55 / 8.10-10.40 µs at 1 / 48 rows, byte-identical but no
  faster: with one block an SM, its epilogue is 96 values a thread at 48
  rows and the cast's instructions are on the critical path. Kept on
  mma.sync.
- **W_UK into L2 from mla_prep** (two launches before absorb; HBM idle
  while q_b's CTAs wait on mla_prep), nsys rank 0, µs at 8 × 64k / 48 × 128k:
  one thread a block issuing 128 KB `cp.async.bulk.prefetch.L2`: mla_prep
  1.68 → 5.12 / 1.73 → 3.30, absorb 4.98 → 4.38 / 5.31 → 5.12, q_b
  unchanged; a `prefetch.global.L2` a line over every thread of the grid:
  mla_prep 4.99 / 3.31, absorb 4.35 / 5.17. Issuing 12.6 MB of requests keeps
  the kernel open about as long as HBM takes to serve them (1.7 µs at full
  rate, longer from few SMs), and absorb gains less than that. D A/B
  (bulk): 3.378 / 3.376 → 3.376 / 3.375, 8 rows worse. Dropped.
- **kda_core: the next KDA layer's state into L2 from this one** (S build:
  one thread a block ends with a 64 KB `cp.async.bulk.prefetch.L2` at
  evict_last of the next layer's slice, under o_proj whose weight is
  already in L2; the next layer's bulk copy reads it with an evict_first
  hint): D A/B 3.376 / 3.378 → 3.391 / 3.392, 8-24 rows +15..35 µs a step.
  nsys: kda_core 6.16 → 6.82 µs at 8 rows, 9.14 → 9.60 at 24; the prefetch's
  issue holds the block (as in mla_prep above) and no read-side gain shows.
  Dropped. What a warm state is worth (harness/attn kda, `STATE_WARM=1`:
  the rows' lines read into L2 after the flush, weights and partial as in
  the step): 1 / 8 / 16 / 24 rows 5.17 / 5.76 / 7.58 / 8.43 → 5.12 / 5.18 /
  6.66 / 7.20 µs (everything warm: 4.45 / 4.51 / 6.34 / 6.59; R build at
  36 / 48 rows 11.04 / 13.07 → 8.48 / 10.40 all warm).
  **Proposal for k3_ar_fused's owner** (r-ar / r-comm): the all-reduce before
  each KDA layer's qkvg (land_add2_attnres_rms_snap) idles HBM while it
  waits; its idle blocks could load the layer's state lines (rows × 12 heads
  × 64 KB, line_index known) beside the qkvg weight they already pull, at
  evict_last, kda_core then reading them evict_first. Worth ~0.6 / 0.9 / 1.2
  µs a call at 8 / 16 / 24 rows (12 calls a 16-layer step: 7 / 11 / 15 µs,
  0.25-0.45%), less what the extra 6.3-19 MB of L2 costs the qkvg prefetch.
- **kda_core's own weights at evict_last** (w_f_b tile by cp.async with an
  L2 cache hint, conv taps / dt_bias / a_log / gamma by ld.global.nc with
  it: ~0.5 MB a layer, all twelve KDA layers resident from step to step):
  one GPU no change (5.54 → 5.55 at 8 rows, 12.98 → 12.96 at 48), D A/B on
  main 21a9174 3.285 / 3.286 → 3.285 / 3.287. The all-warm harness's extra
  0.6 µs over a warm state is not those weights' L2 misses. Dropped.

## For an MLA kernel later: measured building blocks (this session)

- tcgen05 rates by shape (`harness/attn/mma_rate.cu`, above); a TMEM kernel
  is resident one block an SM (`work/mma/occk.cu`); ptxas rejects
  cta_group::1 and ::2 in one kernel.
- **Where a CTA pair's D lands** (`harness/attn/tmem_layout.cu`): at M = 128
  (64 rows a CTA) each CTA keeps its rows twice down the lanes, lanes 0-63
  for columns 0 .. N/2 and lanes 64-127 for N/2 .. N, in N/2 TMEM columns; at
  M = 256 its 128 rows fill lanes 0-127 over N columns. So heads-as-M in a
  pair (64 heads a CTA) holds O (512 latent columns) in 256 TMEM columns and
  two 128-position S buffers in 2 x 64: 384 of 512. That is the budget the
  DSL and Cake's kernel live in; a 1-CTA heads-as-M O alone is 512.
- Verified descriptor forms (absorb on tcgen05, bit-identical to mma.sync):
  K-major SW128 (start + 32 B a k16 step inside the 128 B row, SBO 1 KB) and
  MN-major SW128 (64 elements x 8 rows a 1 KB atom, SBO 1 KB between 8-row
  groups, LBO between 64-element blocks), the latter straight from a TMA box
  of 64 x rows with the 128 B swizzle; manifest tensor maps (`tensormap` in a
  `bytes<128>` pack) for TMA.
- A starting point in plain CUDA: the vendored Cake kernel
  (`source/flashinfer-cake-k3-mla/`) is the pair heads-as-M design (M = 128,
  N = 128 positions, cta_group::2 TMA and multicast commit, softmax warps on
  64 rows) for e4m3; bf16 doubles its q (64 KB) and stage bytes, so its 5 x
  32 KB K stages become a ring of 64-dim chunks. The ceiling for any kernel
  with the DSL's items is the no-math table above (5.9 / 7.5 / 11.5 / 27.9
  µs a call at the bench shapes); a persistent one also removes the second
  wave's chain at 48 rows and the 16-slot grid's empty clusters.
- **The pair design worked through** (not built): CTA c holds heads 64c ..
  64c + 63 (CTA 1's last 32 zero-filled by the q tensor map's bounds), q as
  9 chunks of 64 dims (72 KB). A tile is 128 positions, page 2t + c in CTA c
  (QK^T's B, N-split); PV's B is N-split by latent dims, so CTA c also needs
  the other page's 4 chunks of its PV dims (128c .. +128 and 256 + 128c ..
  +128): 13 chunks of 8 KB a tile (104 KB), 5 of them free after QK^T, 8
  after PV. TMEM: S 2 x 64 columns (lanes 0-63 positions 0-63, 64-127 the
  rest: softmax warps w and w ^ 2 share a head row), O 2 x 128 (lanes 0-63
  the low 128 dims of each 256 group). Shared memory: q 72 + P 16-32 + one
  tile's 104 KB + ~2 chunks: about 1.2 tiles in flight, the DSL's own
  budget, so the tile rate is not where it would win; a persistent grid is.
  Output either the DSL's contract (acc_o f32 [R][128][split_max][512]
  normalized, acc_lse in log2 units, S = ceil(t / ceil(t / bsk)) splits a
  row) or k3_dcp's CAKE build's (bf16 parts, max and sum, uniform splits:
  half the partial bytes).
