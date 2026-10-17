# r-attn: D's attention kernels at their floor (round 3)

Kernels: `k3_kda_core`, MLA (`k3_mla_prep`, `k3_mla_absorb_mma`, the DSL
`mla_decode_h96_p64` and its grid), the DCP exchange (`k3_dcp`).
Previous lane notes: `notes/d-attn.md` (round 1, every idea tried there).

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

A/B shapes, rows × 8k (and 32 × 32k), the branch at 6029a75:

| kernel | 1 | 8 | 32 | 64 | 32 × 32k | floor (1 / 8 / 32 / 64 / 32×32k) | main (roofline) |
|---|---|---|---|---|---|---|---|
| kda_core | 5.65 | 6.11 | 10.90 | 20.59 | 10.90 | 0.29 / 1.87 / 7.29 / 14.52 / 7.29 | 9.5 / 9.6 / 12.2 / 22.8 / 12.2 |
| mla_attn (DSL, main's grid) | 8.58 | 10.60 | 27.29 | 50.07 | 43.79 | 0.42 / 2.48 / 7.36 / 12.99 / 22.80 | 8.8 / 10.5 / 27.2 / 48.1 / 43.7 |
| mla_absorb | 4.79 | 4.89 | 6.03 | 8.37 | 5.99 | 1.72 (12.6 MB) | 6.5 / 6.3 / 7.9 / 10.1 / 7.8 |
| mla_prep | 1.63 | 1.59 | 1.57 | 2.91 | 1.58 | launch | 3.5 / 3.1 / 3.1 / 3.5 / 3.0 |

Bench shapes (8 / 24 / 36 rows × 64k, 48 × 128k): kda 6.45 / 9.15 / 11.65 /
13.68; DSL 21.9 / 46.7 / 65.0 / 160.9 (floors 10.3 / 31 / 46 / 123);
absorb 5.0 / 5.8 / 6.7 / 6.9; prep 1.6-1.9 after 41d420d.

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
loop/d/ranks with an absolute OUT: the ranks run on other hosts.)
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
   many.
- **absorb as a programmatic dependent** (pdl, W_UK's loads before
  griddepcontrol.wait, the query after): bit-identical, but absorb's busy
  time rises (8 rows 5.03 → 5.34 µs, 24 rows 5.80 → 6.21; its predecessor is
  q_b's splitKreduce) and D 3.379 → 3.377 / 3.389. Dropped, as in round 1.
  An L2 prefetch of W_UK from mla_prep has no idle block to run in (every
  block holds a row at TP8) and would cross q_b's 54 MB.
