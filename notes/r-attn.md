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

- **MLA split slots** (e33e3a1): grid z = 16 below 16 rows, 4 from 16 rows
  (`MLA_SLOTS`), the plan capped to match. Cap 2 from 32 rows tried: at 48 x
  128k the plan wants 3 splits (2 waves of 144 clusters); capped at 2 it
  takes 1 split, +65 µs a step. -0.37%.

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
