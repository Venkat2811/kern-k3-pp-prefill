# r-ar: D's all-reduces and residual rows (`k3_ar_fused`)

Target (round 3): `ar_attnres_rms`, `ar_finalize_rms`, `land_add2_attnres_rms(_snap)`, `embed_rms` at their floor.
`k3_head` moved to `r-comm` (main 04a88d8: r-comm owns the exchange protocol, `k3_dcp`, `k3_head`; r-ar owns the
fused kernels' row math).

Bench: `loop/d/bench loop/out/d-l16-tp8.json`. Check: `loop/d/check loop/out/d-l4-tp8.json` (x3 for protocol).
main at the start: 12.9 launches/layer, 3.486-3.491 ms/step (same-lease A/B, below).

## Tools (work/ar/, not committed unless kept)

- `work/ar/arh.cu` + `run.sh`: the kernels alone on 8 ranks over 2 trays (fabric handles exchanged through files,
  CUDA graphs of K launches, a 1-block `--spacer` kernel between calls so every call starts on an idle GPU as in
  the step), cubin variants interleaved, outputs compared bit for bit to the first variant, every rank's hash.
  `--test attn|moe|land|head`, `--nb`, `--snap`, `--rows`. 1-GPU tests (`land`) run `arh` directly with `--world 1`.
  Block stamps: `-DSTAMP2` builds write per-block globaltimer stamps the harness prints.
- `work/ar/variant.sh NAME "DEF=V"`: this tree with a define's default replaced, built and generated into
  `work/ar/out/NAME.json`. `work/ar/abn.sh OUT ROUNDS M1.json M2.json ...`: D benches interleaved in one lease.
  main's manifest: generate from a main worktree (`$K3_LOOP_DATA/base/d-l16-tp8.json` predates kern's schema).
- `work/ar/prof.sh OUT MANIFEST ROWS@CTX`: every rank under nsys (graph nodes), `work/ar/trace.py` per-kernel table.

## Measurements

Harness, rank 0, us per call (with spacer), main:

| call | 1 row | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ar_attnres_rms` nb 1 | 10.6-10.8 | 10.9-11.1 | 11.4 | 12.4-12.7 | 13.5-13.8 | 12.3 | 12.6 |
| `ar_finalize_rms` | 9.65 | 9.8 | 11.2 | 12.8-13.0 | 14.7-15.2 | 12.4 | 12.8 |
| `land_add2` nb 1 (1 GPU) | 5.1 | 5.2 | | | 5.5 | | 5.5 |
| `land_add2` nb 8 | 9.2 | 9.2 | | | 9.5 | | 9.5 |
| empty 1024-thread kernel b2b (no spacer) | 0.9 | | | | | | |

Where `ar_attnres_rms`'s 1-row call goes (block 0, globaltimer from entry): open 1.1 us, **push 2.1 us** (one block
pushes the whole row, 115 KB of remote stores from one SM: ~55 GB/s), the row's data seen at ~6-7 us, the row
routine + exit ~2-3 us. Without the row routine the call is 7.2 us (nb 1) against 10.1. At 32 rows the one-shot
push is 4.6 us: 3.2 MB of egress at ~700 GB/s, NVLink's rate.

In the step (nsys, 48 rows, rank 1): graph launch gaps are 0.16 us; the GEMM after each all-reduce is a PDL
dependent that starts 10-18 us early and still needs 7.5 us (lat_up, 51 MB) / 13.3 us (front, 86 MB) after the
all-reduce ends. The all-reduces' in-step durations (15-35 us) are mostly waiting for the last rank (EP skew:
fc1 85-92 us across ranks).

## Ideas tried

| idea | harness | step | kept |
|---|---|---|---|
| push spread over all 152 blocks (runs of PUSH_RUN vectors dealt in order) | attn 1 row 10.6 -> 8.3 (with the row routine below), MoE 9.65 -> 8.4; flat from 16 rows | see A/B | yes |
| `share()` divided by gridDim.x (runtime, 64-bit) | +0.4 us on the critical path (32-bit: +0.3) | | replaced by the constant KGRID |
| re-poison by the reader right after its poll, two stages, per-block phase | slower from 8 rows: 8x the row's bytes of stores land on the B row blocks | | no: the spread clear pass (3 stages) stays |
| no grid funnel (per-block stage + re-poison count in `tp_ar_state[8 + 2b]`) | 0 (main's funnel costs nothing measurable) | | yes (simpler; no block waits on another) |
| MoE: finalize() before the open's barrier | +0.5-1 us | | no: after the open |
| PUSH_RUN 64 / 128 / 256 / 512 / 1024 | 64-128 best at 1-8 rows, all equal from 16 | | 128 |
| two-shot from 9 / 13 / 17 / 20 / 25 / 33 rows | crossover ~20 rows for both widths (attn: equal at 20, -0.5 us at 24; MoE: -0.4 at 20, -1.2 at 24, -3 at 32) | 33 vs 20: equal (two33 A/B) | 20 |
| row routine: every candidate's (sq, dot) in one block reduction (recursive halving: N-1 shuffles for N values, every warp finishes it, one barrier), softmax over the totals in registers, the norm's reduction | land -0.6-0.7 us at nb 1-8; 64 rows: 2-7 of 917504 outputs differ by 1 bf16 ulp (nb 2-8) | | yes |
| `ar_finalize_rms`'s latent norm through `block_sum<1>` (same tree, one barrier fewer) | bit-identical | | yes |
| **PDL**: both all-reduces and `land_add2_attnres_rms_snap` launch with `pdl: true`; stage / snapshots / sw / gamma / prefix read before `griddepcontrol.wait`, the partials after | | **3.486 -> 3.430 (-1.6%)**, every shape -40..-70 us | yes |
| `land`: the snapshots' scores reduced before the wait, the prefix row's after (two reductions) | +0.3 us alone (nb 1-2), -0.5 at nb 8 | 3.432 -> 3.435 (flat) | no |
| PDL + the idle blocks prefetch the next weight before the wait (all blocks push) | | 3.473 (+1.3%: the spare blocks' pushes wait behind their prefetch issue) | no |
| pushes from the first 64 blocks only; the blocks with neither row nor push prefetch before the wait | | 3.429 / 3.427 vs 3.432 / 3.429 (flat) | no |
| pushes from 32 blocks | | crashed at 36 rows: a share past 1024 vectors (now a static_assert) | no |

Step A/B (same lease, ms/step, 2 rounds): main 3.488 / 3.488, cand (all of the above) 3.490 / 3.491,
PUSH_RUN 1024 3.502 / 3.497, two-shot at 33 3.487 / 3.488. Per shape cand vs main: 8 rows -13..-15 us,
16 rows -4, 24 rows 0, 36 rows +5..+10, 48 rows +13. The exchange's latency does not reach the step: see
"Why the exchange's latency does not show".

## Why the exchange's latency does not show

Round 1 (d-mlp) made the all-reduces' idle blocks prefetch the next GEMM's weight into L2 (-1.9%). That makes an
all-reduce and the GEMM after it one HBM stream of the GEMM's weight from the all-reduce's start: a shorter
all-reduce starts the GEMM earlier with less of its weight in L2.

Measured (same lease, 2 rounds, ms/step): main 3.488 / 3.489, main without the prefetch 3.538 / 3.539; cand
3.485 / 3.485, cand without the prefetch 3.524 / 3.521. Without the prefetch the exchange work is worth
-14..-18 us a step (0.4-0.5%); with it, -3..-4 us. So the exchanges' latency reaches the step only if the weight
stream starts earlier than the exchange, or where the exchange outlasts the stream.
