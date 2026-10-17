# r-ar: D's all-reduces and residual rows (`k3_ar_fused`)

Target (round 3): `ar_attnres_rms`, `ar_finalize_rms`, `land_add2_attnres_rms(_snap)`, `embed_rms` at their floor.
`k3_head` moved to `r-comm` (main 04a88d8: r-comm owns the exchange protocol, `k3_dcp`, `k3_head`; r-ar owns the
fused kernels' row math).

Bench: `loop/d/bench loop/out/d-l16-tp8.json`. Check: `loop/d/check loop/out/d-l4-tp8.json` (x3 for protocol).
main at the start: 12.9 launches/layer, 3.486-3.491 ms/step (same-lease A/B, below).
Rebased onto main 95bc0b0 (r-gemm's L2 loads in the exchanges, 3.411-3.416 ms/step): the branch's first five
commits on it were flat (3.418-3.420: 8-24 rows faster, 36/48 rows +20 us, the spread push delaying the idle
blocks' weight loads); with the spread only below 33 rows, 3.404-3.405 (-0.2%).

## Where it stands (03:25 UTC, rebased on main d85613c)

12.9 launches/layer of mine unchanged (D 207 a step before r-attn's `when` variants). Same-lease A/B vs main
95bc0b0 (3.407-3.417 ms/step), each step cumulative:
- push spread only below 33 rows: 3.404-3.405
- the closing add's prefetch capped at 16 MB (`LAND_PF_MAX`): 3.382-3.390
- the all-reduces' load-window stamp a phase, no l2_close grid count: 3.371-3.374 (on another lease 3.357-3.361)
- AR_GRACE_NS 2500 -> 500: 3.343-3.351
- ar_finalize's top-k staged in smem before the wait: 3.342-3.343 (-0.1%)
- two-shot gather spread over the non-owned rows' blocks: 3.335-3.344 (36-48 rows -30 us)
= -2.0% vs main 95bc0b0. Rebased on main d85613c (3.377-3.380): 3.300-3.302 (-2.4%), then
- L2_INFLIGHT 8 MB for the all-reduces: 3.295-3.296
- two-shot from 14 rows (was 20): 3.299 vs 3.300-3.302 before the in-flight change (16 rows -5..-10 us)
- embed_rms_plan's plan in block B: 3.282-3.284 (vs 3.284-3.285)
- push spread over every block in the one-shot only: 3.273-3.274
- ar_finalize's routed latent before the wait: 3.270-3.271
= main d85613c 3.373 -> 3.270 (-3.0%). **Offer: the branch at its head** (14 commits on d85613c).
All bit-identical vs main (loop/d/check x3; two-shot paths in the harness).

Each exchange's own time in the step (work/ar/own.py: per call, end minus the previous kernel's end, the minimum
over 5 ranks = the last-arriving rank's; us a call, 8 / 48 rows): ar_attnres own 12.6 / 13.5, mean 17.8 / 18.0,
max 21.7 / 20.8; ar_finalize own 11.0 / 12.2, mean 19.5 / 27.3, max 28.3 / 41.3 (EP skew); land 4.1 / 4.9.

What is left in the step (nsys, 8 / 48 rows x 64k, us a step on the frontier): ar_attnres 331 / 306, ar_finalize
237 / 349, land 66 / 79, embed 6 / 6. An all-reduce's frontier time is ~2x its time alone: the wait for the
last rank (the GEMM before it ends 2-6 us apart across ranks; EP skew before ar_finalize) and the load window's
drain. The closing add ends 3.8-4.9 us after lat_up (1-1.5 of it the release of griddepcontrol.wait).

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
| `block_sum` above 8 values: warp w sums value w behind a second barrier (was: all 32 warps re-reduce all N) | land nb 4 / 8: -0.4 / -0.9 us; attn nb 8: -0.2..-0.4; bit-identical | bench unchanged (nb <= 2) | yes |
| `embed_rms_plan`: the plan's tile counts read once into registers, 32-bit integer math, the gather issued before the plan | 5.3-7.1 -> 4.5-5.2 us, identical outputs and split counts | | yes |
| land ablation (nb 1, 1 row): full 4.73, no norm reduction 4.46, no score reduction 4.32, neither 3.92, empty kernel 1.32 us | each reduction ~0.3-0.4 us; the rest is the load -> store chain and the kernel's launch | | |
| GRAM: the mix's sum of squares from the candidates' pairwise dots (snapshot pairs before the wait/poll, the prefix's with the snapshots in its score reduction): one reduction after the wait | alone: nb 1 +0.1, nb 2 +0.4, nb 4 +1.9, nb 8 +7 us (pairwise work, spills); 0.1-0.4% of outputs move 1-2 ulp | at nb <= 2: 3.437 / 3.437 / 3.437 -> 3.435 / 3.434 / 3.434 (-0.09%) | no: not worth a numerics change |
| land over a cluster of 2 / 4 / 8 blocks (a slice of the row a block, totals over DSMEM, cluster barrier) | 1 row nb 1/2/8: 4.64/5.32/7.74 -> cl2 4.37/4.73/6.23, **cl4 3.96/3.96/5.28**, cl8 3.91/4.05/5.69; 64 rows nb 2: 5.56 -> 5.14 / 4.92 / 5.42; <= 0.03% of outputs move 1-2 ulp | see A/B | |
| snapshot rows by TMA (`cp.async.bulk` 14 KB a snapshot, warp c's lane 0, one mbarrier) instead of 16-byte cp.async | land nb 2: -0.1..-0.2 us, nb 4-8: flat (9.3 vs 9.0-9.7) | | no: the ~0.5 us a snapshot is not the load |

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

## The references, as written, next to ours (`--test ref`, work/ar/ref_ar.cu)

FlashInfer / TRT-LLM `allreduce_fusion_kernel_oneshot_lamport` (trt: kARResidualRMSNorm; trtar: kAllReduce): its
token-per-cluster layout and launcher geometry (224 threads x cluster 4 at 7168 wide, 448 x 2 past 38 rows), push,
clear of the buffer two calls back, -0.0 poll, grid counter; SGLang `all_reduce_push_res_kernel` (sgl: residual
add, no norm) with its `multimem.st` replaced by 8 unicast stores (no multicast object in the runtime), its
two-phase per-block buffers, zero-atom marker, poller re-zeroing. Ours: `kern_k3_ar_attnres_rms` at nb 0 (residual
add, the attention residual's mix of nothing, norm). 8 ranks over 2 trays, rank 0, us per call, **spacer 5 us
included** (subtract 5):

| 7168 wide | 1 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours nb 0 | 13.25 | 13.79 | 14.88 | 15.70 | 15.84 | 16.11 | 16.52 |
| ours nb 1 | 14.34 | 14.74 | 15.97 | 17.06 | 16.93 | 17.61 | 17.88 |
| trt (+ residual + rms) | 12.42 | 12.83 | 14.20 | 15.29 | 16.66 | 19.39 | 21.98 |
| trtar (sum only) | 10.79 | 11.33 | 12.56 | 13.93 | 15.29 | 17.75 | 20.21 |
| sgl (+ residual) | 9.80 | 10.51 | 11.74 | 13.11 | 14.48 | 17.20 | 19.80 |

| 10752 wide | 1 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours (`ar_finalize_rms`: + MoE finalize, latent norm) | 14.34 | 14.86 | 16.66 | 17.18 | 17.29 | 17.88 | 18.16 |
| trt (+ residual + rms over 10752) | 12.40 | 13.49 | 15.70 | 17.73 | 19.66 | 23.89 | 27.85 |
| trtar | 10.65 | 11.99 | 13.93 | 15.97 | 17.88 | 22.25 | 26.21 |
| sgl | 9.83 | 11.20 | 13.11 | 15.16 | 17.20 | 21.06 | 25.53 |

From 24 rows ours is the fastest (two-shot from 20 rows, the push spread over the grid); the references are one-shot
to 128 tokens. Below 24 rows ours is 0.4-1 us behind TRT's fused one-shot and ~2.5-3.5 us behind SGLang's push (which
does no norm).

Ablation of ours at 1 / 8 / 16 rows (spacer 0, same lease; us per call): ours 8.23 / 8.85 / 10.24; without the row
epilogue 7.78 / 8.30 / 9.56; also without main's L2 bookkeeping (l2_load_while / l2_demote / l2_arrive / l2_close,
pf_bytes 0) 6.96 / 7.78 / 9.01; trtar 5.60 / 6.42 / 7.78; sgl 4.78 / 5.46 / 6.83; an empty 152-block launch 1.50
(1024 or 128 threads, with or without 112 KB of dynamic smem: the same). So at <= 16 rows our protocol itself is
~2.2 us behind SGLang's push and ~1.4 behind TRT's: l2_close's grid counter (a __syncthreads and an atomic a block
before the grid ends) is 0.8 of it.

## In the step (nsys, rank 0/1/4, work/ar/expo.py: time on the step's frontier, i.e. a kernel's end minus the end
## of the kernel before it)

| us / step | 8 rows x 64k (2.41 ms) | 48 rows x 64k (4.39-4.58 ms) |
|---|---:|---:|
| ar_finalize_rms (16 calls) | 278-344 | 484-507 |
| ar_attnres_rms (17) | 249-298 | 400-418 |
| land_add2_attnres_rms (15) | 99-101 | 119-121 |
| head_argmax (1) | 12-14 | 51-53 |
| embed_rms_plan (1) | 6 | 6 |

- An all-reduce's frontier time is the protocol, the wait for the last rank (EP skew before ar_finalize) and the
  next weight's load window (l2_load_while runs AR_GRACE_NS = 2.5 us past the first row's data): ar_attnres at
  48 rows ends 13-17.5 us after its o_proj ends (alone: ~10 us), the next GEMM ends 13-16 us after it.
- The closing add ends 5.2-7.1 us after lat_up ends (it launches only ~1 us before): alone it is 3.96 / 4.07 /
  4.10 / 4.64 us at 1 / 8 / 32 / 48 rows with no prefetch, and 6.12 / 7.07 / 7.35 / 7.75 us prefetching the
  92 MB qkvg weight (16 MB: +0.3, 48 MB: +1 us): the 576 idle blocks' prefetch issue holds the grid the qkvg GEMM
  waits on.

## Tried and dropped

| idea | harness | step | kept |
|---|---|---|---|
| two-shot owner's gather by TMA (the summed row staged in smem, one cp.async.bulk a peer) | MoE 20-64 rows +1.2..+1.9 us a call | | no: slower than 7 x n stores |
| two-shot gather spread over the non-owned rows' blocks (an equal run of the owned vectors each, before their own poll; the owner's block sums its row again for itself) | attn 20-64 rows 11.20 / 11.29 / 11.33 / 11.84 / 11.97 -> 9.65 / 9.51 / 9.56 / 10.20 / 10.60 us; MoE alone slower when the owned rows are unbalanced (20-23, 28 rows: +1 us), faster at 24 (-0.4) | see below | |
| ar_finalize_rms: the top-k (exp2perm, wts) of the block's <= 2 rows staged in smem before griddepcontrol.wait (written by moe_front, 4 kernels back) | with the spread gather, MoE 20 / 21 / 24 / 28 / 48 rows 11.71 / 11.61 / 11.69 / 11.74 / 12.15 -> 11.47 / 10.65 / 9.69 / 10.78 / 11.70 | see below | |
| ar_attnres_rms's rows over clusters of 2 (152 blocks = 76 clusters: cudaOccupancyMaxActiveClusters at one 1024-thread block an SM says 76 of 2 fit, 36 of 4, 15 of 8) | nb 1-2: flat at 1-16 rows, +0.5..+0.9 us at 24-64 (two-shot); nb 8: -0.4 at 1-16 | | no |
| ar_attnres_rms's row epilogue over a cluster of 4 (kern_k3_ar_attnres_rms_cl, rows <= 38 by `when`) | -0.5..-1.0 us a call at 1-32 rows, bit-identical | 3.404 -> 3.473 (+2%; 36 rows +220 us): clusters of 4 cannot make all 152 blocks co-resident (GPC packing), and the protocol needs every block resident: a deadlock risk | no |

Block 0's stamps at 1 row (ns from the first block's entry, ar_attnres_rms nb 1, harness without PDL): push and
clear issued 2080, the row's data in 5472, exit 6976; 8 rows: 2144 / 5728-6272 / 7776; 32 rows: 1984 / 6720-7872 /
9440. In the step the open (state, handshake, slot pointers) runs before griddepcontrol.wait under the GEMM; after
the wait it is x's load, the push, ~3.4 us to the slowest peer's data, the epilogue (~1.5 us at nb 1) and the exit.
The epilogue sits under the load window (AR_GRACE_NS = 2.5 us past the first row's data), so the grid's end is the
window's, not the row's.

| land prefetch / load-window idea | step (ms) | kept |
|---|---|---|
| closing add's prefetch hints capped at 0 / 8 / 12 / 16 / 24 / 32 / 48 MB (was the whole next weight, 92 MB for qkvg) | main 3.407-3.410; none 3.399-3.400; 3.390 / 3.392 / 3.391 / 3.382-3.390 / 3.397-3.408 / 3.389-3.391 / 3.394-3.397 | 16 MB |
| the hints stop 1 / 2 / 3 us after the block starts (whole weight) | 3.403 / 3.404 / 3.405 | no |
| l2_close's grid count replaced by a stamp a phase | 3.390 -> 3.371 / 3.374 | yes |
| AR_GRACE_NS 2500 -> 0 / 500 / 1000 / 1500 / 2000 (FIN_GRACE_NS 0 / 500 / 1000: no difference) | 3.357-3.361 -> 3.342-3.346 / 3.343-3.344 / 3.344-3.345 / 3.348-3.349 / 3.354-3.355 | 500 |
| the closing add's grid 4B + 64 / 4B + 152 instead of 608 (fewer prefetching blocks) | 3.333-3.338 -> 3.346-3.368 / 3.344-3.345 | no |
| L2_INFLIGHT (the all-reduces' loads in flight) 12 -> 4 / 8 MB | 3.338-3.339 -> 3.340-3.342 / (d85613c) 3.300-3.302 -> 3.295-3.296 | 8 MB |
| the closing add's second partial (the MoE sum, two kernels back) loaded before griddepcontrol.wait | 3.300-3.302 -> 3.301-3.304 | no |
| two-shot from 14 rows (harness: attn wins from 12, MoE from 14, loses at 10-12) | 3.300-3.302 -> 3.299 (16 rows -5..-10 us) | yes |
| push spread (runs over all 152 blocks) below 14 rows only, i.e. one-shot only / at every row count / AR_GRACE_NS 250 | 3.290-3.293 -> 3.283 / 3.292-3.294 / 3.291 | one-shot only |
| L2_INFLIGHT 6 / 10 MB, AR_GRACE_NS 0 (with 8 MB) | 3.273-3.275 -> 3.278 / 3.281-3.291 / 3.276-3.279 | no |
| ar_finalize's routed latent summed before griddepcontrol.wait (fc2 is two kernels back) | 3.272-3.275 -> 3.270-3.271 | yes |
| no push spread at all / PUSH_RUN 64 / 256 (one-shot) | 3.276-3.277 -> 3.278-3.279 (8 rows +10 us) / 3.278-3.280 / 3.278 | no: noise |
| embed_rms_plan's split plan in block B (was: warp 0 of every row block, before the row's norm) | embed alone 4.37-5.14 -> 2.97-3.64 us, the same outputs and splits; step 3.284 -> 3.282 | yes |
| land's mix norm from the candidates' pairwise dots (one cluster reduction after the wait instead of two) | | not retried: -0.09% in round 3's first session, and the only change here that would move outputs (0.1-0.4% by 1-2 ulp) |

## Runtime proposal: multicast for the exchanges

`multimem.st` to a multicast alias of `tp_ar_lamport` makes a one-shot push one copy of the partial out of each rank
(the switch fans it out) instead of 8 unicast copies, and the two-shot's gather one store. Round 2 (agent/x-mcast,
f07e2f9) measured it with k3_step creating the object (rank 0 cuMulticastCreate, its fabric handle shared beside the
NCCL id, every rank cuMulticastAddDevice, binds its own mapping by address, maps the alias, writes the alias into a
`tp_ar_lamport_mc` u64 buffer): one-shot 7168 wide 8.06 / 8.36 / 9.16 / 10.25 -> 6.92 / 8.05 / 9.31 / 10.53 us at
8 / 16 / 24 / 32 rows, two-shot 8.07-8.31 flat; in the step -0.25% (3.484 -> 3.475). What the runtime would need: a
buffer kind `multicast_of: <peer buffer>` that kern creates after the peers' handles are imported and whose value is
the alias, so a manifest can pass it to a launch. Not shipped (the manifest must run on today's kern).

Note on checking: loop/d/check runs 4 rows, i.e. only the one-shot path. Every two-shot change was checked in the
harness instead: outputs at 8-64 rows bit-for-bit equal to the variant before on every rank (no DIFF in any rank's
log). The 16-layer bench only runs nb 1-2 (attnres calls: 13 at nb 1, 4 at nb 2), so work for many snapshots does
not show in the score.
