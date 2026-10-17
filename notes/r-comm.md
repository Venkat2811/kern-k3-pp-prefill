# r-comm: D's exchanges at their floor

Run `r-comm`, round 3. Target: the 8-rank Lamport protocol under `k3_ar_fused`'s two all-reduces,
the DCP exchange (`k3_dcp`) and the head's gather (`k3_head`). Since `fd729e6` the whole exchange
path is this run's, `k3_ar_fused.cu` included.

## State

Main is `fd5c92c` (04:27: r-ar + r-gemm, 3.239 ms/step in the orchestrator's lease). This branch is
rebased onto it and carries:

| commit | what | D 16 layers, weighted (A/B against its parent in one lease, twice) |
|---|---|---|
| `7560058` | dcp: one wait a merge, reduce-scatter v-up, 16 splits' lse at once | 3.295 / 3.296 -> 3.294 / 3.293 ms on `21a9174` |
| `5307a10` | head: the row's key after a gpu-scope fence (-0.6 to -3.9 us a call) | 3.287 / 3.289 -> 3.288 / 3.290 ms (one call a step) |
| `c934878` | the two-shot MoE all-reduce's row block waits after its gather and its shared half | 3.245 / 3.244 -> 3.241 / 3.240 ms on `fd5c92c` (36 rows -13..-19 us) |
| `eb1f5ab` | deadlines in SM cycles (all-reduces, head), as k3_dcp's | 3.241 / 3.240 -> 3.243 / 3.243 ms (noise; a fired deadline traps the step) |
| `a6707bd` | dcp traps without a printf: no stack frame, `harness/run.sh sass` all PASS (k3_dcp failed it on main) | 3.242 / 3.243 -> 3.254 / 3.240 ms (noise) |
| `0f5d1fe`, `02a2685` | `harness/xchg/` (the 8-rank harness), `abn.sh` (interleaved A/B) | |

All bit-identical (`loop/d/check` x3 each; the harness compares main's kernels byte for byte).
The whole branch (`eb1f5ab`) against main `fd5c92c`, interleaved three rounds in one lease:
3.248 / 3.248 / 3.248 -> 3.245 / 3.244 / 3.247 ms/step (-0.1%).

History of the all-reduce line, for the orchestrator's question (03:48):

- `k3_xchg.cuh` (per-block stage records, input / record / peers loaded together, pushes spread
  over the grid, the two-shot gather spread over the grid, the MoE row's shared half summed by the
  grid, deadlines in cycles, switch at 20 / 12 rows) measured against main `95bc0b0`'s protocol:
  fused all-reduces 1.0-3.8 us faster a call in the harness (tables below), -0.8% in the step
  (3.411 / 3.414 -> 3.384 / 3.385). The warp-0 softmax took another -0.1%.
- r-ar's rewrite (merged at `21a9174`) took the same protocol ideas (per-block stage state, pushes
  from every block, owner sums gathered by the other rows' blocks, no grid count) and faster row
  math: its attention all-reduce is 9.3 us at 1 row against the header's 10.4 (4 snapshots). So
  the rewrite was dropped (`r-comm-pre-rebase` keeps it). What it still won on top of r-ar's: the
  one/two-shot switch at 12 rows (r-ar has since moved to 14), and the two-shot MoE all-reduce's
  waits (`c934878`), which the header's grid-wide gather never needed.
- In the orchestrator's 04:27 lease the branch before this rebase (85048e0) measured 3.280 against
  main 3.290 (-0.3%).

## The harness (`harness/xchg/`)

`xbench`: 8 processes over the lease's two hosts (`run.sh`, as `loop/d/ranks`), every peer buffer
a fabric allocation exported through files and mapped by every member, as kern's are (optional
multicast object bound to it). A test is `calls` launches captured in one CUDA graph, replayed twice
to absorb the members' start skew, then the median of 5 replays / calls: us per call. The Lamport
chain keeps the members in step, so a call's time includes the slowest member's.

    harness/xchg/build.sh
    loop/withgpu -n 4 -H 2 harness/xchg/run.sh OUT TEST...

`ours.cu` (this tree's kernels, a TU each), `main_fused.cu` / `main_dcp.cu` (main's, entries
renamed, for byte-for-byte checks), `new.cu` (bare all-reduces on the header), `fi_ar.cu`
(FlashInfer's `trtllm_allreduce_fusion.cuh` as written). `harness/xchg/abn.sh OUT N M1.json M2.json ..`
interleaves D benches of several manifests in one lease (main's from a `git archive main` tree).

## Measured floors (8 GB300, two trays)

| what | us | how |
|---|---:|---|
| one-way flag, ping-pong, same tray / across trays | 1.25 / 1.31 | one thread each side |
| one-way 14 KB row, one block, ping-pong | 1.96 / 1.99 | (16 B: 1.36 / 1.50; 64 KB: 3.1) |
| empty kernel in a graph, per call | 0.53 (1 block) / 0.69 (152 x 1024) | PDL launch: 0.39 / 0.54 |
| 8-rank flag exchange per call, in a graph | 2.58-2.76 | thread q stores to member q, polls slot q |
| ... Lamport 16-byte payload, 3 stages, re-poison | 2.89 | `lx_par_lamport` |
| ... one thread storing and polling all 8 | 4.00 | one thread's serial waits: +1.1 |
| ... one thread polling all 8 at once | 3.41 | +0.5 |
| remote store issue, one warp, 8 x 512 B | 0.25 | acked (fence.sc.sys): 2.3 |
| rank 0 NVLink ingress, 7 members writing at once | ~600-630 GB/s complete, 13.8-27.5 MB | each sender fenced (acknowledged) before the closing flag; unfenced the flag overtakes data in flight and the same bytes read 760-1150 GB/s |
| one block reading a row's 8 slots from L2 | 0.38 | 750 cycles |

The roofline's "2.09 us one-way flag" includes a launch; the flag alone is 1.3 us, within a tray
or across (NVL72: every hop goes through the switch). An empty 8-rank Lamport exchange is 2.9 us
per call in a graph; the protocol's 3.9 is the old grid funnel and serial waits.

**The one-shot's floor at 1-8 rows is a row's work spread over SMs.** The same push-then-poll
kernel at 4 CTAs x 224 threads a 7168 row (FlashInfer's shape) is 4.65 us at 1 row; at 1024
threads a row (one SM) 7.27. One SM storing a row to 8 members and polling its 8 slots is the
limit, not the wire (1 x 7168: 112 KB of egress, 0.16 us).

## k3_xchg.cuh (`51538be`)

Wire unchanged (TRT-LLM's Lamport one-shot / two-shot, rank-order f32 sums: bit-identical).

1. **Per-block records instead of the grid funnel**: block b reads its own 16-byte record {calls,
   clear count} with its inputs at entry and writes it at exit; nothing atomic, no block waits for
   another. Every kernel on one record array runs the same grid (152).
2. **Input, record and peer table load together at entry** (the peer table in registers through a
   select chain: indexing it by a runtime member put it on the stack, +1.6 us).
3. **Spread push**: 32-vector chunks dealt round-robin over the blocks.
4. **Spread two-shot gather**: the owner's rows summed and gathered by the whole grid, not by the
   row's block; the owner's row block sums its row from the slots again.
5. **Spread non-row vectors** (`xchg_spread`): the MoE row's shared half (896 of 1344 vectors) needs
   no row math; the grid sums it. A thread waiting for two vectors in turn pays the wait twice, and
   polling both at once spills at 1024 threads (64 registers, +3 us).
6. **Deadline in SM cycles** (as `k3_dcp.cu`): %globaltimer jumps early in a process.
7. One/two-shot switch: attention from 20 rows (was 33), MoE from 12.
8. main's L2 loads keep working: their words moved past the records, the arrival word alternates by
   call parity (block 0 zeroes the next), so `l2_close`'s grid-wide counter is gone too.

Fused kernels, main vs this branch, one lease (harness, us, 4 snapshots, no prefetch):

| rows | attn main | attn new | moe main | moe new |
|---:|---:|---:|---:|---:|
| 1 | 11.48 | 10.47 | 9.50 | 6.64 |
| 8 | 11.51 | 10.45 | 9.43 | 7.99 |
| 16 | 11.73 | 10.51 | 10.68 | 8.47 |
| 24 | 13.11 | 11.95 | 12.34 | 8.57 |
| 36 | 12.73 | 12.03 | 11.68 | 8.97 |
| 48 | 13.01 | 12.11 | 11.94 | 9.33 |
| 64 | 13.04 | 12.18 | 12.27 | 9.93 |

In the step this is -0.8% (27 us of 32 all-reduces): about half of the harness's gain shows up; the
rest is off the critical path (the MoE all-reduce's faster ranks wait for the slowest's MoE anyway).

## Baselines: the authors' kernels on our buffers (bare all-reduce, us per call)

FlashInfer's `trtllm_allreduce_fusion.cuh` as written (kAllReduce pattern, its own grid choice:
a cluster of 4 x 224 threads a 7168 row, 8 x 168 a 10752 row; workspace laid out as
`trtllm_create_ipc_workspace_for_all_reduce_fusion`), `fi_ar.cu`. Ours: `kern_xar` (the header's
one-shot / two-shot at 152 x 1024), `kern_xar_local` (the header, FlashInfer's shape: a thread pushes
its vector and polls it), main's `kern_peer_allreduce_bf16`, multicast (`kern_xar_mc`, below).

| rows x 7168 | main peer_ar | FI one-shot | FI two-shot | ours one | ours local | ours two | mc one | mc two |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.86 | 5.17 | 17.96 | 5.75 | 4.63 | | 5.20 | |
| 8 | 6.28 | 5.85 | 22.62 | 6.51 | 5.33 | 7.58 | 6.16 | 6.55 |
| 16 | 7.44 | 7.04 | 24.67 | 7.58 | 6.50 | 7.80 | 7.44 | 6.86 |
| 24 | 8.59 | 8.35 | 24.58 | 8.68 | 7.72 | 7.95 | 8.87 | 7.05 |
| 32 | 9.79 | 9.53 | 24.32 | 9.78 | | 8.06 | 10.09 | 7.21 |
| 48 | 12.19 | 12.38 | 24.85 | 12.12 | | 8.27 | 12.63 | 7.74 |
| 64 | 14.77 | 14.94 | 25.36 | 14.48 | | 8.57 | 15.14 | 8.31 |

| rows x 10752 | main peer_ar | FI one-shot | FI two-shot | ours one | ours local | ours two | mc one | mc two |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.91 | 5.10 | 16.93 | 6.23 | 4.61 | | 6.58 | |
| 8 | 6.86 | 6.51 | 21.56 | 7.52 | 6.01 | 7.62 | 8.13 | 7.08 |
| 16 | 8.59 | 8.67 | 24.42 | 9.20 | 7.91 | 7.94 | 10.11 | 7.42 |
| 24 | 10.42 | 10.56 | 24.46 | 11.02 | 9.86 | 8.27 | 12.02 | 7.71 |
| 32 | 12.20 | 12.50 | 24.38 | 12.89 | | 8.53 | 13.96 | 8.09 |
| 48 | 16.42 | 17.09 | 25.00 | 16.60 | | 9.18 | 17.79 | 8.79 |
| 64 | 20.21 | 20.82 | 25.96 | 20.41 | | 9.84 | 21.64 | 9.61 |

- FlashInfer's one-shot beats ours by 0.4-1.1 us at 1-16 rows, and the header in its shape
  (`local`) beats FlashInfer by another 0.5: the shape, a row's push and poll over 4-8 SMs, is the
  difference. Its two-shot (barrier-synchronized, two grid barriers) is 17-26 us: never use it here.
- From 16 rows (10752) / 24 rows (7168) the spread two-shot wins everything unicast.
- Not run: TRT-LLM's copy (the same kernel, per the survey), MNNVL (needs a multicast object for its
  one-shot's single store; its two-shot scatters unicast and multicasts the sum, i.e. `mc two`),
  SGLang `ar_fusion.cuh` push kernel (multimem.st one-shot with per-block phases: `mc one` plus
  per-block records, which ours already has), inkling v4 (multimem.ld_reduce in bf16: not
  rank-order f32, would change numerics), NCCL symmetric (not callable from the harness).

## Multicast (proposal: needs the runtime to create the multicast object)

The orchestrator measured SGLang's K3 small-message protocol (`kimi_k3/comm/ar_fusion.cuh`'s
push / one-shot: one `multimem.st` into slot `rank` of every member, no barrier) on this cluster:
6.92 us at 8 x 7168 on 8 ranks over 2 trays against 8.06 unicast, 5.6 us on 4 ranks in one tray;
crossing the trays costs ~0.3 us; barriers with a system-scope fence or release ~4.4 us each,
relaxed ~2.5.

`kern_xar_mc`: the header with every store all members receive as one `multimem.st` to the stage's
multicast alias: the one-shot's push, and the two-shot owner's sum into an extra slot (so the
owner's own partial in slot `rank` is not overwritten; a stage holds NRANKS + 1 slots). Same bytes
in the slots, same sums: bit-identical.

- Two-shot: 0.3-1.0 us faster than unicast at every row count (gather egress 1/7).
- One-shot: faster at 1-8 rows of 7168 (5.71 -> 5.20, 6.51 -> 6.16), slower at 10752 and from 24
  rows (the switch's replication rate is lower than 8 unicast streams).
- Best of each against best of each: 0.2-0.8 us a call, ~16 us a 16-layer step (~0.5%).
- What the runtime needs: `"multicast": true` on an exported buffer: create the object over the
  group (rank 0 creates, shares the fabric handle, every rank adds its device), bind each rank's
  allocation, map the alias, expose it as `X_mc` (u64[1]) like a peer table. `agent/x-mcast` does
  this in `k3_step`.

## k3_dcp (`d315411`)

Stamps (per-block globaltimer, `stamp_dcp.cu`) put the time in three serial chains:

1. The split merge loaded splits 8-15 one after another (`SPLITS_AT_ONCE 8`): 4.4 us before the
   push at 1 row (16 splits). Now all 16 lse in registers.
2. A merge item waited for the members' lse, then their o: one wait for both (`await_all2`).
3. The v-up GEMV reduced each of a warp's 32 (row, dv) sums by a 5-shuffle butterfly: two 16-value
   reduce-scatters (same xor tree, bit-identical).

us per call (harness, 8192 positions a member, the plan's splits):

| rows | 1 | 8 | 16 | 24 | 36 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 12.0 | 10.5 | 11.5 | 12.7 | 15.7 | 22.4 | 24.9 |
| `d315411` | 9.6 | 9.7 | 10.8 | 12.2 | 15.2 | 20.5 | 22.6 |
| floor (the rank's 7/8 of R x 96 heads x 1 KB at 700 GB/s, + 2.9 us exchange) | 3.0 | 3.9 | 4.9 | 5.8 | 7.2 | 8.6 | 10.6 |

- Grid 304 (two blocks an SM) made the 48-row merge one round (22.4 -> 17.5) but the step slower:
  with every SM full, the o_proj GEMM that launches behind it as a programmatic dependent had
  nowhere to land. Kept 256.
- The merge phase alone, pushes to the owners against the same stores to local memory: 4.04 / 4.01
  (8 rows), 6.33 / 5.46 (24), 8.52 / 6.65 (36), 10.49 / 8.04 us (48). The egress is 0.9-2.5 us of
  it; the rest is reading the attention's f32 split partials (2 KB a split and head: 27.6 MB at 48
  rows x 3 splits, ~3 TB/s achieved in 2-4 rounds a block). Hoisting the splits' loads, a lane per
  split (max by a warp reduction, the sum in order through shuffles; bit-identical), L2-prefetching
  the next round, warp-uniform early exits, software-pipelining the items (the next item's row
  length, split count, lse and first four splits' o loaded before this one merges; 128 registers,
  bit-identical): none moved 24-48 rows. bf16 partials (the Cake path's) would halve the reads: the
  attention's output format, not the exchange's. Caveat: the harness writes the partials once at
  setup; rewriting them just before each call (as the attention does in the step) gives the same
  times (`X_DCP_HOT`: 10.3 / 10.0 / 11.0 / 12.2 / 15.6 / 20.1 / 23.0 us at 1-64 rows).
- Nsight Compute on the merge phase alone (24 rows, its pushes to local memory, one GPU): latency-
  bound, L1TEX scoreboard 47% of the stalls at 16 warps an SM (25% occupancy: 94-128 registers and
  the v-up's 64 KB W stage allow two blocks an SM), SMs active 56% of the elapsed cycles (the
  blocks with 3 items against 2), DRAM 19% (1.5 TB/s, ncu's L2 flushed). More warps need a smaller
  W stage and fewer registers, and room left for the o_proj GEMM launched behind it (grid 304 took
  that room: step slower). Not done.
- Left: at 36-64 rows the push of R x 96 records of 1 KB (5.4 MB at 64 rows, 7.7 us at the
  all-to-all rate) and two merge rounds; at 1-24 rows the merge chain (load lse, weights, o, push)
  then the v-up. Applying W_UV before the exchange would cut the bytes 4x but needs every head's
  W_UV on every rank (+300 MB a rank) and a tensor-core GEMM in the kernel: not done.

## k3_head (`fd8a04b`)

The row's tail ran two system fences in a row (every block's before it counts in, then the last
block's again before it pushes the row's key); stamps put the second at ~2 us (rank 5, 8 rows:
fenced 8.19 -> key out 10.14). The second only needs the other blocks' keys visible on this GPU:
a GPU-scope fence. 8.4 / 14.3 / 22.6 / 29.5 / 40.0 / 51.7 / 64.5 -> 7.8 / 13.3 / 21.2 / 28.4 /
38.4 / 48.0 / 60.6 us. What is left at 24 rows: the senders' fences return at 23.5 us (their
copies taken by rank 0 at ~590 GB/s, against ~800 for the same bytes as plain stores), then the
key's one-way trip. The landing itself is at the floor: the head's exact pattern as plain fenced
stores (`ingress2`) completes in the same ~23 us at 24 rows (one fence a row instead of sixteen
saves 1.5 of it); unfenced copies of the same bytes look 30% faster only because the closing flag
overtakes them.

us per call before it (harness, `kern_k3_head_argmax` as the step runs it) against rank 0's ingress
(7 x B x 80 KiB at 730 GB/s) + the key exchange (~3 us: the senders' system fence round trip, then
one flag):

| rows | 1 | 8 | 16 | 24 | 36 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| measured | 8.4 | 14.3 | 22.6 | 29.5 | 40.0 | 51.7 | 64.5 |
| floor | 3.8 | 9.3 | 15.6 | 21.9 | 31.3 | 40.7 | 53.3 |

PARTS 32 / 64 (more, smaller blocks) is slower; 8 x 512 and 4 x 1024 within +-2 us, no consistent
winner. Rank 0's ingress is the limit; the full-logits gather is the contract (the caller reads
rank 0's logits). Proposals: skip the gather on steps whose caller only needs tokens (roofline
idea 12), or let the lm_head GEMM write rank 0's buffer directly through a peer-mapped output
(needs a peer buffer usable as a GEMM output in the runtime).

## r-ar's all-reduces (main since `21a9174`), measured here

Harness, us a call (`fusedv`, `attnv`, `sweep`; 1 snapshot):

- One-shot vs two-shot: at 16 rows the MoE all-reduce is 10.30 one-shot / 8.54 two-shot, the
  attention's 9.56 / 8.42; at 12 rows 8.96 / 8.56 (attention); 8 and 24 rows unchanged. Both now
  switch at 12 (`08b98c2`, `fb64092`), byte for byte main's outputs.
- Pushes from every block at all sizes (PUSH_SPREAD_ROWS 33 -> 65): 0.4 us faster for the attention
  all-reduce at 36-48 rows in the harness, but in the step it competes with the next weight's L2
  loads: 3.288 -> 3.286-3.306 ms (noisy, not better). Kept r-ar's 33.
- The MoE all-reduce at B % 8 != 0 was 1.0-2.0 us slower than at the multiples of 8 around it
  (24 / 28 / 30 / 32 / 36 / 40 / 44 / 48 rows: 8.6 / 9.8 / 10.6 / 8.8 / 10.5 / 8.9 / 10.5 / 9.1).
  Stamps (`stmoe`) of a build with two `__syncthreads` added made the cause visible: that build
  was faster. Warps that finish a phase early (the gather, the shared half) start spinning on the
  next phase's slots before the data is there, and their polls slow everyone's. Two-shot, the
  row block now waits after each (`23bf79b`): 8.2-9.1 us at every count from 12 to 48. One-shot
  the second wait costs 0.2-0.4 us (fewer slots, all landing together), so not there.
- The same lesson the other way: one-shot, giving the shared half's first 576 vectors to the
  threads past the latent's 448 (so no thread waits twice) is slower, 8.12 -> 8.88 us at 1 row:
  more threads spinning at once. Dealing the gather in chunks over every row block (as
  `k3_xchg.cuh`) is smooth across counts but slower at the multiples of 8 and costs the attention
  all-reduce ~2 us.

## Tried and dropped

- PDL on the all-reduces and the closing add (`"pdl": true`, records / peers / re-poison / snapshot
  copies before `griddepcontrol.wait`): the step went 3.39 -> 3.44 ms. The cuBLAS GEMM before
  them does not trigger early, so nothing overlaps, and the reordered prologue put the snapshot
  copies ahead of the input load.
- Reduce-scatter warp sums in the attention residual's scoring (bit-identical, closing add
  -0.15 / -0.6 us at 4 / 8 snapshots in the harness): the step did not move (3.383 -> 3.388).
- The closing add over a cluster of C = 7 / 4 / 2 CTAs a row, distributed-shared-memory reductions
  that keep the 1024-thread block's trees (bit-identical at every row count and 0-8 snapshots):
  1 snapshot 4.45 -> 4.8-5.4 us, 4 snapshots equal, 8 snapshots -1 us. The cluster launch and its
  two barriers cost what spreading the loads saves. The 16-layer bench has 1-2 snapshots.
- A poll gate (8 lanes wait for one vector a slot, then the row reads once): slower everywhere (on
  the header's one-shot, where every thread waits for one vector only).
- The attention all-reduce over a cluster of C = 4 / 2 CTAs a row (the row's poll, loads and row math
  over C SMs, distributed-shared-memory reductions): C = 2 wins 0.4-0.6 us at 24-48 rows, loses
  0.1-0.4 at 8-16, -0.9 at 1 row: ~-0.2 us a call at the bench's traffic, ~0.1% of the step. Not
  bit-identical from 2 snapshots (1-9 elements of `normed` a call) and not worth finding why.
- A `__nanosleep` backoff in the polls (32 / 100 / 200 ns): slower at every row count, 0.1-0.9 us.
  The same in k3_dcp's `await_all2` (64 / 256 ns, to free issue slots for a co-resident block still
  merging): 0.4-0.6 us slower at 1-36 rows, equal at 48.
- A barrier between a block's pushes and its row poll: neutral (+-0.1 us), worse at 48 MoE rows.
- The shuffle-free softmax (each thread computing all candidates' weights, warp_sum's tree replayed in
  registers; bit-identical): 9 rsqrt / exp / IEEE divides a thread swamp the special-function unit,
  closing add 4.45 -> 5.15 us. Warp 0 alone running the original (`f12cd81`) is what helps.
- The one/two-shot switch at attention 16 / MoE 8 rows instead of 20 / 12: the step identical to
  0.001 ms at every shape (3.383 / 3.384 / 3.384).
- `l2_arrive` as a fire-and-forget `red.min` (thread 0 no longer waits for a load of the arrival word
  before its block's next barrier): step identical (3.381 -> 3.383).
- PDL on the all-reduces and the closing add with nothing moved ahead of `griddepcontrol.wait`
  (on 95bc0b0): 3.385 -> 3.420 ms/step; the launch attribute alone costs here.
- PDL for the DCP exchange: its predecessor, FlashInfer's CuTe-DSL MLA decode, has no PREEXIT
  (`griddepcontrol.launch_dependents`) in its SASS, so its dependents launch at its completion.
- One-shot `xchg_sum_n<2>` (both vectors of a thread polled at once): spills.

## Done table (harness, us a call, 8 GB300 over two trays)

The all-reduce rows are r-ar's kernels as merged (`21a9174`) with this branch's switch points and
waits (`08b98c2`, `fb64092`, `23bf79b`), 1 snapshot; the closing-add row is this branch's before
the rebase (r-ar's now runs a row over a cluster of 4 blocks). `k3_dcp` and `k3_head` are this
branch's.

Floors: an exchange is latency-bound at these sizes. The empty 8-rank Lamport exchange in a graph
is 2.9 us (launch 0.6 + one-way 1.3 + a poll's round trip); a row's push and poll spread over 4 SMs
reaches 4.65 us at 1 x 7168 (`xar_local`), which is the one-shot floor used below; the two-shot's is
two hops plus its bytes. The row routine's own floor is its skeleton (three row loads, the add, two
block reductions: 2.2 us a call, `skel`).

| kernel | rows | measured | floor | x | limit |
|---|---:|---:|---:|---:|---|
| `ar_attnres_rms` (r-ar's, this branch's switch; 1 snapshot) | 1 / 8 | 8.2 / 8.3 | 4.65 + 2.2 | 1.2 | one SM a row: its 8-slot poll (+0.9 measured on the header's one-shot, `xar_spreadpoll`) and its row tail |
| | 16 / 24 / 36 / 48 | 8.7 / 8.8 / 9.5 / 9.5 | two-shot 7.8-8.3 + 0.6 | 1.0-1.1 | at the protocol (the spread two-shot bare) plus its row math; from 33 rows the pushes leave from the first blocks so the rest can load the next weight (spread: 0.4 us faster here, not in the step) |
| `ar_finalize_rms` (r-ar's, this branch's switch and waits) | 1 / 8 | 8.0 / 8.6 | 4.6 / 6.0 (local) + 0.3 | 1.7 / 1.4 | the row's 1344 vectors polled by one block, the shared half then the latent; spreading them over more threads at once is slower (more spinners) |
| | 16 / 24 / 36 / 48 | 8.2 / 8.3 / 9.0 / 8.7 | spread two-shot bare 7.9 / 8.3 / 8.6 / 9.2 | 1.0-1.05 | at the protocol's floor; multicast's two-shot is 0.3-1.0 below it (proposal) |
| `land_add2_attnres_rms` (no exchange) | any | 4.1 / 4.8 / 5.9 / 8.2 (1 / 2 / 4 / 8 snapshots) | 2.2 skeleton | 1.9-3.7 | one SM a row: the snapshots' copies (one SM keeps ~30-50 KB in flight) and 32 warps' scalar work; a cluster of 7 / 4 / 2 SMs a row costs as much (tried) |
| `k3_dcp` | 1 / 8 / 16 / 24 | 9.6 / 9.7 / 10.8 / 12.2 | 3.0 / 3.9 / 4.9 / 5.8 | 3.2-2.1 | the merge phase: reading the attention's f32 split partials at ~3 TB/s in 2-3 rounds a block (registers allow one item in flight; two blocks an SM starve the o_proj GEMM behind it), then the egress |
| | 36 / 48 / 64 | 15.2 / 20.5 / 22.6 | 7.2 / 8.6 / 10.6 | 2.1-2.4 | the same (f32 partials: 27.6 MB at 48 rows), plus two merge rounds from 43 rows |
| `k3_head` | 1 / 8 / 16 / 24 | 7.8 / 13.3 / 21.2 / 28.4 | 3.8 / 9.3 / 15.6 / 21.9 | 2.1-1.3 | rank 0's ingress (730 GB/s measured) + the completion protocol (each sender's system fence round trip, then the key, ~3 us) |
| | 36 / 48 / 64 | 38.4 / 48.0 / 60.6 | 31.3 / 40.7 / 53.3 | 1.23-1.14 | rank 0's ingress |

DCP floor: the rank's 7/8 of R x 96 heads x 1 KB at the 700 GB/s all-to-all + the 2.9 us
exchange. Head floor: 7 x B x 80 KiB into rank 0 at 730 GB/s + 3 us (the complete, fenced transfer
measures ~600-630 GB/s at 13.8-27.5 MB, so the reachable floor is ~20% above this).

## Proposals (runtime or callers; measured where possible)

1. **Multicast objects for exported buffers** (`"multicast": true`, the alias as `X_mc`): the
   two-shot gather and the small one-shot as one `multimem.st` a vector. 0.2-0.8 us an
   all-reduce, ~0.5% of the 16-layer step (table above). `agent/x-mcast` has the k3_step side.
2. **A peer buffer usable as a GEMM output**: lm_head writing rank 0's logits directly over
   NVLink would hide the head's 7 x B x 80 KiB ingress under the GEMM's own weight stream
   (~40 us at 280 MB), leaving the key exchange (~3-4 us) for `k3_head`.
3. **Skip the full-logits gather** on steps whose caller needs only tokens (roofline idea 12):
   the head becomes the 8-rank key exchange, ~3-4 us at any batch.
4. **cuBLAS kernels with an early trigger** (or a PDL-aware GEMM): every kernel of ours runs
   after a GEMM whose dependents launch only at its completion; PDL on ours measured +1.6%
   because nothing overlaps.
5. `loop/count` counts every `when`-gated launch of an op although its docstring says a gated
   launch counts once (it sums `len(launches)`): a variant split by batch looks like more kernels.
