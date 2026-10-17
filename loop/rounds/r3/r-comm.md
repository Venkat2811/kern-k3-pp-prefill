Run `r-comm` · **target: D's exchanges themselves at their floor: the 8-rank Lamport protocol under `k3_ar_fused`'s two all-reduces, the DCP exchange (`k3_dcp`) and the head's logits gather (`k3_head`); a protocol header the fused kernels adopt.** Notes go in `notes/r-comm.md`.

Read [`loop/rounds/r3/ROOF.md`](ROOF.md) first: done-means, references, numerics, GPUs. Use the `KernelWiki` skill (Blackwell kernel techniques and PRs: search it for all-reduce, Lamport, NVLink, multicast, MNNVL) and `ncu-report-skill` before and while you design.

This run is a second, independent line on D's communication, beside `r-ar`, on purpose: do not coordinate with it, do not read its branch. The whole exchange path is yours, `source/k3_ar_fused.cu` included (protocol, row math, thresholds, prefetch), plus `k3_dcp` and `k3_head`. The orchestrator measures both runs' results and merges the faster one.

Where they stand (`roofline/decode.md`): two exchanges a layer, 186 a step, the empty 8-rank one-shot 3.90 µs (2 trays), one-shot all-reduce 6.0 µs at 1 × 7168, 7.3 / 10.2 µs at 8 rows (7168 / 10752 wide), 12.0 / 16.0 at 32, 18.4 / 26.2 at 64; `k3_dcp` 12.0-24.7 µs (1.2-2× a same-size one-shot); `k3_head` 11.7-64 µs. The one-way NVLink flag across trays is 2.09 µs, so an exchange's latency floor is about two of those; 3.90 is this protocol's, not the hardware's.

Order of work:

1. Baselines first, as written by their authors, on our peer buffers, 8 ranks over 2 trays, CUDA graphs, our shapes (7168 and 10752 wide, 1-64 rows): TRT-LLM / FlashInfer Lamport one-shot and two-shot (`flashinfer/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`, `trtllm/.../communicationKernels/allReduceFusionKernels.cu`, `lamportUtils.cuh`), MNNVL all-reduce (`trtllm_mnnvl_allreduce.cuh`, `mnnvlAllreduceKernels.cu`), SGLang's `kimi_k3/comm/ar_fusion.cuh` and `sp_collective.cuh`, `inkling_all_reduce.cuh`, and NCCL 2.30's symmetric / device-side all-reduce if you can call it from the harness. One table, µs per call, next to ours.
2. Then the floor: what limits an exchange across two trays (flag latency, poll scan, the push from one SM, stage clearing, PDL); per-block `globaltimer` stamps tell where the time goes. `r-ar` measured that one block pushing a whole row caps near 55 GB/s and that rows see their data ~6 µs after entry; build on that, do not repeat it.
3. Multicast (`multimem.st` / `multimem.ld_reduce`, round 2's `agent/x-mcast` sets it up in `k3_step`): measure it on our shapes; it needs the runtime to create the multicast object, so it ships only as a proposal with its number.
4. `k3_dcp`: the merge before the push and the merge + W_UV after the poll; FlashInfer's `comm/cake_fused_norm_combine.py` and `dcp_lse_reduce` are references.
5. `k3_head`: rank 0's NVLink ingress at 32-64 rows (7 × m × 80 KiB), the small-message exchange at 1-8.

Check: `loop/d/check loop/out/d-l4-tp8.json` three times for any protocol change. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
