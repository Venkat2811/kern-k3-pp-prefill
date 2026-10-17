# Round 3: every D kernel at its roofline

Rounds 1 and 2 fused D's step to 12.9 launches a layer and tried our own
GEMMs (cuBLAS kept them). `roofline/decode.md` prices every launch of the
full 93-layer step against a floor built from measured ceilings: rank 0's
busy time is 1.7-2.05x its floor, and the gap is spread over many kernels.
This round closes it kernel by kernel. Every run of this round reads this
file; its own `ROUND.md` names the kernels it owns.

## Done means

A kernel is done when one of these holds, shown with numbers in
`notes/<run>.md`:

1. it runs within ~1.15x of its floor at every A/B shape (1, 8, 32, 64 rows
   at 8k; 32 at 32k; the bench's 8-48 rows at 64k / 128k), or
2. what is left is a limit no kernel of ours can move, named and measured:
   the bytes are the minimum (ncu DRAM read / model bytes ~1.00) and the
   achieved rate is the measured ceiling, or the time is a measured protocol
   or launch floor, or the step's structure (EP skew, the runtime) sets it.
   Say which, with the counter or the experiment that shows it.

"It is close enough" is not a reason; "this is the floor, here is the
measurement" is. Floors and ceilings are in `roofline/decode.md`
(HBM 7.33 TB/s read, NVLink 715 GB/s, one-way flag 2.09 us across trays,
empty 8-rank one-shot 3.90 us). If you find the floor itself is wrong
(a better protocol, a tighter byte count), say so and use the better one.

## Learn from the engines first

Before writing a kernel, read how vLLM, SGLang, FlashInfer and TensorRT-LLM
run the same operation on Blackwell, and say in your notes what you took or
why not. `$K3_REF` has them; the most relevant:

| what | where |
|---|---|
| K3's all-reduce fusions, SGLang | `sglang/python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`, `.../comm/sp_collective.cuh`, `.../comm/gemm_ag.cuh` |
| K3's attention residual, SGLang | `sglang/python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`, `sglang/python/sglang/srt/layers/attn_residual.py` |
| all-reduce + residual + norm, MoE finalize + all-reduce | `flashinfer/csrc/trtllm_allreduce_fusion.cu`, `flashinfer/include/flashinfer/comm/trtllm_moe_allreduce_fusion.cuh`, `flashinfer/csrc/cake_moe_finalize_allreduce_fusion/`, `flashinfer/csrc/trtllm_mnnvl_allreduce.cu`, `trtllm/cpp/tensorrt_llm/kernels/communicationKernels/` |
| multicast all-reduce | `sglang/python/sglang/srt/distributed/device_communicators/torch_symm_mem.py`, `flashinfer/flashinfer/comm/mnnvl_cutedsl_ar.py`; branch `agent/x-mcast` (round 2, 2 trays: one-shot 8.06 -> 6.92 us at 8 rows) |
| KDA decode | `sglang/python/sglang/kernels/ops/attention/kda_fused_decode.py`, `kda_packed_decode.py`, `cutedsl_kda.py` |
| MLA decode, DCP combine | `flashinfer/flashinfer/mla/`, `flashinfer/flashinfer/comm/cake_fused_norm_combine.py`, SGLang's FlashMLA / trtllm-gen MLA paths |
| MoE routing | `flashinfer/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_*.cu(h)`, `trtllm/cpp/tensorrt_llm/kernels/moe/trtllmGen/routing/` |
| small-M dense GEMM | `flashinfer/flashinfer/gemm/kernels/tgv_gemm_cute_ext.py` (SGLang runs it at 1 and 8 sequences), `sglang/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py` |
| what SGLang runs per layer at our layout | `decode/sglang-tp8-dcp8-ep8/` (captured graphs) |

`$K3_REF/insights.md` has the round-1 notes on these.

## What is open

- Every D kernel of `d-*` manifests: `source/` kernels, their grids and
  variants, and the generator (`scripts/gen_stage.py`). A launch can carry
  `when: {var, min, max}` to route by row count, which also gives a launch
  expression its `min` (one launch per row bucket).
- The GEMMs as round 2 left them open (`loop/rounds/r2/GEMM.md`: replace,
  fuse, pick algorithms, prefetch their weights), under its bit-identity
  rule.
- Not open: P (`loop/gen` must give P manifests byte-identical to main's),
  quantization, the parallel layout, the kern runtime (a runtime feature is
  a proposal in your notes: what it saves, measured with `k3_step` if you
  can).

## Numerics

As in `loop/TASK.md`: every difference explained, `loop/d/check` PASS. A
GEMM change follows round 2's stricter rule (bit-identical first, else the
difference proved). A change that touches the all-reduce or the Lamport
protocol runs `loop/d/check` three times before it counts: a fired deadline
yields wrong values silently.

## Measuring a kernel

- In the step: `loop/d/bench` (8 GPUs, ~2 min) for the weighted cost; nsys
  on it for each launch's busy time.
- Alone: a harness under `work/` that runs your kernel at the A/B shapes in
  a CUDA graph, through `loop/withgpu -n 1` (or `-n 4 -H 2` for collectives).
  Report us per call, the floor, and the ratio.
- Interleave candidate and main in one lease for A/B; the noise between
  leases is larger than many wins here.

## Done for the round

Each run ends its notes with the table of its kernels: shape, measured,
floor, ratio, and for each that is not within ~1.15x the named limit. The
orchestrator folds those tables into `roofline/decode.md`.
