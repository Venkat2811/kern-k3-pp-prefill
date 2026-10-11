# Round 2: D's GEMMs

Round 1 fused D's non-GEMM kernels down to their bounds
([`../r1/RESULT.md`](../r1/RESULT.md)). What is left in D's step is mostly
GEMM time, so this round opens **D's GEMMs** (decode only). Every run of
this round reads this file; its own `ROUND.md` names the shapes it owns.

## What is open, what is not

- Open: replacing `gemm_f32` (cublasGemmEx, f32 out), `gemm_bf16`
  (cuBLASLt, bf16 out) and the TRT-LLM gen `moe_fc1` / `moe_fc2` cubins **in
  the D manifests** with kernels of ours in `source/`, and fusing a GEMM
  with the kernels around it (a prologue that norms or gathers its input, an
  epilogue that reduces, adds, quantizes or feeds the all-reduce).
- Not open: P (its manifests must come out of `gen_stage.py` byte-identical
  to main's: check with `loop/gen` and a diff), quantization (weights stay
  bf16 / mxfp4 as they are, activations as the manifests have them), the
  parallel layout, the runtime.
- `when: {var, min, max}` on a launch routes by row count: a kernel may win
  only on some row counts and leave cuBLAS the rest.

## Numerics: bit-identical first

The bar is **bit-identical to main** (`loop/d/check` prints `bit-identical`
on the `cand vs main` line). Where that cannot be had at the speed, the
difference is **proved**, not just measured:

1. Name the exact change in summation order (split-K boundaries, the order
   partial sums are added, the MMA instruction's own accumulation, f32 vs
   bf16 rounding of the output) for every row count the step runs.
2. GEMM level, in a harness, on real weights (`$K3_WEIGHTS`) and real
   activations (dump them from a `loop/d/check` run or the oracle): the
   output of the new kernel and of cuBLAS against an fp64 reference. The new
   kernel's error must be no larger than cuBLAS's (max and distribution,
   per output element, in bf16 / f32 ulps), and the bound must follow from
   (1).
3. `loop/d/check` PASS, its `cand vs main` line in the commit.

Reaching bit-identity usually means reproducing cuBLAS's order: find the
kernel cuBLAS actually runs for each shape and row count (nsys / ncu kernel
names carry the tile, split-K and reduction scheme; the result depends on
the cuBLAS version and workspace, so read what main's D step runs, not what
a harness picks), then use the same MMA family (`tcgen05.mma kind::f16` and
`mma.sync` need not round alike), the same K chunking, the same split-K
partition and reduction order, and the same epilogue rounding. Test bit
equality over many random inputs and every row count first; it is cheap.

Deterministic always: same input, same bits, run to run and rank to rank.
No float atomics whose order varies.

## Floors

Every D GEMM at 1-64 rows is bound by reading its weight from HBM (bf16
weights reach the tensor ridge near 200 rows). `harness/d_gemm_floor.cu`
(`N K [m...]`) times kern's cuBLAS call, every cuBLASLt heuristic candidate
(f32 and bf16 out) and plain reads of the same weight bytes (grid-stride
loads and `cp.async.bulk`), the weight rotating over 8 copies so nothing
hits L2. One GPU, alone (2026-10-11), us per call: kern's cuBLAS call at
1 / 8 / 24 / 48 / 64 rows, cuBLASLt's best candidate at 24 rows, and the
best plain read of the weight:

| op | N x K | cuBLAS 1 / 8 / 24 / 48 / 64 | Lt best @24 | read floor | gap @24 x calls |
|---|---|---|---|---|---|
| `front` | 6016 x 7168 | 16.8 / 16.5 / 17.0 / 17.7 / 15.7 | 16.9 | 13.6 | 0.32 ms |
| `qkvg` | 6400 x 7168 | 17.2 / 17.2 / 17.8 / 18.2 / 16.7 | 17.7 | 14.4 | 0.23 ms |
| `wfu` | 3648 x 7168 | 11.4 / 11.6 / 11.9 / 12.1 / 11.7 | 11.2 | 9.0 | 0.07 ms |
| `q_b` | 18432 x 1536 | 10.3 / 10.1 / 10.3 / 10.5 / 10.5 | 9.8 | 9.5 | 0.02 ms |
| `o_proj` | 7168 x 1536 | 5.2 / 5.2 / 5.9 / 6.1 / 5.6 | 5.2 | 5.0 | 0.09 ms |
| `sh_down` | 7168 x 768 | 3.0 / 2.8 / 3.6 / 3.6 / 3.1 | 2.9 | 2.2 | 0.12 ms |
| `lat_up` | 7168 x 3584 | 9.7 / 9.7 / 9.8 / 9.9 / 10.1 | 9.8 | 8.9 | 0.09 ms |
| `lm_head` | 20480 x 7168 | 48.9 / 50.7 / 45.8 / 46.3 / 46.6 | 45.2 | 41.4 | 0.004 ms |

The f32 calls include cuBLAS's split-K reduce. Alone, the dense GEMMs have
~1 ms a step to give at 24 rows (~5% of the 93-layer step), most of it in
`front` and `qkvg`; the read floor at these sizes is 6.2-6.3 TB/s, not 8.
The rest of the case is the step itself and fusion.

Inside the step the same GEMMs measured slower under nsys (`front` 19.5 us
plus a separate ~2.9 us `splitKreduce_kernel`), and the all-reduces' idle
blocks already prefetch the next weight into L2. Why the step is slower than
the harness is not known yet; the `g-floor` run owns that question. Use the
step (`loop/d/bench`) as the judge, the harness to iterate.

## Shapes

Per D step of 93 layers (TP8 rank, from `scripts/gen_stage.py`'s D
manifest), rows 8-48 weighted by `loop/score`:

| op | N x K | out | calls | weight | in-step (nsys, 24 / 48 rows) |
|---|---|---|---|---|---|
| `front` (router 896, shared gate/up 1536, latent down 3584) | 6016 x 7168 | f32, read by `moe_front` | 92 | 86 MB | 4.43 / 3.98 TB/s + reduce |
| `qkvg` | 6400 x 7168 | f32 | 69 | 92 MB | 4.93 / 4.58 + reduce |
| `wfu` | 3648 x 7168 | f32 | 24 | 52 MB | 4.09 / 3.70 + reduce |
| `q_b` | 18432 x 1536 | f32 | 24 | 57 MB | 5.14 / 4.92 |
| `o_proj` | 7168 x 1536 | bf16 | 93 | 22 MB | 3.82 / 3.63 |
| `sh_down` | 7168 x 768 | bf16 | 92 | 11 MB | 2.77 / 2.71 |
| `lat_up` | 7168 x 3584 | bf16 | 92 | 51 MB | 5.32 / 5.24 |
| `lm_head` | 20480 x 7168 | f32 | 1 | 294 MB | 6.27 / 5.95 |
| `wgu`, `w_dn` (layer 0's dense MLP) | 8448 x 7168, 7168 x 4224 | bf16 | 1 each | 121 / 61 MB | |
| `moe_fc1`, `moe_fc2` (mxfp8 x mxfp4, 112 local experts) | expert 17.5 MB | | 92 each | 36.5 / 55.5 GB a step at 24 / 48 rows (bench routing) | 6.2-7.0 TB/s |

Real text routes wider than the bench's tokens: 28.8 / 42.7 active experts
a member at 24 / 48 rows, the busiest member 1.2x the mean
([`../../../notes/d-mlp.md`](../../../notes/d-mlp.md)). The expert-parallel
skew itself is layout, not a kernel target.

## Writing kernels

CUDA C++ with inline PTX where it pays (`cp.async.bulk` / TMA, `mbarrier`,
`tcgen05.*`, `mma.sync`, `griddepcontrol`), built by `loop/build` with the
pinned nvcc. The step already runs with PDL: our kernels trigger their
dependents at entry and cuBLAS(Lt) launches as a programmatic dependent, so
PDL by itself is not a gain over cuBLAS. References: `$K3_REF/flashinfer`
(`csrc/tinygemm2_sm100.cu`: mma.sync, TMA ring, 16 x 8 tiles, no split-K;
`include/flashinfer/gemm/tgv_gemm.cuh`: tcgen05 1-SM, 64/128 x 8-64 tiles,
no split-K; both bf16 out only), `$K3_REF/trtllm`, `$K3_REF/DeepGEMM`,
`$K3_REF/sglang`, `$K3_REF/vllm`, the KernelWiki skill.

## GPUs

`loop/d/check` and `loop/d/bench` lease 8 GPUs over two hosts from your
pool, as in round 1. Harness work (one GPU) goes to the harness pool named
in your `ROUND.md`, never the TP8 pool, so single-GPU leases cannot starve
the TP8 runs: `WITHGPU_HOSTS="<harness pool>" loop/withgpu -n 1 CMD`.
Several runs share both pools: batch changes before a TP8 measurement.

## Ownership

Each run owns its ops' emission in `gen_stage.py` and its kernels; touching
another run's op goes through the orchestrator (a line in your notes). Two
runs (`g-front`, `g-front-mma`) take the same shapes with different designs
on purpose; the orchestrator merges the better one.
