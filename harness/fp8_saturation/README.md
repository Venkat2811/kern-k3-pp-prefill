# FP8 overflow regression

Requires an owned CUDA device, CUDA 13.0 toolchain, C++ compiler and PyTorch. The test caps its allocator at 4 GiB and child runtime at 300 seconds. Select the device with `CUDA_VISIBLE_DEVICES`; no model or server is used. Output directories must be new.

Build the two actual kernels from baseline 3c7bae2 and the candidate with their published commands, naming the resulting cubins `absorb.cubin` and `prep.cubin` in separate directories:

```sh
nvcc -cubin -arch=sm_103a -DQ_FP8=1 -o BUILD/absorb.cubin SOURCE/source/k3_mla_absorb_mma.cu
nvcc -cubin -arch=sm_103a -DINNER=1536 -DKV_FP8=1 -DMLA_FUSED=3648 -o BUILD/prep.cubin SOURCE/source/k3_mla_prep.cu
```

Run the identical test with pristine or fixed candidates. The first command must exit nonzero because finite overflow remains NaN; the second must pass:

```sh
CUDA_VISIBLE_DEVICES=0 python harness/fp8_saturation/run.py --baseline BASELINE_BUILD --candidate BASELINE_BUILD --out /tmp/fp8-before
CUDA_VISIBLE_DEVICES=0 python harness/fp8_saturation/run.py --baseline BASELINE_BUILD --candidate CANDIDATE_BUILD --out /tmp/fp8-after
```

All 65,536 BF16 bit patterns pass through each actual kernel's RoPE conversion, followed by finite overflow in absorbed query/cache latent and normal randomized B24 input equivalence. In-range finite outputs must remain bitwise equal; finite overflow and infinities must become signed 448; existing NaNs must remain NaN (payload identity is not promised). The test reports the first BF16 NOSAT NaN threshold instead of assuming every value above 448 overflows. ABBA timing samples are retained separately from correctness.

This intentionally changes overflow policy from the original Torch-cast parity. It does not repair arbitrary stale page tails, establish full-model quality or claim serving throughput.
