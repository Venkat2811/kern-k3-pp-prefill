# HGX B300 reproduction

This branch starts from companion revision
`0c152765e8cd5b3eb906138fcf61c81cb8921b2e` and preserves the changes exercised
on 148-SM, 1,100 W HGX B300 GPUs with CUDA 13.0.88. Use the matching
[kern runtime branch](https://github.com/Venkat2811/kern/tree/codex/b300-reproduction-2026-10-15),
based on `3eeec9044ba437e2818052a17895b8670a552cee`, and the kernel index from
`304c0d434836d0760f07d77e2b0e20fed8f402ec`.

The branch adds CPU-only manifest adapters and an opt-in KDA source override.
The default generator and pinned baseline cubins retain the author's recipe.
The B300 adaptations are explicit because the original persistent grids can
stall this hardware. They are not a general occupancy auto-tuner.

## Generate and adapt manifests

Follow the root README to build the matching runtime and the pinned kernels.
Use a clean Python environment if an unrelated installed `kernels` package
shadows kern's generator modules. For example, with sibling checkouts:

```sh
# From the matching kern checkout, make the pinned registry available.
git worktree add --detach ../kern-kernel-index 304c0d434836d0760f07d77e2b0e20fed8f402ec

# From this companion checkout, using CUDA 13.0.88:
NVCC=/usr/local/cuda/bin/nvcc python3 scripts/build.py
bash scripts/deploy.sh ../kern ../kern-kernel-index/kernel_index generated
python3 scripts/b300_manifests.py --kind prefill \
  --input generated/p --output generated/p-b300
python3 scripts/b300_manifests.py --kind decode \
  --input generated/d.json --output generated/d-b300.json
```

Choose fresh output paths. The adapter validates every input before writing,
leaves the inputs unchanged, and rejects unexpected launch geometry.

| Launch | Author grid | B300 grid | Measured occupancy contract |
|---|---:|---:|---|
|Large PP MoE router|304|296|512 threads, 64 registers, two CTAs/SM; whole-grid barrier|
|Fused decode attention-residual AR|152|144|1,024 threads, one CTA/SM; resident-grid synchronization|
|Fused decode final AR|152|144|1,024 threads, one CTA/SM; resident-grid synchronization|

The small 64-CTA route and 256-CTA DCP exchange remain unchanged. The decode
adapter accepts the qualified fused peer-AR manifest. The 144-CTA choice
retains the author's multiple-of-eight constraint. These changes only alter
manifest launch geometry; the baseline cubin bytes do not change.

The portable adapters were checked against all eight saved PP stage manifests
and the saved standalone decode manifest from the experiment: the parsed
outputs match exactly. The actual router passed six eager/changed-input graph
comparisons against another safe grid, and the decoder passed the eight-rank
collective and full-model gates described in the runtime guide.

For multiple processes on one HGX host, export the same private
`KERN_LOCAL_IPC_DIR` in every process; see the runtime guide. Use
`generated/p-b300/stageN.json` for PP and `generated/d-b300.json` for D.
Cross-host direct peer mapping is still unsupported by the local-IPC adapter.
The existing SGLang plugin remains a decode adapter, not an IB receiver for
the standalone PP8 state pools.

## Optional KDA52 experiment

The original single-sequence recurrence launch has 96 recurrence CTAs and
56 gate CTAs: 152 blocks on 148 SMs. Both kernels use 512 threads,
128 registers/thread and 182,272 bytes dynamic shared memory, permitting one
CTA/SM. Reducing gate CTAs to 52 fits one sequence in 148 blocks. Unlike the
router barrier, the original recurrence is progress-safe: recurrence CTAs
never wait for gate CTAs.

`source/k3_kda_rec.cu` now allows an explicit compile-time override while
keeping 56 as the default. The default rebuilt cubin was byte-identical to
the pinned baseline. Compile the candidate into the same kernel directory:

```sh
/usr/local/cuda/bin/nvcc -cubin -arch=sm_103a \
  -DK12_LD=49408 -DK12_GATE_CTAS=52 \
  source/k3_kda_rec.cu -o build/kda-gate52.cubin
python3 scripts/b300_kda_manifest.py \
  --input generated/p-b300/stage0.json \
  --cubin build/kda-gate52.cubin \
  --output generated/stage0-kda52.json
```

The selector accepts only the tested candidate SHA256
`f7d840e582d3c2b3bbba91a37f0a5fce0e0d3e1e7dfc02a2a1917b4e1a99a60c` and
baseline module SHA256
`663253f41bd8eaaa4d586ddcedf826d7b3d06defc77f3b95c4cf235a830e81ad`.
It changes only the recurrence module and `nseq*96+56` to `nseq*96+52`.
Its output matches the experiment's saved candidate manifest exactly.

Do not use the candidate manifest as the baseline. It was tested only in
stage 0, after all serving measurements had finished. No full-model serving
gain or downstream quality improvement was measured with it.

### Numerical gate

[diagnostic-state-hashes.patch](diagnostic-state-hashes.patch) is the exact
instrumentation used in a separate kern worktree, not a runtime change
enabled by this branch. Apply it to the original runtime base and build a
separate `kern` executable:

```sh
# From the matching kern checkout; adapt sibling paths if needed.
git worktree add --detach ../kern-b300-diagnostic 3eeec9044ba437e2818052a17895b8670a552cee
git -C ../kern-b300-diagnostic apply \
  ../kern-k3-pp-prefill/b300/diagnostic-state-hashes.patch
cargo build --locked --release --manifest-path ../kern-b300-diagnostic/Cargo.toml -p kern-run
```

For each seed 24301 and 24302, copy [stage0.toml](stage0.toml), set `samples`
to 12 and set the seed. Run reference, independent reference repeat, and
candidate with that same workload, model/tokenizer and `--kernels build`.
Set `KERN_BENCH_STATE_HASH_DIR` to a different fresh directory for each run:

```sh
KERN_BENCH_STATE_HASH_DIR=images-reference-24301 \
  ../kern-b300-diagnostic/target/release/kern bench \
  --manifest generated/p-b300/stage0.json --kernels build \
  --weights "$MODEL" --tokenizer "$MODEL/tokenizer.json" --gpu 0 \
  --workload gate-24301.toml --out gate-reference-24301.json
python3 b300/compare-gates.py images-reference-24301 images-reference-repeat-24301 \
  --label reference-repeat --out repeat-24301.json
python3 b300/compare-gates.py images-reference-24301 images-candidate-24301 \
  --out candidate-24301.json
```

The comparison commands run after all corresponding benchmark calls. Repeat
for the second seed, and verify that changing the seed changes all 13 images
for every shape. In the recorded experiment, all six candidate/reference and
six reference-repeat comparisons passed 13 finite bitwise-identical images
each. State storage and hidden rows are initialized; the benchmark builds the
requested real prefix before restoring the measured preimage. This is a
stage hidden/state check, not a full-model logit or answer-quality test.

### Timing and observed result

Use the uninstrumented runtime, unset `KERN_BENCH_STATE_HASH_DIR`, and run
reference → candidate → candidate → reference with [stage0.toml](stage0.toml)
(128 graph samples per shape, 32 per-call bracket samples). Keep the GPU
exclusive to these runs and retain every sample. Compare the arithmetic means
of the two graph medians per configuration; do not substitute bracket sums
for graph wall time.

| Stage-0 shape | Reference | Candidate | Graph time reduction |
|---|---:|---:|---:|
|Cold 8K|170.3423 ms|169.0506 ms|0.7583%|
|8K after 24K prefix|198.3555 ms|196.2616 ms|1.0556%|
|Four packed 2K sequences|168.4861 ms|168.4844 ms|No meaningful change|

This is a bounded ABBA session with two runs per configuration, not a
confidence interval or a serving-throughput claim. The two prefix pair
savings were 1.420 and 2.767 ms. The single-request `flash_kda` bracket sums
fell about 3.7 ms, but graph wall time fell only 1.29/2.09 ms.

## Serving result and remaining work

The baseline PP8 service ran all 93 layers and 896 experts. A cold diagnostic
used one output token, 2K–32K inputs, 60 seconds warmup plus 300 seconds
measurement, and a 5-second p90 TTFT criterion. Highest tested passing nominal
rates were kern 1.75 requests/s and the corrected SGLang recipe 1.5.
At identical 1.75 traffic, p90 was 3.778 versus 8.142 seconds; kern 1.875
failed at 6.414 seconds. The nominal-point difference is 16.7%, not an estimate
of precise maximum capacity. Topology, chunking and KV/state precision differ
between engines, and the corpus is a disclosed shape-matched diagnostic,
not the author's exact corpus or AgentX session replay.

Stage 0 was about 20% slower than the published 1,400 W GB300 shape study.
Hardware power/clock differences were not isolated causally. Broad quality
equivalence and the 50–100% system-throughput objective remain unestablished.
Full 1P2D AgentX reproduction still requires the cross-HGX model-state
pack/IB-transfer/scatter/visibility/acknowledgement implementation and an
end-to-end workload/SLA run. A successful small GPU RDMA transfer is not that
implementation.
