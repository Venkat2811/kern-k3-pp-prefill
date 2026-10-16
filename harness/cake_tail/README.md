# Partial-page FP8 operand regression

Requires a free, owned SM103 CUDA device, CUDA 13.0 toolchain, C++ compiler and PyTorch. The test bounds its PyTorch allocator at 4 GiB and each child at 300 seconds. Select the device with `CUDA_VISIBLE_DEVICES`; it does not interact with a model or server. Output directories must not already exist.

PyTorch is a test-only dependency for constructing FP8 inputs, managing CUDA
allocations/graphs/events, and computing the FP32 sanity reference. The
attention operation itself is launched directly from the two cubins through
`launch.cpp`; it does not use a PyTorch attention implementation. No serving
dependency is added. This standalone harness is not part of `harness/run.sh`.

The recorded runs used B300, CUDA 13.0.88, Python 3.12 and
PyTorch 2.13.0+cu130. The full run also records ABBA×5 CUDA-event timings
with 20 graph replays per sample (10 samples per arm); these measure this
operator only. The 74/9/3 splits at batch 1/8/24 are the 148-SM B300 plan
proposed in PR #10. This PR's base generator uses 64 splits at batch 1;
the harness supplies its split counts explicitly.

The baseline is the unmodified published wide cubin from 3c7bae2:

```sh
git show 3c7bae2:prebuilt/cake_k3_mla_wide.cubin > /tmp/cake-original.cubin
CUDA_VISIBLE_DEVICES=0 python harness/cake_tail/run.py --baseline /tmp/cake-original.cubin --candidate prebuilt/cake_k3_mla_wide.cubin --out /tmp/cake-fixed
CUDA_VISIBLE_DEVICES=0 python harness/cake_tail/run.py --baseline /tmp/cake-original.cubin --candidate prebuilt/cake_k3_mla_wide.cubin --out /tmp/cake-empty --empty
```

A direct failing-before test passes the pristine cubin as the candidate. It must exit nonzero on poisoned-tail invariance:

```sh
CUDA_VISIBLE_DEVICES=0 python harness/cake_tail/run.py --baseline /tmp/cake-original.cubin --candidate /tmp/cake-original.cubin --out /tmp/cake-before --smoke
```

Require `results.json` to report `status: failed` and an `exception` containing
`tail changed valid result` for length 32769 / splits 74 / `nan_positive`.
An import, compiler or launch-setup error is not the expected regression.

The fixed candidate must pass the same smoke command. The full test adds lengths 1/63/64/65/127/128/129/32768/32769/32800/32801, ragged batches, graph replay with changed lengths/page tables/contents, reused physical pages, both FP8 NaN encodings, single-byte poison, maximal finite tail bytes, input immutability and output guards. Empty-member tests compare only active output partials; inactive outputs are not required to be finite. Independent active-split geometry checks max/sum metadata. Clean and poisoned candidate partials must equal pristine zero-tail partials bitwise. FP32 reference attention additionally uses FlashInfer 3637083a's q1/wide tolerance atol 0.01 / rtol 0.01 as a sanity check. This loose tolerance, particularly at long contexts with small outputs, is not a quality-equivalence criterion; the exact partial comparison is the repair's stronger invariant.

Recorded full and empty matrices total 37 clean/reference cases, 205
tail-invariance checks (168 actual poisoned-tail cases plus 37 zero-tail
controls), 29 discriminating pristine-poison cases and three changed-metadata
graph replays. The earlier matrices used execution scripts checked against
the public harness by source/AST comparison. On 16 October 2026, the public
full and `--empty` entry points were also executed verbatim on B300 with
CUDA 13.0.88 and passed. The public pristine-as-candidate smoke failed on
the expected poisoned-tail invariant, rather than a setup error.

Scope is the q1 wide ABI used here: `m_tiles=1`, `n_full_items=0`. Other query
lengths and row-tile programs are outside this harness. It compares the
candidate's maximal-finite tail against pristine zero-tail output, but does
not separately compare pristine maximal-positive-tail against pristine
zero-tail output. Separate follow-up checks passed nine such pristine
finite-tail controls and 10,000 poisoned graph replays each at B1/splits74
and B24/splits3, checking output bytes and canaries after every replay.
Those supplemental checks are not modes of this public harness and are
not included in its counts. Stress is not a formal memory-model proof;
[the source provenance](../../source/flashinfer-cake-k3-mla/PROVENANCE.md)
documents the existing completion and proxy-fence ordering argument.

Raw JSON, logs, input/cubin hashes and subprocess return codes are retained.
The component test does not establish full-model correctness, stock-model
equivalence or serving performance. A supplemental poisoned and reused
physical-page model check is outside this harness: on the repaired binary,
one batch-1 first decode step remained bitwise equal to a clean run of the
same binary after one unused NaN byte survived physical-page reuse. That
separate integration result is not a model-level before/after repair test.
