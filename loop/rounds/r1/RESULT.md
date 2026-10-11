# Round 1 result (2026-10-10 17:50 to 2026-10-11 01:50 UTC)

Four lanes (p-kda, p-glue, d-attn, d-mlp), each closed on its own notes. Main
3ad4ec9 at the start, ec11125 at the end. Every merge was checked: P against
the stage reference (bit-identical or within the bf16 floor, see
[`notes/error-budget.md`](../../../notes/error-budget.md)), D teacher-forced
against the oracle three times.

| | start | end |
|---|---|---|
| D, 16 layers TP8 (`loop/d/bench`) | 24.7 launches/layer, 4.562 ms/step | 12.9 launches/layer, 3.486 ms/step |
| D, 93 layers at 128k, 24 / 48 rows | 23.88 / 30.87 ms/step | ~19.2 / ~25.8 ms/step |
| P, stage 0 of PP8 (`loop/p/ab`) | 26.3 launches/layer, 132.0 ms/item | 16.7 launches/layer, ~122-124 ms/item |

P's stage is power-capped (1400 W): a non-GEMM kernel's saving shows in its
own in-program time more than in the item's.

## Where the remaining time is

D (93 layers, 128k, 24 rows, ~19.1 ms a step):

- every non-GEMM launch sits between GEMMs or the prebuilt attention and at
  its measured bound: the all-reduces at the fabric, moe_front and the
  residual rows at their reduction latency, kda_core at HBM;
- expert-parallel skew, ~1.5 ms: chance within a step, recovered by expert
  replicas chosen per step (4 a member a layer: 0.6-0.9 ms; 8: 1.2-1.4 ms,
  13 GB more weight a member). A layout change, not a kernel one;
- dense GEMMs at 4.3-5.6 TB/s, ~1.2 ms against ~6.5 TB/s (GEMMs were out of
  scope);
- MLA decode (the prebuilt DSL kernel), within ~13% of its layout's MMA
  instruction bound; a rewrite gains ~0.1-0.2 ms;
- the head's logits gather, 29 / 51 us at 24 / 48 rows (a runtime change:
  return token ids only).

P (stage 0): every glue launch sits between two GEMMs or the prebuilt FMHA /
FlashKDA; the glue kernels are within ~1.35x of the copy rate (floor table in
[`notes/p-glue.md`](../../../notes/p-glue.md)). The FMHA at 8192 rows is
MMA-bound at the power cap. What is left is the GEMMs.
