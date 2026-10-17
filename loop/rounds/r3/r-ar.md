Run `r-ar` · **target: D's all-reduces and residual rows (`k3_ar_fused`: `ar_attnres_rms`, `ar_finalize_rms`, `land_add2_attnres_rms(_snap)`, `embed_rms`), and the head's exchange (`k3_head`), each at its floor.** Notes go in `notes/r-ar.md`.

Read [`loop/rounds/r3/ROOF.md`](ROOF.md) first: done-means, references, numerics, GPUs.

Where they stand (`roofline/decode.md`, "All-reduce and the residual rows", rank 0, µs per call at 1 / 8 / 32 / 64 rows × 8k):

| call | n | measured | floor (protocol 3.9 + bytes) |
|---|---:|---|---|
| `ar_finalize_rms` (MoE partials, 10752 wide) | 92 | 15.5 / 21.5 / 27.5 / 33.6 | ~6 |
| `ar_attnres_rms` (o_proj partials, 7168) | 93 | 12.5 / 12.5 / 14.6 / 20.0 | ~6 |
| `land_add2_attnres_rms_snap` (no exchange) | 91 | 7.1 / 6.6 / 6.8 / 7.3 | 0.02-1.6 |
| `k3_head` argmax + logits gather | 1 | 11.7 / 15.1 / 35.2 / 64.0 | 6.0 / 6.0 / 25 / 51 |

What the doc already measured, to start from:

1. The attention-residual row routine costs 4.3 + 0.55 × snapshots µs flat in rows: one 1024-thread block a row walks a chain of dependent steps (score each candidate's RMS, softmax, mix, norm). It sits in `ar_attnres_rms` and in every closing add. Shorten the chain: score all candidates in one pass with one block reduction, more rows a block at small batches, the snapshot reads issued at entry. SGLang's `kimi_k3/attn_res/fused_tma.cuh` does the same math with TMA: read it.
2. The one/two-shot switch is at 33 rows; the MoE message's two-shot wins from 12-24 rows (round 2's `agent/x-mcast` harness: at 10752 wide, one-shot 8.13 / 9.16 / 10.77 / 12.56 µs at 8 / 16 / 24 / 32 rows against two-shot ~8.3-8.6). Move it, measured in the step.
3. The fastest rank waits in `ar_finalize_rms` for the slowest one's MoE (EP skew, up to 28 µs a launch). That wait is not this kernel's; separate it in your table (the last-arriving rank's time is the kernel's own).
4. Compare the protocol with FlashInfer's / TRT-LLM's allreduce fusion (Lamport one-shot and two-shot, MNNVL) and SGLang's `kimi_k3/comm/ar_fusion.cuh`: anything that gets an 8-rank exchange across two trays under 3.9 µs, or overlaps it with the epilogue, moves the floor.
5. Multicast (`multimem.st`) needs the runtime to create the multicast object (round 2 did it in `k3_step` only): measure it and write it as a runtime proposal; do not ship a manifest that needs it.

Check: `loop/d/check loop/out/d-l4-tp8.json` three times for any protocol change. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
