Run `r-attn` · **target: D's attention kernels at their floor: `k3_kda_core`, MLA (`k3_mla_prep`, `k3_mla_absorb_mma`, `mla_decode_h96_p64` and its grid) and the DCP exchange (`k3_dcp`).** Notes go in `notes/r-attn.md`.

Read [`loop/rounds/r3/ROOF.md`](ROOF.md) first: done-means, references, numerics, GPUs.

Where they stand (`roofline/decode.md`, "KDA", "MLA attention and the DCP exchange"), µs per call at 1 / 8 / 32 / 64 rows × 8k:

| call | n | measured | floor |
|---|---:|---|---|
| `kda_core` | 69 | 9.5 / 9.6 / 12.2 / 22.8 | 0.29 / 1.87 / 7.29 / 14.52 |
| `mla_attn` | 24 | 8.8 / 10.5 / 27.2 / 48.1 | 0.42 / 2.48 / 7.36 / 12.99 |
| `mla_absorb` | 24 | 6.5 / 6.3 / 7.9 / 10.1 | 3.0-3.7× |
| `dcp_exchange` | 24 | 12.0 / 12.0 / 14.2 / 24.7 | 2.0-3.1× |
| `mla_prep` | 24 | 3.1-3.5 | launch latency |

Measured levers, to start from:

1. MLA attention's grid always has 16 split slots a row; at 32-64 rows 960 of 1024 slot clusters launch and exit. Slots = the plan's maximum (16 at 1 row, 8 at 8, 2 at 32-64) measured 48.0 → 24.4 µs at 64 rows (266 / 566 / 179 µs a step at 32 × 8k / 64 × 8k / 32 × 32k). The launch grammar has no `min`; one launch a row bucket under `when` gives it. After that it is 1.6-1.9×: ~7 µs fixed, 96 heads padded to 128 MMA rows. Compare with FlashInfer's MLA decode paths and FlashMLA for a 96-head tiling.
2. `kda_core` at 1-8 rows is a chain of serial loads before the state (line index → conv window → partial → decay tile → state in two passes): issue the state loads at entry. At 32-64 rows it is waves (128 registers, 4 blocks an SM, 768 blocks = a wave and a 160-block tail): a 64-register build or a persistent grid. SGLang's `kda_fused_decode.py` / `kda_packed_decode.py` / `cutedsl_kda.py` are the references.
3. `k3_mla_absorb_mma` reads exactly its 12.6 MB in one wave plus a serial tail (106-163 µs a step of headroom); `k3_mla_prep` takes the generic path at TP8 (3 blocks in y, the fast path needs 4).
4. `k3_dcp` is 1.2-1.35× a same-size one-shot at 32-64 rows and 1.7-2× at 1-8: the split merge before the push and the merge + W_UV after the poll. FlashInfer's `cake_fused_norm_combine.py` is the reference combine.

Check: `loop/d/check loop/out/d-l4-tp8.json` (layer 3 is MLA). Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
