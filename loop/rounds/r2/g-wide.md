Run `g-wide` · **target: D's wide GEMMs, `q_b` (18432 x 1536, f32), `lm_head` (20480 x 7168, f32), layer 0's dense MLP `wgu` / `w_dn` (bf16), and MLA's `mla_absorb` (a batched GEMM written as a kernel, 25 MB at ~3 TB/s).** Notes go in `notes/g-wide.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

Alone: `q_b` 10.3 us at 24 rows against a 9.5 us read floor (x24 a step), `lm_head` 45.8 against 41.4 (x1, and its output feeds the head's argmax: a fused argmax epilogue would drop the f32 logits pass), `mla_absorb` 7.8 / 9.1 us at 24 / 48 rows for 25 MB. Directions: the head (lm_head + argmax, bit-identical tokens and logits where the check reads them), `mla_absorb` to its floor, then `q_b` (its input comes from `mla_prep`; its output feeds `mla_absorb`: fusing q_b's epilogue with the absorb is the larger idea if the data flow allows).

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
