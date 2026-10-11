Run `g-small` · **target: D's bf16-out GEMMs with N=7168, `o_proj` (K 1536), `sh_down` (K 768), `lat_up` (K 3584), and fusing them with their neighbours.** Notes go in `notes/g-small.md`.

Read [`loop/rounds/r2/GEMM.md`](GEMM.md) first: scope, the bit-identity rule, floors, GPUs.

Alone they have ~0.3 ms a step to give at 24 rows (o_proj 0.09, sh_down 0.12, lat_up 0.09), mostly launch ramp: 11-22 MB reads of 3-6 us, where the read floor itself is 2.2-5.0 us. The larger lever is their neighbours:

- `o_proj` follows `kda_core` / `dcp_exchange` and feeds the attention all-reduce `ar_attnres_rms`;
- `sh_down` follows `moe_fc2` (its shared-expert input comes out of `moe_front`'s path) and feeds `ar_finalize_rms`;
- `lat_up` follows `ar_finalize_rms` and feeds the next layer's `land_add2_attnres_rms_snap`.

Directions: a GEMM kernel at the read floor for these shapes (bit-identical or proved), then fusion where it removes a launch or a pass: the GEMM epilogue writing straight into the all-reduce's push buffers, `sh_down` riding in `moe_fc2`'s launch, the norm / residual of the next kernel in `lat_up`'s epilogue. Note `o_proj` is a pinned cuBLASLt algorithm in the manifest: the heuristic's best at 24 rows is 5.2 us against 5.9.

Check: `loop/d/check loop/out/d-l4-tp8.json`. Bench: `loop/d/bench loop/out/d-l16-tp8.json`.
