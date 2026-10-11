# Round 1 end to end: 1P2D AgentX (2026-10-11, GB300)

Round 1's kernels (main 6741274) served through kern-serve, against the same deployment with the kernels
before the round. Only the manifests and kernels change: same kern-serve binary, same hosts layout, same
workload.

- P: full K3 PP8 at 262144 context, 8192-token chunks, 16 sequences a chunk, 128 GiB host tier a stage.
  Generated with `scripts/gen_stage.py --layers 93 --ranks 1 --chunk 8192 --experts 896 --state-per-layer
  --pack 16 --max-ctx 262144`, cut every 12 layers: 16.7-17.9 launches/layer (before: 29.1-30.2).
- D: two tiers, each the 93-layer TP8/EP8/DCP8 step (`--dcp --experts 896 --state-per-layer --max-ctx 327680
  --seqs 64 --peer-ar`): 13.0 launches/layer (before: 25.0).
- Workload: aiperf `inferencex-agentx-mvp`, 256k, seed 20260925, burst phase starts, 900 s a point.
  SLO: TTFT p90 < 5 s, ITL p99 <= 33.3 ms.

## Throughput and latency

| | c64 before | c64 round 1 | c96 round 1 |
|---|---|---|---|
| QPS | 1.86 | 2.03 | 2.68 |
| input tok/s | 166.5k | 183.3k | 223.0k |
| output tok/s a GPU | 92.6 | 103.5 | 134.7 |
| cache hit / dataset bound | 92.83% / 94.70% | 92.79% / 94.63% | 89.49% / 93.82% |
| TTFT p50 / p90 / p99 (s) | 1.87 / 14.7 / 48.2 | 1.93 / 13.8 / 40.5 | 7.10 / 29.7 / 63.0 |
| TTFT p90 past the first 120 s (s) | 3.00 | 2.97 | 18.3 |
| ITL p50 / p99 (ms) | 30.8 / 38.3 | 23.4 / 30.0 | 28.0 / 34.2 |
| goodput | 0.473 | 0.847 | 0.376 |
| D step p50 / p99 (ms) | 27.2 / 38.5 | 22.0 / 29.9 | 26.4 / 33.8 |
| D steps over 33 ms | 22.7% | 0.04% | ~4% |

D's step by rows (median, c64): 1-8 rows 19.0 -> 13.9 ms, 17-24 27.1 -> 22.7, 33-40 36.2 -> 28.9,
41-48 37.4 -> 29.7.

- c64: ITL and ICL now meet the SLO. TTFT p90 misses it only through the burst at the start of the
  profiling phase (64 sessions at once, 16 of them cold first turns of ~110k tokens); past the first 120 s
  p90 is 3.0 s with or without the round's kernels. The scenario defines the burst, so it stays.
- c96: P saturates first (24k computed tokens/s, queue p50 12), the cache hit drops 3.3 points (resident
  hit 0.87 -> 0.77, host evictions doubled), and D's 55-row steps (p99) reach 33.8 ms. Capacity of this
  1P2D lies between c64 and c96.

## Numerics (lm-eval 0.4.13 over the chat endpoint)

- gsm8k, 5-shot as chat turns, greedy: 0.9682 flexible / 0.9689 strict on a cold cache; again with 99.1% of
  prompt tokens served from the cache, 0.9689 / 0.9697. 21 answers flip between the two runs, in both
  directions (net +1). 21% of the outputs equal gsm8k's reference solutions verbatim, so the score cannot
  separate small numeric differences.
- RULER, 15 samples a task: 16k niah_single_2 / niah_multikey_2 / vt 1.00, qa_squad 0.89; 64k 1.00 /
  1.00 / 1.00, qa_squad 0.96. 128k was not run (the deployment went away mid-generation).

## Where P's time goes at c96

From the stages' `pp item` marks over the profiling window: a full stage computes ~27.7 ms per 1k rows, so
P could do ~36k rows/s and does 24.3k (~67%). Stage 0 is 95% busy, about a third of it waiting for stage 1
to take the previous item; stage 7 (9 layers) waits 43% of the time for upstream. The losses:

1. One arrival buffer a stage with unequal items: replaying the longest busy run, measured 1.15x of
   unbounded buffering; a model with two buffers gives 1.02x.
2. Items not full: 63% of stage 0's items are 8192 rows (68% while 5+ requests wait). Admission sits at the
   16-sequences-a-chunk bound, each request is cut at its snapshot and its 10-row generation prompt, and in
   38% of the queued windows the sequence-state slots are full (130 of 131, mostly cached prefixes). The
   logs do not say which of these holds a request back.
3. No work: 2-5%.
