"""Exact upstream-Cake poison regression. Only an isolated GPU component test.
FP32 sanity-reference tolerance is upstream FI3637083a q1/wide test's .01/.01.
Stronger patch contract: every active partial O, max and sum is bitwise equal
between pristine zero-tail, repaired zero-tail, and repaired arbitrary-tail.
"""

import argparse
import ctypes as C
import gc
import hashlib
import json
import math
import pathlib
import statistics
import time

import torch

P = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--empty", action="store_true")
a = ap.parse_args()
outpath = pathlib.Path(a.out)
assert not outpath.exists()
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(
    4 * 2**30 / torch.cuda.get_device_properties(0).total_memory
)
torch.backends.cuda.matmul.allow_tf32 = False
L = C.CDLL(str(P / "launch.so"))
L.setup.argtypes = [C.c_int, C.c_char_p]
L.run.argtypes = [C.c_int] + [C.c_uint64] * 8 + [C.c_int] * 4 + [C.c_uint64]
torch.empty(1, device="cuda")
for i, n in enumerate(["original", "candidate"]):
    assert L.setup(i, str(P / (n + ".cubin")).encode()) == 0
records = []
timings = []
start = time.time()


def sha(t):
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def emit(x):
    records.append(x)
    print(json.dumps(x), flush=True)
    outpath.write_text(
        json.dumps(
            {"status": "in_progress", "records": records, "timings": timings}, indent=2
        )
    )


def guarded(shape, dtype):
    n = math.prod(shape)
    base = torch.full((n + 64,), -113, device="cuda", dtype=dtype)
    return base, base[32 : 32 + n].view(shape)


def canary(t):
    assert (t[:32] == -113).all() and (t[-32:] == -113).all()


class Case:
    def __init__(self, lengths, split, seed=645001):
        self.lengths = list(lengths)
        self.B = len(lengths)
        self.split = split
        self.width = (max(lengths) + 127) // 64 + 1
        self.pool = self.B * self.width + 8
        self.gen = torch.Generator(device="cuda").manual_seed(seed)
        self.q = (
            (torch.randn((self.B * 96, 576), generator=self.gen, device="cuda") * 0.5)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        self.base = (
            (torch.randn((self.pool, 64, 576), generator=self.gen, device="cuda") * 0.5)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        self.kv = self.base.clone()
        self.pt = (
            torch.randperm(self.pool, generator=self.gen, device="cuda")[
                : self.B * self.width
            ]
            .view(self.B, self.width)
            .to(torch.int32)
        )
        self.lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        self.cu = torch.arange(self.B + 1, device="cuda", dtype=torch.int32)
        self.ob, self.o = guarded((self.B, 96, split, 512), torch.bfloat16)
        self.mb, self.m = guarded((self.B, 96, split), torch.float32)
        self.sb, self.s = guarded((self.B, 96, split), torch.float32)
        self.graph = None

    def prepare(self, tail=0, single=None):
        self.kv.view(torch.uint8).copy_(self.base.view(torch.uint8))
        for b, n in enumerate(self.lengths):
            pages = self.pt[b].long()
            valid = (torch.arange(self.width * 64, device="cuda") < n).view(
                self.width, 64
            )
            selected = self.kv.view(torch.uint8)[pages].clone()
            selected[~valid] = tail
            if single is not None:
                selected[~valid] = 0
                row = n
                selected[row // 64, row % 64, single] = 127
            self.kv.view(torch.uint8)[pages] = selected

    def inputs(self):
        return [self.q, self.kv, self.lens, self.cu, self.pt]

    def launch(self, which):
        rc = L.run(
            which,
            *[
                x.data_ptr()
                for x in [
                    self.q,
                    self.kv,
                    self.o,
                    self.m,
                    self.s,
                    self.lens,
                    self.cu,
                    self.pt,
                ]
            ],
            self.pool * 64,
            self.width,
            self.split,
            self.B,
            torch.cuda.current_stream().cuda_stream,
        )
        assert rc == 0, rc

    def active(self):
        # Independently derive nonempty splits from logical lengths and 128-token
        # tiles. Inactive output storage may stay untouched under the kernel ABI.
        lens = torch.tensor(self.lengths, device="cuda")
        tiles = (lens + 127) // 128
        per = (tiles + self.split - 1) // self.split
        return (
            torch.arange(self.split, device="cuda")[None, :] * per[:, None]
            < tiles[:, None]
        )[:, None, :].expand(-1, 96, -1)

    def run(self, which, graph=False):
        self.o.fill_(float("nan"))
        self.m.fill_(float("nan"))
        self.s.fill_(float("nan"))
        if graph:
            if self.graph is None:
                self.launch(which)
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with (
                    torch.cuda.stream(stream),
                    torch.cuda.graph(self.graph, stream=stream),
                ):
                    self.launch(which)
                torch.cuda.synchronize()
            self.graph.replay()
        else:
            self.launch(which)
        torch.cuda.synchronize()
        for x in [self.ob, self.mb, self.sb]:
            canary(x)
        act = self.active()
        assert torch.equal(self.s > 0, act)
        assert torch.isfinite(self.s).all()
        assert not torch.isnan(self.m).any()
        return {"o": self.o[act].clone(), "m": self.m.clone(), "s": self.s.clone()}

    def merged(self):
        act = self.active()
        lse = torch.where(act, self.m + torch.log2(self.s), -torch.inf)
        top = lse.max(dim=2, keepdim=True).values
        w = torch.where(act, torch.exp2(lse - top), 0)
        v = torch.where(act[:, :, :, None], self.o.float(), 0)
        den = w.sum(2)[:, :, None]
        return torch.where(den > 0, (v * w[:, :, :, None]).sum(2) / den, 0).to(
            torch.bfloat16
        )

    def reference(self):
        # Reconstruct logical K/V rows through the randomized physical page table.
        # This FP32 sanity check does not reuse the split or masking logic;
        # its tolerance alone is not a numerical-quality acceptance criterion.
        refs = []
        for b, n in enumerate(self.lengths):
            if n == 0:
                refs.append(torch.zeros((96, 512), device="cuda", dtype=torch.bfloat16))
                continue
            v = (
                self.kv.view(torch.uint8)[self.pt[b, : ((n + 63) // 64)].long()]
                .view(torch.float8_e4m3fn)
                .reshape(-1, 576)[:n]
                .float()
            )
            q = self.q[b * 96 : (b + 1) * 96].float()
            probs = torch.softmax(q @ v.T / math.sqrt(192), -1)
            refs.append((probs @ v[:, :512]).to(torch.bfloat16))
        return torch.stack(refs)


def equal(x, y):
    return all(torch.equal(x[k].view(torch.uint8), y[k].view(torch.uint8)) for k in x)


def checkcase(lengths, split, graph=False, extra=False):
    c = Case(lengths, split)
    c.prepare()
    inputs = [sha(t) for t in c.inputs()]
    ref = c.run(0)
    oldmerged = c.merged()
    fpref = c.reference()
    torch.testing.assert_close(oldmerged.float(), fpref.float(), atol=0.01, rtol=0.01)
    new = c.run(1, graph)
    assert equal(ref, new), "clean candidate differs"
    assert inputs == [sha(t) for t in c.inputs()]
    emit(
        {
            "kind": "clean_reference",
            "lengths": lengths,
            "split": split,
            "graph": graph,
            "bitwise_partials": True,
            "reference_atol": 0.01,
            "reference_rtol": 0.01,
            "max_abs_reference": float((oldmerged.float() - fpref.float()).abs().max()),
            "output_sha256": sha(new["o"]),
        }
    )
    patterns = [
        ("zero", 0, None),
        ("nan_positive", 127, None),
        ("nan_negative", 255, None),
        ("max_positive", 126, None),
        ("max_negative", 254, None),
    ]
    if extra:
        patterns += [
            ("single_" + str(i), 0, i)
            for i in [0, 127, 128, 255, 256, 383, 384, 511, 512, 575]
        ]
    for label, tail, single in patterns:
        c.prepare(tail, single)
        before = [sha(t) for t in c.inputs()]
        got = c.run(1, graph)
        if not equal(ref, got):
            torch.save(
                {
                    "q": c.q,
                    "kv": c.kv,
                    "pt": c.pt,
                    "lens": c.lens,
                    "expected": ref,
                    "actual": got,
                },
                outpath.with_suffix(".failure.pt"),
            )
            raise AssertionError((lengths, split, label, "tail changed valid result"))
        assert before == [sha(t) for t in c.inputs()]
        assert all(torch.isfinite(x).all() for x in [got["o"], got["s"]])
        emit(
            {
                "kind": "tail_invariance",
                "lengths": lengths,
                "split": split,
                "graph": graph,
                "pattern": label,
                "bitwise": True,
                "inputs_unchanged": True,
                "output_canaries": True,
            }
        )
    if any(n % 64 for n in lengths):
        c.prepare(127)
        old = c.run(0)
        bad = int((~torch.isfinite(old["o"])).sum())
        assert bad > 0, "original poison not discriminating"
        emit(
            {
                "kind": "original_poison_failure",
                "lengths": lengths,
                "split": split,
                "nonfinite_active": bad,
            }
        )
    if graph:
        # Reuse same physical pages and graph with changed lengths, page ordering and contents.
        c.lengths = [max(1, n - 17) for n in reversed(lengths)]
        c.lens.copy_(torch.tensor(c.lengths, device="cuda", dtype=torch.int32))
        c.pt.copy_(c.pt.flip(1))
        c.base.view(torch.uint8).copy_(
            (torch.randn(c.base.shape, generator=c.gen, device="cuda") * 0.5)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        c.prepare()
        r = c.run(0)
        rr = c.reference()
        torch.testing.assert_close(c.merged().float(), rr.float(), atol=0.01, rtol=0.01)
        c.prepare(255)
        v = c.run(1, True)
        assert equal(r, v)
        emit(
            {
                "kind": "reused_page_changed_graph_metadata",
                "lengths": c.lengths,
                "split": split,
                "bitwise": True,
                "reference_atol": 0.01,
                "reference_rtol": 0.01,
            }
        )
    del c
    gc.collect()
    torch.cuda.empty_cache()


def timing(lengths, split):
    c = Case(lengths, split)
    c.prepare()
    graphs = []
    for which in [0, 1]:
        c.launch(which)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            c.launch(which)
        torch.cuda.synchronize()
        graphs.append(g)
    samples = {0: [], 1: []}
    for repeat in range(5):
        for which in [0, 1, 1, 0]:
            for _ in range(3):
                graphs[which].replay()
            st, en = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            st.record()
            for _ in range(20):
                graphs[which].replay()
            en.record()
            en.synchronize()
            samples[which].append(st.elapsed_time(en) * 1000 / 20)
    rec = {
        "lengths": lengths,
        "split": split,
        "order": "ABBA x5",
        "replays_per_sample": 20,
        "units": "us_per_launch",
        "samples": samples,
        "median_original": statistics.median(samples[0]),
        "median_candidate": statistics.median(samples[1]),
    }
    timings.append(rec)
    emit({"kind": "timing", **rec})


try:
    if a.empty:
        for lengths in [[0], [0, 1, 65], [32769, 0, 64]]:
            for split in [1, 74]:
                checkcase(lengths, split, graph=lengths == [0, 1, 65])
    elif a.smoke:
        checkcase([32769], 74, extra=True)
    else:
        for lengths in [
            [1],
            [63],
            [64],
            [65],
            [127],
            [128],
            [129],
            [32768],
            [32769],
            [32800],
            [32801],
            [1, 63, 64],
            [65, 127, 129],
            [32769, 32800, 32801],
            [1, 64, 65, 128],
        ]:
            for split in [1, 74]:
                checkcase(lengths, split, extra=lengths == [32769])
        checkcase([32769, 32800, 32801], 74, graph=True)
        for lengths, split in [
            ([32768], 74),
            ([32769], 74),
            ([32769] * 8, 9),
            ([32769] * 24, 3),
        ]:
            timing(lengths, split)
    result = {
        "status": "passed",
        "elapsed_seconds": time.time() - start,
        "records": records,
        "timings": timings,
        "torch_peak_allocated": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved": torch.cuda.max_memory_reserved(),
        "reference": "FI3637083a tests/mla/test_cake_kimi_k3_mla.py q1/wide replay tolerance, float32 attention using actual scale1/sqrt192",
        "scope": "operator only; no full-model or serving claim",
    }
except Exception as e:
    result = {
        "status": "failed",
        "exception": repr(e),
        "elapsed_seconds": time.time() - start,
        "records": records,
        "timings": timings,
    }
    outpath.write_text(json.dumps(result, indent=2))
    raise
outpath.write_text(json.dumps(result, indent=2))
print(
    json.dumps({k: v for k, v in result.items() if k not in ["records", "timings"]}),
    flush=True,
)
