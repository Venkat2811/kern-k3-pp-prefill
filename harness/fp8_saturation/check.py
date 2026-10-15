import argparse, ctypes as C, hashlib, json, math, pathlib, statistics, time
import torch

P = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
outpath = pathlib.Path(a.out)
assert not outpath.exists()
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(
    4 * 2**30 / torch.cuda.get_device_properties(0).total_memory
)
L = C.CDLL(str(P / "launch.so"))
L.setup.argtypes = [C.c_int, C.c_char_p, C.c_char_p]
L.absorb.argtypes = [C.c_int] + [C.c_uint64] * 3 + [C.c_int, C.c_uint64]
L.prep.argtypes = [C.c_int] + [C.c_uint64] * 7 + [C.c_int, C.c_uint64]
torch.empty(1, device="cuda")
for i, (name, entry) in enumerate(
    [
        ("absorb-original", "kern_k3_mla_absorb_mma"),
        ("absorb-candidate", "kern_k3_mla_absorb_mma"),
        ("prep-original", "kern_k3_mla_prep"),
        ("prep-candidate", "kern_k3_mla_prep"),
    ]
):
    assert L.setup(i, str(P / (name + ".cubin")).encode(), entry.encode()) == 0
records = []
start = time.time()


def emit(x):
    records.append(x)
    print(json.dumps(x), flush=True)


def same(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def check_conversion(before, old, new, kind):
    v = before.float()
    finite = torch.isfinite(v)
    inside = finite & (v.abs() <= 448)
    overflow = finite & (v.abs() > 448)
    nan = torch.isnan(v)
    inf = torch.isinf(v)
    assert torch.equal(
        old.contiguous().view(torch.uint8)[inside],
        new.contiguous().view(torch.uint8)[inside],
    )
    assert torch.isfinite(new.float()[finite | inf]).all()
    assert torch.isnan(new.float()[nan]).all()
    # Independent value-domain oracle: clamp before conversion. Check source
    # NaNs separately, without requiring a particular NaN payload.
    expected = v.clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(
        new.contiguous().view(torch.uint8)[~nan],
        expected.contiguous().view(torch.uint8)[~nan],
    )
    assert torch.isfinite(new.float()[overflow]).all()
    emit(
        {
            "kind": kind,
            "total": v.numel(),
            "in_range": int(inside.sum()),
            "finite_overflow": int(overflow.sum()),
            "infinity": int(inf.sum()),
            "nan": int(nan.sum()),
            "original_overflow_nan": int(torch.isnan(old.float()[overflow]).sum()),
            "first_positive_bf16_nan": float(
                v[finite & (v > 0) & torch.isnan(old.float())].min()
            ),
            "first_negative_bf16_nan_toward_zero": float(
                v[finite & (v < 0) & torch.isnan(old.float())].max()
            ),
            "in_range_bitwise": True,
            "finite_saturation_exact": True,
            "existing_nan_preserved": True,
        }
    )


def absorb_case(b, qp=None, w=None):
    if qp is None:
        qp = torch.randn((b, 96, 192), device="cuda") * 0.2
    if w is None:
        w = (torch.randn((96, 256, 512), device="cuda") * 0.1).to(torch.bfloat16)
    outs = [
        torch.empty((b, 96, 576), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(2)
    ]

    def call(i):
        assert (
            L.absorb(
                i,
                qp.data_ptr(),
                w.data_ptr(),
                outs[i].data_ptr(),
                b,
                torch.cuda.current_stream().cuda_stream,
            )
            == 0
        )

    return qp, w, outs, call


def prep_case(b, p=None, gk=None):
    if p is None:
        p = torch.randn((b, 3648), device="cuda") * 0.2
    gq = torch.ones(1536, device="cuda", dtype=torch.bfloat16)
    gk = torch.ones(512, device="cuda", dtype=torch.bfloat16) if gk is None else gk
    slots = torch.arange(b, device="cuda", dtype=torch.int64)
    os = [
        torch.empty(
            ((b + 63) // 64 * 64, 576), device="cuda", dtype=torch.float8_e4m3fn
        )
        for _ in range(2)
    ]
    qs = [torch.empty((b, 1536), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    gs = [torch.empty((b, 1536), device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def call(i):
        assert (
            L.prep(
                i + 2,
                *[x.data_ptr() for x in [p, gq, gk, slots, os[i], qs[i], gs[i]]],
                b,
                torch.cuda.current_stream().cuda_stream,
            )
            == 0
        )

    return p, gq, gk, slots, os, qs, gs, call


try:
    torch.manual_seed(615)
    # Every BF16 bit pattern through the real query kernel's rope conversion.
    bits = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.uint16)
        .view(torch.bfloat16)
    )
    q = torch.zeros((64, 96, 192), device="cuda")
    vals = bits.repeat(6).view(64, 96, 64)
    q[:, :, 128:] = vals.float()
    w = torch.zeros((96, 256, 512), device="cuda", dtype=torch.bfloat16)
    q, w, os, call = absorb_case(64, q, w)
    for i in [0, 1]:
        call(i)
    torch.cuda.synchronize()
    check_conversion(vals, os[0][:, :, 512:], os[1][:, :, 512:], "absorb_all_bf16_bits")
    # Every bit pattern through actual cache-append rope conversion.
    p = torch.ones((1024, 3648), device="cuda")
    p[:, 2048:2112] = bits.view(1024, 64).float()
    pc = prep_case(1024, p)
    for i in [0, 1]:
        pc[-1](i)
    torch.cuda.synchronize()
    check_conversion(
        bits.view(1024, 64),
        pc[4][0][:1024, 512:],
        pc[4][1][:1024, 512:],
        "prep_all_bf16_bits",
    )
    # Finite overflow in absorbed latent math, normalized cache latent and rope.
    for sign in [1, -1]:
        q = torch.zeros((3, 96, 192), device="cuda")
        q[:, :, 0] = 1
        q[:, :, 128:] = sign * 500
        w = torch.zeros((96, 256, 512), device="cuda", dtype=torch.bfloat16)
        w[:, 0, :] = sign * 500
        ac = absorb_case(3, q, w)
        pc = prep_case(
            3,
            torch.ones((3, 3648), device="cuda"),
            torch.full((512,), sign * 500, device="cuda", dtype=torch.bfloat16),
        )
        pc[0][:, 2048:2112] = sign * 500
        for i in [0, 1]:
            ac[-1](i)
            pc[-1](i)
        torch.cuda.synchronize()
        for label, old, new in [
            ("absorb", ac[2][0], ac[2][1]),
            ("prep", pc[4][0][:3], pc[4][1][:3]),
        ]:
            assert torch.isnan(old.float()).all()
            assert (new.float() == sign * 448).all()
            emit(
                {
                    "kind": "actual_kernel_overflow",
                    "kernel": label,
                    "sign": sign,
                    "original_all_nan": True,
                    "candidate_exact_signed448": True,
                }
            )
    # In-range regression and actual-kernel ABBA timings, normal tensor inputs.
    for kind, obj in [("absorb", absorb_case(24)), ("prep", prep_case(24))]:
        call = obj[-1]
        for i in [0, 1]:
            call(i)
        torch.cuda.synchronize()
        if kind == "absorb":
            assert same(obj[2][0], obj[2][1])
        else:
            for outputs in obj[4:7]:
                assert same(outputs[0][:24], outputs[1][:24])
        gs = []
        for i in [0, 1]:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
                call(i)
            torch.cuda.synchronize()
            gs.append(g)
        samples = {0: [], 1: []}
        for rep in range(5):
            for i in [0, 1, 1, 0]:
                for _ in range(3):
                    gs[i].replay()
                st, en = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                st.record()
                for _ in range(30):
                    gs[i].replay()
                en.record()
                en.synchronize()
                samples[i].append(st.elapsed_time(en) * 1000 / 30)
        emit(
            {
                "kind": "normal_in_range_and_timing",
                "kernel": kind,
                "batch": 24,
                "bitwise": True,
                "order": "ABBA x5",
                "replays_per_sample": 30,
                "unit": "us_per_launch",
                "samples": samples,
                "medians": {i: statistics.median(x) for i, x in samples.items()},
            }
        )
    result = {
        "status": "passed",
        "elapsed_seconds": time.time() - start,
        "records": records,
        "torch_peak_reserved": torch.cuda.max_memory_reserved(),
        "scope": "actual conversion kernels only; no full-model quality claim",
    }
except Exception as e:
    result = {"status": "failed", "error": repr(e), "records": records}
    outpath.write_text(json.dumps(result, indent=2))
    raise
outpath.write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
