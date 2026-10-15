#!/usr/bin/env python3
"""Select the separately tested 52-gate-CTA KDA cubin in a fresh stage manifest.

The cubin must be in the directory passed to kern's --kernels. This does not
enable the candidate in other manifests or qualify full-model serving quality.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

BASE_SHA = "663253f41bd8eaaa4d586ddcedf826d7b3d06defc77f3b95c4cf235a830e81ad"
CANDIDATE_SHA = "f7d840e582d3c2b3bbba91a37f0a5fce0e0d3e1e7dfc02a2a1917b4e1a99a60c"


def adapt(manifest, cubin_name, cubin_sha):
    old = "k3_kda_rec+K12_LD=49408"
    new = old + "+K12_GATE_CTAS=52"
    if cubin_sha != CANDIDATE_SHA:
        raise ValueError("Candidate cubin differs from the numerically tested CUDA 13.0.88 build")
    if manifest["modules"][old]["sha256"] != BASE_SHA or new in manifest["modules"]:
        raise ValueError("Unexpected baseline recurrence module")
    result = copy.deepcopy(manifest)
    del result["modules"][old]
    result["modules"][new] = {"source": cubin_name, "sha256": cubin_sha}
    changed = []
    for name, op in result["ops"].items():
        for index, launch in enumerate(op["impl"]["launches"]):
            if launch.get("module") != old:
                continue
            if launch["entry"] != "kern_k3_kda_rec" or launch["grid"] != [{"add": [{"mul": ["seqs", 96]}, 56]}, 1, 1]:
                raise ValueError("Unexpected recurrence launch geometry")
            launch["module"] = new
            launch["grid"] = [{"add": [{"mul": ["seqs", 96]}, 52]}, 1, 1]
            changed.append((name, index))
    if changed != [("flash_kda", 1)]:
        raise ValueError(f"Unexpected recurrence launch set: {changed}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    cubin_sha = hashlib.sha256(args.cubin.read_bytes()).hexdigest()
    result = adapt(json.loads(args.input.read_text()), args.cubin.name, cubin_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), cubin_sha256=cubin_sha)))


if __name__ == "__main__":
    main()
