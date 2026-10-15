#!/usr/bin/env python3
"""Apply the measured 148-SM HGX B300 grid adaptations to a fresh output.

CPU-only. Inputs, cubins and generator defaults remain unchanged. This is
deliberately restricted to the qualified full-K3 PP8 / peer-AR decode shapes.
"""
import argparse
import copy
import json
from pathlib import Path


def adapt(manifest, kind):
    result = copy.deepcopy(manifest)
    changes = []
    if kind == "prefill":
        def visit(value, path=""):
            if isinstance(value, dict):
                if value.get("entry") == "kern_k3g_moe_route":
                    grid = value.get("grid")
                    if grid == [64, 1, 1]:
                        return  # Qualified small-row path is unchanged.
                    if grid != [304, 1, 1]:
                        raise ValueError(f"Unexpected route grid at {path}: {grid}")
                    value["grid"] = [296, 1, 1]
                    changes.append(path)
                for key, child in value.items():
                    visit(child, path + "/" + key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, path + "/" + str(index))
        visit(result)
        if len(changes) != 1:
            raise ValueError(f"Expected one large routing launch, got {changes}")
    else:
        if "tp_allreduce_bf16" in result["ops"]:
            raise ValueError("This adapter qualifies the fused peer-AR decode manifest only")
        for name in ("ar_attnres_rms", "ar_finalize_rms"):
            launches = result["ops"][name]["impl"]["launches"]
            if len(launches) != 1 or launches[0]["grid"] != [152, 1, 1]:
                raise ValueError(f"Unexpected {name} launch geometry")
            launches[0]["grid"] = [144, 1, 1]
            changes.append(f"/ops/{name}/impl/launches/0")
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("prefill", "decode"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    if args.kind == "prefill":
        sources = [args.input / f"stage{i}.json" for i in range(8)]
    else:
        sources = [args.input]
    prepared = [(source, *adapt(json.loads(source.read_text()), args.kind)) for source in sources]
    if args.kind == "prefill":
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    receipts = []
    for source, manifest, changes in prepared:
        destination = args.output / source.name if args.kind == "prefill" else args.output
        with destination.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        receipts.append(dict(source=str(source), destination=str(destination), changes=changes))
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
