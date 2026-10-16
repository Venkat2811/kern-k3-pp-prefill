"""Run the component regression on an owned CUDA device; no serving processes."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

p = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True, type=pathlib.Path)
ap.add_argument("--candidate", required=True, type=pathlib.Path)
ap.add_argument("--out", required=True, type=pathlib.Path)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--empty", action="store_true")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
o = a.out.resolve()
shutil.copyfile(p / "check.py", o / "gate.py")
shutil.copyfile(p / "launch.cpp", o / "launch.cpp")
shutil.copyfile(a.baseline, o / "original.cubin")
shutil.copyfile(a.candidate, o / "candidate.cubin")
cuda = pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
cmd = [
    os.environ.get("CXX", "c++"),
    "-shared",
    "-fPIC",
    "-O2",
    "-I" + str(cuda / "include"),
    str(o / "launch.cpp"),
    "-L" + str(cuda / "lib64/stubs"),
    "-lcuda",
    "-o",
    str(o / "launch.so"),
]
subprocess.run(cmd, check=True)
argv = [sys.executable, str(o / "gate.py"), "--out", str(o / "results.json")]
argv += ["--smoke"] if a.smoke else []
argv += ["--empty"] if a.empty else []
receipt = {
    "argv": argv,
    "build": cmd,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "hashes": {
        x.name: hashlib.sha256(x.read_bytes()).hexdigest()
        for x in o.iterdir()
        if x.is_file()
    },
    "timeout_seconds": 300,
    "start": time.time(),
}
with (o / "stdout.log").open("w") as log:
    child = subprocess.Popen(
        argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    receipt["pid"] = child.pid
    (o / "launch.json").write_text(json.dumps(receipt, indent=2))
    try:
        rc = child.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        rc = child.wait()
        receipt["timeout"] = True
receipt.update(exit_code=rc, end=time.time())
(o / "completion.json").write_text(json.dumps(receipt, indent=2))
print((o / "stdout.log").read_text())
sys.exit(rc)
