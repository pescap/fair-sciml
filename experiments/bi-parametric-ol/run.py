import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(f"{HERE}/../../src")

p = argparse.ArgumentParser(description="Run the training runs of runs.json whose name matches a pattern.")
p.add_argument("pattern", nargs="?", default=".")
p.add_argument("--out", default=f"{HERE}/reproduced")
p.add_argument("--tensorpils", default=f"{HERE}/TensorPILS")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
out = os.path.abspath(args.out)

for e in json.load(open(f"{HERE}/runs.json")):
    run = e["run"]
    if not re.search(args.pattern, run):
        continue
    tpils = "tensorpils.cli" in e["command"]
    result = f"{out}/results/{run}/result.json" if tpils else f"{out}/results/{run}.json"
    if os.path.exists(result):
        continue
    cmd = e["command"].replace(f"runs/{run}" if tpils else f"runs/{run}.json", f"{out}/runs/{run}" if tpils else result)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [SRC, os.path.abspath(args.tensorpils), os.environ.get("PYTHONPATH")])))
    env.update(kv.split("=", 1) for kv in e["env"].replace("<repo>/src", SRC).split())
    print(run, e["env"], cmd, flush=True)
    if args.dry_run:
        continue
    subprocess.run(shlex.split(cmd), cwd=args.tensorpils if tpils else HERE, env=env, check=True)
    if tpils:
        os.makedirs(os.path.dirname(result), exist_ok=True)
        shutil.copy(glob.glob(f"{out}/runs/{run}/results/*.json")[0], result)
