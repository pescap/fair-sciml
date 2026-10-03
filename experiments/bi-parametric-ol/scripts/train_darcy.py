import argparse
import json
import os

import numpy as np

from preconditioning.darcy import darcy_dataset, train

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=65)
p.add_argument("--loss", choices=["data", "ls", "pls"], default="pls")
p.add_argument("--prec", choices=["fixed", "fixed_diag", "own"], default="own")
p.add_argument("--sigma", type=float, default=0.5)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--device", default="cuda")
p.add_argument("--cache", default="data")
p.add_argument("--out", required=True)
args = p.parse_args()

os.makedirs(args.cache, exist_ok=True)
cache = f"{args.cache}/darcy_n{args.n}_s{args.sigma}.npz"
if not os.path.exists(cache):
    a, u = darcy_dataset(args.n, args.sigma, 1024 + 128 + 256)
    tmp = f"{cache}.{os.getpid()}.npz"
    np.savez(tmp, a=a, u=u)
    os.replace(tmp, cache)
d = np.load(cache)
res = train(d["a"], d["u"], args.sigma, args.loss, args.prec, args.epochs, args.lr, args.seed, args.device)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(dict(vars(args), **res), open(args.out, "w"))
print("test", 100 * res["test_rel"], "mem", res["max_mem_gb"], "s/epoch", sum(res["epoch_times"]) / len(res["epoch_times"]))
