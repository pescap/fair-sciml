import argparse
import json
import os

import numpy as np
import torch

from preconditioning.darcy import darcy_dataset, train


class SensorSubsample(torch.nn.Module):
    def __init__(self, net, n, sensors):
        super().__init__()
        self.net, self.stride, self.n = net, (n - 1) // (sensors - 1), n
        x = torch.linspace(0.0, 1.0, n)
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("coords", torch.stack([xx.reshape(-1), yy.reshape(-1)], -1))

    def forward(self, f):
        return self.net.forward_at(f[..., ::self.stride, ::self.stride], self.coords).reshape(-1, 1, self.n, self.n)


def build(model, n, seed):
    if model == "gaot":
        from tensorpils.gaot import GAOTModel
        return lambda: GAOTModel(grid_size=(n, n))
    if model == "deeponet":
        from tensorpils.baselines import DeepONetModel
        return lambda: SensorSubsample(DeepONetModel(grid_size=(33, 33), p=128, width=256, depth=4, trunk_fourier=64,
                                                     fourier_scale=2.0, seed=seed), n, 33)
    return None

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=65)
p.add_argument("--loss", choices=["data", "ls", "pls"], default="pls")
p.add_argument("--prec", choices=["fixed", "fixed_diag", "own"], default="own")
p.add_argument("--sigma", type=float, default=0.5)
p.add_argument("--model", choices=["fno", "gaot", "deeponet"], default="fno")
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
res = train(d["a"], d["u"], args.sigma, args.loss, args.prec, args.epochs, args.lr, args.seed, args.device,
            build=build(args.model, args.n, args.seed))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(dict(vars(args), **res), open(args.out, "w"))
print("test", 100 * res["test_rel"], "mem", res["max_mem_gb"], "s/epoch", sum(res["epoch_times"]) / len(res["epoch_times"]))
