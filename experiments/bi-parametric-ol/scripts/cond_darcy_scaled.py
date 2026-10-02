import json
import numpy as np
import torch

from preconditioning import VCycle, interior, kappa_pls, lognormal_coef, stiffness
from preconditioning.fem import assemble


def nodal(n, c):
    w = assemble(n, c, local=np.full((4, 4), 0.25)).diagonal()
    cnt = assemble(n, None, local=np.full((4, 4), 0.25)).diagonal()
    return (w / cnt)[interior(n)]


class Scaled(torch.nn.Module):
    def __init__(self, V, s):
        super().__init__()
        self.V, self.s = V, torch.from_numpy(s)

    def forward(self, r):
        return self.V(r / self.s) / self.s


out = {}
for sigma in [0.5, 1.0, 1.5]:
    for n in [33, 65, 129, 257]:
        c = lognormal_coef(n, sigma, np.random.default_rng(0))
        cbar = np.exp(np.log(c).mean())
        A = stiffness(n, c)
        V0 = VCycle(stiffness(n, np.full_like(c, cbar)), n)
        s = np.sqrt(nodal(n, c) / cbar)
        out[f"{sigma}_{n}"] = {"contrast": float(c.max() / c.min()),
                               "mean": kappa_pls(A, V0),
                               "scaled": kappa_pls(A, Scaled(V0, s)),
                               "own": kappa_pls(A, VCycle(A, n))}
        print(sigma, n, out[f"{sigma}_{n}"], flush=True)
json.dump(out, open("cond_darcy_scaled.json", "w"), indent=1)
