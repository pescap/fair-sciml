import sys

import numpy as np
import scipy.sparse.linalg as sla
import torch

from preconditioning import VCycle, interior, kappa_pls, stiffness
from tensorpils.data import create_datasets


def xi(n, seed):
    xc = (np.arange(n - 1) + 0.5) / (n - 1)
    if seed == 0:
        return np.outer(np.sin(2 * np.pi * xc), np.sin(2 * np.pi * xc)).ravel()
    return np.random.RandomState(seed).uniform(-1, 1, (n - 1) ** 2)


class Perturbed(torch.nn.Module):
    def __init__(self, V, B, mu):
        super().__init__()
        self.V, self.B, self.mu = V, B, mu

    def forward(self, r):
        p = self.V(r)
        return p + self.mu * self.V(torch.from_numpy(self.B @ p[0].numpy())[None])


for n in [int(a) for a in sys.argv[2:]]:
    _, _, te = create_datasets(1024, 128, 256, K=10, grid_resolution=n, seed=42, solution="analytic")
    I = interior(n)
    M = te.problem.M.to_scipy_coo().tocsr()
    U = torch.stack(te.us).double().numpy()
    F = torch.stack(te.fs).double().numpy()
    A = stiffness(n)
    V = VCycle(A, n)
    row = {}
    for nu in (0.0, 0.01, 0.03, 0.1, 0.3):
        Anu = stiffness(n, 1 + nu * xi(n, int(sys.argv[1])))
        lu = sla.splu(Anu.tocsc())
        rel = []
        for u, f in zip(U, F):
            uh = np.zeros_like(u)
            uh[I] = lu.solve((M @ f)[I])
            e = uh - u
            rel.append(np.sqrt((e @ (M @ e)) / (u @ (M @ u))))
        row[f"nu{nu}"] = {"floor": round(100 * float(np.mean(rel)), 3), "kappa": round(kappa_pls(Anu, V), 4)}
    for mu in (0.3, 0.6, 0.9):
        Bm = stiffness(n, xi(n, 4321))
        row[f"mu{mu}"] = {"kappa": round(kappa_pls(A, Perturbed(V, Bm, mu)), 4)}
    print(n, row, flush=True)
