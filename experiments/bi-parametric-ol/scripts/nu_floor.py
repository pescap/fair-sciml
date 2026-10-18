import sys

import numpy as np
import scipy.sparse.linalg as sla
import torch

from preconditioning import interior, stiffness
from tensorpils.data import create_datasets

nu0 = float(sys.argv[1])
for n in [int(a) for a in sys.argv[2:]]:
    nu = nu0 * (32 / (n - 1)) ** 2
    _, _, te = create_datasets(1024, 128, 256, K=10, grid_resolution=n, seed=42, solution="analytic")
    I = interior(n)
    M = te.problem.M.to_scipy_coo().tocsr()
    U = torch.stack(te.us).double().numpy()
    F = torch.stack(te.fs).double().numpy()
    xc = (np.arange(n - 1) + 0.5) / (n - 1)
    lu = sla.splu(stiffness(n, 1 + nu * np.outer(np.sin(2 * np.pi * xc), np.sin(2 * np.pi * xc)).ravel()).tocsc())
    rel = []
    for u, f in zip(U, F):
        uh = np.zeros_like(u)
        uh[I] = lu.solve((M @ f)[I])
        e = uh - u
        rel.append(np.sqrt((e @ (M @ e)) / (u @ (M @ u))))
    print(n, {"nu": nu, "floor": round(100 * float(np.mean(rel)), 4)}, flush=True)
