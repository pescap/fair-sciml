import os
import sys

import numpy as np
import torch

from tensorpils.preconditioners.multigrid import GeometricMultigrid
from preconditioning import gram_kappa, interior, sine_basis, stiffness

for n in [int(a) for a in sys.argv[1:]]:
    I = interior(n)
    A = stiffness(n)
    S = sine_basis(n, 16)
    L = int(np.log2((n - 1) / 8)) + 1
    variants = {"s2": (dict(n_levels=L), {}), "s1": (dict(n_levels=L, pre_smooth=1, post_smooth=1), {}), "ngp1": (dict(n_levels=L, ngp=1), {}),
                "twogrid": (dict(n_levels=2), {"TPILS_COARSE_SWEEPS": "4"})}
    row = {}
    for name, (kw, env) in variants.items():
        os.environ["TPILS_COARSE_SWEEPS"] = env.get("TPILS_COARSE_SWEEPS", "0")
        G = GeometricMultigrid(n, n, omega=8 / 9, **kw).double()

        def PA(X):
            R = np.zeros((X.shape[1], n * n))
            R[:, I] = (A @ X).T
            with torch.no_grad():
                Y = G(torch.from_numpy(R)).numpy()
            return Y[:, I].T

        Q = PA(S)
        row[name] = round(gram_kappa(Q), 3)
    print(n, row, flush=True)
