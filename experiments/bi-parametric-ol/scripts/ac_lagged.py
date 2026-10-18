import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch

from preconditioning import VCycle, interior, mass, stiffness
from tensorpils.data import ACDataset

a, eps, dt = 1.0, 32.0, 0.01
c = 1 / dt + 3 * eps ** 2


def measure(J, P, Pt):
    m = J.shape[0]
    E = lambda x: x - P(J @ x)
    Et = lambda y: y - J.T @ Pt(y)
    rho = np.sqrt(sla.eigsh(sla.LinearOperator((m, m), matvec=lambda x: Et(E(x)), dtype=np.float64),
                            k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0])
    H = sla.LinearOperator((m, m), matvec=lambda x: J.T @ Pt(P(J @ x)), dtype=np.float64)
    hi = sla.eigsh(H, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    lo = sla.eigsh(H, k=1, which="SA", tol=1e-6, return_eigenvectors=False)[0]
    return round(float(rho), 4), round(float(hi / lo), 2)


def vcycle(S, n):
    V = VCycle(S, n)
    f = lambda x: V(torch.from_numpy(x)[None])[0].detach().numpy()
    return f, f


for n in [int(x) for x in sys.argv[1:]]:
    ds = ACDataset(num_samples=2, K=4, seed=42, grid_resolution=n, dt=dt, n_steps=10, a=a, eps=eps, r=0.5)
    I = interior(n)
    A, M = a * a * stiffness(n), mass(n)
    ML = sp.diags(np.asarray(M.sum(axis=1)).ravel())
    frozen = vcycle((A + c * M).tocsr(), n)
    rows = []
    for k in (1, 5, 10):
        u = ds.trajs[0][k].double().numpy()[I]
        up = ds.trajs[0][k - 1].double().numpy()[I]
        J = (M / dt + A + 3 * eps ** 2 * M @ sp.diags(u ** 2)).tocsr()
        Jp = (M / dt + A + 3 * eps ** 2 * M @ sp.diags(up ** 2)).tocsc()
        lu = sla.splu(Jp)
        exact = (lu.solve, lambda y: lu.solve(y, trans="T"))
        lagged = vcycle((M / dt + A + 3 * eps ** 2 * ML @ sp.diags(up ** 2)).tocsr(), n)
        rows.append({"k": k, "frozen": measure(J, *frozen), "lagged_exact": measure(J, *exact), "lagged_vcycle": measure(J, *lagged)})
    print(n, rows, flush=True)
