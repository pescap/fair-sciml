import sys

import numpy as np
import scipy.sparse.linalg as sla

from preconditioning import VCycle, stiffness
from preconditioning.diagnostics import as_numpy


def xi(n, seed):
    xc = (np.arange(n - 1) + 0.5) / (n - 1)
    if seed == 0:
        return np.outer(np.sin(2 * np.pi * xc), np.sin(2 * np.pi * xc)).ravel()
    return np.random.RandomState(seed).uniform(-1, 1, (n - 1) ** 2)


def kappa_energy(A, Anu, P):
    m = A.shape[0]
    lu = sla.splu(A.tocsc())
    Q = lambda x: P(Anu @ x)
    Qt = lambda y: Anu.T @ P(y)
    H = sla.LinearOperator((m, m), matvec=lambda x: Qt(A @ Q(x)), dtype=np.float64)
    Ainv = sla.LinearOperator((m, m), matvec=lu.solve, dtype=np.float64)
    hi = sla.eigsh(H, k=1, M=A, Minv=Ainv, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    lo = sla.eigsh(H, k=1, M=A, Minv=Ainv, which="SA", tol=1e-6, return_eigenvectors=False)[0]
    return float(hi / lo)


def kappa_euclid(Anu, P):
    m = Anu.shape[0]
    H = sla.LinearOperator((m, m), matvec=lambda x: Anu.T @ P(P(Anu @ x)), dtype=np.float64)
    hi = sla.eigsh(H, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    lo = sla.eigsh(H, k=1, which="SA", tol=1e-6, return_eigenvectors=False)[0]
    return float(hi / lo)


for n in [int(a) for a in sys.argv[1:]]:
    A = stiffness(n)
    V = VCycle(A, n)
    Pv = as_numpy(V)
    row = {}
    for tag, seed in (("iid", 1234), ("smooth", 0)):
        for nu in (0.1, 0.3):
            Anu = stiffness(n, 1 + nu * xi(n, seed))
            row[f"nu_{tag}{nu}"] = (round(kappa_euclid(Anu, Pv), 2), round(kappa_energy(A, Anu, Pv), 2))
    B = stiffness(n, xi(n, 4321))
    for mu in (0.3, 0.6, 0.9):
        Pm = lambda r, mu=mu: (lambda p: p + mu * Pv(B @ p))(Pv(r))
        row[f"mu{mu}"] = (round(kappa_euclid(A, Pm), 2), round(kappa_energy(A, A, Pm), 2))
    print(n, row, flush=True)
