import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from oseen_dual import convection, orient, top, wind
from tensorpils.meshing import obstacle_mesh
from tensorpils.physics import StokesProblem

for h in [float(a) for a in sys.argv[1:]]:
    mesh = obstacle_mesh(chara_length=h, order=2)
    P = StokesProblem(mesh)
    K0 = P.K.to_scipy_coo().tocsr().astype(np.float64)
    free = np.flatnonzero(~P.dirichlet_mask.numpy())
    nu = int((free < P.off_p).sum())
    fv = free[:nu]
    Av = K0[fv][:, fv].tocsc()
    lu = sla.splu(Av)
    w, wl = wind(P, mesh, 0.0), wind(P, mesh, 0.3)
    rows = orient(mesh, P)

    def block(ww):
        C = convection(mesh, ww)
        C = C if rows else C.T
        return sp.kron(C, sp.eye(2), format="csr")[fv][:, fv]

    Cw, Cl = block(w), block(wl)
    n = Av.shape[0]
    v0 = np.random.default_rng(0).standard_normal(n)
    norm = lambda C: np.sqrt(top(sla.LinearOperator((n, n), matvec=lambda x: lu.solve(C.T @ lu.solve(C @ x)), dtype=float), v0))
    M = P.M.to_scipy_coo().tocsr().astype(np.float64)
    dw = w - wl
    l2 = np.sqrt(sum(dw[:, i] @ (M @ dw[:, i]) for i in range(2)))
    print(h, n, "|C(w)|", round(norm(Cw), 5), "|C(w)-C(wl)|", round(norm(Cw - Cl), 5),
          "L2(w-wl)", round(float(l2), 4), "mean|w|", round(float(np.linalg.norm(w, axis=1).mean()), 4),
          "Kdiag mean", float(Av.diagonal().mean()), flush=True)
