import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
from tensormesh import ElementAssembler

from tensorpils.meshing import obstacle_mesh
from tensorpils.physics import StokesProblem

TOL = 1e-6


class Convection(ElementAssembler):
    def forward(self, gradu, v, w):
        return (gradu @ w) * v


def top(op, v0):
    return float(np.real(sla.eigs(op, k=1, which="LM", tol=TOL, v0=v0, return_eigenvectors=False)[0]))


def wind(P, mesh, tilt):
    x, y = mesh.points.numpy().astype(np.float64).T
    f = torch.as_tensor(np.stack([np.sin(np.pi * y), tilt * np.sin(np.pi * x)], 1))[None]
    u, _ = P.fem_reference(f)
    u = u[0].numpy()
    return u / np.linalg.norm(u, axis=1).max()


def convection(mesh, w):
    C = Convection.from_mesh(mesh, quadrature_order=5)(mesh.points, point_data={"w": torch.as_tensor(w)})
    return C.to_scipy_coo().tocsr().astype(np.float64)


def orient(mesh, P):
    C = convection(mesh, np.tile([1.0, 0.0], (P.n_u, 1)))
    x = mesh.points.numpy()[:, 0].astype(np.float64)
    M1 = P.M.to_scipy_coo().tocsr().astype(np.float64) @ np.ones(P.n_u)
    inner = ~P.dirichlet_mask.numpy()[:2 * P.n_u:2]
    e = lambda y: np.abs((y - M1)[inner]).max()
    return e(C @ x) <= e(C.T @ x)


def run(h, Us):
    t = time.time()
    mesh = obstacle_mesh(chara_length=h, order=2)
    P = StokesProblem(mesh)
    rows = orient(mesh, P)
    K0 = P.K.to_scipy_coo().tocsr().astype(np.float64)
    Mp = P.M_p.to_scipy_coo().tocsr().astype(np.float64)
    free = np.flatnonzero(~P.dirichlet_mask.numpy())
    nu = int((free < P.off_p).sum())
    n = len(free)
    w, wl = wind(P, mesh, 0.0), wind(P, mesh, 0.3)
    Cw, Cl = convection(mesh, w), convection(mesh, wl)
    Cw, Cl = (Cw, Cl) if rows else (Cw.T, Cl.T)

    def oseen(C, U):
        Cv = sp.kron(C, sp.eye(2), format="csr")
        Z = sp.csr_matrix((K0.shape[0] - Cv.shape[0], K0.shape[0] - Cv.shape[0]))
        return (K0 + U * sp.block_diag([Cv, Z])).tocsr()[free][:, free].tocsc()

    Av = K0.tocsr()[free][:, free][:nu][:, :nu].tocsc()
    G = sp.block_diag([Av, Mp]).tocsc()
    lu_A, lu_M, lu_G = sla.splu(Av), sla.splu(Mp.tocsc()), sla.splu(G)
    B = lambda r: np.r_[lu_A.solve(r[:nu]), lu_M.solve(r[nu:])]
    Binv = lambda x: np.r_[Av @ x[:nu], Mp @ x[nu:]]
    wt = np.asarray(Mp.sum(axis=0)).ravel()

    def gproj(x):
        y = x.copy()
        y[nu:] -= (wt @ y[nu:]) / wt.sum()
        return y

    v0 = gproj(np.random.default_rng(0).standard_normal(n))
    L = lambda f: sla.LinearOperator((n, n), matvec=f, dtype=float)
    out = []
    for U in Us:
        K = oseen(Cw, U)
        Kl = oseen(Cl, U)
        keep = np.arange(n - 1)
        lu = sla.splu(K[keep][:, keep].tocsc())
        lul = sla.splu(Kl[keep][:, keep].tocsc())
        solve = lambda r, trans="N": np.r_[lu.solve(r[:-1], trans=trans), 0.0]

        def hinv(x):
            y = solve(x, "T")
            y[nu:] -= (wt @ y[nu:]) / wt.sum()
            return gproj(solve(Binv(y)))

        Hm = lambda x: K.T @ B(K @ x)
        gmx = top(L(lambda x: gproj(lu_G.solve(Hm(gproj(x))))), v0)
        gmn = 1 / top(L(lambda x: hinv(G @ gproj(x))), v0)
        Pl = lambda r: gproj(np.r_[lul.solve(r[:-1]), 0.0])
        E = lambda x: gproj(x - Pl(K @ gproj(x)))
        Et = lambda y: gproj(y - K.T @ np.r_[lul.solve(gproj(y)[:-1], trans="T"), 0.0])
        rho = np.sqrt(max(top(L(lambda x: gproj(lu_G.solve(Et(G @ E(x))))), v0), 0.0))
        out.append({"U": U, "kG_stokes_weight": round(gmx / gmn, 1), "rho_lagged": round(float(rho), 4)})
    print(h, n, round(time.time() - t), out, flush=True)


if __name__ == "__main__":
    for h in [float(a) for a in sys.argv[1:]]:
        run(h, [0.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
