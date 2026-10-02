import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from tensorpils.meshing import obstacle_mesh
from tensorpils.physics import StokesProblem

TOL = 1e-6


def top(op, n, v0):
    return float(np.real(sla.eigs(op, k=1, which="LM", tol=TOL, v0=v0, return_eigenvectors=False)[0]))


def setup(h):
    mesh = obstacle_mesh(chara_length=h, order=2)
    P = StokesProblem(mesh)
    K = P.K.to_scipy_coo().tocsr().astype(np.float64)
    Mp = P.M_p.to_scipy_coo().tocsr().astype(np.float64)
    free = np.flatnonzero(~P.dirichlet_mask.numpy())
    Kf = K[free][:, free].tocsc()
    nu = int((free < P.off_p).sum())
    pts = mesh.points.numpy()
    cells = mesh.cells["triangle6"].numpy()[:, :3]
    e = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    hmax = float(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1).max())
    return Kf, Mp, nu, hmax, P.m_p_lumped.double().numpy()


def run(h, w_code):
    t = time.time()
    Kf, Mp, nu, hmax, ml = setup(h)
    n = Kf.shape[0]
    Av = Kf[:nu][:, :nu].tocsc()
    G = sp.block_diag([Av, Mp]).tocsc()
    nul = np.r_[np.zeros(nu), np.ones(n - nu)]
    asym = sla.norm(Kf - Kf.T) / sla.norm(Kf)
    knul = np.linalg.norm(Kf @ nul) / np.linalg.norm(nul)
    keep = np.r_[np.arange(n - 1)]
    lu_pin = sla.splu(Kf[keep][:, keep].tocsc())
    lu_A = sla.splu(Av)
    lu_M = sla.splu(Mp.tocsc())
    lu_G = sla.splu(G)

    def ksolve(r):
        return np.r_[lu_pin.solve(r[:-1]), 0.0]

    def eucl(x):
        return x - nul * (x @ nul) / (nul @ nul)

    def gproj(x):
        y = x.copy()
        y[nu:] -= (Mp @ y[nu:]).sum() / Mp.sum()
        return y

    preconds = {"B": (lambda r: np.r_[lu_A.solve(r[:nu]), lu_M.solve(r[nu:])],
                      lambda w: np.r_[Av @ w[:nu], Mp @ w[nu:]], np.asarray(Mp.sum(axis=0)).ravel()),
                "B_code": (lambda r: np.r_[lu_A.solve(r[:nu]), w_code * r[nu:] / ml],
                           lambda w: np.r_[Av @ w[:nu], ml * w[nu:] / w_code], ml / w_code)}
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    out = {"h": h, "hmax": hmax, "n": n, "nu": nu, "asym": asym, "knul": knul}
    gmax = top(sla.LinearOperator((n, n), matvec=lambda x: G @ x, dtype=float), n, v0)
    gmin = 1 / top(sla.LinearOperator((n, n), matvec=lambda x: lu_G.solve(x), dtype=float), n, v0)
    out["kG_X"] = gmax / gmin
    for name, (Bap, Binv, wt) in preconds.items():
        def hinv(r, proj):
            wv = ksolve(r)
            wv[nu:] -= (wt @ wv[nu:]) / wt.sum()
            return proj(ksolve(Binv(wv)))
        Hm = lambda x: Kf.T @ Bap(Kf @ x)
        lmax = top(sla.LinearOperator((n, n), matvec=Hm, dtype=float), n, eucl(v0))
        lmin = 1 / top(sla.LinearOperator((n, n), matvec=lambda x: hinv(eucl(x), eucl), dtype=float), n, eucl(v0))
        gmx = top(sla.LinearOperator((n, n), matvec=lambda x: lu_G.solve(Hm(x)), dtype=float), n, gproj(v0))
        gmn = 1 / top(sla.LinearOperator((n, n), matvec=lambda x: hinv(G @ gproj(x), gproj), dtype=float), n, gproj(v0))
        out[name] = (lmax / lmin, gmx / gmn, lmax, lmin, gmx, gmn)
    out["sec"] = time.time() - t
    return out


w_code = 16.0
print(f"tol={TOL}  B_code pressure block = {w_code:g}*mu/diag(M_p) (lumped), velocity block exact LU")
for h in [float(x) for x in sys.argv[1:]]:
    o = run(h, w_code)
    print(f"cl={o['h']:g} hmax={o['hmax']:.4f} free={o['n']} (u {o['nu']}, p {o['n']-o['nu']}) "
          f"asym={o['asym']:.1e} |K_f 1_p|={o['knul']:.1e} kappa2(G_X)={o['kG_X']:.4e} time={o['sec']:.0f}s")
    for name in ["B", "B_code"]:
        k2, kg, a, b, c, d = o[name]
        print(f"   {name:7s} kappa2(H)={k2:.4e} [{a:.4e}/{b:.4e}]  kappaG(H)={kg:.4e} [{c:.4e}/{d:.4e}]")
    sys.stdout.flush()
