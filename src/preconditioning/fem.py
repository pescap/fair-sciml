import numpy as np
import scipy.sparse as sp

KE = np.array([[4, -1, -2, -1],
               [-1, 4, -1, -2],
               [-2, -1, 4, -1],
               [-1, -2, -1, 4]], dtype=np.float64) / 6.0

ME = np.array([[4, 2, 1, 2],
               [2, 4, 2, 1],
               [1, 2, 4, 2],
               [2, 1, 2, 4]], dtype=np.float64) / 36.0


def element_nodes(n):
    iy, ix = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    k = (iy * n + ix).ravel()
    return np.stack([k, k + 1, k + n + 1, k + n], axis=1)


def assemble(n, coef=None, local=KE, scale=1.0):
    """Assemble a Q1 matrix on the uniform n x n grid of the unit square, one coefficient per element."""
    en = element_nodes(n)
    c = np.ones(len(en)) if coef is None else np.asarray(coef, dtype=np.float64).ravel()
    rows = np.repeat(en, 4, axis=1).ravel()
    cols = np.tile(en, (1, 4)).ravel()
    vals = (c[:, None, None] * local[None]).reshape(len(en), 16).ravel() * scale
    return sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))


def interior(n):
    iy, ix = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.flatnonzero(((iy > 0) & (iy < n - 1) & (ix > 0) & (ix < n - 1)).ravel())


def stiffness(n, coef=None):
    """Stiffness matrix of -div(coef grad u) on the interior nodes."""
    I = interior(n)
    return assemble(n, coef)[I][:, I].tocsr()


def mass(n):
    """Mass matrix on the interior nodes."""
    h = 1.0 / (n - 1)
    I = interior(n)
    return assemble(n, local=ME, scale=h * h)[I][:, I].tocsr()


def prolongation_1d(nf):
    nc = (nf + 1) // 2
    mf, mc = nf - 2, nc - 2
    rows, cols, vals = [], [], []
    for i in range(1, nf - 1):
        if i % 2 == 0:
            rows.append(i - 1); cols.append(i // 2 - 1); vals.append(1.0)
        else:
            for j in ((i - 1) // 2, (i + 1) // 2):
                if 0 < j < nc - 1:
                    rows.append(i - 1); cols.append(j - 1); vals.append(0.5)
    return sp.csr_matrix((vals, (rows, cols)), shape=(mf, mc))


def prolongation(nf):
    """Bilinear prolongation from the (nf + 1) / 2 grid to the nf grid, interior nodes."""
    p = prolongation_1d(nf)
    return sp.kron(p, p, format="csr")


def lognormal_coef(n, sigma, rng, modes=4):
    """Element coefficients exp(g) of a smooth Gaussian field g with standard deviation sigma."""
    xc = (np.arange(n - 1) + 0.5) / (n - 1)
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    g = np.zeros_like(X)
    for i in range(1, modes + 1):
        for j in range(1, modes + 1):
            g += rng.standard_normal() * np.cos(np.pi * i * X) * np.cos(np.pi * j * Y) / (i * i + j * j)
    g *= sigma / g.std()
    return np.exp(g)
