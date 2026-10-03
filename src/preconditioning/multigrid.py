import numpy as np
import scipy.sparse as sp
import torch

from .fem import prolongation


def to_csr(S, dtype=torch.float64, device="cpu"):
    S = S.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(S.indptr.astype(np.int64)),
                                   torch.from_numpy(S.indices.astype(np.int64)),
                                   torch.from_numpy(S.data), size=S.shape,
                                   dtype=dtype, device=device)


class _SymMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, St, x):
        ctx.St = St
        return (S @ x.T).T

    @staticmethod
    def backward(ctx, g):
        return None, None, (ctx.St @ g.T.contiguous()).T


def mm(S, St, x):
    return _SymMM.apply(S, St, x.contiguous())


class VCycle(torch.nn.Module):
    """Galerkin V-cycle for a matrix A on the interior nodes, damped Jacobi smoothing, exact coarsest solve."""

    def __init__(self, A, n, coarsest=9, nu=2, omega=8 / 9, coarse_sweeps=0, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.nu, self.omega, self.coarse_sweeps = nu, omega, coarse_sweeps
        As, Ps = [sp.csr_matrix(A)], []
        m = n
        while m > coarsest:
            P = prolongation(m)
            Ps.append(P)
            As.append((P.T @ As[-1] @ P).tocsr())
            m = (m + 1) // 2
        kw = dict(dtype=dtype, device=device)
        self.A = [to_csr(a, **kw) for a in As]
        self.P = [to_csr(p, **kw) for p in Ps]
        self.R = [to_csr(p.T, **kw) for p in Ps]
        self.D = [torch.as_tensor(a.diagonal(), **kw) for a in As]
        self.coarse_inv = None if coarse_sweeps else torch.as_tensor(np.linalg.inv(As[-1].toarray()), **kw)
        self.levels = len(As)

    def smooth(self, l, e, r):
        for _ in range(self.nu):
            e = e + self.omega * (r - mm(self.A[l], self.A[l], e)) / self.D[l]
        return e

    def cycle(self, l, r):
        if l == self.levels - 1:
            if self.coarse_sweeps:
                e = torch.zeros_like(r)
                for _ in range(self.coarse_sweeps):
                    e = e + self.omega * (r - mm(self.A[l], self.A[l], e)) / self.D[l]
                return e
            return r @ self.coarse_inv.T
        e = self.smooth(l, torch.zeros_like(r), r)
        rc = mm(self.R[l], self.P[l], r - mm(self.A[l], self.A[l], e))
        e = e + mm(self.P[l], self.R[l], self.cycle(l + 1, rc))
        return self.smooth(l, e, r)

    def forward(self, r):
        return self.cycle(0, r)
