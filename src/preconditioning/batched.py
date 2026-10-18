import torch
import torch.nn.functional as F

KE = torch.tensor([[4, -1, -2, -1],
                   [-1, 4, -1, -2],
                   [-2, -1, 4, -1],
                   [-1, -2, -1, 4]], dtype=torch.float64) / 6.0
W = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float64)


def interior_mask(n, device):
    m = torch.zeros(n, n, dtype=torch.bool, device=device)
    m[1:-1, 1:-1] = True
    return m


def stiffness_apply(a, u):
    """Matrix-free product of the Q1 stiffness matrices of a batch of element coefficients a with nodal fields u."""
    U = torch.stack([u[:, :-1, :-1], u[:, :-1, 1:], u[:, 1:, 1:], u[:, 1:, :-1]], 1)
    L = a[:, None] * torch.einsum("ij,bjxy->bixy", KE.to(u), U)
    out = torch.zeros_like(u)
    out[:, :-1, :-1] += L[:, 0]
    out[:, :-1, 1:] += L[:, 1]
    out[:, 1:, 1:] += L[:, 2]
    out[:, 1:, :-1] += L[:, 3]
    return out


def stiffness_diag(a):
    d = torch.zeros(a.shape[0], a.shape[1] + 1, a.shape[2] + 1, dtype=a.dtype, device=a.device)
    d[:, :-1, :-1] += a
    d[:, :-1, 1:] += a
    d[:, 1:, 1:] += a
    d[:, 1:, :-1] += a
    return 2 * d / 3


class BMG:
    """Batched matrix-free V-cycle for -div(a grad u) + c u with one coefficient pair per sample."""

    def __init__(self, n, coarsest=9, s=2, omega=8 / 9):
        self.n, self.coarsest, self.s, self.omega = n, coarsest, s, omega

    def setup(self, a, c=None):
        B, n = a.shape[0], self.n
        c = torch.zeros(B, n, n, dtype=a.dtype, device=a.device) if c is None else c
        self.a, self.c, self.h, self.m = [a], [c], [1.0 / (n - 1)], [interior_mask(n, a.device)]
        while self.a[-1].shape[1] + 1 > self.coarsest:
            al, cl = self.a[-1], self.c[-1]
            self.a.append(0.25 * (al[:, ::2, ::2] + al[:, 1::2, ::2] + al[:, ::2, 1::2] + al[:, 1::2, 1::2]))
            self.c.append(cl[:, ::2, ::2])
            self.h.append(2 * self.h[-1])
            self.m.append(interior_mask(self.a[-1].shape[1] + 1, a.device))
        self.d = [stiffness_diag(al) + cl * hl * hl for al, cl, hl in zip(self.a, self.c, self.h)]
        nc = self.a[-1].shape[1] + 1
        idx = self.m[-1].flatten().nonzero().flatten()
        k = len(idx)
        E = torch.zeros(k, nc * nc, dtype=a.dtype, device=a.device)
        E[torch.arange(k), idx] = 1
        U = E.view(1, k, nc, nc).expand(B, -1, -1, -1).reshape(B * k, nc, nc)
        ar, cr = self.a[-1].repeat_interleave(k, 0), self.c[-1].repeat_interleave(k, 0)
        Ae = (stiffness_apply(ar, U) + cr * self.h[-1] ** 2 * U) * self.m[-1]
        self.coarse_inv = torch.linalg.inv(Ae.view(B, k, nc * nc)[:, :, idx].transpose(1, 2))
        self.coarse_idx = idx
        return self

    def op(self, l, u):
        return (stiffness_apply(self.a[l], u) + self.c[l] * self.h[l] ** 2 * u) * self.m[l]

    def restrict(self, r):
        return F.conv2d(r[:, None], W.to(r)[None, None], stride=2, padding=1)[:, 0]

    def prolong(self, e):
        return F.conv_transpose2d(e[:, None], W.to(e)[None, None], stride=2, padding=1)[:, 0]

    def cycle(self, l, r):
        if l == len(self.a) - 1:
            nc = r.shape[1]
            x = torch.einsum("bij,bj->bi", self.coarse_inv, r.flatten(1)[:, self.coarse_idx])
            e = torch.zeros(r.shape[0], nc * nc, dtype=r.dtype, device=r.device)
            e[:, self.coarse_idx] = x
            return e.view(-1, nc, nc)
        e = torch.zeros_like(r)
        for _ in range(self.s):
            e = e + self.omega * (r - self.op(l, e)) / self.d[l] * self.m[l]
        rc = self.restrict(r - self.op(l, e)) * self.m[l + 1]
        e = e + self.prolong(self.cycle(l + 1, rc)) * self.m[l]
        for _ in range(self.s):
            e = e + self.omega * (r - self.op(l, e)) / self.d[l] * self.m[l]
        return e

    def __call__(self, r):
        n = self.n
        return self.cycle(0, r.reshape(-1, n, n) * self.m[0]).reshape(r.shape)
