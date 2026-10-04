import copy
import math
import time

import numpy as np
import scipy.sparse.linalg as sla
import torch
from neuralop.models import FNO

from .batched import BMG, interior_mask, stiffness_apply
from .fem import interior, lognormal_coef, stiffness


def darcy_dataset(n, sigma, count, seed=0):
    """Coefficients [count, n-1, n-1] and Q1 solutions [count, n, n] of -div(a grad u) = 1, u = 0 on the boundary."""
    rng = np.random.default_rng(seed)
    I = interior(n)
    h = 1.0 / (n - 1)
    a_all, u_all = [], []
    for _ in range(count):
        a = lognormal_coef(n, sigma, rng)
        u = np.zeros(n * n)
        u[I] = sla.spsolve(stiffness(n, a).tocsc(), np.full(len(I), h * h))
        a_all.append(a.reshape(n - 1, n - 1))
        u_all.append(u.reshape(n, n))
    return np.array(a_all, dtype=np.float32), np.array(u_all, dtype=np.float32)


def nodal(a):
    """Average of the element coefficients around each node."""
    q = torch.nn.functional.pad(a, (1, 1, 1, 1), mode="replicate")
    return 0.25 * (q[:, :-1, :-1] + q[:, 1:, :-1] + q[:, :-1, 1:] + q[:, 1:, 1:])


def train(a, u, sigma, loss="pls", prec="own", epochs=300, lr=None, seed=42, device="cuda",
          splits=(1024, 128, 256), bs=32, build=None):
    """Train a neural operator a -> u with the data, least-squares or preconditioned residual loss.

    build returns the network, mapping [B, 1, n, n] to [B, 1, n, n]; the default is an FNO.
    prec selects the preconditioner of the residual: "fixed" is one cycle for the geometric mean
    coefficient, "fixed_diag" adds a diagonal scaling by the local coefficient, "own" is one cycle
    for the coefficient of each sample.
    """
    ntr, nva, nte = splits
    n, dev = u.shape[-1], device
    cuda = torch.device(dev).type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    A = torch.as_tensor(a, device=dev)
    U = torch.as_tensor(u, device=dev)
    m = interior_mask(n, dev)
    h = 1.0 / (n - 1)
    b = (h * h) * m.float()
    abar = float(torch.exp(torch.log(A[:ntr]).mean()))
    X = (torch.log(nodal(A)) / sigma)[:, None]
    torch.manual_seed(seed)
    model = (build or (lambda: FNO(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1, n_layers=5)))().to(dev)
    lr = lr or (3e-3 if loss == "ls" else 3e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (ntr // bs), eta_min=lr / 10)
    fixed = BMG(n).setup(torch.full((bs, n - 1, n - 1), abar, device=dev))
    scale = 1.0 / U[:ntr].abs().max().item()

    def predict(x):
        return model(x)[:, 0] * m / scale

    def precondition(ai, r):
        if prec == "own":
            return BMG(n).setup(ai)(r)
        if prec == "fixed_diag":
            s = torch.sqrt(nodal(ai) / abar)
            return fixed(r / s) / s
        return fixed(r)

    def loss_fn(idx):
        ai, ui = A[idx], predict(X[idx])
        if loss == "data":
            return 0.5 * ((ui - U[idx]) ** 2).sum((1, 2)).mean() * scale ** 2
        r = (stiffness_apply(ai, ui) - b) * m
        if loss == "pls":
            r = precondition(ai, r)
        return 0.5 * (r ** 2).sum((1, 2)).mean()

    def rel_err(lo, hi):
        with torch.no_grad():
            e = torch.cat([((predict(X[i:i + bs]) - U[i:i + bs]).flatten(1).norm(dim=1)
                            / U[i:i + bs].flatten(1).norm(dim=1)) for i in range(lo, hi, bs)])
        return e.mean().item()

    val, times, best, best_state = [], [], math.inf, None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        sync()
        t0 = time.perf_counter()
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, bs):
            opt.zero_grad(set_to_none=True)
            loss_fn(perm[i:i + bs]).backward()
            opt.step()
            sched.step()
        sync()
        times.append(time.perf_counter() - t0)
        v = rel_err(ntr, ntr + nva)
        val.append(v)
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
        print(f"epoch {ep + 1} val {100 * v:.2f}% time {times[-1]:.2f}s", flush=True)
    best_state.pop("_metadata", None)
    model.load_state_dict(best_state)
    return dict(abar=abar, test_rel=rel_err(ntr + nva, ntr + nva + nte), val=val, epoch_times=times,
                max_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None)
