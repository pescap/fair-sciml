import math
import os
import sys
import time

import torch

from tensorpils.preconditioners.multigrid import GeometricMultigrid


def full_levels(n, coarsest=9):
    return int(math.log2((n - 1) / (coarsest - 1))) + 1


def variants(n):
    L = full_levels(n)
    return {
        "s2": dict(n_levels=L),
        "s1": dict(n_levels=L, pre_smooth=1, post_smooth=1),
        "s2_bf16": dict(n_levels=L, dtype=torch.bfloat16),
        "s1_bf16": dict(n_levels=L, pre_smooth=1, post_smooth=1, dtype=torch.bfloat16),
        "s2_fp16": dict(n_levels=L, dtype=torch.float16),
        "s2_ngp1": dict(n_levels=L, ngp=1),
        "s2_ngp2": dict(n_levels=L, ngp=2),
        "c17_jac4": dict(n_levels=full_levels(n, 17), sweeps=4),
        "c33_jac8": dict(n_levels=full_levels(n, 33), sweeps=8),
        "two_grid_jac4": dict(n_levels=2, sweeps=4),
    }


def timed(f, reps=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return sorted(t)[reps // 2]


def rho(G, mask, dtype, its=60):
    x = torch.randn(1, mask.numel(), device="cuda", dtype=torch.float64) * ~mask
    A = lambda v: G._mm("A", 0, v.to(dtype)).to(torch.float64) * ~mask
    P = lambda v: G(v.to(dtype)).to(torch.float64) * ~mask
    for _ in range(its):
        y = x - P(A(x))
        x = y - A(P(y))
        lam = x.norm()
        x = x / lam
    return math.sqrt(lam.item())


def run(n, name, kw, mask):
    dtype = kw.pop("dtype", torch.float32)
    os.environ["TPILS_COARSE_SWEEPS"] = str(kw.pop("sweeps", 0))
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    G = GeometricMultigrid(n, n, omega=8 / 9, **kw).cuda().to(dtype)
    setup = time.perf_counter() - t0
    r0 = torch.randn(B, n * n, device="cuda") * ~mask
    apply_t = timed(lambda: G(r0.to(dtype)))

    def step():
        r = r0.clone().requires_grad_(True)
        p = G(r.to(dtype)).float()
        (0.5 * (p * p).sum(-1).mean()).backward()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    mem = (torch.cuda.max_memory_allocated() - base) / 1e6
    step_t = timed(step)
    print(n, name, {"levels": G.n_levels, "setup_s": round(setup, 3), "apply_ms": round(1e3 * apply_t, 3),
                    "step_ms": round(1e3 * step_t, 3), "mem_MB": round(mem, 1),
                    "rhoI": round(rho(G, mask, dtype), 3)}, flush=True)


B = 32
for n in [int(a) for a in sys.argv[1:]]:
    mask = torch.zeros(n, n, dtype=torch.bool, device="cuda")
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    mask = mask.flatten()
    for name, kw in variants(n).items():
        try:
            run(n, name, dict(kw), mask)
        except Exception as e:
            print(n, name, "failed:", type(e).__name__, str(e)[:120], flush=True)


