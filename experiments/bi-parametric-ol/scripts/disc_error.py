import sys

import torch

from tensorpils.data import create_datasets

for n in [int(a) for a in sys.argv[1:]]:
    res = {}
    for sol in ("analytic", "fem"):
        _, _, te = create_datasets(1024, 128, 256, K=10, grid_resolution=n, seed=42, solution=sol)
        res[sol] = torch.stack(te.us).double()
    M = te.problem.M.to_scipy_coo().tocsr()
    e = (res["fem"] - res["analytic"]).numpy()
    u = res["analytic"].numpy()
    rel = [((ei @ (M @ ei)) / (ui @ (M @ ui))) ** 0.5 for ei, ui in zip(e, u)]
    print(n, "mean rel L2 of u_h - u:", 100 * sum(rel) / len(rel), "%", flush=True)
