import glob
import os
import sys

import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from preconditioning import interior, stiffness
from tensorpils.cli import _build_model, build_parser
from tensorpils.data import create_datasets
from tensorpils.preconditioners.multigrid import GeometricMultigrid

runs, arch, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ns = [int(a) for a in sys.argv[4:]]
dev = "cuda"
trained = "rough/n129_s2" if arch == "fno" else f"{arch}/hsweep_s42/n129_pls"
os.environ.setdefault("TPILS_DEEPONET_SENSORS", "33")
tr, _, _ = create_datasets(1024, 128, 256, K=10, grid_resolution=129, seed=42, solution="analytic")
state = torch.load(glob.glob(f"{runs}/{trained}/checkpoints/*_best.pth")[0], map_location=dev, weights_only=False)["model_state_dict"]


def network(n, load):
    args = build_parser().parse_args(["--model", arch, "--grid_resolution", str(n), "--mg_levels", "2"])
    torch.manual_seed(42)
    model = _build_model(args, 1, train_ds=tr).to(dev)
    if load:
        params = dict(model.named_parameters())
        model.load_state_dict({key: v for key, v in state.items() if key in params}, strict=False)
        assert all(torch.equal(params[key], state[key]) for key in params)
    return model.eval()


def tangent(model, f, I):
    params = {name: p.detach() for name, p in model.named_parameters()}
    g = torch.Generator(device=dev).manual_seed(5)
    cols = []
    for _ in range(k):
        v = {name: torch.randn(p.shape, dtype=p.dtype, generator=g, device=dev) for name, p in params.items()}
        with sdpa_kernel(SDPBackend.MATH):
            _, t = jvp(lambda q: functional_call(model, q, (f,)), (params,), (v,))
        cols.append(t.flatten().double().cpu().numpy()[I])
    return np.stack(cols, 1)


def cycle(n, levels, sweeps):
    I = interior(n)

    def P(X):
        os.environ["TPILS_COARSE_SWEEPS"] = str(sweeps)
        G = GeometricMultigrid(n, n, omega=8 / 9, n_levels=levels).double()
        R = np.zeros((X.shape[1], n * n))
        R[:, I] = X.T
        with torch.no_grad():
            return G(torch.from_numpy(R)).numpy()[:, I].T
    return P


def spectrum(Q):
    return np.sort(np.linalg.eigvalsh(Q.T @ Q))[::-1]


for n in ns:
    I = interior(n)
    A = stiffness(n)
    B = stiffness(n, np.random.RandomState(4321).uniform(-1, 1, (n - 1) ** 2))
    P = cycle(n, int(np.log2((n - 1) / 8)) + 1, 0)
    P2 = cycle(n, 2, 4)
    _, _, te = create_datasets(1024, 128, 256, K=10, grid_resolution=n, seed=42, solution="analytic")
    f = te.fs[1].float().view(1, 1, n, n).to(dev)
    for name, load in (("init", False), ("trained", True)):
        Y = tangent(network(n, load), f, I)
        PAY = P(A @ Y)
        losses = {"A": A @ Y / A.diagonal().mean(), "CA": PAY, "CA_mu0.9": PAY + 0.9 * P(B @ PAY), "CA_twogrid": P2(A @ Y)}
        lamD = spectrum(Y)
        U = np.linalg.qr(Y)[0]
        row = {}
        for lname, Q in losses.items():
            r = np.abs(np.log(spectrum(Q) / lamD))
            row[lname] = {"dist100": round(float(r[:100].max()), 3),
                          "kappa_T": round(float(np.linalg.cond(Q @ np.linalg.lstsq(Y, U, rcond=None)[0]) ** 2), 2)}
        r = np.abs(np.log(np.sort(np.linalg.eigvalsh(Y.T @ (A @ Y)))[::-1] / A.diagonal().mean() / lamD))
        row["energy"] = {"dist100": round(float(r[:100].max()), 3), "kappa_T": round(float(np.linalg.cond(U.T @ (A @ U))), 2)}
        print(n, name, row, flush=True)
