import glob
import os
import sys

import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from preconditioning import gram_kappa, interior, sine_basis, stiffness
from tensorpils.cli import _build_model, build_parser
from tensorpils.data import create_datasets
from tensorpils.models import FNOModel
from tensorpils.preconditioners.multigrid import GeometricMultigrid

runs, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
arch = sys.argv[4] if len(sys.argv) > 4 else "fno"
dev = "cuda"
L = int(np.log2((n - 1) / 8)) + 1
variants = {
    "s2": ("rough/n{n}_s2", dict(n_levels=L), 0, 0.0),
    "s1": ("rough/n{n}_s1", dict(n_levels=L, pre_smooth=1, post_smooth=1), 0, 0.0),
    "ngp1": ("rough/n{n}_ngp1", dict(n_levels=L, ngp=1), 0, 0.0),
    "twogrid": ("rough/n{n}_twogrid", dict(n_levels=2), 4, 0.0),
    "s0": ("degraded/n{n}_pls_nu0", dict(n_levels=L, pre_smooth=0, post_smooth=0), 0, 0.0),
    "mu0.3": ("coef/n{n}_pls_mu0.3", dict(n_levels=L), 0, 0.3),
    "mu0.6": ("coef/n{n}_pls_mu0.6", dict(n_levels=L), 0, 0.6),
    "mu0.9": ("coef/n{n}_pls_mu0.9", dict(n_levels=L), 0, 0.9),
}

tr, _, te = create_datasets(1024, 128, 256, K=10, grid_resolution=n, seed=42, solution="analytic")
f = te.fs[0].float().view(1, 1, n, n).to(dev)
I = interior(n)
A = stiffness(n)
B = stiffness(n, np.random.RandomState(4321).uniform(-1, 1, (n - 1) ** 2))
cfg = dict(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1, n_layers=5)


def build():
    if arch == "fno":
        return FNOModel(**cfg)
    os.environ.setdefault("TPILS_DEEPONET_SENSORS", "33")
    args = build_parser().parse_args(["--model", arch, "--grid_resolution", str(n), "--mg_levels", str(L)])
    return _build_model(args, 1, train_ds=tr)


def network(run):
    torch.manual_seed(42)
    model = build().to(dev)
    if run:
        run = run if arch == "fno" else f"{arch}/" + run.replace("rough/n{n}_s2", "hsweep_s42/n{n}_pls")
        path = glob.glob(f"{runs}/{run.format(n=n)}/checkpoints/*_best.pth")[0]
        state = torch.load(path, map_location=dev, weights_only=False)["model_state_dict"]
        state.pop("_metadata", None)
        model.load_state_dict(state)
    return model.eval()


def tangent(model):
    params = {name: p.detach() for name, p in model.named_parameters()}
    g = torch.Generator(device=dev).manual_seed(0)
    cols = []
    for _ in range(k):
        v = {name: torch.randn(p.shape, dtype=p.dtype, generator=g, device=dev) for name, p in params.items()}
        with sdpa_kernel(SDPBackend.MATH):
            _, t = jvp(lambda q: functional_call(model, q, (f,)), (params,), (v,))
        cols.append(t.flatten().double().cpu().numpy()[I])
    return np.linalg.qr(np.stack(cols, 1))[0]


def operator(kw, sweeps, mu):
    os.environ["TPILS_COARSE_SWEEPS"] = str(sweeps)
    G = GeometricMultigrid(n, n, omega=8 / 9, **kw).double()

    def P(X):
        R = np.zeros((X.shape[1], n * n))
        R[:, I] = X.T
        with torch.no_grad():
            return G(torch.from_numpy(R)).numpy()[:, I].T

    return lambda U: (lambda Y: Y + mu * P(B @ Y))(P(A @ U))


S = sine_basis(n, 16)
U0 = tangent(network(None))
for name, (run, kw, sweeps, mu) in variants.items():
    PA = operator(kw, sweeps, mu)
    Ut = tangent(network(run))
    print(n, name, {"kappa16": round(gram_kappa(PA(S)), 3), "tangent_init": round(gram_kappa(PA(U0)), 3),
                    "tangent_trained": round(gram_kappa(PA(Ut)), 3)}, flush=True)
