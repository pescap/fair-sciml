import numpy as np
import pytest
import scipy.sparse as sp
import torch

from preconditioning import BMG, VCycle, interior, kappa_pls, lognormal_coef, mass, rho2, stiffness


def power_rho(P, A, mask, its=80):
    v = torch.randn(1, mask.numel(), dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * mask
    for _ in range(its):
        w = v - P(A(v))
        w = w - A(P(w))
        lam = w.norm()
        v = w / lam
    return lam.sqrt().item()


def as_full(V, n):
    I = torch.as_tensor(interior(n))

    def apply(v):
        out = torch.zeros_like(v)
        out[:, I] = V(v[:, I])
        return out
    return apply


def test_stiffness_and_mass_of_a_bubble():
    n = 33
    A, M = stiffness(n), mass(n)
    assert abs(A - A.T).max() < 1e-15 and abs(M - M.T).max() < 1e-15
    x = np.arange(1, n - 1) / (n - 1)
    X, Y = np.meshgrid(x, x)
    u = (X * (1 - X) * Y * (1 - Y)).ravel()
    assert abs(u @ (M @ u) * 900 - 1) < 1e-2
    assert abs(u @ (A @ u) * 45 - 1) < 1e-2


@pytest.mark.parametrize("n", [17, 33])
def test_batched_operator_matches_assembly(n):
    torch.manual_seed(0)
    a = torch.rand(2, n - 1, n - 1, dtype=torch.float64) + 0.5
    c = torch.rand(2, n, n, dtype=torch.float64) * 100
    P = BMG(n).setup(a, c)
    u = torch.randn(2, n, n, dtype=torch.float64) * P.m[0]
    I, h = interior(n), 1 / (n - 1)
    for b in range(2):
        ref = stiffness(n, a[b].numpy().ravel()) + sp.diags(c[b].numpy().ravel()[I] * h * h)
        got = P.op(0, u)[b].flatten()[I].numpy()
        assert np.abs(got - ref @ u[b].flatten()[I].numpy()).max() < 1e-12


def test_batched_cycle_is_symmetric():
    n = 33
    P = BMG(n).setup(torch.rand(2, n - 1, n - 1, dtype=torch.float64) + 0.5)
    m = P.m[0].flatten()
    x, y = torch.randn(2, n * n, dtype=torch.float64) * m, torch.randn(2, n * n, dtype=torch.float64) * m
    assert ((P(x) * y).sum() - (x * P(y)).sum()).abs() < 1e-12 * (P(x) * y).sum().abs()


def test_batched_cycle_equals_galerkin_cycle_for_constant_coefficient():
    n = 65
    one = BMG(n).setup(torch.ones(1, n - 1, n - 1, dtype=torch.float64))
    A = lambda v: one.op(0, v.view(-1, n, n)).flatten(1)
    own = as_full(VCycle(stiffness(n), n), n)
    m = one.m[0].flatten()
    assert abs(power_rho(one, A, m) - power_rho(own, A, m)) < 1e-3
    assert power_rho(one, A, m) < 0.15


def test_batched_cycle_contracts_for_rough_coefficient():
    n = 65
    P = BMG(n).setup(torch.as_tensor(lognormal_coef(n, 0.5, np.random.default_rng(0)))[None])
    assert power_rho(P, lambda v: P.op(0, v.view(-1, n, n)).flatten(1), P.m[0].flatten()) < 0.2


def test_preconditioned_loss_is_mesh_independent():
    kappas = []
    for n in (17, 33, 65):
        A = stiffness(n)
        V = VCycle(A, n)
        assert rho2(A, V) < 0.2
        kappas.append(kappa_pls(A, V))
    assert max(kappas) < 2 and max(kappas) / min(kappas) < 1.2
