# Operator preconditioning for physics-informed operator learning

Preconditioners and diagnostics for training neural operators with the preconditioned residual loss

$$\mathcal L(\theta) = \tfrac12 \lVert P\,(A u_\theta - b)\rVert^2 ,$$

where $A$ is a finite element matrix and $P \approx A^{-1}$ is one multigrid cycle. The loss is mesh independent
when $P$ is spectrally equivalent to $A^{-1}$; the cycle may be rough (few smoothing steps, low quadrature, half
precision) as long as that equivalence holds on the range of the network.

| file | content |
|---|---|
| `fem.py` | $Q_1$ stiffness and mass matrices on the unit square, bilinear prolongation, log-normal coefficients |
| `multigrid.py` | `VCycle`: Galerkin V-cycle for an assembled sparse matrix, as a differentiable `torch` module |
| `batched.py` | `BMG`: matrix-free V-cycle for $-\nabla\cdot(a\nabla u) + c\,u$ with one coefficient pair per sample of a batch |
| `diagnostics.py` | condition number of the Gauss-Newton matrix (`kappa_pls`), contraction of the cycle (`rho2`, `rhoA`), conditioning on the lowest modes (`sine_basis`, `gram_kappa`) |
| `darcy.py` | FNO for the Darcy problem trained with the data, residual or preconditioned residual loss |

## Usage

```python
import torch
from preconditioning import BMG, VCycle, kappa_pls, stiffness

A = stiffness(65)
print(kappa_pls(A, VCycle(A, 65)))

a = torch.rand(32, 64, 64) + 0.5
P = BMG(65).setup(a)
r = torch.randn(32, 65, 65)
z = P(r)
```

`P(r)` is differentiable with respect to `r`, so `0.5 * (P(r) ** 2).sum()` can be used directly as a loss.
The batched cycle builds its hierarchy by averaging the coefficients, so a new `BMG(n).setup(a)` per batch
gives a preconditioner adapted to each sample at the cost of one setup.

The experiments of the paper that introduced this module are in
[`experiments/bi-parametric-ol`](../../experiments/bi-parametric-ol/README.md). Tests: `pytest tests/test_preconditioning.py`.
