Operator Preconditioning
========================

Overview
--------

The ``preconditioning`` module provides multigrid preconditioners and diagnostics for training neural
operators with the preconditioned residual loss

.. math::

   \mathcal L(\theta) = \tfrac12 \lVert P\,(A u_\theta - b)\rVert^2 ,

where :math:`A` is a finite element matrix and :math:`P \approx A^{-1}` is one multigrid cycle. The training
problem is mesh independent when :math:`P` is spectrally equivalent to :math:`A^{-1}` on the range of the
network, which allows rough cycles: few smoothing steps, low quadrature for the coarse operators or half
precision.

Usage Example
-------------

.. code-block:: python

    import torch
    from preconditioning import BMG, VCycle, kappa_pls, stiffness

    A = stiffness(65)
    print(kappa_pls(A, VCycle(A, 65)))

    a = torch.rand(32, 64, 64) + 0.5
    P = BMG(65).setup(a)
    loss = 0.5 * (P(torch.randn(32, 65, 65)) ** 2).sum()

Reproducing the paper
---------------------

The directory ``experiments/bi-parametric-ol`` holds the runs, results and logs of the paper on bi-parametric
operator preconditioning, and ``reproduce.sh`` regenerates every table from them.

Contents
--------

- ``fem``: :math:`Q_1` stiffness and mass matrices on the unit square (``stiffness``, ``mass``), bilinear
  prolongation, log-normal coefficients (``lognormal_coef``).
- ``multigrid``: ``VCycle``, a Galerkin V-cycle for an assembled sparse matrix, as a differentiable ``torch``
  module.
- ``batched``: ``BMG``, a matrix-free V-cycle for :math:`-\nabla\cdot(a\nabla u) + c\,u` with one coefficient
  pair per sample of a batch.
- ``diagnostics``: condition number of the Gauss--Newton matrix (``kappa_pls``), contraction of the cycle
  (``rho2``, ``rhoA``), conditioning on the lowest modes (``sine_basis``, ``gram_kappa``).
- ``darcy``: FNO for the Darcy problem trained with the data, residual or preconditioned residual loss
  (``darcy_dataset``, ``train``).

Tests: ``pytest tests/test_preconditioning.py``.
