import numpy as np
import scipy.sparse.linalg as sla
import torch


def as_numpy(V):
    """Wrap a torch preconditioner acting on [B, m] tensors as a function of one numpy vector."""
    def apply(x):
        with torch.no_grad():
            return V(torch.from_numpy(x)[None])[0].numpy()
    return apply


def kappa_pls(A, V):
    """Condition number of the Gauss-Newton matrix A^T V V A of the preconditioned residual loss."""
    m = A.shape[0]
    Vx = as_numpy(V)
    H = sla.LinearOperator((m, m), matvec=lambda x: A.T @ Vx(Vx(A @ x)), dtype=np.float64)
    hi = sla.eigsh(H, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]
    lo = sla.eigsh(H, k=1, which="SA", tol=1e-6, return_eigenvectors=False)[0]
    return float(hi / lo)


def rho2(A, V):
    """Euclidean norm of the error propagator I - V A."""
    m = A.shape[0]
    Vx = as_numpy(V)
    E = lambda x: x - Vx(A @ x)
    Et = lambda y: y - A.T @ Vx(y)
    op = sla.LinearOperator((m, m), matvec=lambda x: Et(E(x)), dtype=np.float64)
    return float(np.sqrt(sla.eigsh(op, k=1, which="LA", tol=1e-6, return_eigenvectors=False)[0]))


def rhoA(A, V):
    """Spectral radius of I - V A, the contraction in the energy norm for symmetric V."""
    m = A.shape[0]
    Vx = as_numpy(V)
    VA = sla.LinearOperator((m, m), matvec=lambda x: Vx(A @ x), dtype=np.float64)
    hi = sla.eigs(VA, k=1, which="LR", tol=1e-6, return_eigenvectors=False)[0].real
    low = sla.eigs(VA, k=1, which="SR", tol=1e-6, return_eigenvectors=False)[0].real
    return float(max(abs(1 - hi), abs(1 - low)))


def sine_basis(n, k):
    """Orthonormal basis of the k x k lowest discrete sine modes on the interior nodes."""
    x = np.arange(1, n - 1) / (n - 1)
    S1 = np.sqrt(2.0 / (n - 1)) * np.sin(np.pi * np.outer(x, np.arange(1, k + 1)))
    return np.kron(S1, S1)


def gram_kappa(Q):
    """Condition number of Q^T Q, the Gauss-Newton matrix restricted to the span of the columns."""
    w = np.linalg.eigvalsh(Q.T @ Q)
    return float(w[-1] / w[0])
