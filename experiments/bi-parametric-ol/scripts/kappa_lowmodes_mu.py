import sys

import numpy as np
import torch

from preconditioning import VCycle, gram_kappa, sine_basis, stiffness

for n in [int(a) for a in sys.argv[1:]]:
    A = stiffness(n)
    V = VCycle(A, n)
    B = stiffness(n, np.random.RandomState(4321).uniform(-1, 1, (n - 1) ** 2))
    S = sine_basis(n, 16)
    apply = lambda X: V(torch.from_numpy(np.ascontiguousarray(X.T))).detach().numpy().T
    row = {}
    for mu in (0.0, 0.3, 0.6, 0.9):
        Y = apply(A @ S)
        Q = Y + mu * apply(B @ Y)
        row[mu] = round(gram_kappa(Q), 3)
    print(n, row, flush=True)
