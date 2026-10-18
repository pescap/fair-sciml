import sys

import numpy as np

for n in [int(a) for a in sys.argv[1:]]:
    h = 1 / (n - 1)
    t = np.arange(1, n - 1) * np.pi * h
    k, m = (2 / h) * (1 - np.cos(t)), (h / 3) * (2 + np.cos(t))
    lam = np.outer(k, m) + np.outer(m, k)
    print(n, "kappa_2(A^T A) = %.2e" % ((lam.max() / lam.min()) ** 2))
