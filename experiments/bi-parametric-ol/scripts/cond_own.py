import json

import numpy as np

from preconditioning import VCycle, kappa_pls, lognormal_coef, stiffness

out = {"poisson": {}, "darcy": {}}
for n in [17, 33, 65, 129, 257]:
    A = stiffness(n)
    out["poisson"][n] = kappa_pls(A, VCycle(A, n))
    print("poisson", n, out["poisson"][n], flush=True)

for sigma in [0.5, 1.0]:
    for n in [33, 65, 129, 257]:
        rng = np.random.default_rng(0)
        c = lognormal_coef(n, sigma, rng)
        A = stiffness(n, c)
        V = VCycle(stiffness(n, np.full_like(c, np.exp(np.log(c).mean()))), n)
        out["darcy"][f"{sigma}_{n}"] = {"kappa": kappa_pls(A, V), "contrast": float(c.max() / c.min())}
        print("darcy", sigma, n, out["darcy"][f"{sigma}_{n}"], flush=True)

json.dump(out, open("cond_own.json", "w"), indent=1)
