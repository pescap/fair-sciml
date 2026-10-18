import os
import sys

root = sys.argv[1]

p = f"{root}/tensorpils/physics.py"
s = open(p).read()
old = """            r = r + nu * self._perturbation(u_bc) * u_bc
"""
new = """            r = r + nu * self._perturbation(u_bc) * u_bc
        nuc = float(os.environ.get("TPILS_NUC", "0"))
        if nuc:
            r = r + nuc * self._spmm(self.coef_stiffness(1234, u_bc), u_bc)
        nus = float(os.environ.get("TPILS_NUS", "0"))
        if nus:
            r = r + nus * self._spmm(self.coef_stiffness(0, u_bc), u_bc)
"""
assert old in s
s = s.replace(old, new, 1)
old = """    def _perturbation(self, like):
"""
new = """    def coef_stiffness(self, seed, like):
        cache = getattr(self, "_coef", {})
        self._coef = cache
        if seed not in cache or cache[seed].device != like.device or cache[seed].dtype != like.dtype:
            import numpy as np
            n = int(round(self.n_nodes ** 0.5))
            ke = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0
            iy, ix = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
            k = (iy * n + ix).ravel()
            en = np.stack([k, k + 1, k + n + 1, k + n], axis=1)
            xc = (np.arange(n - 1) + 0.5) / (n - 1)
            smooth = np.outer(np.sin(2 * np.pi * xc), np.sin(2 * np.pi * xc)).ravel()
            xi = smooth if seed == 0 else np.random.RandomState(seed).uniform(-1, 1, len(en))
            rows = np.repeat(en, 4, axis=1).ravel()
            cols = np.tile(en, (1, 4)).ravel()
            vals = (xi[:, None, None] * ke[None]).reshape(len(en), 16).ravel()
            idx = torch.as_tensor(np.stack([rows, cols]))
            B = torch.sparse_coo_tensor(idx, torch.as_tensor(vals), (n * n, n * n)).coalesce()
            cache[seed] = B.to(device=like.device, dtype=like.dtype)
        return cache[seed]

    def _perturbation(self, like):
"""
assert old in s
s = s.replace(old, new, 1)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)

p = f"{root}/tensorpils/losses.py"
s = open(p).read()
old = """            Pr = Pr * (1 + mu * self._xi.to(Pr.device, Pr.dtype))
"""
new = """            Pr = Pr * (1 + mu * self._xi.to(Pr.device, Pr.dtype))
        muc = float(os.environ.get("TPILS_MUC", "0"))
        if muc:
            Bp = self.problem._spmm(self.problem.coef_stiffness(4321, Pr), Pr) * (~self.problem.boundary_mask)
            Pr = Pr + muc * self.precond(Bp)
        if os.environ.get("TPILS_WA"):
            return 0.5 * (Pr * self.problem._spmm(self.problem.A, Pr)).sum(dim=-1).mean()
"""
assert old in s
s = s.replace(old, new, 1)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
