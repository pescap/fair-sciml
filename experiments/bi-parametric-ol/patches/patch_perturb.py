import sys

root = sys.argv[1]

p = f"{root}/tensorpils/physics.py"
s = open(p).read()
old = """        u_bc = apply_zero_boundary(u, self.boundary_mask)
        r = self._spmm(self.A, u_bc) - self.load_vector(f)
        return apply_zero_boundary(r, self.boundary_mask)
"""
new = """        u_bc = apply_zero_boundary(u, self.boundary_mask)
        r = self._spmm(self.A, u_bc) - self.load_vector(f)
        nu = float(os.environ.get("TPILS_NU", "0"))
        if nu:
            r = r + nu * self._perturbation(u_bc) * u_bc
        return apply_zero_boundary(r, self.boundary_mask)

    def _perturbation(self, like):
        if getattr(self, "_pert", None) is None or self._pert.device != like.device:
            d = torch.as_tensor(self.A.to_scipy_coo().diagonal(), dtype=like.dtype)
            g = torch.Generator().manual_seed(1234)
            xi = 2 * torch.rand(d.shape[0], generator=g, dtype=like.dtype) - 1
            self._pert = (d * xi).to(like.device)
        return self._pert
"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("import torch\nimport torch.nn as nn\n", "import os\n\nimport torch\nimport torch.nn as nn\n", 1)
open(p, "w").write(s)

p = f"{root}/tensorpils/losses.py"
s = open(p).read()
old = """        Pr = self.precond(r)                    # ≈ A⁻¹ r
"""
new = """        Pr = self.precond(r)                    # ≈ A⁻¹ r
        mu = float(os.environ.get("TPILS_MU", "0"))
        if mu:
            if getattr(self, "_xi", None) is None or self._xi.shape[-1] != Pr.shape[-1]:
                g = torch.Generator().manual_seed(4321)
                self._xi = 2 * torch.rand(Pr.shape[-1], generator=g) - 1
            Pr = Pr * (1 + mu * self._xi.to(Pr.device, Pr.dtype))
"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("from typing import Optional\n", "import os\nfrom typing import Optional\n", 1)
open(p, "w").write(s)
