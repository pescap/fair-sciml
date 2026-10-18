import sys

path = sys.argv[1]
src = open(path).read()

helper = '''class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, St, x):
        ctx.St = St
        return torch.sparse.mm(S, x.T).T

    @staticmethod
    def backward(ctx, g):
        return None, None, torch.sparse.mm(ctx.St, g.T.contiguous()).T


class GeometricMultigrid(Preconditioner):'''
src = src.replace("class GeometricMultigrid(Preconditioner):", helper, 1)

methods = '''    def _csr(self, kind: str, L: int):
        cache = self.__dict__.setdefault("_csr_cache", {})
        dev = getattr(self, f"diag_{L}").device
        key = (kind, L, dev)
        if key not in cache:
            build = {"A": self._A_sparse, "P": self._P_sparse, "R": self._R_sparse}[kind]
            cache[key] = build(L).coalesce().to_sparse_csr()
        return cache[key]

    def _mm(self, kind: str, L: int, x: torch.Tensor) -> torch.Tensor:
        tr = {"A": "A", "P": "R", "R": "P"}[kind]
        return _SpMM.apply(self._csr(kind, L), self._csr(tr, L), x.contiguous())

    # -------------------- V-cycle building blocks --------------------'''
src = src.replace("    # -------------------- V-cycle building blocks --------------------", methods, 1)

old_smooth = '''    def _smooth(self, A_sp, diag, e, r, n_iter: int):
        """Weighted Jacobi: ``e ← e + ω D⁻¹ (r − A e)``. Batched ``[B, N]``."""
        for _ in range(n_iter):
            Ae = torch.sparse.mm(A_sp, e.T).T
            e = e + self.omega * (r - Ae) / diag.unsqueeze(0)
        return e'''
new_smooth = '''    def _smooth(self, level, diag, e, r, n_iter: int):
        """Weighted Jacobi: ``e ← e + ω D⁻¹ (r − A e)``. Batched ``[B, N]``."""
        for _ in range(n_iter):
            Ae = self._mm("A", level, e)
            e = e + self.omega * (r - Ae) / diag.unsqueeze(0)
        return e'''
assert old_smooth in src
src = src.replace(old_smooth, new_smooth, 1)

old_rec = src[src.index("        A_sp = self._A_sparse(level)\n        diag = getattr(self, f\"diag_{level}\")"):]
new_rec = '''        diag = getattr(self, f"diag_{level}")

        e = torch.zeros_like(r)
        e = self._smooth(level, diag, e, r, self.pre_smooth)

        res = r - self._mm("A", level, e)
        res_c = self._mm("R", level, res)
        e_c = self._v_cycle_rec(level + 1, res_c)
        e = e + self._mm("P", level, e_c)
        e = self._smooth(level, diag, e, r, self.post_smooth)
        return e
'''
src = src.replace(old_rec, new_rec, 1)
open(path, "w").write(src)
