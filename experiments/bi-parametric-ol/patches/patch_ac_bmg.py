import os
import sys

root = sys.argv[1]
p = f"{root}/tensorpils/losses.py"
s = open(p).read()
pairs = [
    ("""            if self.precond is not None:
                r = self.precond(r)                                   # ½‖P R‖², P ≈ (a²A+cM)⁻¹
""", """            mode = os.environ.get("TPILS_AC_BMG")
            if mode:
                r = self._bmg(seq_node[:, k].detach(), mode)(r)
            elif self.precond is not None:
                r = self.precond(r)                                   # ½‖P R‖², P ≈ (a²A+cM)⁻¹
"""),
    ("""        return total / (L - 1)
""", """        return total / (L - 1)

    def _bmg(self, u, mode):
        if mode == "frozen" and getattr(self, "_frozen", None) is not None and self._frozen[0] == u.shape[0]:
            return self._frozen[1]
        sys.path.insert(0, os.environ["TPILS_OWN"])
        from preconditioning import BMG
        B, n = u.shape[0], int(round(u.shape[1] ** 0.5))
        a = torch.full((B, n - 1, n - 1), self.a ** 2, dtype=u.dtype, device=u.device)
        w = u.view(B, n, n) ** 2 if mode == "lagged" else torch.ones(B, n, n, dtype=u.dtype, device=u.device)
        P = BMG(n).setup(a, 1 / self.dt + 3 * self.eps ** 2 * w)
        if mode == "frozen":
            self._frozen = (B, P)
        return P
"""),
    ("import os\n", "import os\nimport sys\n"),
]
for old, new in pairs:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
