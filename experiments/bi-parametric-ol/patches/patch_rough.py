import os
import sys

root = sys.argv[1]


def patch(p, pairs):
    s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    open(p + ".new", "w").write(s)
    os.replace(p + ".new", p)


patch(f"{root}/tensorpils/preconditioners/multigrid.py", [
    ("""        super().__init__()
        self.n_levels = n_levels
""", """        super().__init__()
        ngp = int(os.environ.get("TPILS_MG_NGP", ngp))
        self.n_levels = n_levels
"""),
    ("""        \"\"\"Apply the preconditioner (one V-cycle). Alias of :meth:`v_cycle`.\"\"\"
        return self.v_cycle(r)
""", """        \"\"\"Apply the preconditioner (one V-cycle). Alias of :meth:`v_cycle`.\"\"\"
        if os.environ.get("TPILS_MG_FP16"):
            if not getattr(self, "_fp16", False):
                self.half()
                self.__dict__.pop("_csr_cache", None)
                self._fp16 = True
            return self.v_cycle(r.half()).to(r.dtype)
        return self.v_cycle(r)
"""),
])
