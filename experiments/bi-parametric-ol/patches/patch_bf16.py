import os
import sys

p = f"{sys.argv[1]}/tensorpils/preconditioners/multigrid.py"
s = open(p).read()
old = """        if os.environ.get("TPILS_MG_FP16"):
            if not getattr(self, "_fp16", False):
                self.half()
                self.__dict__.pop("_csr_cache", None)
                self._fp16 = True
            return self.v_cycle(r.half()).to(r.dtype)
"""
new = """        low = torch.float16 if os.environ.get("TPILS_MG_FP16") else torch.bfloat16 if os.environ.get("TPILS_MG_BF16") else None
        if low is not None:
            if not getattr(self, "_fp16", False):
                self.to(low)
                self.__dict__.pop("_csr_cache", None)
                self._fp16 = True
            return self.v_cycle(r.to(low)).to(r.dtype)
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
