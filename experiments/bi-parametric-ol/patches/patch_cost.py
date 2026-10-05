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
    ("""            if L == n_levels - 1:
                self.register_buffer("A_inv_coarsest\"""",
     """            if L == n_levels - 1 and not int(os.environ.get("TPILS_COARSE_SWEEPS", "0")):
                self.register_buffer("A_inv_coarsest\""""),
    ("""        if level == self.n_levels - 1:
            return r @ self.A_inv_coarsest.T""",
     """        if level == self.n_levels - 1:
            sweeps = int(os.environ.get("TPILS_COARSE_SWEEPS", "0"))
            if sweeps:
                return self._smooth(level, getattr(self, f"diag_{level}"), torch.zeros_like(r), r, sweeps)
            return r @ self.A_inv_coarsest.T"""),
    ("import numpy as np\n", "import os\n\nimport numpy as np\n"),
])

patch(f"{root}/tensorpils/trainer.py", [
    ("""            "test_mse": test_mse, "test_l2": test_l2, "test_rl2": test_rl2,
            "stats": asdict(self.stats),
        }
        path = f"{self.output_dir}/results/{self._file_prefix()}.json\"""",
     """            "test_mse": test_mse, "test_l2": test_l2, "test_rl2": test_rl2,
            "stats": asdict(self.stats),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        }
        path = f"{self.output_dir}/results/{self._file_prefix()}.json\""""),
])
