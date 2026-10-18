import sys
from pathlib import Path

path = Path(sys.argv[1]) / "tensorpils" / "losses.py"
src = path.read_text()
old = """        residual = self.problem.residual(u_pred_node, f_node)
        return 0.5 * (residual ** 2).sum(dim=-1).mean()
"""
new = """        residual = self.problem.residual(u_pred_node, f_node)
        if os.environ.get("TPILS_ENERGY"):
            minus_b = self.problem.residual(torch.zeros_like(u_pred_node), f_node)
            return 0.5 * (u_pred_node * (residual + minus_b)).sum(dim=-1).mean()
        return 0.5 * (residual ** 2).sum(dim=-1).mean()
"""
if "TPILS_ENERGY" not in src:
    assert old in src
    src = src.replace(old, new)
path.write_text(src)
