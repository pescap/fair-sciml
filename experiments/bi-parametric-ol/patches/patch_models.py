import sys
from pathlib import Path

path = Path(sys.argv[1]) / "tensorpils" / "cli.py"
src = path.read_text()
if "\nimport os\n" not in src:
    src = src.replace("\nimport numpy as np\n", "\nimport os\n\nimport numpy as np\n", 1)
old = """    if args.model == "deeponet" and args.loss != "pideeponet":"""
src = src.replace(old, """    if not os.environ.get("TPILS_DEEPONET_ANY") and args.model == "deeponet" and args.loss != "pideeponet":""")
old = """        if args.model == "gaot":
            raise SystemExit("--pde ac is run with --model fno (or deeponet for pideeponet).")"""
src = src.replace(old, old.replace('if args.model == "gaot":', 'if not os.environ.get("TPILS_GAOT_AC") and args.model == "gaot":'))
old = "        model = DeepONetModel(**cfg)\n"
new = """        sensors = int(os.environ.get("TPILS_DEEPONET_SENSORS", 0))
        model = DeepONetModel(**(cfg | dict(grid_size=(sensors, sensors)) if sensors else cfg))
        if sensors:
            model = SensorSubsample(model, nx, sensors)
"""
if "TPILS_DEEPONET_SENSORS" not in src:
    src = src.replace(old, new)
    src = src.replace("\ndef build_parser", '''
class SensorSubsample(torch.nn.Module):
    def __init__(self, net, n, sensors):
        super().__init__()
        self.net, self.stride = net, (n - 1) // (sensors - 1)
        x = torch.linspace(0.0, 1.0, n)
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        self.register_buffer("coords", torch.stack([xx.reshape(-1), yy.reshape(-1)], -1))
        self.n = n

    def forward(self, f):
        u = self.net.forward_at(f[..., ::self.stride, ::self.stride], self.coords)
        return u.reshape(-1, 1, self.n, self.n)


def build_parser''', 1)
for key in ("TPILS_DEEPONET_ANY", "TPILS_GAOT_AC", "TPILS_DEEPONET_SENSORS", "class SensorSubsample", "\nimport os\n", "\ndef build_parser"):
    assert key in src, key
path.write_text(src)
