import ast
from decimal import ROUND_HALF_UP, Decimal
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def result(run):
    p = f"{ROOT}/results/{run}/result.json" if not run.startswith("darcy/") else f"{ROOT}/results/{run}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def err(run, key="test_rl2"):
    r = result(run)
    return None if r is None else 100 * r.get(key, r.get("test_rel", float("nan")))


def epochs_to(run, tol, key="val_rel_l2_errors"):
    r = result(run)
    if r is None:
        return "--"
    v = r["stats"].get(key) or []
    return next((i + 1 for i, x in enumerate(v) if x <= tol), "--")


def log(name):
    rows = {}
    for line in open(f"{ROOT}/logs/{name}"):
        line = re.sub(r"np\.float64\(([^)]*)\)", r"\1", line.strip())
        m = re.match(r"^([0-9.]+(?:\s+[0-9.]+)?)\s+([\[{].*)$", line)
        if m:
            rows[tuple(m.group(1).split())] = ast.literal_eval(m.group(2))
    return rows


def fmt(x, d=2):
    if x is None:
        return "--"
    return x if isinstance(x, str) else str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), ROUND_HALF_UP))


def mean_range(runs):
    v = [e for e in (err(r) for r in runs) if e is not None]
    if not v:
        return "--"
    m = sum(v) / len(v)
    return fmt(m, 1) if len(v) == 1 else f"{fmt(m, 1)} +- {fmt((max(v) - min(v)) / 2, 1)} ({len(v)} seeds)"


def table(title, header, rows):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


N = (33, 65, 129, 257)
seeds = (42, 43, 44)


def tab_rho():
    c = log("contraction.log")
    table("tab:rho", ["n", "s=1 rho_A", "s=1 rho_I", "s=1 kappa", "s=2 rho_A", "s=2 rho_I", "s=2 kappa"],
          [[k[0]] + [v[s][q] for s in (1, 2) for q in ("rhoA", "rho2", "kappa")] for k, v in c.items()])


def tab_h():
    floor = {**{int(k[0]): v["nu0.0"]["floor"] for k, v in log("dichotomy_theory.log").items()},
             **{int(k[0]): v["nu0.0"]["floor"] for k, v in log("dichotomy_theory_smooth.log").items()}}
    table("tab:h", ["n", "D", "CA s=2", "A", "||u_h-u||/||u|| (%)"],
          [[n] + [mean_range([f"hsweep_s{s}/n{n}_{l}" for s in seeds]) for l in ("data", "pls", "galerkin")]
           + [fmt(floor.get(n))] for n in N])


def tab_epochs():
    table("tab:epochs", ["n"] + [f"s{s} {l}" for s in seeds for l in ("D", "CA")],
          [[n] + [f"{epochs_to(f'hsweep_s{s}/n{n}_{l}', 0.2)} / {epochs_to(f'hsweep_s{s}/n{n}_{l}', 0.1)}"
                  for s in seeds for l in ("data", "pls")] for n in N])


def tab_dichotomy():
    fl = {int(k[0]): v for k, v in log("dichotomy_theory_smooth.log").items()}
    kd = {int(k[0]): v for k, v in log("dichotomy_theory.log").items()}
    k16 = {int(k[0]): v for k, v in log("kappa_lowmodes_mu.log").items()}
    rows = [["nu=0"] + sum([[fmt(fl[n]["nu0.0"]["floor"]), "", fmt(err(f"hsweep_s42/n{n}_pls"))] for n in N[:3]], [])]
    for nu in (0.03, 0.1, 0.3):
        rows.append([f"nu={nu}"] + sum([[fmt(fl[n][f"nu{nu}"]["floor"]), "", fmt(err(f"coef/n{n}_pls_nus{nu}"))] for n in N[:3]], []))
    for mu in (0.3, 0.6, 0.9):
        rows.append([f"mu={mu}"] + sum([[fmt(kd[n][f"mu{mu}"]["kappa"], 1), fmt(k16.get(n, {}).get(mu), 1),
                                          fmt(err(f"coef/n{n}_pls_mu{mu}"))] for n in N[:3]], []))
    table("tab:dichotomy (nu rows: floor, -, error; mu rows: kappa_2, kappa_16, error)",
          ["", "n=33", "", "", "n=65", "", "", "n=129", "", ""], rows)


def tab_rough():
    rho_i = {}
    for line in open(f"{ROOT}/logs/bench_precond.log") if os.path.exists(f"{ROOT}/logs/bench_precond.log") else []:
        m = re.match(r"^129 (\S+) (\{.*\})$", line.strip())
        if m:
            rho_i[m.group(1)] = ast.literal_eval(m.group(2))["rhoI"]
    kl = {int(k[0]): v for k, v in log("kappa_lowmodes.log").items()}.get(129, {})
    spec = [("exact inverse", "", ["degraded/n65_pls_exact", "degraded/n129_pls_exact", None], "0", "1"),
            ("s=2", "s2", ["rough/n65_s2", "rough/n129_s2", "rough_hpc/n257_s2"], None, None),
            ("s=1", "s1", ["rough/n65_s1", "rough/n129_s1", "rough_hpc/n257_s1"], None, None),
            ("half precision", "s2_fp16", ["rough/n65_fp16", "rough/n129_fp16", "rough_hpc/n257_fp16"], None, None),
            ("bfloat16 (H100)", "", [None, None, "rough_hpc/n257_bf16"], "--", "--"),
            ("two Gauss points", "s2_ngp2", ["rough/n65_ngp2", "rough/n129_ngp2", "rough_hpc/n257_ngp2"], None, None),
            ("one Gauss point", "s2_ngp1", ["rough/n65_ngp1", "rough/n129_ngp1", None], None, None),
            ("two-grid", "two_grid_jac4", ["rough/n65_twogrid", "rough/n129_twogrid", None], None, None),
            ("no smoothing", "", ["degraded/n65_pls_nu0", "degraded/n129_pls_nu0", None], ">=1", "inf"),
            ("weight W=A", "", ["coef/n65_pls_WA", "coef/n129_pls_WA", None], "--", "--")]
    kmap = {"s2": "s2", "s1": "s1", "s2_ngp2": "s2", "s2_ngp1": "ngp1", "two_grid_jac4": "twogrid"}
    rows = []
    for name, b, runs, r0, k0 in spec:
        rows.append([name, r0 or fmt(rho_i.get(b), 3), k0 or fmt(kl.get(kmap.get(b, "")), 2)] + [fmt(err(r)) if r else "--" for r in runs])
    table("tab:rough (rho_I and kappa_16 at n=129)", ["cycle", "rho_I", "kappa_16", "n=65", "n=129", "n=257"], rows)


def tab_darcy():
    dl = log("cond_darcy_scaled.log")
    table("tab:darcy (condition number of H)", ["sigma, n", "fixed", "fixed + diagonal scaling", "own"],
          [[f"{k[0]}, {k[1]}", fmt(v["mean"], 1), fmt(v["scaled"], 1), fmt(v["own"], 2)] for k, v in dl.items()])


def tab_darcy_train():
    rows = []
    for loss in ("data_fixed", "ls_fixed", "pls_fixed", "pls_fixed_diag", "pls_own"):
        rows.append([loss] + [fmt(err(f"darcy/n{n}_s{s}_{loss}")) for s, n in (("0.5", 65), ("0.5", 129), ("0.5", 257), ("1.0", 65), ("1.0", 129))])
    table("tab:darcy_train", ["loss", "s0.5 n65", "s0.5 n129", "s0.5 n257", "s1.0 n65", "s1.0 n129"], rows)


def tab_ac():
    al = {int(k[0]): {d["k"]: d for d in v} for name in ("ac_lagged_small.log", "ac_lagged.log")
          if os.path.exists(f"{ROOT}/logs/{name}") for k, v in log(name).items()}
    table("tab:ac", ["n", "frozen k=5 rho0", "kappa", "lagged k=5 rho0", "kappa", "frozen k=1 kappa", "lagged k=1 kappa"],
          [[n, v[5]["frozen"][0], v[5]["frozen"][1], v[5]["lagged_vcycle"][0], v[5]["lagged_vcycle"][1],
            v[1]["frozen"][1], v[1]["lagged_vcycle"][1]] for n, v in al.items()])


def tab_ac_train():
    rows = []
    for name, tag in (("D", "ac_hsweep/n{n}_data"), ("A", "ac_hsweep/n{n}_galerkin"), ("CA frozen", "ac_hsweep/n{n}_pls"),
                      ("CA frozen, batched cycle", "ac_bmg_hpc/n{n}_frozen"), ("CA lagged", "ac_bmg_hpc/n{n}_lagged")):
        row = [name]
        for n in (65, 129):
            run = tag.format(n=n)
            r = result(run)
            et = r["stats"].get("epoch_times") if r else None
            row += [fmt(err(run, "test_st_rel_l2")), epochs_to(run, 0.1, "val_errors"), fmt(sum(et) / len(et) if et else None)]
        rows.append(row)
    table("tab:ac_train (space-time error %, epochs to 10%, s/epoch)",
          ["loss", "n=65 error", "epochs", "s/epoch", "n=129 error", "epochs", "s/epoch"], rows)


def tab_stokes():
    rows, cur = [], None
    for line in open(f"{ROOT}/logs/stokes_dual.log"):
        m = re.search(r"hmax=(\S+) free=(\d+).*kappa2\(G_X\)=(\S+)", line)
        if m:
            cur = [m.group(1), m.group(2), m.group(3)]
        m = re.match(r"\s+(B|B_code)\s+kappa2\(H\)=(\S+).*kappaG\(H\)=(\S+?) ", line)
        if m and cur:
            cur += [m.group(2), m.group(3)]
            if m.group(1) == "B_code":
                rows.append(cur)
    table("tab:stokes", ["hmax", "N", "kappa_2(G_X)", "exact kappa_2", "exact kappa_G", "code kappa_2", "code kappa_G"], rows)


def tab_oseen():
    rows = []
    for line in open(f"{ROOT}/logs/oseen_dual.log"):
        m = re.match(r"^(0\.\d+) (\d+) \d+ (\[.*\])$", line.strip())
        if m:
            r = ast.literal_eval(m.group(3))
            rows.append([m.group(1), m.group(2)] + [f"{d['kG_stokes_weight']:.3g} / {d['rho_lagged']:.2f}" for d in r])
    table("tab:oseen (kappa_G with the Stokes weight / rho of the lagged preconditioner)",
          ["h", "N", "U=0", "U=10", "U=30", "U=100", "U=300", "U=1000"], rows)


for t in (tab_rho, tab_h, tab_epochs, tab_dichotomy, tab_rough, tab_darcy, tab_darcy_train, tab_ac, tab_ac_train, tab_stokes, tab_oseen):
    try:
        t()
    except (FileNotFoundError, KeyError) as e:
        print(f"\n## {t.__name__[4:]}: missing input {e}")
