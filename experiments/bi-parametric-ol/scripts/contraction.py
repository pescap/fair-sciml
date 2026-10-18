import sys

from preconditioning import VCycle, kappa_pls, rho2, rhoA, stiffness

for n in [int(a) for a in sys.argv[1:]]:
    A = stiffness(n)
    row = {}
    for s in (1, 2):
        V = VCycle(A, n, nu=s)
        row[s] = {"kappa": round(kappa_pls(A, V), 4), "rho2": round(rho2(A, V), 4), "rhoA": round(rhoA(A, V), 4)}
    print(n, row, flush=True)
