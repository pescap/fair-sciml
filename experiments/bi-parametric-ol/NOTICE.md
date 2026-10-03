# Third-party code

The training runs use [TensorPILS](https://github.com/camlab-ethz/TensorPILS) (Wen, Mishra and Zeinhofer,
ETH Zurich), Apache License 2.0, at commit `6bef5ef`. This directory does not redistribute TensorPILS. It
contains patch scripts (`patches/`) that modify a fresh checkout of it:

| patch | change |
|---|---|
| `patch_mg.py`, `patch_mg_fast.py` | assemble the multigrid hierarchy in sparse (CSR) format |
| `patch_perturb.py` | nodal perturbations `TPILS_NU`, `TPILS_MU` (not used in the paper) |
| `patch_coef.py` | coefficient perturbations `TPILS_NUC`, `TPILS_NUS`, `TPILS_MUC`, energy weight `TPILS_WA` |
| `patch_cost.py` | Jacobi coarse solve `TPILS_COARSE_SWEEPS`, peak memory in the result file |
| `patch_rough.py`, `patch_bf16.py` | quadrature of the coarse operators `TPILS_MG_NGP`, `TPILS_MG_FP16`, `TPILS_MG_BF16` |
| `patch_ac_bmg.py` | frozen or lagged batched multigrid for Allen--Cahn, `TPILS_AC_BMG`; the cycle is `preconditioning.BMG`, imported from `TPILS_OWN` |

Without these environment variables the patched code behaves as the original, except for the sparse
hierarchy, which reproduces the original cycle to a relative difference of `1e-7`.
