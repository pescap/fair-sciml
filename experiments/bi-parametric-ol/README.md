# Bi-parametric operator preconditioning for physics-informed operator learning

Runs, results and logs of

> P. Escapil-Inchauspé and C. Jerez-Hanckes, *Bi-parametric operator preconditioning for physics-informed
> operator learning*, preprint, 2026.

The preconditioners and diagnostics are the [`preconditioning`](../../src/preconditioning/README.md) module of
this repository. This directory holds what is specific to the paper: the scripts that produce each table, the
configuration of every training run, and their outputs.

## Reproducing the paper

```bash
./reproduce.sh tables                    # every table from the archived results/ and logs/
./reproduce.sh setup                     # dependencies, TensorPILS at 6bef5ef with the patches, md5 check
./reproduce.sh diagnostics               # condition numbers, contractions, floors -> reproduced/logs
./reproduce.sh train 'hsweep_s42/n65_'   # the training runs whose name matches -> reproduced/results
./reproduce.sh tables                    # tables from reproduced/, compared with the archived ones
```

`tables` writes `reproduced/tables.md` and `reproduced/tables_archived.md` and prints their differences; a table
whose inputs have not been reproduced yet is reported as missing. `ONLY="contraction cond_own" ./reproduce.sh
diagnostics` runs a subset of the diagnostics. `python run.py --dry-run` lists every training run with its
command. On a Slurm cluster, `sbatch --export=ALL,RUN='darcy/n129_' scripts/run_one.sbatch` runs a group of
runs on one GPU.

## Contents

| path | content |
|---|---|
| `reproduce.sh` | setup, diagnostics, training runs and tables |
| `run.py` | runs the entries of `runs.json` that match a pattern |
| `make_tables.py` | every table of the paper from a `results/` and `logs/` directory |
| `runs.json` | every training run: name, GPU used, environment variables and command |
| `results/` | the result file of each run (test error, validation curve, time per epoch, memory) |
| `logs/` | the output of every diagnostic |
| `scripts/` | diagnostics, the Darcy trainer, the Stokes and Oseen diagnostics |
| `patches/` | modifications of TensorPILS (see `NOTICE.md`); `patched.md5` holds the checksums of the patched files |

## Tables and their sources

| table | training runs | diagnostics |
|---|---|---|
| contraction of the cycles | -- | `contraction`, `kappa_stiffness` |
| mesh independence, epochs | `hsweep_s42`, `hsweep_s43`, `hsweep_s44` | `dichotomy_theory_smooth` (errors of $u_h$) |
| residual and preconditioner perturbations | `coef/*_nus*`, `coef/*_mu*` | `dichotomy_theory`, `dichotomy_theory_smooth`, `kappa_lowmodes_mu`, `kappa_energy` |
| rough cycles | `rough`, `rough_hpc`, `degraded`, `coef/*_WA` | `bench_precond` (GPU), `kappa_lowmodes` |
| Darcy | `darcy` | `cond_darcy_scaled`, `cond_own` |
| Allen--Cahn | `ac_hsweep`, `ac_bmg_hpc` | `ac_lagged_small`, `ac_lagged` |
| Stokes, Oseen | -- | `stokes_dual`, `oseen_dual`, `oseen_norms` |

The diagnostics that import `tensorpils` (`dichotomy_theory*`, `disc_error`, `kappa_lowmodes`, `bench_precond`,
`ac_lagged*`, `stokes_dual`, `oseen_dual`, `oseen_norms`) measure the multigrid cycle and the problems of TensorPILS; the others
use only the `preconditioning` module.

## Environment

Python 3.11, the versions in `requirements.txt` (CUDA 12.1 to 12.4 builds of torch 2.5.1), and
`PYTHONPATH=<repo>/src`, which `reproduce.sh` and `run.py` set. The runs of TensorPILS execute from the
checkout made by `setup_tensorpils.sh`; `TPILS_OWN=<repo>/src` lets its Allen--Cahn loss import the batched
cycle. GPUs: Quadro RTX 8000 (Poisson up to $n=129$, Allen--Cahn baselines), H100 (Poisson at $n=257$ for the
cheaper cycles, Allen--Cahn with the batched cycle), A100 and A30 MIG (Darcy); `runs.json` records the GPU of
each run. Times per epoch depend on the GPU and on the load of the node.

The archived results were produced before the code moved to `src/preconditioning`. The move kept the
numerics: on the same GPU, the Darcy trainer reproduces the validation curve and the test error of the archived
version bit for bit, the Darcy dataset is identical, and the diagnostics reproduce the archived logs up to the last digits of
the eigenvalue solver (tolerance `1e-6`; the trailing digits change with the number of threads), far below the
precision of the tables. The
seed-42 Poisson sweep predates the later patches, which are inactive without their environment variables; a
rerun of `n65` with the final code reproduces its test error exactly.

## License

Released under the license of the repository. TensorPILS is Apache-2.0; see `NOTICE.md`.
