#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${OUT:-$here/reproduced}
export PYTHONPATH=$here/../../src${PYTHONPATH:+:$PYTHONPATH}

diag() {
  local name=$1
  shift
  [[ -n ${ONLY:-} && " $ONLY " != *" $name "* ]] && return 0
  echo "$name: $*"
  (cd "$out/logs" && python "$here/scripts/$@" > "$name.log" 2>&1)
}

diagnostics() {
  mkdir -p "$out/logs"
  diag contraction contraction.py 33 65 129 257 513
  diag kappa_stiffness kappa_stiffness.py 33 65 129 257 513
  diag cond_own cond_own.py
  diag cond_darcy_scaled cond_darcy_scaled.py
  diag kappa_lowmodes_mu kappa_lowmodes_mu.py 33 65 129
  diag kappa_energy kappa_energy.py 65 129 257
  diag dichotomy_theory_smooth dichotomy_theory.py 0 33 65 129
  diag dichotomy_theory dichotomy_theory.py 1234 33 65 129 257
  diag disc_error disc_error.py 33 65 129 257
  diag kappa_lowmodes kappa_lowmodes.py 65 129 257
  diag ac_lagged_small ac_lagged.py 33 65
  diag ac_lagged ac_lagged.py 129 257
  diag stokes_dual stokes_dual.py 0.07 0.035 0.0175 0.00875
  diag oseen_dual oseen_dual.py 0.07 0.035 0.0175
  diag bench_precond bench_precond.py 129 257 513
}

case ${1:-} in
  setup)
    pip install -r "$here/requirements.txt"
    "$here/setup_tensorpils.sh"
    ;;
  diagnostics)
    diagnostics
    ;;
  train)
    python "$here/run.py" "${2:-.}" --out "$out"
    ;;
  tables)
    python "$here/make_tables.py" "$out" > "$out/tables.md"
    python "$here/make_tables.py" > "$out/tables_archived.md"
    diff "$out/tables_archived.md" "$out/tables.md" && echo "tables match the archived results"
    ;;
  *)
    echo "usage: $0 setup | diagnostics | train [pattern] | tables" >&2
    exit 2
    ;;
esac
