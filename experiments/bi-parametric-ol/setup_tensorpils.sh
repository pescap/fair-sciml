#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dest=${1:-$here/TensorPILS}
git clone -q https://github.com/camlab-ethz/TensorPILS.git "$dest"
git -C "$dest" checkout -q 6bef5ef
python "$here/patches/patch_mg.py" "$dest/tensorpils/preconditioners/multigrid.py"
python "$here/patches/patch_mg_fast.py" "$dest/tensorpils/preconditioners/multigrid.py"
for p in patch_perturb patch_coef patch_cost patch_rough patch_bf16 patch_ac_bmg; do
  python "$here/patches/$p.py" "$dest"
done
pip install --no-deps -e "$dest"
(cd "$dest" && md5sum -c "$here/patches/patched.md5")
