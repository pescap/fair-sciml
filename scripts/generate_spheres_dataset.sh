#!/bin/bash
set -e
SHARDS=${SHARDS:-16}
PER_SHARD=${PER_SHARD:-32}
OUT=${OUT:-simulations/spheres}
mkdir -p "$OUT"
for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
        python src/simulators/spheres_helmholtz_simulator.py \
        --num_simulations "$PER_SHARD" --output_directory "$OUT/shard_$i" "$@" \
        > "$OUT/shard_$i.log" 2>&1 &
done
wait
