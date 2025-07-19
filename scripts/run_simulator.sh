#!/bin/bash

# Script to run simulations in the Docker container
# Usage: ./scripts/run_simulator.sh [simulator_type] [parameters]

set -e

# Default values
SIMULATOR_TYPE=${1:-"poisson"}
NUM_SIMULATIONS=${2:-10}
MESH_SIZE=${3:-32}
OUTPUT_DIR=${4:-"simulations"}

# Create necessary directories
mkdir -p simulations
mkdir -p data

echo "Running $SIMULATOR_TYPE simulator with:"
echo "  - Number of simulations: $NUM_SIMULATIONS"
echo "  - Mesh size: $MESH_SIZE"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

# Build and run the container using plain Docker
case $SIMULATOR_TYPE in
    "poisson")
        echo "Running Poisson simulator..."
        docker run --rm \
            -v $(pwd)/src:/app/src:ro \
            -v $(pwd)/simulations:/app/simulations \
            -v $(pwd)/data:/app/data \
            -e OMP_NUM_THREADS=1 \
            -e OPENBLAS_NUM_THREADS=1 \
            -e MKL_NUM_THREADS=1 \
            -e VECLIB_MAXIMUM_THREADS=1 \
            -e NUMEXPR_NUM_THREADS=1 \
            fair-sciml-fenics:latest \
            bash -c "source /dolfinx-env/bin/activate && cd /app && python -c 'import sys; sys.path.append(\"/app/src\"); from simulators.poisson_simulator import PoissonSimulator; import argparse; import numpy as np; from mpi4py import MPI; from dolfinx.mesh import create_unit_square; args = argparse.Namespace(source_strength_min=10.0, source_strength_max=20.0, neumann_coefficient_min=5.0, neumann_coefficient_max=10.0, num_simulations=$NUM_SIMULATIONS, mesh_size=$MESH_SIZE, output_directory=\"$OUTPUT_DIR\"); mesh = create_unit_square(MPI.COMM_WORLD, args.mesh_size, args.mesh_size); simulator = PoissonSimulator(mesh_size=args.mesh_size, output_directory=args.output_directory); parameter_ranges = {\"source_strength\": (args.source_strength_min, args.source_strength_max), \"neumann_coefficient\": (args.neumann_coefficient_min, args.neumann_coefficient_max)}; simulator.run_session(mesh, parameter_ranges, num_simulations=args.num_simulations)'"
        ;;
    "biharmonic")
        echo "Running Biharmonic simulator..."
        docker run --rm \
            -v $(pwd)/src:/app/src:ro \
            -v $(pwd)/simulations:/app/simulations \
            -v $(pwd)/data:/app/data \
            -e OMP_NUM_THREADS=1 \
            -e OPENBLAS_NUM_THREADS=1 \
            -e MKL_NUM_THREADS=1 \
            -e VECLIB_MAXIMUM_THREADS=1 \
            -e NUMEXPR_NUM_THREADS=1 \
            fair-sciml-fenics:latest \
            bash -c "source /dolfinx-env/bin/activate && cd /app && python -c 'import sys; sys.path.append(\"/app/src\"); from simulators.biharmonic_simulator import BiharmonicSimulator; import argparse; import numpy as np; from mpi4py import MPI; from dolfinx.mesh import create_unit_square; args = argparse.Namespace(coefficient_min=1.0, coefficient_max=5.0, num_simulations=$NUM_SIMULATIONS, mesh_size=$MESH_SIZE, output_directory=\"$OUTPUT_DIR\"); mesh = create_unit_square(MPI.COMM_WORLD, args.mesh_size, args.mesh_size); simulator = BiharmonicSimulator(mesh_size=args.mesh_size, output_directory=args.output_directory); parameter_ranges = {\"coefficient\": (args.coefficient_min, args.coefficient_max)}; simulator.run_session(mesh, parameter_ranges, num_simulations=args.num_simulations)'"
        ;;
    "helmholtz")
        echo "Running Helmholtz simulator..."
        docker run --rm \
            -v $(pwd)/src:/app/src:ro \
            -v $(pwd)/simulations:/app/simulations \
            -v $(pwd)/data:/app/data \
            -e OMP_NUM_THREADS=1 \
            -e OPENBLAS_NUM_THREADS=1 \
            -e MKL_NUM_THREADS=1 \
            -e VECLIB_MAXIMUM_THREADS=1 \
            -e NUMEXPR_NUM_THREADS=1 \
            fair-sciml-fenics:latest \
            bash -c "source /dolfinx-env/bin/activate && cd /app && python -c 'import sys; sys.path.append(\"/app/src\"); from simulators.helmholtz_simulator import HelmholtzSimulator; import argparse; import numpy as np; from mpi4py import MPI; from dolfinx.mesh import create_unit_square; args = argparse.Namespace(coefficient_min=1.0, coefficient_max=3.0, num_simulations=$NUM_SIMULATIONS, mesh_size=$MESH_SIZE, output_directory=\"$OUTPUT_DIR\"); mesh = create_unit_square(MPI.COMM_WORLD, args.mesh_size, args.mesh_size); simulator = HelmholtzSimulator(mesh_size=args.mesh_size, output_directory=args.output_directory); parameter_ranges = {\"coefficient\": (args.coefficient_min, args.coefficient_max)}; simulator.run_session(mesh, parameter_ranges, num_simulations=args.num_simulations)'"
        ;;
    *)
        echo "Unknown simulator type: $SIMULATOR_TYPE"
        echo "Available simulators: poisson, biharmonic, helmholtz"
        exit 1
        ;;
esac

echo ""
echo "Simulation completed! Results saved in $OUTPUT_DIR/" 