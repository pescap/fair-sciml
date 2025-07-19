#!/bin/bash

# Script to test the FEniCS simulator setup
# Usage: ./scripts/test_simulator.sh

set -e

echo "Testing FEniCS simulator setup..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if the image exists
if ! docker images | grep -q "fair-sciml-fenics"; then
    echo "Building FEniCS simulator image..."
    ./scripts/build_fenics.sh
fi

echo "Running basic tests..."

# Test 1: Check if container can start
echo "Test 1: Container startup..."
docker run --rm \
    -v $(pwd)/src:/app/src:ro \
    fair-sciml-fenics:latest \
    bash -c "source /dolfinx-env/bin/activate && python -c 'import dolfinx; print(\"✓ DOLFINx imported successfully\")'"

# Test 2: Check if all simulators can be imported
echo "Test 2: Simulator imports..."
docker run --rm \
    -v $(pwd)/src:/app/src:ro \
    fair-sciml-fenics:latest \
    bash -c "source /dolfinx-env/bin/activate && cd /app && python -c '
import sys
sys.path.append(\"/app/src\")
from simulators.base_simulator import BaseSimulator
from simulators.poisson_simulator import PoissonSimulator
from simulators.biharmonic_simulator import BiharmonicSimulator
from simulators.helmholtz_simulator import HelmholtzSimulator
print(\"✓ All simulators imported successfully\")
'"

# Test 3: Run a small Poisson simulation
echo "Test 3: Small Poisson simulation..."
mkdir -p test_simulations
docker run --rm \
    -v $(pwd)/src:/app/src:ro \
    -v $(pwd)/test_simulations:/app/test_simulations \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e VECLIB_MAXIMUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    fair-sciml-fenics:latest \
    bash -c "source /dolfinx-env/bin/activate && cd /app && python -c '
import sys
sys.path.append(\"/app/src\")
from simulators.poisson_simulator import PoissonSimulator
import argparse
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_unit_square

# Create a simple test
mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
simulator = PoissonSimulator(mesh_size=8, output_directory=\"test_simulations\")
parameter_ranges = {\"source_strength\": (1.0, 2.0), \"neumann_coefficient\": (1.0, 2.0)}
simulator.run_session(mesh, parameter_ranges, num_simulations=1)
print(\"✓ Poisson simulation completed successfully\")
'"

# Test 4: Check if output file was created
if [ -f "test_simulations/poisson_equation.h5" ]; then
    echo "✓ Poisson simulation completed successfully"
else
    echo "✗ Poisson simulation failed - no output file found"
    exit 1
fi

# Test 5: Check HDF5 file structure
echo "Test 4: HDF5 file structure..."
docker run --rm \
    -v $(pwd)/test_simulations:/app/test_simulations \
    fair-sciml-fenics:latest \
    bash -c "source /dolfinx-env/bin/activate && python -c '
import h5py
with h5py.File(\"test_simulations/poisson_equation.h5\", \"r\") as f:
    print(\"✓ HDF5 file structure:\")
    print(\"  - Number of sessions:\", len(f.keys()))
    print(\"  - Session keys:\", list(f.keys())[:2])  # Show first 2 session keys
    # Check if any session has simulations
    has_simulations = any(\"simulation_\" in str(key) for session in f.values() for key in session.keys())
    print(\"  - Contains simulations:\", has_simulations)
'"

# Cleanup
echo ""
echo "Cleaning up test files..."
rm -rf test_simulations

echo ""
echo "✓ All tests passed! The FEniCS simulator setup is working correctly."
echo ""
echo "You can now run simulations using:"
echo "  ./scripts/run_simulator.sh [simulator_type]" 