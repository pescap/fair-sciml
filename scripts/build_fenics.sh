#!/bin/bash

# Script to build the FEniCS Docker image
# Usage: ./scripts/build_fenics.sh

set -e

echo "Building FEniCS simulator Docker image..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the image using plain Docker
echo "Building image with tag: fair-sciml-fenics:latest"
docker build -f Dockerfile.fenics -t fair-sciml-fenics:latest .

echo ""
echo "Build completed successfully!"
echo ""
echo "To run a simulation, use:"
echo "  ./scripts/run_simulator.sh [simulator_type] [num_simulations] [mesh_size] [output_dir]"
echo ""
echo "Examples:"
echo "  ./scripts/run_simulator.sh poisson 10 32 simulations"
echo "  ./scripts/run_simulator.sh biharmonic 5 64 data"
echo "  ./scripts/run_simulator.sh helmholtz 20 16 test_simulations"
echo ""
echo "To enter the container for development:"
echo "  docker run -it --rm -v \$(pwd)/src:/app/src:ro -v \$(pwd)/utils:/app/utils:ro -v \$(pwd)/simulations:/app/simulations -v \$(pwd)/data:/app/data fair-sciml-fenics:latest" 