# FEniCS Simulator Setup

This directory contains the **FEniCS simulator components** of the `fair-sciml` project, packaged in a reproducible Docker environment.

## Overview

The FEniCS simulator provides modular PDE solvers for:
- **Poisson Equation**: With parameterized source terms and Neumann boundary conditions
- **Biharmonic Equation**: Using C0 Interior Penalty Galerkin method
- **Helmholtz Equation**: With transmission conditions
- **Base Simulator**: Abstract framework for extensible PDE solvers

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Docker

### Build the Image

```bash
# Build the FEniCS simulator Docker image
./scripts/build_fenics.sh
```

### Run Simulations

```bash
# Run Poisson simulations (default: 10 simulations, mesh size 32)
./scripts/run_simulator.sh poisson

# Run Biharmonic simulations with custom parameters
./scripts/run_simulator.sh biharmonic 5 64 simulations

# Run Helmholtz simulations
./scripts/run_simulator.sh helmholtz 20 16 test_data
```

## Detailed Usage

### Command Line Interface

The `run_simulator.sh` script accepts the following parameters:

```bash
./scripts/run_simulator.sh [simulator_type] [num_simulations] [mesh_size] [output_dir]
```

**Parameters:**
- `simulator_type`: `poisson`, `biharmonic`, or `helmholtz`
- `num_simulations`: Number of simulations to run (default: 10)
- `mesh_size`: Mesh resolution (default: 32)
- `output_dir`: Output directory for HDF5 files (default: simulations)

### Direct Docker Usage

For advanced usage, you can run the container directly:

```bash
# Enter the container for development
docker-compose -f docker-compose.fenics.yml run --rm fenics-simulator

# Run a specific simulation from inside the container
python src/simulators/poisson_simulator.py \
    --source_strength_min 10.0 \
    --source_strength_max 20.0 \
    --neumann_coefficient_min 5.0 \
    --neumann_coefficient_max 10.0 \
    --num_simulations 100 \
    --mesh_size 32 \
    --output_directory simulations
```

## Simulator Details

### Poisson Simulator

Solves the Poisson equation with parameterized source terms:

```python
-∇²u = f  in  Ω
```

**Parameters:**
- `source_strength`: Controls the intensity of the source term
- `neumann_coefficient`: Defines boundary condition coefficient

### Biharmonic Simulator

Solves the Biharmonic equation using discontinuous Galerkin methods:

```python
∇⁴u = f  in  Ω
```

**Parameters:**
- `coefficient`: Controls the coefficient in the source term

### Helmholtz Simulator

Solves the Helmholtz equation with transmission conditions:

```python
-∇²u - k²u = f  in  Ω
```

**Parameters:**
- `wavenumber`: Wave number k
- `ref_ind`: Refractive index
- `direction`: Direction of incident wave

## Output Format

All simulations produce HDF5 files with the following structure:

```
simulation_file.h5
├── session_metadata/
│   ├── num_simulations
│   ├── mesh_size
│   └── timestamp
└── simulations/
    ├── simulation_1/
    │   ├── coordinates
    │   ├── values
    │   ├── field_input_f
    │   ├── field_input_g (if applicable)
    │   └── metadata
    └── simulation_2/
        └── ...
```

## Reproducibility Features

### Environment Consistency

- **Base Image**: Uses official `fenicsproject/stable:latest`
- **Pinned Dependencies**: All Python packages have fixed versions
- **Resource Limits**: CPU and memory limits for consistent performance
- **Thread Control**: Environment variables prevent thread variations

### Data Persistence

- **Volume Mounts**: Source code and data directories are mounted
- **Output Directory**: Simulations are saved to host filesystem
- **Development Mode**: Code changes are reflected immediately

## Development

### Adding New Simulators

1. Create a new simulator class inheriting from `BaseSimulator`
2. Implement the required abstract methods:
   - `_get_equation_name()`
   - `setup_problem()`
   - `solve_problem()`
3. Add the simulator to the `run_simulator.sh` script

### Testing

```bash
# Run tests inside the container
docker-compose -f docker-compose.fenics.yml run --rm fenics-simulator pytest

# Run specific simulator tests
docker-compose -f docker-compose.fenics.yml run --rm fenics-simulator \
    python -m pytest src/simulators/test_poisson_simulator.py
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Increase Docker memory limit to 8GB
2. **Build Failures**: Ensure Docker has enough disk space
3. **Permission Errors**: Check file permissions on mounted volumes

### Performance Optimization

- **Mesh Size**: Smaller meshes run faster but are less accurate
- **Number of Simulations**: Adjust based on available resources
- **Parallel Processing**: The container supports MPI for parallel simulations

## Dependencies

### Core FEniCS Stack
- **DOLFINx**: Main FEniCS library
- **UFL**: Unified Form Language
- **PETSc**: Linear algebra backend
- **MPI4py**: Parallel computing

### Python Dependencies
- **NumPy**: Numerical computing
- **H5py**: HDF5 file handling
- **SciPy**: Scientific computing
- **Matplotlib**: Visualization (optional)

## License

This simulator setup is part of the FAIR Scientific Machine Learning project. See the main LICENSE file for details. 