# FAIR Scientific Machine Learning

This repository is dedicated to advancing the principles of **Findable, Accessible, Interoperable, and Reusable (FAIR)** data in the field of scientific machine learning, with a particular focus on solving partial differential equations (PDEs). The project provides modular simulators, machine learning models, and datasets to support reproducible research and collaborative development of PDE-solving algorithms.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Simulators](#simulators)
  - [Base Simulator](#base-simulator)
  - [Poisson Simulator](#poisson-simulator)
  - [Biharmonic Simulator](#biharmonic-simulator)
  - [Helmholtz Simulator](#helmholtz-simulator)
- [Neural Operators](#neural-operators)
  - [DeepONet Training](#deeponet-training)
  - [Fourier Neural Operator (FNO) Training](#fourier-neural-operator-fno-training)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Structure

```bash
fair-sciml/
│
├── src/                     # Source code for the project
│   ├── simulators/          # Different PDE simulators
│   │   ├── base_simulator.py
│   │   ├── poisson_simulator.py
│   │   ├── biharmonic_simulator.py
│   │   ├── helmholtz_simulator.py
│   │   └── helmholtz_transmission_simulator.py
│   └── ml/                  # Machine learning models (e.g., DeepONet)
│       ├── deeponet_trainer.py
│       └── fno_2d.py
├── docs/                    # Documentation files (for ReadTheDocs)
│   ├── conf.py
│   ├── index.rst
│   ├── simulators.rst
│   └── deeponet.rst
├── utils/                   # Handles data and metadata
│   ├── h5_handler.py
│   └── metadata.py
├── scripts/                 # Docker automation scripts
│   ├── build_fenics.sh
│   ├── run_simulator.sh
│   └── test_simulator.sh
├── Dockerfile.fenics        # FEniCS simulator Docker image
├── requirements-simulator.txt # FEniCS dependencies
├── Makefile                 # Convenient commands
├── requirements.txt         # List of dependencies
├── README.md                # Project overview and instructions (this file)
└── LICENSE                  # Project license
```

---

## **Features**

### Modular PDE Simulators:
  - **BaseSimulator**: Provides a framework for reusable simulation logic, including metadata collection and HDF5 data handling.
  - **Poisson Simulator**: Solves the Poisson equation with parameterized source strength and Neumann boundary conditions.
  - **Biharmonic Simulator**: Solves the Biharmonic equation using a C0 Interior Penalty Galerkin method with continuous Lagrange elements with parameterized coefficients.
  - **Helmholtz Simulator**: Solves the Helmholtz equation with parameterized coefficients and boundary conditions.
  - **Helmholtz Transmission Simulator**: Advanced Helmholtz solver with transmission conditions and analytical solutions.
  - **Spheres Helmholtz Simulator**: Helmholtz transmission by arrays of many disjoint spheres in 3D, solved with the multiple traces formulation of [biosspheres](https://github.com/iamartineza/biosspheres). No mesh, no FEniCS; one sphere has the Mie series as analytical solution.
  - **Input Fields**: Supports the use of `field_input_f` (and others if applicable) for parameterized simulations.

### Neural Operator Training:
- **DeepONet**: Trains dual-architecture branch and trunk networks to learn mappings from field inputs and spatial coordinates to PDE solutions.
- **Fourier Neural Operator (FNO)**: Leverages spectral convolutions for efficient PDE solution approximation. Supports multi-field inputs and adaptive discretizations.

### FAIR Principles:
  - **Data Formats**: Simulation results are stored in HDF5 format with hierarchical organization, metadata, and FAIR compliance.
  - **Findable and Reusable Data**: Datasets can be accessed from local files or cloud platforms (e.g., Hugging Face).
  - **Interoperable Code**: Modular design allows for easy extensions to new PDEs, machine learning models, and data workflows.
  - **Reproducible Environment**: Docker containers ensure consistent execution across different platforms.

---

## **Installation**

### Option 1: Local Installation
1. Clone the repository:
```bash
git clone https://github.com/pescap/fair-sciml.git
cd fair-sciml
```

2. Create a virtual environment (recommended)
```bash
python3 -m venv env
source env/bin/activate  # On Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Setup (Recommended for FEniCS Simulators)
The FEniCS simulators are containerized for full reproducibility. See the [Docker Setup](#docker-setup) section below.

---

## **Docker Setup**

The FEniCS simulators are packaged in a fully reproducible Docker environment using DOLFINx (the modern FEniCS version).

### Quick Start
```bash
# Build the Docker image
make build

# Run tests to verify everything works
make test

# Run a simulation
./scripts/run_simulator.sh poisson 10 32 simulations
```

### Available Commands
```bash
# Build and test
make build          # Build Docker image
make test           # Run comprehensive tests
make clean          # Clean Docker resources

# Run simulations
make poisson        # Run Poisson simulator
make biharmonic     # Run Biharmonic simulator  
make helmholtz      # Run Helmholtz simulator

# Development
make shell          # Enter container for development
```

### Manual Commands
```bash
# Build image
./scripts/build_fenics.sh

# Run simulations
./scripts/run_simulator.sh [simulator_type] [num_simulations] [mesh_size] [output_dir]

# Examples
./scripts/run_simulator.sh poisson 10 32 simulations
./scripts/run_simulator.sh biharmonic 5 64 data
./scripts/run_simulator.sh helmholtz 20 16 test_data

# Test the setup
./scripts/test_simulator.sh
```

### Features
- **Fully Reproducible**: Pinned dependencies and consistent environment
- **DOLFINx Integration**: Modern FEniCS with proper environment activation
- **HDF5 Output**: Structured data storage for ML training
- **Multiple Simulators**: Poisson, Biharmonic, Helmholtz
- **Resource Control**: Thread limits for reproducible performance
- **Rich Metadata**: Hardware info, execution time, parameters

For detailed Docker documentation, see [README-FENICS.md](README-FENICS.md).

---

## **Usage**

### Local Usage
Run a Poisson simulation: 
```bash
python3 src/simulators/poisson_simulator.py \
    --source_strength_min 10.0 \
    --source_strength_max 20.0 \
    --neumann_coefficient_min 5.0 \
    --neumann_coefficient_max 10.0 \
    --num_simulations 100 \
    --mesh_size 32 \
    --output_path "poisson_results.h5"
```

Run a Biharmonic simulation:
```bash
python3 src/simulators/biharmonic_simulator.py \
    --coefficient_min 1.0 \
    --coefficient_max 5.0 \
    --num_simulations 50 \
    --mesh_size 32 \
    --output_path "biharmonic_results.h5"
```

### Docker Usage
```bash
# Quick simulation runs
./scripts/run_simulator.sh poisson 10 32 simulations
./scripts/run_simulator.sh biharmonic 5 64 data
./scripts/run_simulator.sh helmholtz 20 16 test_data
```

---

## **Simulators**

### **Base Simulator**

The `BaseSimulator` defines reusable logic for running PDE simulations, handling metadata, and storing results in a structured HDF5 format.

### **Poisson Simulator**

The `PoissonSimulator` solves the Poisson equation. Key features include:

  - Parameterized source term (`source_strength`).
  - Neumann boundary conditions (`neumann_coefficient`).
  - Input field values (`field_input_f`, `field_input_g`) generated dynamically.

### **Biharmonic Simulator**

The `BiharmonicSimulator` solves the Biharmonic equation using a Discontinuous Galerkin method. Supports parameterized coefficients for flexibility in simulation generation.

### **Helmholtz Simulator**

The `HelmholtzSimulator` solves the Helmholtz equation with parameterized coefficients and boundary conditions.

### **Helmholtz Transmission Simulator**

The `HelmholtzTransmissionSimulator` provides advanced Helmholtz solving capabilities with transmission conditions and analytical solutions.

### **Spheres Helmholtz Simulator**

The `SpheresHelmholtzSimulator` solves the scattering of a plane wave by a cubic lattice of `n_side**3` disjoint spheres with random radii and positions, using [biosspheres](https://github.com/iamartineza/biosspheres) (spherical harmonics up to degree `big_l`, local multiple traces formulation). Each simulation samples the wavenumber, the refractive index of the spheres, the direction of incidence and the geometry. It stores:

  - `coordinates`, `values`: the real part of the total field on the plane `y = 0`, and `field_values_imag` its imaginary part.
  - `field_input_k`: the local wavenumber on a coarse 3D grid of sensors, which encodes geometry and material and is the branch input of the DeepONet.
  - `field_centers`, `field_radii`: the geometry.

With `--analytical` it uses the Mie series for one sphere instead of solving. It needs `pip install biosspheres` only (no FEniCS). To generate a dataset in parallel shards and train a DeepONet on it:

```bash
SHARDS=16 PER_SHARD=32 ./scripts/generate_spheres_dataset.sh --n_side 3 --big_l 8
PYTHONPATH=src DDE_BACKEND=pytorch python src/ml/train_spheres_deeponet.py --data "simulations/spheres/*/*.h5"
```

---

## **DeepONet Training**

The `DeepONetTrainer` trains a **Deep Operator Network (DeepONet)** using data generated by the PDE simulators. The trainer supports:

  - **Branch network**: Encodes `field_input_f` values from simulations.
  - **Trunk network**: Encodes spatial coordinates for PDE solutions.
  - **Output mapping**: Maps combined branch and trunk inputs to the solution space.

## **Fourier Neural Operator (FNO) Training**

The `FNOTrainer` trains Fourier Neural Operators using spectral convolutions. Features include:

  - Multi-field support for complex PDEs.
  - Efficient learning of high-dimensional solution mappings.
  - Robust to varying resolutions and discretizations.

---

## **Documentation**

Complete documentation for this project is available on [ReadTheDocs](https://fair-sciml.readthedocs.io/)

For detailed FEniCS simulator Docker setup documentation, see [README-FENICS.md](README-FENICS.md).

---

## Cite This Work

If you use this data or code for your research, please cite this GitHub repository:

```bibtex
@misc{fair_sciml2024,
  title   = {FAIR Scientific Machine Learning},
  author  = {Paul Escapil and Eduardo Álvarez and Adolfo Parra and Vicente Iligaray and Vicente Opazo and Danilo Aballay},
  year    = {2024},
  url     = {https://github.com/pescap/fair-sciml}
}
```
---

## **Acknowledgments**

- Built with [FEniCS](https://fenicsproject.org/) and [DOLFINx](https://docs.fenicsproject.org/dolfinx/).
- Deep learning with [DeepXDE](https://github.com/lululxvi/deepxde).
- Scattering by arrays of spheres with [biosspheres](https://github.com/iamartineza/biosspheres) (Martínez-Ávila, Jerez-Hanckes, Escapil-Inchauspé, Gebäck). If you use the spheres dataset, please also cite biosspheres (see its `CITATION.cff`).
- Documentation powered by [ReadTheDocs](https://fair-sciml.readthedocs.io/).
- H5py: For efficient hierarchical data storage. [H5Py](https://docs.h5py.org/en/stable/index.html)
- NeuralOperator: For advanced neural operator architectures such as FNO. [neuraloperator](https://neuraloperator.github.io/dev/).
