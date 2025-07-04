# FAIR Scientific Machine Learning

This repository is dedicated to advancing the principles of **Findable, Accessible, Interoperable, and Reusable (FAIR)** data in the field of scientific machine learning, with a particular focus on solving partial differential equations (PDEs). The project provides modular simulators, machine learning models, and datasets to support reproducible research and collaborative development of PDE-solving algorithms.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Simulators](#simulators)
  - [Base Simulator](#base-simulator)
  - [Poisson Simulator](#poisson-simulator)
  - [Biharmonic Simulator](#biharmonic-simulator)
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

fair-sciml/ │ ├── src/ # Source code for the project │ ├── simulators/ # PDE simulators │ │ ├── base_simulator.py │ │ ├── poisson_simulator.py │ │ └── biharmonic_simulator.py │ └── ml/ # Machine learning models │ ├── deeponet_trainer.py │ └── fno_2d.py ├── docs/ # Documentation files (for ReadTheDocs) │ ├── conf.py │ ├── index.rst │ ├── simulators.rst │ ├── deeponet.rst │ └── fno.rst ├── requirements.txt # Dependencies ├── README.md # Project overview and instructions └── LICENSE # Project license

fair-sciml/
│
├── src/                     # Source code for the project
│   ├── simulators/          # Different PDE simulators
│   │   ├── base_simulator.py
│   │   ├── poisson_simulator.py
│   │   └── biharmonic_simulator.py
│   └── ml/                  # Machine learning models (e.g., DeepONet)
│       ├── deeponet_trainer.py
│       └── fno_2d.py
├── docs/                    # Documentation files (for ReadTheDocs)
│   ├── conf.py
│   ├── index.rst
│   ├── simulators.rst
│   └── deeponet.rst
├── utils/                  # Handles data and metadata
│   ├── h5_handler.py
│   └── metadata.py
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
  - **Input Fields**: Supports the use of `field_input_f` (and others if applicable) for parameterized simulations.

### Neural Operator Training:
- **DeepONet**: Trains dual-architecture branch and trunk networks to learn mappings from field inputs and spatial coordinates to PDE solutions.
- **Fourier Neural Operator (FNO)**: Leverages spectral convolutions for efficient PDE solution approximation. Supports multi-field inputs and adaptive discretizations.

### FAIR Principles:
  - **Data Formats**: Simulation results are stored in HDF5 format with hierarchical organization, metadata, and FAIR compliance.
  - **Findable and Reusable Data**: Datasets can be accessed from local files or cloud platforms (e.g., Hugging Face).
  - **Interoperable Code**: Modular design allows for easy extensions to new PDEs, machine learning models, and data workflows.

---

## **Installation**
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

---

## **Usage**

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

Complete documentation for this projecty is available on [ReadTheDocs](https://fair-sciml.readthedocs.io/)

---

## Cite This Work

If you use this data or code for your research, please cite this GitHub repository:

```bibtex
@misc{fair_sciml2024,
  title   = {FAIR Scientific Machine Learning},
  author  = {Paul Escapil, Eduardo Álvarez and Adolfo Parra},
  year    = {2024},
  url     = {https://github.com/pescap/fair-sciml}
}
```
---

## **Acknowledgments**

- Built with [FEniCS](https://fenicsproject.org/).
- Deep learning with [DeepXDE](https://github.com/lululxvi/deepxde).
- Documentation powered by [ReadTheDocs](https://fair-sciml.readthedocs.io/).
- H5py: For efficient hierarchical data storage. [H5Py](https://docs.h5py.org/en/stable/index.html)
- NeuralOperator: For advanced neural operator architectures such as FNO. [neuraloperator](https://neuraloperator.github.io/dev/).
