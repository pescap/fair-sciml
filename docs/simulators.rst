Simulators Documentation
========================

Overview
--------

The simulators in this project are built to solve different Partial Differential Equations (PDEs). Each simulator extends the `BaseSimulator` class and implements the problem-specific logic for a particular PDE. Simulation results, including metadata and input-output pairs, are saved in an HDF5 format, making them suitable for training machine learning models like DeepONet and FNO.

Base Simulator
--------------

The `BaseSimulator` class serves as the foundation for all PDE simulators. It implements reusable methods for setting up, running, and saving simulations, ensuring a modular and extensible structure. 

**Key Features:**
- **Metadata Management**: Collects and saves metadata such as simulation parameters, execution time, and hardware information.
- **HDF5 File Handling**: Stores simulation results hierarchically in an HDF5 file.
- **Reusability**: Abstract methods allow flexibility for specific PDE implementations.

**Example Constructor:**

.. code-block:: python

    class BaseSimulator:
        def __init__(self, mesh_size: int = 32, output_path: str = "simulations.h5"):
            self.mesh_size = mesh_size
            self.output_path = output_path

**Key Methods:**
- `setup_problem()`: Abstract method to set up the specific PDE problem.
- `run_simulation()`: Executes a single simulation and stores results.
- `run_session()`: Runs multiple simulations with varying parameters and saves them in the HDF5 format.

Poisson Simulator
-----------------

The `PoissonSimulator` solves the Poisson equation using FEniCS, incorporating Dirichlet and Neumann boundary conditions. It supports parameterization of the source term and Neumann coefficient, enabling dynamic simulations for various configurations.

**Key Parameters:**
- **Source Strength**: Controls the intensity of the source term.
- **Neumann Coefficient**: Defines the boundary condition coefficient.

**Example Usage:**

.. code-block:: python

    from simulators.poisson_simulator import PoissonSimulator

    simulator = PoissonSimulator(mesh_size=32, output_path="poisson_results.h5")
    simulator.run_session(
        parameter_ranges={
            'source_strength': (10.0, 20.0),
            'neumann_coefficient': (5.0, 10.0)
        },
        num_simulations=10
    )

**Output Data:**
- **Field Inputs**: `field_input_f` (source term) and `field_input_g` (Neumann boundary).
- **Solutions**: `values` containing PDE solutions at grid points.
- **Metadata**: Parameter values, execution time, and hardware information.

Biharmonic Simulator
--------------------

The `BiharmonicSimulator` solves the biharmonic equation using a discontinuous Galerkin method. It supports parameterization of the source term for flexible simulation setups.

**Key Features:**
- Implements a penalty-based discontinuous Galerkin formulation.
- Allows parameterization of the source term.

**Key Parameter:**
- **Source Coefficient**: Controls the amplitude of the source term in the PDE.

**Example Usage:**

.. code-block:: python

    from simulators.biharmonic_simulator import BiharmonicSimulator

    simulator = BiharmonicSimulator(mesh_size=32, output_path="biharmonic_results.h5")
    simulator.run_session(
        parameter_ranges={'source_coefficient': (1.0, 5.0)},
        num_simulations=10
    )

**Output Data:**
- **Field Inputs**: `field_input_f` representing the source term.
- **Solutions**: `values` containing the PDE solutions at grid points.
- **Metadata**: Includes source coefficient, execution time, and hardware details.

HDF5 File Structure
-------------------

The simulation results are saved in an HDF5 format with the following hierarchical structure:

.. code-block:: text

    simulations/
    ├── session_1/
    │   ├── simulation_1/
    │   │   ├── coordinates            # Spatial grid coordinates
    │   │   ├── values                 # Solution values at grid points
    │   │   ├── field_inpu             # Input field data
    │   │   └── attributes/            # Metadata
    │   │       ├── paramete
    │   │       ├── mesh_size
    │   │       ├── execution_time
    │   └── attributes/
    │       ├── number_of_simulations
    │       ├── timestamp
    │       └── hardware_info
    ├── session_2/
    │   ...

This structure ensures interoperability and reusability of the data.

Advanced Usage
--------------

### Custom Parameter Ranges
You can easily modify the parameter ranges for any simulator to explore different PDE configurations. For example:

.. code-block:: python

    parameter_ranges = {'source_strength': (5.0, 15.0), 'neumann_coefficient': (3.0, 7.0)}

### Multiple Field Inputs
Both `PoissonSimulator` and `BiharmonicSimulator` support handling multiple input fields. Simply define the fields in the `setup_problem` method and store them using the HDF5 file handler.

Contact
-------

For further assistance or feedback, contact the project maintainers:

- GitHub Issues: https://github.com/pescap/fair-sciml/issues
