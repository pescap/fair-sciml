Simulators Documentation
========================

Overview
--------

The simulators in this project are built to solve different Partial Differential Equations (PDEs).
Each simulator extends the `BaseSimulator` class and implements the problem-specific logic.

Base Simulator
--------------

The `BaseSimulator` class serves as the foundation for all simulators. It defines common methods
for setting up and running simulations.

**Example: `BaseSimulator` constructor**

::

    class BaseSimulator:
        def __init__(self, mesh_size: int = 32, output_path: str = "simulations.h5"):
            self.mesh_size = mesh_size
            self.output_path = output_path

Poisson Simulator
-----------------

The `PoissonSimulator` class solves the Poisson equation using FEniCS. Below is a usage example:

**Example usage:**

::

    from simulators.poisson_simulator import PoissonSimulator

    simulator = PoissonSimulator(mesh_size=32, output_path="poisson_results.h5")
    simulator.run_session(
        parameter_ranges={'source_strength': (10.0, 20.0), 'neumann_coefficient': (5.0, 10.0)},
        num_simulations=5
    )

Biharmonic Simulator
--------------------

The `BiharmonicSimulator` solves the biharmonic equation using a discontinuous Galerkin method.

**Example usage:**

::

    from simulators.biharmonic_simulator import BiharmonicSimulator

    simulator = BiharmonicSimulator(mesh_size=32, output_path="biharmonic_results.h5")
    simulator.run_session(
        parameter_ranges={'alpha': (8.0, 12.0)},
        num_simulations=3
    )
