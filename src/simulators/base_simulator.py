from abc import ABC, abstractmethod
from time import time
from typing import Dict, Any
import uuid
import os
import numpy as np
from utils.metadata import MetadataCollector
from utils.h5_handler import H5Handler
from dolfinx.mesh import Mesh


class BaseSimulator(ABC):
    """Abstract base class for PDE simulators."""

    def __init__(self, mesh_size: int = 32, output_directory: str = "simulations"):
        self.mesh_size = mesh_size
        self.equation_name = self._get_equation_name()

        # Automatically define file path using equation name
        self.output_path = os.path.join(output_directory, f"{self.equation_name}.h5")

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        self.h5_handler = H5Handler(self.output_path)

    @abstractmethod
    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        pass

    @abstractmethod
    def setup_problem(self, **parameters) -> Any:
        """Set up the problem to be solved. Can return any data structure."""
        pass

    @abstractmethod
    def solve_problem(self, problem_data: Any) -> Dict[str, Any]:
        """Solve the problem. Returns a dictionary with solution data."""
        pass

    def collect_metadata(
        self, parameters: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Collect metadata specific to the simulation."""
        metadata = MetadataCollector.collect_simulation_metadata(
            parameters, execution_time
        )
        metadata["mesh_size"] = self.mesh_size  # Include mesh size in metadata
        return metadata

    def run_simulation(
        self, mesh: Mesh, session_id: str, simulation_index: int, **parameters
    ) -> None:
        """Run a single simulation, using custom problem setup and solver."""
        start_time = time()

        # Setup and solve the problem
        problem_data = self.setup_problem(mesh, **parameters)
        solution_data = self.solve_problem(problem_data)

        # Collect metadata
        execution_time = time() - start_time
        simulation_metadata = self.collect_metadata(parameters, execution_time)

        # Save results with dynamic naming for fields
        self.h5_handler.save_simulation_data(
            session_id=session_id,
            solution_data=solution_data,
            simulation_metadata=simulation_metadata,
        )

        print(
            f"Simulation {simulation_index} completed in {execution_time:.2f} seconds."
        )

    def run_simulation_analytical(
        self, mesh: Mesh, session_id: str, simulation_index: int, **parameters
    ) -> None:
        """Run a single simulation, obtaining the analytical solution."""
        start_time = time()

        # Setup and solve the problem
        # problem_data = self.setup_problem(mesh, **parameters)
        solution_data = self.analytical_solution(mesh, **parameters)

        # Collect metadata
        execution_time = time() - start_time
        simulation_metadata = self.collect_metadata(parameters, execution_time)

        # Save results with dynamic naming for fields
        self.h5_handler.save_simulation_data(
            session_id=session_id,
            solution_data=solution_data,
            simulation_metadata=simulation_metadata,
        )

        print(
            f"Simulation {simulation_index} completed in {execution_time:.2f} seconds."
        )

    def run_session(
        self,
        mesh: Mesh,
        parameter_ranges: Dict[str, tuple],
        num_simulations: int,
        **mesh_parameters,
    ) -> None:
        """Run a session of multiple simulations with varying parameters."""
        session_id = str(uuid.uuid4())
        session_metadata = MetadataCollector.collect_session_metadata(
            num_simulations, self.mesh_size
        )

        # Save session metadata without `equation_name`
        self.h5_handler.save_session_metadata(session_id, session_metadata)

        # Generate parameter combinations
        parameter_values = {
            param: np.random.uniform(range_vals[0], range_vals[1], num_simulations)
            for param, range_vals in parameter_ranges.items()
        }

        # Run each simulation with generated parameters
        for i in range(num_simulations):
            params = {param: values[i] for param, values in parameter_values.items()}
            params = dict(params, **mesh_parameters)  # Merge mesh parameters
            self.run_simulation(mesh, session_id, simulation_index=i + 1, **params)

        print(f"All simulations for {self.equation_name} saved to {self.output_path}")

    def run_session_analytical(
        self,
        mesh: Mesh,
        parameter_ranges: Dict[str, tuple],
        num_simulations: int,
        **mesh_parameters,
    ) -> None:
        """Run a session of multiple simulations with varying parameters."""
        session_id = str(uuid.uuid4())
        session_metadata = MetadataCollector.collect_session_metadata(
            num_simulations, self.mesh_size
        )

        # Save session metadata without `equation_name`
        self.h5_handler.save_session_metadata(session_id, session_metadata)

        # Generate parameter combinations
        parameter_values = {
            param: np.random.uniform(range_vals[0], range_vals[1], num_simulations)
            for param, range_vals in parameter_ranges.items()
        }

        # Run each simulation with generated parameters
        for i in range(num_simulations):
            params = {param: values[i] for param, values in parameter_values.items()}
            params = dict(params, **mesh_parameters)
            self.run_simulation_analytical(
                mesh, session_id, simulation_index=i + 1, **params
            )

        print(f"All simulations for {self.equation_name} saved to {self.output_path}")
