from abc import ABC, abstractmethod
from time import time
from typing import Dict, Any, Optional
import uuid
from utils.metadata import MetadataCollector
from utils.h5_handler import H5Handler

class BaseSimulator(ABC):
    """Abstract base class for PDE simulators."""

    def __init__(self, mesh_size: int = 32, output_path: str = "simulations.h5"):
        self.mesh_size = mesh_size
        self.h5_handler = H5Handler(output_path)
        self.equation_name = self._get_equation_name()

    @abstractmethod
    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        pass

    @abstractmethod
    def setup_problem(self, **parameters) -> Any:
        """Set up the problem to be solved. Can return any data structure."""
        pass

    @abstractmethod
    def solve_problem(self, problem_data: Any) -> Any:
        """Solve the problem. Returns the solution."""
        pass

    def collect_metadata(self, parameters: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Collect metadata specific to the simulation."""
        return MetadataCollector.collect_simulation_metadata(parameters, execution_time)

    def run_simulation(self, session_id: str, **parameters) -> None:
        """Run a single simulation, using custom problem setup and solver."""
        start_time = time()

        # Setup and solve the problem
        problem_data = self.setup_problem(**parameters)
        solution = self.solve_problem(problem_data)

        # Collect metadata and save results
        execution_time = time() - start_time
        simulation_metadata = self.collect_metadata(parameters, execution_time)

        self.h5_handler.save_simulation_data(
            self.equation_name, session_id, solution, simulation_metadata
        )

    def run_session(self, parameter_ranges: Dict[str, tuple], num_simulations: int) -> None:
        """Run a session of multiple simulations with varying parameters."""
        session_id = str(uuid.uuid4())
        session_metadata = MetadataCollector.collect_session_metadata(num_simulations)

        # Save session metadata
        self.h5_handler.save_session_metadata(self.equation_name, session_id, session_metadata)

        # Generate parameter combinations
        import numpy as np
        parameter_values = {
            param: np.linspace(range_vals[0], range_vals[1], num_simulations)
            for param, range_vals in parameter_ranges.items()
        }

        # Run each simulation with generated parameters
        for i in range(num_simulations):
            params = {param: values[i] for param, values in parameter_values.items()}
            self.run_simulation(session_id, **params)
