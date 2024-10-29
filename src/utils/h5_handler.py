import h5py
import uuid
import dolfin
from typing import Dict, Any


class H5Handler:
    """Handles HDF5 file operations for simulation data."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def save_simulation_data(
        self,
        equation_name: str,
        session_id: str,
        u: dolfin.Function,
        simulation_metadata: Dict[str, Any],
    ) -> None:
        """Save simulation results and metadata to HDF5 file."""
        with h5py.File(self.filepath, "a") as h5file:
            eq_group = h5file.require_group(equation_name)
            session_group = eq_group.require_group(f"session_{session_id}")
            sim_group = session_group.create_group(f"simulation_{uuid.uuid4()}")

            # Save solution data
            mesh = u.function_space().mesh()
            coordinates = mesh.coordinates()
            values = u.vector().get_local()

            sim_group.create_dataset("coordinates", data=coordinates)
            sim_group.create_dataset("values", data=values)

            # Save metadata
            for key, value in simulation_metadata["parameters"].items():
                sim_group.attrs[key] = str(value)
            sim_group.attrs["execution_time"] = simulation_metadata["execution_time"]

    def save_session_metadata(
        self, equation_name: str, session_id: str, session_metadata: Dict[str, Any]
    ) -> None:
        """Save session metadata to HDF5 file."""
        with h5py.File(self.filepath, "a") as h5file:
            eq_group = h5file.require_group(equation_name)
            session_group = eq_group.require_group(f"session_{session_id}")

            session_group.attrs["timestamp"] = session_metadata["timestamp"]
            session_group.attrs["num_simulations"] = str(
                session_metadata["num_simulations"]
            )
            for key, value in session_metadata["hardware"].items():
                session_group.attrs[key] = str(value)
