import h5py
import uuid
from typing import Dict, Any


class H5Handler:
    """Handles HDF5 file operations for simulation data."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def save_simulation_data(
        self,
        session_id: str,
        solution_data: Dict[str, Any],
        simulation_metadata: Dict[str, Any],
    ) -> None:
        """
        Save simulation results and metadata to an HDF5 file.

        Args:
            session_id (str): Unique ID for the simulation session.
            solution_data (Dict[str, Any]): Solution data containing fields with..
            simulation_metadata (Dict[str, Any]): Metadata for the simulation.
        """
        with h5py.File(self.filepath, "a") as h5file:
            session_group = h5file.require_group(f"session_{session_id}")
            sim_group = session_group.create_group(f"simulation_{uuid.uuid4()}")

            # Save solution data
            for key, data in solution_data.items():
                if key.startswith("field_") or key in ["coordinates", "values"]:
                    sim_group.create_dataset(key, data=data)

            # Save metadata
            for key, value in simulation_metadata.items():
                if isinstance(value, dict):  # Handle parameters separately
                    for sub_key, sub_value in value.items():
                        sim_group.attrs[sub_key] = sub_value
                else:
                    sim_group.attrs[key] = str(value)

            print(f"Saved simulation to {sim_group.name}")

    def save_session_metadata(
        self, session_id: str, session_metadata: Dict[str, Any]
    ) -> None:
        """Save session metadata to HDF5 file."""
        with h5py.File(self.filepath, "a") as h5file:
            session_group = h5file.require_group(f"session_{session_id}")

            session_group.attrs["timestamp"] = session_metadata["timestamp"]
            session_group.attrs["num_simulations"] = str(
                session_metadata["num_simulations"]
            )
            session_group.attrs["mesh_size"] = session_metadata["mesh_size"]

            # Convert hardware dictionary to a JSON string
            for key, value in session_metadata["hardware"].items():
                session_group.attrs[key] = str(value)
