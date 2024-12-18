Utils Documentation
===================

Overview
--------

The `utils` directory contains utility modules designed to support the main functionalities of the `fair-sciml` project. These modules include tools for handling HDF5 files and managing metadata for simulations.

Modules
-------

- **`h5_handler.py`**: Provides utility functions to handle HDF5 file creation, data storage, and retrieval.
- **`metadata.py`**: Handles metadata collection, including session-level and simulation-specific metadata.

metadata.py
-----------

The `metadata.py` module is responsible for collecting and structuring metadata associated with simulation sessions and individual simulations. This metadata is essential for ensuring traceability and reproducibility of the simulation data.

**Key Features:**

- **Session Metadata**: Collects hardware and environment details for the simulation session, including system information, CPU details, and memory usage.
- **Simulation Metadata**: Collects simulation-specific details such as input parameters and execution time.
- **Parameter Flattening**: Automatically flattens parameter dictionaries for consistent storage.

**Example Usage:**

::

    from utils.metadata import MetadataCollector

    # Collect session-level metadata
    session_metadata = MetadataCollector.collect_session_metadata(
        num_simulations=10, mesh_size=32
    )

    print(session_metadata)
    # Output:
    # {
    #     "timestamp": "2024-12-17T12:00:00",
    #     "hardware": {
    #         "machine": "x86_64",
    #         "processor": "Intel(R) Core(TM) i7-10700K",
    #         "system": "Linux",
    #         "version": "5.15.0-73-generic",
    #         "physical_cores": 8,
    #         "logical_cores": 16,
    #         "total_memory": "32.00 GB"
    #     },
    #     "num_simulations": 10,
    #     "mesh_size": 32
    # }

    # Collect simulation-specific metadata
    simulation_metadata = MetadataCollector.collect_simulation_metadata(
        parameters={"source_strength": 15.0, "neumann_coefficient": 7.0},
        execution_time=12.34,
    )

    print(simulation_metadata)
    # Output:
    # {
    #     "parameter_source_strength": 15.0,
    #     "parameter_neumann_coefficient": 7.0,
    #     "execution_time": 12.34
    # }

h5_handler.py
-------------

The `h5_handler.py` module provides methods for managing the hierarchical structure of HDF5 files. It ensures efficient storage of simulation data and metadata.

**Key Features:**

- **Create HDF5 File**: Initialize an HDF5 file with the appropriate structure for storing simulation sessions and results.
- **Add Simulation Data**: Append new simulations, including fields, coordinates, and results, to the file.
- **Retrieve Data**: Extract stored data for training models or analysis.

**Example Usage:**

::

    from utils.h5_handler import H5Handler

    # Initialize the handler
    h5_handler = H5Handler("simulations.h5")

    # Add a new simulation
    h5_handler.add_simulation(
        session_id="session_1",
        simulation_id="simulation_1",
        field_input_f=[...],
        coordinates=[...],
        values=[...],
        metadata={"parameter": 5.0, "execution_time": 12.5}
    )

    # Retrieve data for analysis
    data = h5_handler.get_simulation("session_1", "simulation_1")

Contributing
------------

If you wish to extend or improve the `utils` directory, ensure that new functionalities adhere to the modular design of the project. For significant changes, consider creating a new utility module and documenting it here.
