## **Utilities**

The `utils` folder contains helper modules for managing HDF5 files, metadata, and the hierarchical structure used throughout the project. These utilities enable seamless integration of metadata with simulation data and ensure the generated datasets adhere to FAIR (Findable, Accessible, Interoperable, and Reusable) principles.

### **Modules in `utils`**

### `h5_handler.py`
This module provides utilities for creating and manipulating HDF5 files, including structured storage of simulation data and efficient data retrieval.

#### **Key Features:**
- **File Creation**: Initializes an HDF5 file with the appropriate hierarchy for simulations.
- **Data Storage**: Saves simulation results, including coordinates, solution values, and input fields (e.g., `field_input_f`).
- **Efficient Data Handling**: Facilitates reading and writing large simulation datasets.
- **Structure Compliance**: Ensures the HDF5 files follow a consistent structure:

#### **Key Functions:**
- `create_h5_file(output_path)`: Initializes a new HDF5 file.
- `add_simulation(session_name, simulation_name, data)`: Adds simulation data (coordinates, field inputs, solution values) to a specific session.
- `read_simulation(session_name, simulation_name)`: Retrieves data for a specific simulation.

### `metadata.py`
This module provides functionality for collecting and managing metadata associated with simulations.

#### **Key Features:**
- **Session Metadata**: Tracks attributes such as timestamp, number of simulations, and hardware information.
- **Simulation Metadata**: Records parameters used in each simulation (e.g., `source_strength`, `neumann_coefficient`) and execution time.
- **FAIR Compliance**: Ensures all metadata is machine-readable and linked directly to the relevant data.

#### **Key Functions:**
- `add_session_metadata(h5_file, session_name, metadata)`: Adds metadata for a specific simulation session.
- `add_simulation_metadata(h5_file, session_name, simulation_name, metadata)`: Adds metadata for a specific simulation.
- `collect_hardware_info()`: Collects hardware information (e.g., CPU, memory, OS) for reproducibility.
- `generate_timestamp()`: Generates a timestamp for session creation.

### **Example Usage**

#### HDF5 File Creation and Data Storage
```python
from utils.h5_handler import create_h5_file, add_simulation

# Initialize an HDF5 file
output_path = "simulations/poisson_equation.h5"
h5_file = create_h5_file(output_path)

# Add a simulation
data = {
  "coordinates": coordinates_array,
  "values": solution_array,
  "field_input_f": input_field_array
}
add_simulation(h5_file, session_name="session_1", simulation_name="simulation_1", data=data)
```

#### **Adding Metadata**

```python
from utils.metadata import add_session_metadata, add_simulation_metadata

# Add session metadata
session_metadata = {
    "timestamp": generate_timestamp(),
    "number_of_simulations": 10,
    "hardware_info": collect_hardware_info()
}
add_session_metadata(h5_file, session_name="session_1", metadata=session_metadata)

# Add simulation metadata
simulation_metadata = {
    "parameter_source_strength": 15.0,
    "parameter_neumann_coefficient": 8.0,
    "execution_time": 0.245
}
add_simulation_metadata(h5_file, session_name="session_1", simulation_name="simulation_1", metadata=simulation_metadata)
```

---

## **Structure and Compliance**
The utilities in utils ensure that all simulation data and metadata follow a clear, hierarchical structure, making datasets interoperable and reusable in line with FAIR principles.
