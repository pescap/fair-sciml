# **Simulators: PDE Solvers in fair-sciml**

This folder contains the **simulators** used to solve different **Partial Differential Equations (PDEs)**. The simulators are built on top of the **BaseSimulator**, ensuring that each simulator implements common functionality, such as metadata collection, saving results in HDF5 format, and running simulation sessions.

Currently, the following simulators are implemented:

- **BaseSimulator**: Abstract class providing reusable logic for all PDE solvers.
- **PoissonSimulator**: Solver for the Poisson equation.
- **BiharmonicSimulator**: Solver for the Biharmonic equation.

---

## **How to Use the Simulators**

Each simulator inherits from the **BaseSimulator** class. The general workflow for using a simulator is as follows:

1. **Setup parameters** specific to the PDE.
2. **Run simulations** by providing parameter ranges and other configurations.
3. **Store the results** in an HDF5 file with relevant metadata.

We will use the **Poisson equation** as a working example in this guide.

---

## **Simulators Overview**

### **BaseSimulator**

The `BaseSimulator` class serves as the foundation for all simulators. It defines shared logic such as:

- **`setup_problem()`**: Abstract method for setting up the variational form of the PDE.
- **`run_simulation()`**: Runs a single simulation.
- **`run_session()`**: Runs multiple simulations with varying parameters.
- **HDF5 file handling**: Saves simulation results and metadata.

Each specific simulator (e.g., **PoissonSimulator**) overrides the `setup_problem()` method to define the appropriate PDE.

---

## **PoissonSimulator: A Detailed Example**

The **Poisson equation** is a second-order PDE used in many fields, such as electrostatics, fluid dynamics, and heat transfer. In this project, the Poisson equation is solved using the FEniCS library.

### **Equation**

The Poisson equation on the unit square Ω = [0,1] × [0,1] is:

    -∇²u = f  in  Ω

With boundary conditions:

    u = 0  on  ∂Ω

---

### **How to Run the Poisson Simulator**

Below is an example of how to run the **Poisson simulator** using the command line.

### **Command Example**

```bash
python3 poisson_simulator.py \
    --source_strength_min 10.0 \
    --source_strength_max 20.0 \
    --neumann_coefficient_min 5.0 \
    --neumann_coefficient_max 10.0 \
    --num_simulations 5 \
    --mesh_size 32 \
    --output_path "poisson_results.h5"
```
---

### **How the Poisson Simulator Works** ###
1. Parameters:

- **`source_strength`**: Controls the intensity of the source term 
𝑓
- **`newmann_coefficient`**: Coefficient for the Neumann boundary condition.
- **`mesh_size`**: Resolution of the mesh used for solving the PDE.
- **`num_simulations`**: Number of simulations to run, with varying parameter values.

2. Simulation Flow:

- A session ID is generated to group all simulations.
- The metadata (hardware, timestamp, etc.) is collected for each session.
- The solution for each parameter combination is computed and stored in the specified HDF5 file.

---

## Accessing Pre-Generated Simulations

If you prefer not to generate the simulations locally, pre-generated datasets are available on [Hugging Face](https://huggingface.co/datasets/Aledhf/Simulations). These datasets include:

- Simulations of the Poisson equation.
- Simulations of the Biharmonic equation.

Each dataset is stored in HDF5 format and includes:
- Field inputs.
- Spatial coordinates of the simulation domain.
- Solution values for each simulation point.

To use these datasets:
1. Download the desired dataset from Hugging Face.
2. Place the downloaded `.h5` file in your desired directory.
3. Update the file path in the training or analysis scripts.

---

### **HDF5 File Structure** ##

Below is a simplified version of the PoissonSimulator class:

```bash
poisson_results.h5
└── session_<session_id>
    ├── simulation_<uuid>
    │   ├── coordinates: Mesh coordinates
    │   ├── values: PDE solution values
    │   ├── input_field: The solution of an input field such as f, g 
    │   └── Metadata (parameters, execution time)
```

---

### **How to Add a New Simulator** ##

You can create new simulators by subclassing the BaseSimulator. For example, the BiharmonicSimulator follows the same pattern but implements a different PDE.

Create a new Python file in the simulators/ folder (e.g., biharmonic_simulator.py).
Subclass the BaseSimulator and implement the setup_problem() and solve_problem() methods.
