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

The Poisson equation on the unit square \(\Omega = [0,1] \times [0,1]\) is:

\[
-\nabla^2 u = f \quad \text{in} \quad \Omega
\]

With boundary conditions:

\[
u = 0 \quad \text{on} \quad \partial \Omega
\]

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

### **HDF5 File Structure** ##

Below is a simplified version of the PoissonSimulator class:

```bash
poisson_results.h5
└── poisson_equation
    └── session_<session_id>
        ├── simulation_<uuid>
        │   ├── coordinates: Mesh coordinates
        │   ├── values: PDE solution values
        │   └── Metadata (parameters, execution time)
```

---

### **How to Add a Mew Simulator** ##

You can create new simulators by subclassing the BaseSimulator. For example, the BiharmonicSimulator follows the same pattern but implements a different PDE.

Create a new Python file in the simulators/ folder (e.g., biharmonic_simulator.py).
Subclass the BaseSimulator and implement the setup_problem() and solve_problem() methods.
