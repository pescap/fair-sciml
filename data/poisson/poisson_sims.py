import json
import uuid
from datetime import datetime
import platform
import psutil
import cpuinfo
import argparse
import h5py
import numpy as np
from time import time
from dolfin import *


''' 
Script to run simulations of the poisson equation and store them in a hdf5 format.
The poisson equation is parametized by source strength and Neumann coefficient.
Metadata for both a simulation session (hardware information, timestamp, number of simulations)
and sessions themselves (coefficients, execution time) are stored and can be accessed as the attributes
of the groups in the resulting simulations.h5 file.
'''

# Collect session-level metadata (hardware, timestamp, number of simulations)
def collect_session_metadata(num_simulations):
    cpu_info = cpuinfo.get_cpu_info()
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "machine": platform.machine(),
            "processor": cpu_info.get('brand_raw', 'Unknown'),
            "architecture": cpu_info.get('arch', 'Unknown'),
            "system": platform.system(),
            "version": platform.version(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
        },
        "num_simulations": num_simulations
    }
    return metadata

# Collect simulation-specific parameters and metadata
def collect_simulation_metadata(parameters, execution_time):
    # Convert numpy types to native Python types and add parameter prefix
    parameters = {f"parameter_{k}": (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in parameters.items()}
    metadata = {
        "parameters": parameters,
        "execution_time": f"{execution_time:.4f} seconds"
    }
    return metadata

# Save simulation data to HDF5 format within a session
def save_simulation_data(h5file, equation_name, session_id, u, simulation_metadata):
    # Create or access the equation group
    eq_group = h5file.require_group(equation_name)
    
    # Create or access the session group within the equation group
    session_group = eq_group.require_group(f"session_{session_id}")
    
    # Create a group for the simulation within the session
    sim_group = session_group.create_group(f"simulation_{uuid.uuid4()}")
    
    # Extract data from FEniCS function 'u'
    mesh = u.function_space().mesh()
    coordinates = mesh.coordinates()
    values = u.vector().get_local()
    
    # Save datasets within the simulation group
    sim_group.create_dataset("coordinates", data=coordinates)
    sim_group.create_dataset("values", data=values)
    
    # Save simulation-specific parameters and metadata as attributes of the simulation group
    for key, value in simulation_metadata["parameters"].items():
        sim_group.attrs[key] = str(value)
    
    # Add execution time as an attribute
    sim_group.attrs["execution_time"] = simulation_metadata["execution_time"]

# Save session-level metadata (timestamp, hardware, number of simulations) as attributes
def save_session_metadata(session_group, session_metadata):
    session_group.attrs["timestamp"] = session_metadata["timestamp"]
    session_group.attrs["num_simulations"] = str(session_metadata["num_simulations"])
    for key, value in session_metadata["hardware"].items():
        session_group.attrs[key] = str(value)

# Main simulation function
def run_simulation(h5file, equation_name, session_id, source_strength, neumann_coefficient):
    # Fixed mesh size
    mesh_size = 32
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(mesh_size, mesh_size)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary condition
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Parametrized source term: Changing the scaling factor in the source term (f)
    f = Expression(f"{source_strength}*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    
    # Parametrized Neumann boundary condition (g)
    g = Expression(f"sin({neumann_coefficient}*x[0])", degree=2)
    
    # Bilinear and linear forms
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # Track start time
    start_time = time()

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Calculate execution time
    execution_time = time() - start_time

    # Collect parameters for the simulation
    parameters = {
        "source_strength": source_strength,
        "neumann_coefficient": neumann_coefficient
    }
    simulation_metadata = collect_simulation_metadata(parameters, execution_time)

    # Save simulation data and metadata to the HDF5 file
    save_simulation_data(h5file, equation_name, session_id, u, simulation_metadata)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Poisson Equation Simulation")
    parser.add_argument("--source_strength_min", type=float, default=10.0, help="Minimum strength of the source term")
    parser.add_argument("--source_strength_max", type=float, default=20.0, help="Maximum strength of the source term")
    parser.add_argument("--neumann_coefficient_min", type=float, default=5.0, help="Minimum Neumann boundary coefficient")
    parser.add_argument("--neumann_coefficient_max", type=float, default=10.0, help="Maximum Neumann boundary coefficient")
    parser.add_argument("--num_simulations", type=int, default=5, help="Number of simulations to run")
    args = parser.parse_args()

    # Create ranges for source strength and Neumann boundary coefficient
    source_strengths = np.linspace(args.source_strength_min, args.source_strength_max, args.num_simulations)
    neumann_coefficients = np.linspace(args.neumann_coefficient_min, args.neumann_coefficient_max, args.num_simulations)

    # Equation name for grouping
    equation_name = "poisson_equation"

    # Generate a session ID for this batch of simulations
    session_id = str(uuid.uuid4())

    # Collect session-level metadata (hardware, timestamp, number of simulations)
    session_metadata = collect_session_metadata(args.num_simulations)

    # Open the HDF5 file once and store all simulations within the session
    with h5py.File("/app/output/simulations.h5", "a") as h5file:
        # Create or access the session group
        eq_group = h5file.require_group(equation_name)
        session_group = eq_group.require_group(f"session_{session_id}")
        
        # Save session metadata
        save_session_metadata(session_group, session_metadata)

        # Iterate over both source strength and Neumann coefficient values
        for source_strength, neumann_coefficient in zip(source_strengths, neumann_coefficients):
            print(f"Running simulation with source_strength={source_strength} and neumann_coefficient={neumann_coefficient}")
            run_simulation(h5file, equation_name, session_id, source_strength, neumann_coefficient)
