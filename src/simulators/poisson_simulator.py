import argparse
import uuid
import numpy as np
from simulators.base_simulator import BaseSimulator
import dolfin as df
from typing import Dict, Any, Tuple

class PoissonSimulator(BaseSimulator):
    """Implementation of the Poisson equation simulator."""
    
    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "poisson_equation"
    
    def setup_problem(self, **parameters) -> Dict[str, Any]:
        """Set up the Poisson equation with given parameters."""
        source_strength = parameters.get('source_strength', 1.0)
        neumann_coefficient = parameters.get('neumann_coefficient', 1.0)

        # Create mesh and function space
        mesh = df.UnitSquareMesh(self.mesh_size, self.mesh_size)
        V = df.FunctionSpace(mesh, "Lagrange", 1)

        # Define boundary condition
        def boundary(x):
            return x[0] < df.DOLFIN_EPS or x[0] > 1.0 - df.DOLFIN_EPS

        bc = df.DirichletBC(V, df.Constant(0.0), boundary)

        # Define variational problem components
        u = df.TrialFunction(V)
        v = df.TestFunction(V)

        f = df.Expression(
            f"{source_strength}*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", 
            degree=2
        )
        g = df.Expression(f"sin({neumann_coefficient}*x[0])", degree=2)

        a = df.inner(df.grad(u), df.grad(v)) * df.dx
        L = f * v * df.dx + g * v * df.ds

        u = df.Function(V)
        return {"a": a, "L": L, "u": u, "bc": bc}

    def solve_problem(self, problem_data: Dict[str, Any]) -> df.Function:
        """Solve the Poisson equation."""
        a, L, u, bc = problem_data["a"], problem_data["L"], problem_data["u"], problem_data["bc"]
        df.solve(a == L, u, bc)
        return u

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Poisson Equation Simulation")
    parser.add_argument("--source_strength_min", type=float, default=10.0, help="Minimum source term strength")
    parser.add_argument("--source_strength_max", type=float, default=20.0, help="Maximum source term strength")
    parser.add_argument("--neumann_coefficient_min", type=float, default=5.0, help="Minimum Neumann coefficient")
    parser.add_argument("--neumann_coefficient_max", type=float, default=10.0, help="Maximum Neumann coefficient")
    parser.add_argument("--num_simulations", type=int, default=5, help="Number of simulations to run")
    parser.add_argument("--mesh_size", type=int, default=32, help="Size of the mesh")
    parser.add_argument("--output_path", type=str, default="poisson_simulations.h5", help="Output path for HDF5 file")
    return parser.parse_args()

def main():
    """Main function to run the Poisson simulations."""
    args = parse_arguments()

    # Create the simulator with the given mesh size and output path
    simulator = PoissonSimulator(mesh_size=args.mesh_size, output_path=args.output_path)

    # Define parameter ranges
    parameter_ranges = {
        'source_strength': (args.source_strength_min, args.source_strength_max),
        'neumann_coefficient': (args.neumann_coefficient_min, args.neumann_coefficient_max)
    }

    # Run simulation session
    simulator.run_session(parameter_ranges, num_simulations=args.num_simulations)

if __name__ == "__main__":
    main()
