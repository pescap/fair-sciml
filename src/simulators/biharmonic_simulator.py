from simulators.base_simulator import BaseSimulator
from dolfin import *
import argparse
import uuid
from typing import Dict, Any, Tuple

class BiharmonicSimulator(BaseSimulator):
    """Implementation of the Biharmonic equation simulator."""
    
    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "biharmonic_equation"

    def setup_problem(self, **parameters) -> Dict[str, Any]:
        """Set up the Biharmonic equation with given parameters."""
        # Extract penalty parameter (default value: 8.0)
        alpha_value = parameters.get('alpha', 8.0)

        # Create mesh and function space (Quadratic elements)
        mesh = UnitSquareMesh(self.mesh_size, self.mesh_size)
        V = FunctionSpace(mesh, "CG", 2)  # CG = Continuous Galerkin

        # Define boundary condition
        class DirichletBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        u0 = Constant(0.0)
        bc = DirichletBC(V, u0, DirichletBoundary())

        # Define trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Define the source term: f = 4π⁴ sin(πx) sin(πy)
        f = Expression("4.0*pow(pi, 4)*sin(pi*x[0])*sin(pi*x[1])", degree=2)

        # Define normal component, mesh size, and penalty parameter
        h = CellDiameter(mesh)
        h_avg = (h('+') + h('-')) / 2.0
        n = FacetNormal(mesh)
        alpha = Constant(alpha_value)

        # Define bilinear and linear forms
        a = (inner(div(grad(u)), div(grad(v))) * dx
             - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
             - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
             + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS)

        L = f * v * dx

        # Return problem components
        return {"a": a, "L": L, "u": u, "bc": bc}

    def solve_problem(self, problem_data: Dict[str, Any]) -> Function:
        """Solve the Biharmonic equation."""
        a, L, u, bc = problem_data["a"], problem_data["L"], Function(problem_data["u"].function_space()), problem_data["bc"]
        
        # Solve the problem
        solve(a == L, u, bc)
        return u

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Biharmonic Equation Simulation")
    parser.add_argument("--alpha", type=float, default=8.0, help="Penalty parameter value")
    parser.add_argument("--mesh_size", type=int, default=32, help="Size of the mesh")
    parser.add_argument("--num_simulations", type=int, default=1, help="Number of simulations to run")
    parser.add_argument("--output_path", type=str, default="biharmonic_simulations.h5", help="Output path for HDF5 file")
    args = parser.parse_args()

    # Create the simulator
    simulator = BiharmonicSimulator(mesh_size=args.mesh_size, output_path=args.output_path)

    # Define parameter ranges (for alpha, in this case)
    parameter_ranges = {'alpha': (args.alpha, args.alpha)}  # Static value, no range needed

    # Run the simulation session
    simulator.run_session(parameter_ranges, num_simulations=args.num_simulations)
