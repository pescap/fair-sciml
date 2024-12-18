from simulators.base_simulator import BaseSimulator
import dolfin as df
import numpy as np
import argparse
from typing import Dict, Any


class BiharmonicSimulator(BaseSimulator):
    """Implementation of the Biharmonic equation simulator."""

    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "biharmonic_equation"

    def setup_problem(self, **parameters) -> Dict[str, Any]:
        """Set up the Biharmonic equation with given parameters."""
        # Extract coefficient parameter for f
        coefficient_value = parameters.get("coefficient", 1.0)

        # Define the source term (field input): f = coefficient * 4π⁴ sin(πx) sin(πy)
        f_expression = df.Expression(
            f"{coefficient_value}*4.0*pow(pi, 4)*sin(pi*x[0])*sin(pi*x[1])", degree=2
        )

        # Create mesh and function space (Quadratic elements)
        mesh = df.UnitSquareMesh(self.mesh_size, self.mesh_size)
        V = df.FunctionSpace(mesh, "CG", 2)  # CG = Continuous Galerkin

        # Define boundary condition
        class DirichletBoundary(df.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary

        u0 = df.Constant(0.0)
        bc = df.DirichletBC(V, u0, DirichletBoundary())

        # Define trial and test functions
        u = df.TrialFunction(V)
        v = df.TestFunction(V)

        # Define normal component, mesh size
        h = df.CellDiameter(mesh)
        h_avg = (h("+") + h("-")) / 2.0
        n = df.FacetNormal(mesh)

        # Define bilinear and linear forms
        a = (
            df.inner(df.div(df.grad(u)), df.div(df.grad(v))) * df.dx
            - df.inner(df.avg(df.div(df.grad(u))), df.jump(df.grad(v), n)) * df.dS
            - df.inner(df.jump(df.grad(u), n), df.avg(df.div(df.grad(v)))) * df.dS
            + 8.0
            / h_avg
            * df.inner(df.jump(df.grad(u), n), df.jump(df.grad(v), n))
            * df.dS
        )  # Fixed penalty parameter

        L = f_expression * v * df.dx

        # Return problem components
        return {
            "mesh": mesh,
            "a": a,
            "L": L,
            "u": u,
            "bc": bc,
            "field_input_f": f_expression,
        }

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the Biharmonic equation."""
        a, L, u, bc = (
            problem_data["a"],
            problem_data["L"],
            df.Function(problem_data["u"].function_space()),
            problem_data["bc"],
        )

        # Solve the problem
        df.solve(a == L, u, bc)

        # Extract coordinates, solution values, and field input values
        coordinates = problem_data["mesh"].coordinates()
        values = u.vector().get_local()
        f_values = np.array([problem_data["field_input_f"](x) for x in coordinates])

        return {"coordinates": coordinates, "values": values, "field_input_f": f_values}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Biharmonic Equation Simulation")
    parser.add_argument(
        "--coefficient_min",
        type=float,
        default=1.0,
        help="Minimum coefficient for the source term",
    )
    parser.add_argument(
        "--coefficient_max",
        type=float,
        default=5.0,
        help="Maximum coefficient for the source term",
    )
    parser.add_argument("--mesh_size", type=int, default=32, help="Size of the mesh")
    parser.add_argument(
        "--num_simulations", type=int, default=10, help="Number of simulations to run"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="simulations",
        help="Output directory for HDF5 file",
    )
    return parser.parse_args()


def main():
    """Main function to run the Biharmonic simulations."""
    args = parse_arguments()

    # Create the simulator
    simulator = BiharmonicSimulator(
        mesh_size=args.mesh_size, output_directory=args.output_directory
    )

    # Define parameter ranges (for coefficient of f)
    parameter_ranges = {"coefficient": (args.coefficient_min, args.coefficient_max)}

    # Run the simulation session
    simulator.run_session(parameter_ranges, num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()
