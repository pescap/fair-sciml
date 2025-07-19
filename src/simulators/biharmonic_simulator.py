import argparse
import numpy as np
from simulators.base_simulator import BaseSimulator
import ufl
from ufl import dx, dS, inner, grad, div, jump, avg, CellDiameter, FacetNormal
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem as fem
from dolfinx.mesh import Mesh, create_unit_square
from dolfinx.fem.petsc import LinearProblem
from typing import Dict, Any


class BiharmonicSimulator(BaseSimulator):
    """Implementation of the Biharmonic equation simulator."""

    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "biharmonic_equation"

    def setup_problem(self, mesh: Mesh, **parameters) -> Dict[str, Any]:
        """Set up the Biharmonic equation with given parameters."""
        # Extract coefficient parameter for f
        coefficient_value = parameters.get("coefficient", 1.0)

        # Define the source term (field input): f = coefficient * 4π⁴ sin(πx) sin(πy)
        def f_expression(x):
            return (
                coefficient_value
                * 4.0
                * np.pi**4
                * np.sin(np.pi * x[0])
                * np.sin(np.pi * x[1])
            )

        # Create mesh and function space (Quadratic elements)
        V = fem.functionspace(mesh, ("Lagrange", 2))

        # Define boundary condition
        def dirichlet_boundary(x):
            return (
                np.isclose(x[0], 0.0)
                | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0)
                | np.isclose(x[1], 1.0)
            )

        u0 = fem.Constant(mesh, PETSc.ScalarType(0.0))
        dirichlet_dofs = fem.locate_dofs_geometrical(V, dirichlet_boundary)
        bc = fem.dirichletbc(u0, dirichlet_dofs, V)

        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Define normal component, mesh size
        h = CellDiameter(mesh)
        h_avg = (h("+") + h("-")) / 2.0
        n = FacetNormal(mesh)

        f_interpolated = fem.Function(V)
        f_interpolated.interpolate(f_expression)

        # Define bilinear and linear forms
        a = (
            inner(div(grad(u)), div(grad(v))) * dx
            - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
            - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
            + 15.0 / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
        )  # Fixed penalty parameter

        u_sol = fem.Function(V)

        L = inner(f_interpolated, v) * dx

        # Return problem components
        return {
            "mesh": mesh,
            "a": a,
            "L": L,
            "u": u_sol,
            "bc": bc,
            "field_input_f": f_interpolated,
        }

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the Biharmonic equation."""
        a, L, u, bc = (
            problem_data["a"],
            problem_data["L"],
            problem_data["u"],
            problem_data["bc"],
        )

        # Solve the problem
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            u=u,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="biharmonic_",
        )
        problem.solve()

        # Extract coordinates, solution values, and field input values
        coordinates = problem_data["mesh"].geometry.x
        values = np.real(u.x.array)
        f_values = np.real(problem_data["field_input_f"].x.array)
        # f_values = np.array([problem_data["field_input_f"](x) for x in coordinates])

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

    mesh = create_unit_square(MPI.COMM_WORLD, args.mesh_size, args.mesh_size)
    # Run the simulation session
    simulator.run_session(mesh, parameter_ranges, num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()
