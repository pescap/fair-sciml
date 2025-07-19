import argparse
import numpy as np
from simulators.base_simulator import BaseSimulator
import ufl
from ufl import dx, inner, grad
from petsc4py import PETSc
from dolfinx.mesh import Mesh, create_unit_square
from mpi4py import MPI
import dolfinx.fem as fem
from dolfinx.fem.petsc import LinearProblem
from typing import Dict, Any


class HelmholtzSimulator(BaseSimulator):
    """Implementation of the Helmholtz equation simulator."""

    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "helmholtz_equation"

    def setup_problem(self, mesh: Mesh, **parameters) -> Dict[str, Any]:
        """Set up the Helmholtz equation with given parameters."""
        # Extract coefficient parameter for kappa
        n = parameters.get("coefficient", 1.0)
        print(f"Using coefficient n = {n}")
        kappa = 2 * n * np.pi

        def f_expression(x):
            return kappa**2 * np.sin(n * np.pi * x[0]) * np.sin(n * np.pi * x[1])

        # Create mesh and function space
        V = fem.functionspace(mesh, ("Lagrange", 1))

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

        f_interpolated = fem.Function(V)
        f_interpolated.interpolate(f_expression)

        # Define bilinear and linear forms
        a = inner(grad(u), grad(v)) * dx - kappa**2 * inner(u, v) * dx
        L = inner(f_interpolated, v) * dx

        u_sol = fem.Function(V)

        return {
            "mesh": mesh,
            "a": a,
            "L": L,
            "u": u_sol,
            "bc": bc,
            "field_input_f": f_interpolated,
            "kappa": kappa,
        }

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the Helmholtz problem."""
        a, L, u, bc = (
            problem_data["a"],
            problem_data["L"],
            problem_data["u"],
            problem_data["bc"],
        )

        # Create the linear problem
        problem = LinearProblem(
            a,
            L,
            bcs=[bc],
            u=u,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix="helmholtz_",
        )
        problem.solve()

        # Solve the problem
        problem.solve()

        # Extract coordinates and solution values
        coordinates = problem_data["mesh"].geometry.x
        values = np.real(u.x.array)
        f_values = np.real(problem_data["field_input_f"].x.array)

        return {"coordinates": coordinates, "values": values, "field_input_f": f_values}

    def analytical_solution(self, mesh: Mesh, **parameters) -> Dict[str, Any]:
        """Compute the analytical solution for the Helmholtz equation."""
        V = fem.functionspace(mesh, ("Lagrange", 1))
        n = parameters.get("coefficient", 1.0)
        kappa = 2 * n * np.pi

        def f_expression(x):
            return kappa**2 * np.sin(n * np.pi * x[0]) * np.sin(n * np.pi * x[1])

        def u_analytical(x):
            return -2 * np.sin(n * np.pi * x[0]) * np.sin(n * np.pi * x[1])

        # Create a function to hold the analytical solution
        u_analytical_func = fem.Function(V)
        u_analytical_func.interpolate(u_analytical)

        f_interpolated = fem.Function(V)
        f_interpolated.interpolate(f_expression)

        coordinates = mesh.geometry.x
        values = np.real(u_analytical_func.x.array)
        f_values = np.real(f_interpolated.x.array)

        return {"coordinates": coordinates, "values": values, "field_input_f": f_values}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Helmholtz Equation Simulation")
    parser.add_argument(
        "--coefficient_min",
        type=float,
        default=1.0,
        help="Minimum coefficient for the wavenumber",
    )
    parser.add_argument(
        "--coefficient_max",
        type=float,
        default=5.0,
        help="Maximum coefficient for the wavenumber",
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
    """Main function to run the Helmholtz simulations."""
    args = parse_arguments()

    # Create the simulator
    simulator = HelmholtzSimulator(
        mesh_size=args.mesh_size, output_directory=args.output_directory
    )

    # Define parameter ranges (for coefficient of f)
    parameter_ranges = {"coefficient": (args.coefficient_min, args.coefficient_max)}

    mesh = create_unit_square(MPI.COMM_WORLD, args.mesh_size, args.mesh_size)
    # Run the simulation session
    simulator.run_session(mesh, parameter_ranges, num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()
