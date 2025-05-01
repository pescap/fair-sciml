import argparse
import numpy as np
from simulators.base_simulator import BaseSimulator
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx 
import dolfinx.mesh
import dolfinx.fem as fem
import dolfinx.io
from dolfinx.mesh import locate_entities_boundary, create_unit_square, meshtags
from dolfinx.fem.petsc import LinearProblem
from typing import Dict, Any


class PoissonSimulator(BaseSimulator):
    """Implementation of the Poisson equation simulator."""

    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "poisson_equation"

    def setup_problem(self, **parameters) -> Dict[str, Any]:
        """Set up the Poisson equation with given parameters."""
        source_strength = parameters.get("source_strength", 1.0)
        neumann_coefficient = parameters.get("neumann_coefficient", 1.0)

        # Create mesh and function space
        mesh = create_unit_square(MPI.COMM_WORLD, self.mesh_size, self.mesh_size)
        V = fem.functionspace(mesh, ("Lagrange", 1))

        # Define boundary condition
        def dirichlet_boundary(x):
            return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

        
        # Locate DOFs for Dirichlet BC
        u_bc_value = fem.Constant(mesh, PETSc.ScalarType(0.0))
        dirichlet_dofs = fem.locate_dofs_geometrical(V, dirichlet_boundary)
        bc = fem.dirichletbc(u_bc_value, dirichlet_dofs, V)

        # Define variational problem
        def f(x):
            return source_strength * np.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
        
        def g(x):
            return np.sin(neumann_coefficient * x[0])

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Source term f
        f_interpolated = fem.Function(V)
        f_interpolated.interpolate(f)

        # Neumann term g
        g_interpolated = fem.Function(V)
        g_interpolated.interpolate(g)

        # Weak form
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f_interpolated, v) * ufl.dx + ufl.inner(g_interpolated, v) * ufl.ds  # Only Neumann boundary

        u_sol = fem.Function(V)

        return {
            "mesh": mesh,
            "a": a,
            "L": L,
            "u": u_sol,
            "bc": bc,
            "field_input_f": f_interpolated,
            "field_input_g": g_interpolated,
        }

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the Poisson equation."""
        a, L, u, bc = (
            problem_data["a"],
            problem_data["L"],
            problem_data["u"],
            problem_data["bc"],
        )
        problem = LinearProblem(a, L, bcs=[bc], u=u, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()

        # Extract coordinates and solution values
        coordinates = problem_data["mesh"].geometry.x
        values = np.real(u.x.array)

        f_values = np.zeros(coordinates.shape[0], dtype=PETSc.ScalarType)
        g_values = np.zeros(coordinates.shape[0], dtype=PETSc.ScalarType)

        f_values = np.real(problem_data["field_input_f"].x.array)
        g_values = np.real(problem_data["field_input_g"].x.array)


        return {
            "coordinates": coordinates,
            "values": values,
            "field_input_f": f_values,
            "field_input_g": g_values,
        }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Poisson Equation Simulation")
    parser.add_argument(
        "--source_strength_min",
        type=float,
        default=10.0,
        help="Minimum source term strength",
    )
    parser.add_argument(
        "--source_strength_max",
        type=float,
        default=20.0,
        help="Maximum source term strength",
    )
    parser.add_argument(
        "--neumann_coefficient_min",
        type=float,
        default=5.0,
        help="Minimum Neumann coefficient",
    )
    parser.add_argument(
        "--neumann_coefficient_max",
        type=float,
        default=10.0,
        help="Maximum Neumann coefficient",
    )
    parser.add_argument(
        "--num_simulations", type=int, default=10, help="Number of simulations to run"
    )
    parser.add_argument("--mesh_size", type=int, default=32, help="Size of the mesh")
    parser.add_argument(
        "--output_directory",
        type=str,
        default="simulations",
        help="Output directory for HDF5 files",
    )
    return parser.parse_args()


def main():
    """Main function to run the Poisson simulations."""
    args = parse_arguments()

    simulator = PoissonSimulator(
        mesh_size=args.mesh_size, output_directory=args.output_directory
    )

    # Define parameter ranges
    parameter_ranges = {
        "source_strength": (args.source_strength_min, args.source_strength_max),
        "neumann_coefficient": (
            args.neumann_coefficient_min,
            args.neumann_coefficient_max,
        ),
    }

    simulator.run_session(parameter_ranges, num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()
