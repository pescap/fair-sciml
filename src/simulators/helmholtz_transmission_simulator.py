import argparse
import numpy as np
from simulators.base_simulator import BaseSimulator
from ufl import dx, inner, grad, TestFunction, TrialFunction
from mpi4py import MPI
import dolfinx
import dolfinx.mesh
import dolfinx.fem as fem
from dolfinx.mesh import Mesh
from dolfinx.fem.petsc import LinearProblem
from typing import Dict, Any
from scipy.special import (
    spherical_jn as jn,
    spherical_yn as yn,
    eval_legendre,
)


class HelmholtzTransmissionSimulator(BaseSimulator):
    """Helmholtz equation simulator with transmission conditions."""

    def _get_equation_name(self) -> str:
        """Return the name of the equation being simulated."""
        return "helmholtz_transmission_equation"

    def setup_problem(self, mesh: Mesh, **parameters) -> Dict[str, Any]:
        """Set up the Helmholtz equation with given parameters."""
        # Problem parameters
        k0 = parameters.get("wavenumber", 1.0)  # wavenumber
        ref_ind = parameters.get("ref_ind", 1.0)  # refractive index of scatterer
        angle = parameters.get("direction", 0.0)  # direction of incident wave
        dim_x = parameters.get("dim_x", 10.0)  # width of computational domain

        wave_len = 2 * np.pi / k0  # wavelength
        radius = 0.5  # 4 * wave_len    # scatterer radius

        """    Discretization parameters: polynomial degree and mesh resolution     """
        degree = 3  # polynomial degree

        """                   Adiabatic absorber settings                           """
        # The adiabatic absorber is a PML-type layer in which absorption is used to
        # attenutate outgoing waves. Adiabatic absorbers aren't as perfect as PMLs so
        # must be slightly wider: typically 2-5 wavelengths gives adequately small
        # reflections.
        d_absorb = 2 * wave_len  # depth of absorber

        # Increase the absorption within the layer gradually, as a monomial:
        # sigma(x) = sigma_0 * x^d; choices d=2,3 are popular choices.
        deg_absorb = 2  # degree of absorption monomial

        # The constant sigma_0 is chosen to achieve a specified "round-trip" reflection
        # of a wave that through the layer, reflects and returns back into the domain.
        # See Oskooi et al. (2008) for more details.
        RT = 1.0e-6  # round-trip reflection
        sigma0 = -(deg_absorb + 1) * np.log(RT) / (2.0 * d_absorb)

        """        Incident field, wavenumber and adiabatic absorber functions      """

        def incident(x):
            # Plane wave travelling in positive x-direction
            direction = np.array([np.cos(angle), np.sin(angle)])
            return np.exp(1.0j * k0 * (x[0] * direction[0] + x[1] * direction[1]))

        def wavenumber(x):
            # Wavenumber function, k0 outside scatterer and (k0*ref_ind) inside.
            # This function also defines the shape as a circle. Modify for different
            # shapes.
            r = np.sqrt((x[0]) ** 2 + (x[1]) ** 2)
            inside = r <= radius
            outside = r > radius
            return inside * ref_ind * k0 + outside * k0

        def adiabatic_layer(x):
            """Contribution to wavenumber k in absorbing layers"""
            # In absorbing layer, have k = k0 + 1j * sigma
            # => k^2 = (k0 + 1j*sigma)^2 = k0^2 + 2j*sigma*k0 - sigma^2
            # Therefore, the 2j*sigma - sigma^2 piece must be included in the layer.

            # Find borders of width d_absorb in x- and y-directions
            in_absorber_x = np.abs(x[0]) >= dim_x / 2 - d_absorb
            in_absorber_y = np.abs(x[1]) >= dim_x / 2 - d_absorb

            # Function sigma_0 * x^d, where x is depth into adiabatic layer
            sigma_x = (
                sigma0
                * ((np.abs(x[0]) - (dim_x / 2 - d_absorb)) / d_absorb) ** deg_absorb
            )
            sigma_y = (
                sigma0
                * ((np.abs(x[1]) - (dim_x / 2 - d_absorb)) / d_absorb) ** deg_absorb
            )

            # 2j*sigma - sigma^2 in absorbing layers
            x_layers = in_absorber_x * (2j * sigma_x * k0 - sigma_x**2)
            y_layers = in_absorber_y * (2j * sigma_y * k0 - sigma_y**2)

            return x_layers + y_layers

        # Define function space
        V = fem.functionspace(mesh, ("Lagrange", degree))

        # Interpolate wavenumber k onto V
        k = fem.Function(V)
        k.interpolate(wavenumber)

        # Interpolate absorbing layer piece of wavenumber k_absorb onto V
        k_absorb = fem.Function(V)
        k_absorb.interpolate(adiabatic_layer)

        # Interpolate incident wave field onto V
        ui = fem.Function(V)
        ui.interpolate(incident)

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)

        a = (
            inner(grad(u), grad(v)) * dx
            - k**2 * inner(u, v) * dx
            - k_absorb * inner(u, v) * dx
        )

        L = inner((k**2 - k0**2) * ui, v) * dx

        u_sol = fem.Function(V)

        return {
            "mesh": mesh,
            "a": a,
            "L": L,
            "u": u_sol,
            "bc": None,  # No Dirichlet BC for this problem
            "field_input_f": ui,
            "kappa": k0,
            "k": ref_ind * k0,
            "incident_fun": incident,
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
            petsc_options_prefix="helmholtz_transmission_",
        )
        problem.solve()

        # Extract coordinates and solution values
        coordinates = problem_data["mesh"].geometry.x
        values = np.real(u.x.array)

        f_values = np.real(problem_data["field_input_f"].x.array)

        return {
            "coordinates": coordinates,
            "values": values,
            "field_input_f": f_values,
        }

    def coefficients_for_transmission(
        self,
        k_ext: np.float32,
        k_int: np.float32,
        rho_ext: np.float32,
        rho_int: np.float32,
        r: np.float32,
        max_ite: int = 50,
    ):
        def h2n(n, a, derivative=False):
            return jn(n, a, derivative=derivative) - 1j * yn(
                n, a, derivative=derivative
            )

        psca_coef = np.zeros(max_ite, dtype=np.complex128)
        pint_coef = np.zeros(max_ite, dtype=np.complex128)

        rho = rho_int / rho_ext
        k = k_ext / k_int

        n = 0
        while n < max_ite:
            jn_int = jn(n, k_int * r)
            jn_ext = jn(n, k_ext * r)
            h2n_ext = h2n(n, k_ext * r)

            d_jn_int = jn(n, k_int * r, derivative=True)
            d_jn_ext = jn(n, k_ext * r, derivative=True)
            d_h2n_ext = h2n(n, k_ext * r, derivative=True)

            tau_num = (
                (2 * n + 1)
                * (-1j) ** n
                * (d_jn_int * jn_ext - rho * k * jn_int * d_jn_ext)
            )
            tau_den = rho * k * jn_int * d_h2n_ext - d_jn_int * h2n_ext

            tau_n = tau_num / tau_den
            ups_n = ((2 * n + 1) * (-1j) ** n * jn_ext + tau_n * h2n_ext) / jn_int

            psca_coef[n] = tau_n
            pint_coef[n] = ups_n

            n += 1

        return psca_coef, pint_coef

    def analytical_solution(self, mesh, **parameters) -> Dict[str, Any]:
        """Compute the analytical solution for the Helmholtz equation."""

        def h2n(n, a, derivative=False):
            return jn(n, a, derivative=derivative) - 1j * yn(
                n, a, derivative=derivative
            )

        k0 = parameters.get("wavenumber", 1.0)  # wavenumber
        ref_ind = parameters.get("ref_ind", 1.0)  # refractive index of scatterer
        angle = parameters.get("direction", 0.0)  # direction of incident wave

        radius = 0.5  # 4 * wave_len    # scatterer radius

        coef = self.coefficients_for_transmission(k0, k0 * ref_ind, 1, 1, 0.5)

        coordinates = mesh.geometry.x
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        coordinates = np.dot(coordinates[:, 0:2], R.T)
        x = coordinates[:, 0]
        z = coordinates[:, 1]
        rs = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)
        ps = coordinates[:, 1] / rs
        if np.any(rs == 0):
            ind = np.where(rs == 0)[0]
            ps[ind] = 0.0
        ext = rs > radius
        pinc = np.zeros_like(rs, dtype=np.complex128)
        pinc[ext] = np.exp(-1j * k0 * z[ext])

        psca = np.zeros_like(x, dtype=np.complex128)
        pint = np.zeros_like(x, dtype=np.complex128)

        for n, (an, bn) in enumerate(zip(*coef)):
            psca[ext] += an * h2n(n, k0 * rs[ext]) * eval_legendre(n, ps[ext])

            pint[~ext] += (
                bn * jn(n, ref_ind * k0 * rs[~ext]) * eval_legendre(n, ps[~ext])
            )

        ptot = psca + pinc + pint

        coordinates = mesh.geometry.x
        values = np.real(ptot)
        # f_values = np.real(f_interpolated.x.array)

        return {"coordinates": coordinates, "values": values}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Helmholtz Transmission Problem Simulation"
    )
    parser.add_argument(
        "--wavenumber_min",
        type=float,
        default=1.0,
        help="Minimum coefficient for the wavenumber",
    )
    parser.add_argument(
        "--wavenumber_max",
        type=float,
        default=5.0,
        help="Maximum coefficient for the wavenumber",
    )
    parser.add_argument(
        "--ref_ind_min",
        type=float,
        default=1.0,
        help="Minimum coefficient for the refractive index",
    )
    parser.add_argument(
        "--ref_ind_max",
        type=float,
        default=5.0,
        help="Maximum coefficient for the refractive index",
    )
    parser.add_argument(
        "--angle_min",
        type=float,
        default=0.0,
        help="Minimum value for the angle of incidence",
    )
    parser.add_argument(
        "--angle_max",
        type=float,
        default=2 * np.pi,
        help="Maximum value for the angle of incidence",
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
    """Main function to run the Helmholtz transmission simulations."""
    args = parse_arguments()

    # Create the simulator
    simulator = HelmholtzTransmissionSimulator(
        mesh_size=args.mesh_size, output_directory=args.output_directory
    )

    # Define parameter ranges (for coefficient of f)
    parameter_ranges = {
        "wavenumber": (args.wavenumber_min, args.wavenumber_max),
        "ref_ind": (args.ref_ind_min, args.ref_ind_max),
        "angle": (args.angle_min, args.angle_max),
    }

    radius = 0.5  # 4 * wave_len    # scatterer radius
    dim_in = 4 * radius
    dim_x = dim_in + 4 * (2 * np.pi / args.wavenumber_min)

    n_wave = 5  # min number of mesh elements per wavelength
    h_elem = (2 * np.pi / args.wavenumber_max) / n_wave
    n_elem = int(np.round(dim_x / h_elem))

    # Create mesh
    mesh_obj = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-dim_x / 2, -dim_x / 2]), np.array([dim_x / 2, dim_x / 2])],
        [n_elem, n_elem],
        dolfinx.mesh.CellType.triangle,
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )

    mesh_parameters = {"dim_x": dim_x}

    # Run the simulation session
    simulator.run_session(
        mesh_obj,
        parameter_ranges,
        num_simulations=args.num_simulations,
        **mesh_parameters
    )


if __name__ == "__main__":
    main()
