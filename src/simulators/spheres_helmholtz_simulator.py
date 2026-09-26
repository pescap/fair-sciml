import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import special

import biosspheres.formulations.massmatrices as mass
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.formulations.mtf.reconstructions as reconstructions
import biosspheres.formulations.mtf.righthands as righthands
import biosspheres.helmholtz.crossinteractions as helmholtzcross
import biosspheres.helmholtz.selfinteractions as helmholtzself
from simulators.base_simulator import BaseSimulator


def sphere_array(
    n_side: int,
    spacing: float,
    radius_min: float,
    radius_max: float,
    jitter: float,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Cubic lattice of n_side**3 disjoint spheres with random radii and
    random displacements of the centers."""
    if 2.0 * (radius_max + np.sqrt(3.0) * jitter) >= spacing:
        raise ValueError(
            "spacing must exceed 2 (radius_max + sqrt(3) jitter) so that the "
            "spheres stay disjoint."
        )
    grid = (np.arange(n_side) - (n_side - 1) / 2) * spacing
    centers = [
        np.array([x, y, z]) + rng.uniform(-jitter, jitter, 3)
        for x in grid
        for y in grid
        for z in grid
    ]
    radii = rng.uniform(radius_min, radius_max, len(centers))
    return centers, radii


def rotation_to_z(angle: float) -> np.ndarray:
    """Rotation that maps the direction (sin angle, 0, cos angle) to e_z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def plane_grid(half_width: float, resolution: int) -> np.ndarray:
    """Points of the plane y = 0 in [-half_width, half_width]**2."""
    xs = np.linspace(-half_width, half_width, resolution)
    x, z = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=1)


def volume_grid(half_width: float, resolution: int) -> np.ndarray:
    """Points of the cube [-half_width, half_width]**3."""
    xs = np.linspace(-half_width, half_width, resolution)
    x, y, z = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def inside_spheres(
    points: np.ndarray, centers: List[np.ndarray], radii: np.ndarray
) -> np.ndarray:
    """True for the points inside some sphere."""
    inside = np.zeros(len(points), dtype=bool)
    for center, radius in zip(centers, radii):
        inside |= np.linalg.norm(points - center, axis=1) < radius
    return inside


def local_wavenumber(
    points: np.ndarray,
    centers: List[np.ndarray],
    radii: np.ndarray,
    k_exterior: float,
    k_interior: float,
) -> np.ndarray:
    """k_interior inside the spheres and k_exterior outside."""
    return np.where(inside_spheres(points, centers, radii), k_interior, k_exterior)


def incident_wave(points: np.ndarray, k_exterior: float, angle: float) -> np.ndarray:
    """Plane wave of direction (sin angle, 0, cos angle)."""
    direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
    return np.exp(1j * k_exterior * points @ direction)


def mie_total_field(
    points: np.ndarray,
    radius: float,
    k_exterior: float,
    k_interior: float,
    angle: float,
    big_l: int,
) -> np.ndarray:
    """Total field of a plane wave of direction (sin angle, 0, cos angle)
    scattered by one penetrable sphere at the origin (Mie series)."""
    eles = np.arange(0, big_l + 1)
    ka, kb = k_exterior * radius, k_interior * radius
    j0 = special.spherical_jn(eles, ka)
    j0p = special.spherical_jn(eles, ka, derivative=True)
    h0 = j0 + 1j * special.spherical_yn(eles, ka)
    h0p = j0p + 1j * special.spherical_yn(eles, ka, derivative=True)
    j1 = special.spherical_jn(eles, kb)
    j1p = special.spherical_jn(eles, kb, derivative=True)
    alpha = 1j**eles * (2 * eles + 1)
    determinant = -h0 * k_interior * j1p + j1 * k_exterior * h0p
    c = alpha * (j0 * k_interior * j1p - j1 * k_exterior * j0p) / determinant
    b = alpha * (k_exterior * (h0p * j0 - h0 * j0p)) / determinant
    rotated = points @ rotation_to_z(angle).T
    rho = np.linalg.norm(rotated, axis=1)
    cos_theta = np.divide(rotated[:, 2], rho, out=np.ones_like(rho), where=rho > 0.0)
    legendre = special.eval_legendre(eles[:, np.newaxis], cos_theta)
    inside = rho < radius
    u = np.empty(len(points), dtype=np.complex128)
    kr = k_exterior * rho[~inside]
    h = special.spherical_jn(eles[:, np.newaxis], kr) + 1j * special.spherical_yn(
        eles[:, np.newaxis], kr
    )
    u[~inside] = np.exp(1j * k_exterior * rotated[~inside, 2]) + np.sum(
        c[:, np.newaxis] * h * legendre[:, ~inside], axis=0
    )
    jb = special.spherical_jn(eles[:, np.newaxis], k_interior * rho[inside])
    u[inside] = np.sum(b[:, np.newaxis] * jb * legendre[:, inside], axis=0)
    return u


class SpheresHelmholtzSimulator(BaseSimulator):
    """Helmholtz transmission problem for an array of disjoint spheres,
    solved with the multiple traces formulation of biosspheres."""

    def _get_equation_name(self) -> str:
        return "helmholtz_spheres_equation"

    def setup_problem(self, mesh: Any, **parameters) -> Dict[str, Any]:
        rng = np.random.default_rng(int(parameters["geometry_seed"]))
        centers, radii = sphere_array(
            int(parameters["n_side"]),
            parameters["spacing"],
            parameters["radius_min"],
            parameters["radius_max"],
            parameters["jitter"],
            rng,
        )
        half_width = (
            int(parameters["n_side"]) * parameters["spacing"] / 2 + parameters["margin"]
        )
        return {
            "centers": centers,
            "radii": radii,
            "k_exterior": parameters["wavenumber"],
            "k_interior": parameters["wavenumber"] * parameters["ref_ind"],
            "angle": parameters["angle"],
            "big_l": int(parameters["big_l"]),
            "points": plane_grid(half_width, int(parameters["resolution"])),
            "sensors": volume_grid(half_width, int(parameters["sensors"])),
        }

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        centers, radii = problem_data["centers"], problem_data["radii"]
        k_exterior, k_interior = problem_data["k_exterior"], problem_data["k_interior"]
        big_l, points = problem_data["big_l"], problem_data["points"]
        u = self.total_field(
            points, centers, radii, k_exterior, k_interior, problem_data["angle"], big_l
        )
        return self.solution_data(problem_data, u)

    def total_field(
        self,
        points: np.ndarray,
        centers: List[np.ndarray],
        radii: np.ndarray,
        k_exterior: float,
        k_interior: float,
        angle: float,
        big_l: int,
    ) -> np.ndarray:
        """Solves the MTF with the incident direction rotated to e_z and
        evaluates the total field at the points."""
        rotation = rotation_to_z(angle)
        rotated_centers = [rotation @ center for center in centers]
        n = len(radii)
        kii = np.concatenate([[k_exterior], np.full(n, k_interior)])
        x_dia, x_dia_inv = mtf.x_diagonal_with_its_inv(
            n, big_l, radii, np.ones(n), azimuthal=False
        )
        b = righthands.b_vector_n_spheres_mtf_plane_wave(
            n,
            big_l,
            rotated_centers,
            0.0,
            k_exterior,
            1.0,
            radii,
            x_dia,
            mass.n_two_j_blocks(big_l, radii, azimuthal=False),
        )
        a_0_self, a_n = helmholtzself.a_0_a_n_sparse_matrices(
            n, big_l, radii, kii, azimuthal=False
        )
        eles = np.arange(0, big_l + 1)
        kr = k_exterior * radii[:, np.newaxis]
        cross = helmholtzcross.all_cross_interactions_n_spheres_from_v_2d(
            n,
            big_l,
            2 * big_l + 10,
            k_exterior,
            radii,
            rotated_centers,
            special.spherical_jn(eles, kr),
            special.spherical_jn(eles, kr, derivative=True),
        )
        traces = np.linalg.solve(
            mtf.mtf_n_matrix(cross, a_0_self, a_n, x_dia, x_dia_inv), b
        )
        rotated_points = points @ rotation.T
        u = reconstructions.rf_helmholtz_n_spheres(
            rotated_points, n, radii, rotated_centers, kii, big_l, traces
        )
        outside = ~inside_spheres(points, centers, radii)
        u[outside] += incident_wave(points[outside], k_exterior, angle)
        return u

    def solution_data(self, problem_data: Dict[str, Any], u: np.ndarray) -> Dict:
        sensors = problem_data["sensors"]
        incident = incident_wave(
            sensors, problem_data["k_exterior"], problem_data["angle"]
        )
        return {
            "coordinates": problem_data["points"],
            "values": np.real(u),
            "field_values_imag": np.imag(u),
            "field_input_k": local_wavenumber(
                sensors,
                problem_data["centers"],
                problem_data["radii"],
                problem_data["k_exterior"],
                problem_data["k_interior"],
            ),
            "field_input_f": np.real(incident),
            "field_input_f_imag": np.imag(incident),
            "field_sensor_coordinates": sensors,
            "field_centers": np.array(problem_data["centers"]),
            "field_radii": problem_data["radii"],
        }

    def analytical_solution(self, mesh: Any, **parameters) -> Dict[str, Any]:
        """One sphere of radius radius_max at the origin: Mie series."""
        problem_data = self.setup_problem(mesh, **dict(parameters, n_side=1, jitter=0))
        problem_data["centers"] = [np.zeros(3)]
        problem_data["radii"] = np.array([parameters["radius_max"]])
        u = mie_total_field(
            problem_data["points"],
            parameters["radius_max"],
            problem_data["k_exterior"],
            problem_data["k_interior"],
            problem_data["angle"],
            problem_data["big_l"],
        )
        return self.solution_data(problem_data, u)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Helmholtz transmission by an array of spheres (biosspheres)"
    )
    parser.add_argument("--num_simulations", type=int, default=10)
    parser.add_argument("--n_side", type=int, default=3)
    parser.add_argument("--spacing", type=float, default=1.2)
    parser.add_argument("--radius_min", type=float, default=0.25)
    parser.add_argument("--radius_max", type=float, default=0.4)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--big_l", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--sensors", type=int, default=12)
    parser.add_argument("--wavenumber_min", type=float, default=1.0)
    parser.add_argument("--wavenumber_max", type=float, default=4.0)
    parser.add_argument("--ref_ind_min", type=float, default=1.1)
    parser.add_argument("--ref_ind_max", type=float, default=2.0)
    parser.add_argument("--angle_min", type=float, default=0.0)
    parser.add_argument("--angle_max", type=float, default=np.pi)
    parser.add_argument("--analytical", action="store_true")
    parser.add_argument("--output_directory", type=str, default="simulations")
    return parser.parse_args()


def main():
    args = parse_arguments()
    simulator = SpheresHelmholtzSimulator(
        mesh_size=args.resolution, output_directory=args.output_directory
    )
    parameter_ranges = {
        "wavenumber": (args.wavenumber_min, args.wavenumber_max),
        "ref_ind": (args.ref_ind_min, args.ref_ind_max),
        "angle": (args.angle_min, args.angle_max),
        "geometry_seed": (0, 2**31 - 1),
    }
    fixed_parameters = {
        key: getattr(args, key)
        for key in [
            "n_side",
            "spacing",
            "radius_min",
            "radius_max",
            "jitter",
            "margin",
            "big_l",
            "resolution",
            "sensors",
        ]
    }
    run = simulator.run_session_analytical if args.analytical else simulator.run_session
    run(None, parameter_ranges, args.num_simulations, **fixed_parameters)


if __name__ == "__main__":
    main()
