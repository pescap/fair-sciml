import h5py
import numpy as np
import pytest

from simulators.spheres_helmholtz_simulator import (
    SpheresHelmholtzSimulator,
    mie_total_field,
    plane_grid,
    sphere_array,
)


@pytest.mark.parametrize("angle", [0.0, 0.7, 2.5])
def test_mtf_one_sphere_matches_mie(angle):
    simulator = SpheresHelmholtzSimulator(output_directory="/tmp/unused")
    points = plane_grid(1.5, 15)
    radius, k_exterior, k_interior, big_l = 0.6, 3.0, 4.5, 20
    u = simulator.total_field(
        points, [np.zeros(3)], np.array([radius]), k_exterior, k_interior, angle, big_l
    )
    exact = mie_total_field(points, radius, k_exterior, k_interior, angle, big_l)
    assert np.max(np.abs(u - exact)) / np.max(np.abs(exact)) < 1e-10


def test_sphere_array_is_disjoint():
    centers, radii = sphere_array(4, 1.0, 0.2, 0.4, 0.05, np.random.default_rng(0))
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            assert np.linalg.norm(centers[i] - centers[j]) > radii[i] + radii[j]
    with pytest.raises(ValueError):
        sphere_array(2, 1.0, 0.2, 0.5, 0.05, np.random.default_rng(0))


@pytest.mark.parametrize("analytical", [False, True])
def test_session_writes_the_dataset(tmp_path, analytical):
    simulator = SpheresHelmholtzSimulator(mesh_size=8, output_directory=str(tmp_path))
    fixed = dict(
        n_side=2,
        spacing=1.2,
        radius_min=0.3,
        radius_max=0.4,
        jitter=0.05,
        margin=0.5,
        big_l=3,
        resolution=8,
        sensors=4,
    )
    ranges = {
        "wavenumber": (1.0, 2.0),
        "ref_ind": (1.1, 1.5),
        "angle": (0.0, np.pi),
        "geometry_seed": (0, 1000),
    }
    run = simulator.run_session_analytical if analytical else simulator.run_session
    run(None, ranges, 2, **fixed)
    with h5py.File(simulator.output_path, "r") as h5file:
        simulations = [s for g in h5file.values() for s in g.values()]
        assert len(simulations) == 2
        for sim in simulations:
            assert sim["coordinates"].shape == (64, 3)
            assert sim["values"].shape == (64,)
            assert sim["field_values_imag"].shape == (64,)
            assert sim["field_input_k"].shape == (64,)
            assert sim["field_input_f"].shape == (64,)
            assert np.all(np.isfinite(sim["values"][:]))
            assert "parameter_wavenumber" in sim.attrs
