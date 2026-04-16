import numpy as np
import torch
import time

from magrec.prop.Propagator import CurrentDipolePropagator


def _make_scene(dtype=torch.float32):
    r_source = torch.tensor(
        [[0.0, 0.0, -0.5], [0.4, -0.2, -0.3]],
        dtype=dtype,
    )
    r_sensor = torch.tensor(
        [[0.1, 0.0, 0.2], [0.3, 0.4, 0.6], [-0.2, 0.1, 0.7]],
        dtype=dtype,
    )
    J = torch.tensor(
        [[0.0, 1.0, 0.2], [0.5, -0.1, 0.3]],
        dtype=dtype,
    )
    return r_source, r_sensor, J


def test_current_dipole_matrix_shape_and_reshape():
    r_source, r_sensor, _ = _make_scene()
    prop = CurrentDipolePropagator(r_source, r_sensor, backend="torch", method="matrix")
    assert prop.ffm.shape == (r_sensor.shape[0], r_source.shape[0], 3, 3)

    ffm_matrix = CurrentDipolePropagator.reshape_ffm_to_matrix(prop.ffm)
    assert ffm_matrix.shape == (3 * r_sensor.shape[0], 3 * r_source.shape[0])


def test_current_dipole_zero_for_parallel_geometry():
    r_source = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    r_sensor = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    J = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    prop = CurrentDipolePropagator(r_source, r_sensor, method="matrix", dtype=torch.float64)
    B = prop(J)
    torch.testing.assert_close(B, torch.zeros_like(B), rtol=1e-10, atol=1e-12)


def test_current_dipole_sign_and_linearity():
    r_source, r_sensor, J = _make_scene(dtype=torch.float64)
    prop = CurrentDipolePropagator(r_source, r_sensor, method="matrix", dtype=torch.float64)
    B = prop(J)
    B_neg = prop(-J)
    torch.testing.assert_close(B_neg, -B, rtol=1e-9, atol=1e-11)

    alpha = 3.5
    B_scaled = prop(alpha * J)
    torch.testing.assert_close(B_scaled, alpha * B, rtol=1e-9, atol=1e-11)


def test_current_dipole_superposition():
    r_source, r_sensor, J = _make_scene(dtype=torch.float64)
    prop = CurrentDipolePropagator(r_source, r_sensor, method="matrix", dtype=torch.float64)
    J1 = J.clone()
    J1[1] = 0.0
    J2 = J.clone()
    J2[0] = 0.0

    B_full = prop(J)
    B_split = prop(J1) + prop(J2)
    torch.testing.assert_close(B_full, B_split, rtol=1e-9, atol=1e-11)


def test_current_dipole_instance_matches_static_helper():
    r_source, r_sensor, J = _make_scene(dtype=torch.float64)
    prop = CurrentDipolePropagator(r_source, r_sensor, method="matrix", dtype=torch.float64)
    B_instance = prop.get_B_from_J(J)
    B_static = CurrentDipolePropagator.get_B_at_pts_from_J_at_pts(J, r_source, r_sensor)
    torch.testing.assert_close(B_instance, B_static, rtol=1e-9, atol=1e-11)


def test_current_dipole_unsupported_method_raises():
    r_source, r_sensor, _ = _make_scene()
    try:
        CurrentDipolePropagator(r_source, r_sensor, method="unknown")
    except ValueError as exc:
        assert "Unsupported method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported method")


def test_current_dipole_torch_matrix_vs_numba_iterative():
    r_source, r_sensor, J = _make_scene(dtype=torch.float32)
    prop_matrix = CurrentDipolePropagator(r_source, r_sensor, backend="torch", method="matrix")
    prop_numba = CurrentDipolePropagator(r_source, r_sensor, backend="numba", method="iterative")

    B_matrix = prop_matrix(J).detach().cpu().numpy()
    B_numba = prop_numba(J.detach().cpu().numpy())
    np.testing.assert_allclose(B_numba, B_matrix, rtol=3e-4, atol=1e-6)


def test_current_dipole_memory_guard():
    r_source = torch.zeros((50, 3), dtype=torch.float32)
    r_sensor = torch.zeros((50, 3), dtype=torch.float32)
    old_limit = CurrentDipolePropagator.MAX_FFM_SIZE_IN_MB
    CurrentDipolePropagator.MAX_FFM_SIZE_IN_MB = 1e-6
    try:
        try:
            CurrentDipolePropagator(r_source, r_sensor, method="matrix")
        except RuntimeError as exc:
            assert "Expected size of the forward-field matrix" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for too-large matrix")
    finally:
        CurrentDipolePropagator.MAX_FFM_SIZE_IN_MB = old_limit


def test_current_dipole_flat_current_surface_matrix():
    """Test the current dipole propagator with a flat current surface. For a flat current surface, 
    the magnetic field depends only on the surface current density. 

    B = mu0 / 2 * kappa
    where kappa is the surface current density.
    
    We create:
    - source slab centered at z=0 with size 1 cm x 1 cm and thickness 0.1 mm
    - uniform volume current density j along +x
    - one sensor at slab center, 2 mm above the slab

    For volume current density j = 1 mA/mm^2 in x direction, the surface current density is
    
    kappa = j * thickness * dy
    
    for thickness = 0.1 mm, the target for the value is:
    
    B = mu0 / 2 * 0.1 mA/mm^2 * 0.1 mm = 6.2831853071795862e-6 mT
    
    with B directed along -y for +x current and +z observation side.
    """
    # Geometry in magrec default length units (mm).
    a = 10.0          # 1 cm
    thickness = 0.1   # 0.1 mm
    nx, ny = 100, 100
    sensor_z = 2.0    # 2 mm above slab center

    # Uniform volume current density in A/mm^2: 1 mA/mm^2.
    j = 1.0e-3

    x = torch.linspace(-0.5 * a, 0.5 * a, nx, dtype=torch.float64)
    y = torch.linspace(-0.5 * a, 0.5 * a, ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.zeros_like(xx)
    r_source = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)

    # Each discrete source carries current dipole moment j * dV along +x.
    dx = float(a / (nx - 1))
    dy = float(a / (ny - 1))
    dV = dx * dy * thickness
    J = torch.zeros((r_source.shape[0], 3), dtype=torch.float64)
    J[:, 0] = j * dV

    r_sensor = torch.tensor([[0.0, 0.0, sensor_z]], dtype=torch.float64)

    prop = CurrentDipolePropagator(r_source, r_sensor, method="matrix", dtype=torch.float64)
    B = prop(J)[0]

    # Direction checks from J x tau symmetry.
    assert B[1] < 0.0
    assert abs(float(B[0])) < 1e-12
    assert abs(float(B[2])) < 1e-12

    # Magnitude check versus infinite-sheet approximation in mT.
    mu0 = 1.25663706212  # [mT * mm / A]
    kappa = j * thickness
    expected = mu0 * kappa / 2.0
    measured = abs(float(B[1]))

    # Finite-size slab should stay in the same order and close at center.
    np.testing.assert_allclose(measured, expected, rtol=0.40, atol=0.0)
    

def test_current_dipole_flat_current_surface_iterative():
    """Test the current dipole propagator with a flat current surface via 'iterative' method. 
    May take long time to run! Outputs the time taken to run the computation.
    
    For a flat current surface, the magnetic field depends only on the surface current density. 

    B = mu0 / 2 * kappa
    where kappa is the surface current density.
    
    We create:
    - source slab centered at z=0 with size 1 cm x 1 cm and thickness 0.1 mm
    - uniform volume current density j along +x
    - one sensor at slab center, 2 mm above the slab

    For volume current density j = 1 mA/mm^2 in x direction, the surface current density is
    
    kappa = j * thickness * dy
    
    for thickness = 0.1 mm, the target for the value is:
    
    B = mu0 / 2 * 0.1 mA/mm^2 * 0.1 mm = 6.2831853071795862e-6 mT
    
    with B directed along -y for +x current and +z observation side.
    """
    # Geometry in magrec default length units (mm).
    a = 10.0          # 1 cm
    thickness = 0.1   # 0.1 mm
    nx, ny = 100, 100
    sensor_z = 2.0    # 2 mm above slab center

    # Uniform volume current density in A/mm^2: 1 mA/mm^2.
    j = 1.0e-3

    x = torch.linspace(-0.5 * a, 0.5 * a, nx, dtype=torch.float64)
    y = torch.linspace(-0.5 * a, 0.5 * a, ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.zeros_like(xx)
    r_source = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)

    # Each discrete source carries current dipole moment j * dV along +x.
    dx = float(a / (nx - 1))
    dy = float(a / (ny - 1))
    dV = dx * dy * thickness
    J = torch.zeros((r_source.shape[0], 3), dtype=torch.float64)
    J[:, 0] = j * dV

    r_sensor = torch.tensor([[0.0, 0.0, sensor_z]], dtype=torch.float64)

    prop = CurrentDipolePropagator(r_source, r_sensor, method="iterative", dtype=torch.float64)
    t0 = time.perf_counter()
    B = prop(J)[0]
    dt = time.perf_counter() - t0
    print(f"[timing] iterative forward: {dt:.6f} s (n_source={r_source.shape[0]}, n_sensor={r_sensor.shape[0]})")

    # Direction checks from J x tau symmetry.
    assert B[1] < 0.0
    assert abs(float(B[0])) < 1e-12
    assert abs(float(B[2])) < 1e-12

    # Magnitude check versus infinite-sheet approximation in mT.
    mu0 = 1.25663706212  # [mT * mm / A]
    kappa = j * thickness
    expected = mu0 * kappa / 2.0
    measured = abs(float(B[1]))

    # Finite-size slab should stay in the same order and close at center.
    np.testing.assert_allclose(measured, expected, rtol=0.40, atol=0.0)