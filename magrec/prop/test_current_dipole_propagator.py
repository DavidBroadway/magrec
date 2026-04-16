import numpy as np
import torch

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
