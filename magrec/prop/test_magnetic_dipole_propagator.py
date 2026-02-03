import numpy as np
import torch
import matplotlib.pyplot as plt

from magrec.prop.Propagator import MagneticDipolePropagator
from magrec.misc.sampler import GridSampler
from magrec.misc.test_helpers import auto_reference_plot


def _make_scene():
    """
    Small, deterministic setup so indexing can be checked visually and numerically.
    """
    r_source = torch.tensor([[0.2, -0.1, -0.5]], dtype=torch.float32)
    r_sensor = GridSampler.sample_grid(
        nx_points=8,
        ny_points=6,
        origin=[-1.0, -1.0],
        diagonal=[1.0, 1.0],
        z=0.0,
    ).to(torch.float32)
    m = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    return r_source, r_sensor, m


def test_magnetic_dipole_forward_torch_and_numba():
    r_source, r_sensor, m = _make_scene()

    prop_torch = MagneticDipolePropagator(r_source, r_sensor, backend="torch", method="matrix")
    B_torch = prop_torch(m)

    prop_numba = MagneticDipolePropagator(r_source, r_sensor, backend="numba")
    B_numba = prop_numba(m.detach().cpu().numpy())

    assert isinstance(B_torch, torch.Tensor), "Torch backend should return torch.Tensor"
    assert isinstance(B_numba, np.ndarray), "Numba backend with numpy input should return numpy.ndarray"
    assert B_torch.shape == B_numba.shape, "Torch/numba outputs must have same shape"

    # Numerical agreement
    np.testing.assert_allclose(B_torch.detach().cpu().numpy(), B_numba, rtol=1e-4, atol=1e-6)

    # Indexing check: maximum |B| should occur at closest sensor point to the source
    B_mag = torch.norm(B_torch, dim=1)
    idx_max = int(torch.argmax(B_mag).item())
    dists = torch.norm(r_sensor - r_source[0], dim=1)
    idx_min = int(torch.argmin(dists).item())
    assert idx_max == idx_min, "Indexing/order mismatch: max field not at closest sensor"


@auto_reference_plot
def test_magnetic_dipole_forward_plot():
    r_source, r_sensor, m = _make_scene()

    prop_torch = MagneticDipolePropagator(r_source, r_sensor, backend="torch", method="matrix")
    B_torch = prop_torch(m)

    # Project to Bz and reshape into grid for visual inspection of index order
    Bz = B_torch[:, 2]
    grid = GridSampler.pts_to_grid(Bz, nx_points=8, ny_points=6)[0]
    xy_grid = GridSampler.pts_to_grid(r_sensor[:, :2], nx_points=8, ny_points=6)
    x_grid = xy_grid[0]
    y_grid = xy_grid[1]

    # Find max index (should align with closest sensor to source)
    idx_max = int(torch.argmax(Bz).item())
    max_pt = r_sensor[idx_max]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(x_grid, y_grid, grid, shading="auto", cmap="viridis")
    ax.scatter([r_source[0, 0]], [r_source[0, 1]], c="r", marker="x", label="source")
    ax.scatter([max_pt[0]], [max_pt[1]], c="w", marker="o", edgecolors="k", label="max |B|")
    ax.set_title("MagneticDipolePropagator Bz grid (indexing check)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Bz")
    fig.tight_layout()
    return fig
