import numpy as np
import pyvista as pv
import torch

from magrec.misc.data import Pipeset


def test_getitem_array_from_single_block_returns_tensor():
    grid = pv.ImageData(dimensions=(4, 4, 1))
    arr = np.arange(grid.n_points, dtype=float)
    grid.point_data["a"] = arr
    pipe = Pipeset(grid)

    out = pipe["a"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.detach().cpu().numpy(), arr, rtol=0, atol=0)


def test_getitem_block_array_exact():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["array1"] = np.arange(a.n_points, dtype=float)
    b = pv.ImageData(dimensions=(3, 3, 1))
    b.point_data["array1"] = np.arange(b.n_points, dtype=float) + 100.0

    pipe = Pipeset()
    pipe.add("a", a)
    pipe.add("b", b)

    out = pipe["a.array1"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.detach().cpu().numpy(), np.asarray(a.point_data["array1"]), rtol=0, atol=0)


def test_getitem_block_array_glob_within_block():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["array1"] = np.arange(a.n_points, dtype=float)
    a.point_data["array2"] = np.arange(a.n_points, dtype=float) + 10.0
    a.point_data["blah"] = np.arange(a.n_points, dtype=float) + 20.0
    b = pv.ImageData(dimensions=(3, 3, 1))

    pipe = Pipeset()
    pipe.add("a", a)
    pipe.add("b", b)

    out = pipe["a.array*"]
    assert isinstance(out, list)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, torch.Tensor)


def test_getitem_block_and_array_glob_across_blocks():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["array1"] = np.arange(a.n_points, dtype=float)
    a.point_data["array2"] = np.arange(a.n_points, dtype=float) + 10.0
    b = pv.ImageData(dimensions=(3, 3, 1))
    b.point_data["array1"] = np.arange(b.n_points, dtype=float) + 100.0

    pipe = Pipeset()
    pipe.add("a", a)
    pipe.add("b", b)

    out = pipe["*.array*"]
    assert isinstance(out, list)
    assert len(out) == 3
    for item in out:
        assert isinstance(item, torch.Tensor)


def test_getitem_nested_dotted_glob_and_exact():
    c_block = pv.ImageData(dimensions=(2, 2, 1))
    c_block.point_data["c"] = np.arange(c_block.n_points, dtype=float)
    c1_block = pv.ImageData(dimensions=(2, 2, 1))
    c1_block.point_data["c"] = np.arange(c1_block.n_points, dtype=float) + 10.0
    c2_block = pv.ImageData(dimensions=(2, 2, 1))
    c2_block.point_data["c2"] = np.arange(c2_block.n_points, dtype=float) + 20.0

    a_nested = pv.MultiBlock()
    a_nested.append(c_block, "b")
    a_nested.append(c1_block, "b1")
    a_nested.append(c2_block, "b2")

    pipe = Pipeset()
    pipe.add("a", a_nested)

    glob_out = pipe["a.b*.c*"]
    assert isinstance(glob_out, list)
    assert len(glob_out) == 3
    for item in glob_out:
        assert isinstance(item, torch.Tensor)

    exact_out = pipe["a.b.c"]
    assert isinstance(exact_out, torch.Tensor)
    np.testing.assert_allclose(
        exact_out.detach().cpu().numpy(),
        np.asarray(c_block.point_data["c"]),
        rtol=0,
        atol=0,
    )
