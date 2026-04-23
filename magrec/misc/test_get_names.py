import numpy as np
import pyvista as pv
from magrec.misc.data import Pipeset


def test_get_names_single_block_grid_arrays_glob_and_exact():
    grid = pv.ImageData(dimensions=(10, 10, 1))
    for name in ("a", "b", "c", "a1", "a2"):
        grid.point_data[name] = np.random.default_rng(0).random(grid.n_points)

    pipe = Pipeset(grid)

    assert pipe.get_names("a*") == ["a", "a1", "a2"]
    assert pipe.get_names() == ["a", "a1", "a2", "b", "c"]
    assert pipe.get_names("b") == ["b"]


def test_get_names_includes_field_data_but_skips_units_suffix():
    grid = pv.ImageData(dimensions=(2, 2, 1))
    grid.point_data["x"] = np.zeros(grid.n_points)
    grid.field_data["note"] = np.array([1.0])
    grid.field_data["note_units"] = ["m"]

    pipe = Pipeset(grid)

    assert pipe.get_names() == ["note", "x"]


def test_get_names_three_top_level_blocks_returns_block_names_only():
    s1 = pv.ImageData(dimensions=(3, 3, 1))
    s1.point_data["B"] = np.zeros(s1.n_points)
    s1.cell_data["J"] = np.zeros(s1.n_cells)
    s2 = pv.ImageData(dimensions=(3, 3, 1))
    s2.point_data["B"] = np.zeros(s2.n_points)
    s2.cell_data["J"] = np.zeros(s2.n_cells)
    src = pv.ImageData(dimensions=(2, 2, 1))
    src.point_data["m"] = np.zeros(src.n_points)

    pipe = Pipeset()
    pipe.add("sensor1", s1)
    pipe.add("sensor2", s2)
    pipe.add("source", src)

    assert pipe.get_names() == ["sensor1", "sensor2", "source"]
    assert pipe.get_names("sensor*") == ["sensor1", "sensor2"]


def test_get_names_empty_pipeset():
    pipe = Pipeset()
    assert pipe.get_names() == []


def test_get_names_single_multiblock_name():
    """If Pipeset wraps a named MultiBlock, that name is returned."""
    inner = pv.MultiBlock()
    inner.append(pv.ImageData(dimensions=(2, 2, 1)), "left")
    inner.append(pv.ImageData(dimensions=(2, 2, 1)), "right")
    outer = Pipeset()
    outer.add("wrap", inner)

    assert outer.get_names() == ["wrap"]
