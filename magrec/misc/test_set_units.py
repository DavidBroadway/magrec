import numpy as np
import pyvista as pv

from magrec.misc.data import Pipeset


def test_set_units_single_block_assigns_array_and_coordinates():
    grid = pv.ImageData(dimensions=(3, 3, 1))
    grid.point_data["B"] = np.arange(grid.n_points, dtype=float)
    pipe = Pipeset(grid)

    pipe.set_units(B="T", coordinates="m")

    assert pipe.units["B"] == "T"
    assert pipe.units["coordinates"] == "m"


def test_set_units_block_selector_assigns_same_array_on_multiple_blocks():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["B"] = np.arange(a.n_points, dtype=float)
    b = pv.ImageData(dimensions=(3, 3, 1))
    b.point_data["B"] = np.arange(b.n_points, dtype=float)
    c = pv.ImageData(dimensions=(3, 3, 1))
    c.point_data["B"] = np.arange(c.n_points, dtype=float)

    pipe = Pipeset()
    pipe.add("sensor1", a)
    pipe.add("sensor2", b)
    pipe.add("source", c)

    pipe.set_units(block="sensor*", B="uT")

    assert pipe.units["sensor1.B"] == "uT"
    assert pipe.units["sensor2.B"] == "uT"
    assert "source.B" not in pipe.units


def test_set_units_dotted_glob_assigns_multiple_arrays_on_one_block():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["array1"] = np.arange(a.n_points, dtype=float)
    a.point_data["array2"] = np.arange(a.n_points, dtype=float) + 1.0
    a.point_data["other"] = np.arange(a.n_points, dtype=float) + 2.0

    pipe = Pipeset()
    pipe.add("a", a)

    pipe.set_units(**{"a.array*": "mT"})

    assert pipe.units["array1"] == "mT"
    assert pipe.units["array2"] == "mT"
    assert "other" not in pipe.units


def test_set_units_dotted_glob_assigns_across_multiple_blocks():
    a = pv.ImageData(dimensions=(3, 3, 1))
    a.point_data["array1"] = np.arange(a.n_points, dtype=float)
    a.point_data["array2"] = np.arange(a.n_points, dtype=float) + 10.0
    b = pv.ImageData(dimensions=(3, 3, 1))
    b.point_data["array1"] = np.arange(b.n_points, dtype=float) + 100.0

    pipe = Pipeset()
    pipe.add("a", a)
    pipe.add("b", b)

    pipe.set_units(**{"*.array*": "A/m"})

    assert pipe.units["a.array1"] == "A/m"
    assert pipe.units["a.array2"] == "A/m"
    assert pipe.units["b.array1"] == "A/m"
