import numpy as np
import pyvista as pv
import pytest

from magrec.misc.data import Pipeset


def _make_pipe_with_points():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    pipe = Pipeset()
    pipe.add("sensor", pv.PolyData(pts))
    return pipe


def test_to_units_converts_coordinates_mm_to_cm():
    pipe = _make_pipe_with_points()
    pipe.set_units(coordinates="mm")
    pipe.to_units(coordinates="cm")

    pts = np.asarray(pipe.points)
    np.testing.assert_allclose(pts[1], np.array([1.0, 0.0, 0.0]), rtol=0, atol=1e-12)
    assert pipe.units["coordinates"] == "cm"


def test_scale_coordinates_shorthand_source_keyword():
    pipe = _make_pipe_with_points()
    pipe.set_units(coordinates="mm")
    pipe.scale("coordinates", to_units="cm")

    pts = np.asarray(pipe.points)
    np.testing.assert_allclose(pts[1], np.array([1.0, 0.0, 0.0]), rtol=0, atol=1e-12)
    assert pipe.units["coordinates"] == "cm"


def test_scale_to_units_delegates_for_point_data():
    pipe = _make_pipe_with_points()
    block = pipe.resolve_name(None)
    block.point_data["B"] = np.array([1.0, 2.0], dtype=float)
    pipe.set_units(B="mT")

    pipe.scale("B", to_units="uT")
    np.testing.assert_allclose(np.asarray(block.point_data["B"]), np.array([1000.0, 2000.0]), rtol=0, atol=1e-12)
    assert pipe.units["B"] == "uT"


def test_coordinates_to_incompatible_units_raises():
    pipe = _make_pipe_with_points()
    pipe.set_units(coordinates="mm")
    with pytest.raises(ValueError, match="base dimensions differ"):
        pipe.scale("coordinates", to_units="uT")
