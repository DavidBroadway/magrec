import torch
import numpy as np
from magrec.misc.plot import plot_vector_field_2d


def test_vector_field_2d():
    """Tests that the function visualize_vector_field_2d works as expected."""
    # Create a diverging flow field
    W, H = 20, 20
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    u = x - W // 2  # Adjusting x to make the flow diverge from the central y-axis
    v = 0.3 * y + 0.01 * y ** 2

    current_distribution = np.array([u, v])
    plot_vector_field_2d(current_distribution)