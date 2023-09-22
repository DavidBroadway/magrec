import torch
from magrec.misc.test_helpers import auto_reference_plot


@auto_reference_plot
def test_vector_field_2d():
    """Tests that the function visualize_vector_field_2d works as expected."""
    from magrec.misc.plot import plot_vector_field_2d

    # Create a diverging flow field
    W, H = 30, 20
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
    assert x.shape == (W, H), "x has the wrong shape of {}".format(x.shape)
    assert y.shape == (W, H), "y has the wrong shape of {}".format(y.shape)
    u = x - W // 2  # Adjusting x to make the flow diverge from the central y-axis
    v = 0.3 * y + 0.01 * y**2

    current_distribution = torch.stack([u, v], dim=0)
    assert current_distribution.shape == (
        2,
        W,
        H,
    ), "current_distribution has the wrong shape of {}".format(
        current_distribution.shape
    )
    fig = plot_vector_field_2d(current_distribution, show=True)
    return fig


@auto_reference_plot
def test_vector_field_2d_sampled():
    """Tests that the function visualize_vector_field_2d works as expected for the case of
    sampled input vector field."""
    from magrec.misc.plot import plot_vector_field_2d

    # Create a diverging flow field
    W, H = 30, 20
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
    assert x.shape == (W, H), "x has the wrong shape of {}".format(x.shape)
    assert y.shape == (W, H), "y has the wrong shape of {}".format(y.shape)
    u = x - W // 2  # Adjusting x to make the flow diverge from the central y-axis
    v = 0.3 * y + 0.01 * y**2

    current_distribution = torch.stack([u, v], dim=0)
    assert current_distribution.shape == (
        2,
        W,
        H,
    ), "current_distribution has the wrong shape of {}".format(
        current_distribution.shape
    )
    fig = plot_vector_field_2d(current_distribution, num_arrows=10, show=True)
    return fig
