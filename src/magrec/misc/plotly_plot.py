"""
Plotting functions that use Plotly backend, mostly for interactions and 3d plots.
"""

import plotly.graph_objects as go
import torch
import numpy as np


def get_cones(xyz: torch.Tensor | np.ndarray, uvw: torch.Tensor | np.ndarray,
              **kwargs) -> go.Cone:
    """
    Create Cones object from plotly.graphical_objects to be added as a trace to a figure.

    Cones represent direction and amplitude of a vector field with vector coordinates [u, v, w] at positions [x, y, z]

    Parameters
    ----------
    uvw : Vector field components in the shape (3, n1, n2, n3), where (n1, n2, n3) is the shape of the grid of positions
    xyz : Vector of the grid coordinates in the shape (3, n1, n2, n3)
    kwargs

    Returns
    -------
    go.Cones
    """
    uvw = torch.Tensor(uvw).refine_names('component', 'x', 'y', 'z')
    xyz = torch.Tensor(xyz).refine_names('component', 'x', 'y', 'z')
    u, v, w = uvw.flatten(['x', 'y', 'z'], 'point').align_to('component',
                                                             'point')
    x, y, z = xyz.flatten(['x', 'y', 'z'], 'point').align_to('component',
                                                             'point')

    default_kwargs = dict(colorscale='Blues',
                          sizemode="scaled",
                          sizeref=0.7,
                          anchor='center',)

    for key, value in default_kwargs.items():
        kwargs.setdefault(key, value)

    return go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        **kwargs
    )


def plot_vector_field(xyz, uvw, show=False, **kwargs):
    """
    Plot vector field defined by vectors with components uvw at positions xyz.  
    """
    import plotly.graph_objects as go

    cones = get_cones(xyz, uvw, **kwargs)
    
    fig = go.Figure(data=cones)

    if show:
        fig.show()

    return fig
