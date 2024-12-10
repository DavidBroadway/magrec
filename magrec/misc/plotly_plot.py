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

def plot_vector_field_streamtube(xyz, uvw, starts = None, show=False, **kwargs):
    """
    Plot vector field defined by vectors with components uvw at positions xyz.  
    """
    import plotly.graph_objects as go
    
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]
    
    # Use provided `starts`, but if None, select randomly provided points from `xyz` and corresponding `uvw`
    if starts is None:
        idx = np.random.choice(len(x), 16, replace=False)
        starts = dict(
            x = x[idx],
            y = y[idx],
            z = z[idx]
        )
    
    tube = go.Streamtube(    
        x = x,
        y = y,
        z = z,
        u = u,
        v = v,
        w = w,
        starts = starts,
        sizeref = 1,
        colorscale = 'Portland',
        showscale = False,
        # maxdisplayed = 3000
    )
    
    fig = go.Figure(data=tube)

    if show:
        fig.show()

    return fig


def visualize_vector_fields(magnetic_field, current_distribution, z1, z0):
    """
    Visualizes the magnetic field and current distribution in 3D.

    Parameters:
    - magnetic_field: tensor of shape (3, W, H) representing magnetic field vectors
    - current_distribution: tensor of shape (2, W, H) representing current distribution vectors
    - z1: z-coordinate for the magnetic field
    - z0: z-coordinate for the current distribution
    """

    # Grid points in x and y direction
    W, H = magnetic_field.shape[1], magnetic_field.shape[2]
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')

    # Magnetic field vectors
    x_m, y_m, z_m = x.flatten(), y.flatten(), np.full(W*H, z1)
    u_m, v_m, w_m = magnetic_field[0].flatten(), magnetic_field[1].flatten(), magnetic_field[2].flatten()

    # Norm of the magnetic field vectors
    norm_m = np.sqrt(u_m**2 + v_m**2 + w_m**2)
    
    # Hover text for magnetic field
    hover_text_m = ['Norm: {:.4f}<br>Coordinates: ({}, {}, {})<br>Components: ({:.4f}, {:.4f}, {:.4f})'.format(
        norm, x, y, z1, u, v, w) for norm, x, y, u, v, w in zip(norm_m, x_m, y_m, u_m, v_m, w_m)]

    # Current distribution vectors
    x_c, y_c, z_c = x.flatten(), y.flatten(), np.full(W*H, z0)
    u_c, v_c = current_distribution[0].flatten(), current_distribution[1].flatten()

    # Norm of the current distribution vectors
    norm_c = np.sqrt(u_c**2 + v_c**2)
    
    # Hover text for current distribution
    hover_text_c = ['Norm: {:.4f}<br>Coordinates: ({}, {}, {})<br>Components: ({:.4f}, {:.4f})'.format(
        norm, x, y, z0, u, v) for norm, x, y, u, v in zip(norm_c, x_c, y_c, u_c, v_c)]

    # Create figure
    fig = go.Figure()

    # Add magnetic field vectors as cones
    fig.add_trace(go.Cone(x=x_m, y=y_m, z=z_m,
                          u=u_m, v=v_m, w=w_m,
                          sizeref=1, anchor="tail", showscale=False,
                          name="Magnetic Field",
                          text=hover_text_m, hoverinfo='text'))

    # Add current distribution vectors as cones
    fig.add_trace(go.Cone(x=x_c, y=y_c, z=z_c,
                          u=u_c, v=v_c, w=np.zeros(W*H),
                          sizeref=1, anchor="tail", showscale=False,
                          name="Current Distribution",
                          text=hover_text_c, hoverinfo='text'))

    # Set the 3D scene configurations
    fig.update_scenes(aspectmode="cube")

    zrange = z1 - z0

    # Set layout and title
    fig.update_layout(title="Vector Fields Visualization in 3D",
                      scene=dict(zaxis=dict(range=[z0 - zrange * 0.1, z1 + zrange * 0.1])))

    # Show figure
    fig.show()
