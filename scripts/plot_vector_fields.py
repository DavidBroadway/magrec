import numpy as np
import pyvista as pv
import torch
from magrec.plot.plot import plot_vector_field_2d

"""
This script is to visualize vector field obtained from measuring the magnetic field. 
It optionally shows the underlaying current distribution, if provided. 

The principle is the same as in `plot_vector_field_2d` in `magrec/misc/plot.py` but for 3D, 
in particular, 

- it shows color map of the magnitude of the vector field
- it shows downsampled arrows representing the vector field direction
"""

def visualize_vector_fields(magnetic_field, current_distribution=None, z1=None, z0=None, 
                           num_arrows=20, show_inset=False, inset_region=None, 
                           window_size=(800, 600), interactive=True):
    """
    Visualizes the magnetic field and current distribution using PyVista library.
    Creates a color map of amplitude with arrows on top showing direction.

    Parameters:
    - magnetic_field: tensor of shape (3, W, H) representing magnetic field vectors
    - current_distribution: tensor of shape (2, W, H) representing current distribution vectors (optional)
    - z1: z-coordinate for the magnetic field (optional)
    - z0: z-coordinate for the current distribution (optional)
    - num_arrows: approximate number of arrows to display (default: 20)
    - show_inset: whether to show an inset zoom region (default: False)
    - inset_region: tuple of ((x0, y0), (x1, y1)) for inset region (default: None)
    - window_size: tuple of (width, height) for the window (default: (800, 600))
    - interactive: whether to show interactive window (default: True)
    """
    
    # Convert tensors to numpy if needed
    if isinstance(magnetic_field, torch.Tensor):
        magnetic_field = magnetic_field.detach().cpu().numpy()
    if current_distribution is not None and isinstance(current_distribution, torch.Tensor):
        current_distribution = current_distribution.detach().cpu().numpy()
    
    # Get dimensions
    W, H = magnetic_field.shape[1], magnetic_field.shape[2]
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    # Calculate magnitudes for color mapping
    if magnetic_field.shape[0] == 3:
        # 3D magnetic field
        mag_magnitudes = np.sqrt(magnetic_field[0]**2 + magnetic_field[1]**2 + magnetic_field[2]**2)
    else:
        # 2D magnetic field
        mag_magnitudes = np.sqrt(magnetic_field[0]**2 + magnetic_field[1]**2)
    
    # Create PyVista plotter
    plotter = pv.Plotter(window_size=window_size)
    
    mag_grid = pv.ImageData(dimensions=(W, H, 1))
    mag_grid.point_data['magnitude'] = mag_magnitudes.T.flatten()
    mag_grid = mag_grid.points_to_cells()
    plotter.add_mesh(mag_grid, scalars='magnitude', cmap='plasma', show_edges=False)
    
    # Create arrow grid for magnetic field
    step_size = max(W, H) // num_arrows
    x_centers = np.arange(step_size // 2, W, step_size)
    y_centers = np.arange(step_size // 2, H, step_size)
    
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # Average magnetic field vectors for arrows
    avg_u_m = np.zeros((len(x_centers), len(y_centers)))
    avg_v_m = np.zeros((len(x_centers), len(y_centers)))
    avg_w_m = np.zeros((len(x_centers), len(y_centers)))
    avg_m_m = np.zeros((len(x_centers), len(y_centers)))
    
    for i, x_center in enumerate(x_centers):
        for j, y_center in enumerate(y_centers):
            x_low, x_high = x_center - step_size // 2, x_center + step_size // 2 + 1
            y_low, y_high = y_center - step_size // 2, y_center + step_size // 2 + 1
            
            avg_u_m[i, j] = np.mean(magnetic_field[0, x_low:x_high, y_low:y_high])
            avg_v_m[i, j] = np.mean(magnetic_field[1, x_low:x_high, y_low:y_high])
            if magnetic_field.shape[0] == 3:
                avg_w_m[i, j] = np.mean(magnetic_field[2, x_low:x_high, y_low:y_high])
            avg_m_m[i, j] = np.sqrt(avg_u_m[i, j]**2 + avg_v_m[i, j]**2 + avg_w_m[i, j]**2)
    
    # Scale arrows appropriately (similar to plot_vector_field_2d)
    scale_factor = 1.1 * avg_m_m.max() / num_arrows  # Scale to make longest arrow fit within spacing
    
    # Add magnetic field arrows
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            if avg_m_m[i, j] > 0:  # Only create arrows where magnitude is non-zero
                if z1 is not None:
                    start_pos = [x_grid[i, j], y_grid[i, j], z1]
                    direction = [avg_u_m[i, j] / scale_factor, 
                               avg_v_m[i, j] / scale_factor, 
                               avg_w_m[i, j] / scale_factor]
                else:
                    start_pos = [x_grid[i, j], y_grid[i, j], 0]
                    direction = [avg_u_m[i, j] / scale_factor, 
                               avg_v_m[i, j] / scale_factor, 
                               0]
                
                # Create arrow with appropriate size
                arrow = pv.Arrow(
                    start=start_pos, 
                    direction=direction, 
                    scale=scale_factor,  # Use scale=1.0 since we already scaled the direction
                    tip_length=0.25,
                    tip_radius=0.1,
                    shaft_radius=0.05
                )
                plotter.add_mesh(arrow, color='black', opacity=0.8)
    
    # Add current distribution if provided
    if current_distribution is not None:
        # Calculate current magnitudes
        current_magnitudes = np.sqrt(current_distribution[0]**2 + current_distribution[1]**2)
        
        # Add current distribution magnitude as a surface
        if z0 is not None:
            z_coords_current = np.full((W, H), z0)
            current_grid = pv.StructuredGrid()
            current_grid.points = np.column_stack([x.flatten(), y.flatten(), z_coords_current.flatten()])
            current_grid.dimensions = [W, H, 1]
            current_grid.point_data['magnitude'] = current_magnitudes.flatten()
            plotter.add_mesh(current_grid, scalars='magnitude', cmap='viridis', show_edges=False)
        else:
            current_grid = pv.StructuredGrid()
            current_grid.points = np.column_stack([x.flatten(), y.flatten(), (np.zeros_like(x) - 0.1).flatten()])
            current_grid.dimensions = [W, H, 1]
            current_grid.point_data['magnitude'] = current_magnitudes.flatten()
            plotter.add_mesh(current_grid, scalars='magnitude', cmap='viridis', show_edges=False)
        
        # Create arrow grid for current distribution
        avg_u_c = np.zeros((len(x_centers), len(y_centers)))
        avg_v_c = np.zeros((len(x_centers), len(y_centers)))
        avg_m_c = np.zeros((len(x_centers), len(y_centers)))
        
        for i, x_center in enumerate(x_centers):
            for j, y_center in enumerate(y_centers):
                x_low, x_high = x_center - step_size // 2, x_center + step_size // 2 + 1
                y_low, y_high = y_center - step_size // 2, y_center + step_size // 2 + 1
                
                avg_u_c[i, j] = np.mean(current_distribution[0, x_low:x_high, y_low:y_high])
                avg_v_c[i, j] = np.mean(current_distribution[1, x_low:x_high, y_low:y_high])
                avg_m_c[i, j] = np.sqrt(avg_u_c[i, j]**2 + avg_v_c[i, j]**2)
        
        # Scale current arrows (similar to plot_vector_field_2d)
        scale_factor_current = 1.1 * avg_m_c.max() * num_arrows
        
        # Add current distribution arrows
        for i in range(len(x_centers)):
            for j in range(len(y_centers)):
                if avg_m_c[i, j] > 0:  # Only create arrows where magnitude is non-zero
                    if z0 is not None:
                        start_pos = [x_grid[i, j], y_grid[i, j], z0]
                        direction = [avg_u_c[i, j] / scale_factor_current, 
                                   avg_v_c[i, j] / scale_factor_current, 
                                   0]
                    else:
                        start_pos = [x_grid[i, j], y_grid[i, j], -0.1]  # Slightly below magnetic field
                        direction = [avg_u_c[i, j] / scale_factor_current, 
                                   avg_v_c[i, j] / scale_factor_current, 
                                   0]
                    
                    # Create arrow with appropriate size
                    arrow = pv.Arrow(
                        start=start_pos, 
                        direction=direction, 
                        scale=1.0,  # Use scale=1.0 since we already scaled the direction
                        tip_length=0.25,
                        tip_radius=0.1,
                        shaft_radius=0.05
                    )
                    plotter.add_mesh(arrow, color='red', opacity=0.8)
    
    # Add inset if requested
    if show_inset and inset_region is not None:
        x0, y0 = inset_region[0]
        x1, y1 = inset_region[1]
        
        # Create inset region
        inset_magnitudes = mag_magnitudes[x0:x1, y0:y1]
        inset_x = x[x0:x1, y0:y1]
        inset_y = y[x0:x1, y0:y1]
        
        if z1 is not None:
            inset_z = np.full_like(inset_x, z1 + 0.1)  # Slightly above main surface
            inset_grid = pv.StructuredGrid()
            inset_grid.points = np.column_stack([inset_x.flatten(), inset_y.flatten(), inset_z.flatten()])
            inset_grid.dimensions = [inset_x.shape[0], inset_x.shape[1], 1]
            inset_grid.point_data['magnitude'] = inset_magnitudes.flatten()
            plotter.add_mesh(inset_grid, scalars='magnitude', cmap='plasma', show_edges=False)
        else:
            inset_grid = pv.StructuredGrid()
            inset_grid.points = np.column_stack([inset_x.flatten(), inset_y.flatten(), (np.zeros_like(inset_x) + 0.1).flatten()])
            inset_grid.dimensions = [inset_x.shape[0], inset_x.shape[1], 1]
            inset_grid.point_data['magnitude'] = inset_magnitudes.flatten()
            plotter.add_mesh(inset_grid, scalars='magnitude', cmap='plasma', show_edges=False)
    
    # Set camera and show
    if z1 is not None or z0 is not None:
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.5)
    
    if interactive:
        plotter.show(interactive=True)
    
    return plotter


if __name__ == "__main__":
    # Load data
    from magrec import __datapath__
    from magrec.data.Jerschow.scripts.scripts import get_datadict_from_file
    
    datadict = get_datadict_from_file(__datapath__ / "Jerschow" / "Sine_wire.txt")
    B = datadict["B"]
    
    # Create visualization
    plotter = visualize_vector_fields(B, num_arrows=15)
    
