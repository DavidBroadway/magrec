# Vedo Vector Field Visualization

This directory contains enhanced vector field visualization tools using the Vedo library for interactive 3D plotting.

## Overview

The `visualize_vector_fields` function in `plot_vector_fields.py` has been rewritten to use Vedo instead of Plotly, providing:

- **Color-coded magnitude maps**: Amplitude information displayed as colored surfaces
- **Direction arrows**: Vector direction shown with scaled arrows
- **Interactive 3D visualization**: Rotate, zoom, and pan the visualization
- **Support for both 2D and 3D data**: Handles magnetic fields and current distributions
- **Optional inset regions**: Zoom into specific areas of interest
- **Standalone windows**: Works from command line without requiring Jupyter

## Features

### Key Improvements over Plotly Version

1. **Better Visual Clarity**: Color maps show magnitude while arrows show direction
2. **Inspired by `plot_vector_field_2d`**: Uses similar averaging and scaling techniques
3. **Interactive 3D Controls**: Mouse and keyboard controls for exploration
4. **Flexible Data Support**: Works with both PyTorch tensors and NumPy arrays
5. **Configurable Arrow Density**: Control the number of arrows displayed
6. **Standalone Windows**: Creates interactive windows when run from command line

### Visualization Components

- **Magnetic Field**: Displayed with plasma colormap and black arrows
- **Current Distribution**: Displayed with viridis colormap and red arrows (if provided)
- **Inset Regions**: Optional zoom regions for detailed inspection

## Usage

### Basic Usage

```python
from plot_vector_fields import visualize_vector_fields

# Create visualization
plotter = visualize_vector_fields(
    magnetic_field=B,  # Shape: (3, W, H) or (2, W, H)
    current_distribution=J,  # Shape: (2, W, H), optional
    z1=1.0,  # Z-coordinate for magnetic field, optional
    z0=0.0,  # Z-coordinate for current distribution, optional
    num_arrows=20,  # Number of arrows to display
    show_inset=True,  # Show inset region
    inset_region=((20, 20), (44, 44)),  # Inset region coordinates
    window_size=(800, 600),  # Window size
    interactive=True  # Show interactive window
)
```

### Example Script

Run the example script to see the visualization in action:

```bash
cd scripts
conda activate currec  # Make sure vedo is available
python example_vedo_visualization.py
```

This will create sample dipole magnetic field and circular current distribution data and display both 2D and 3D visualizations in interactive windows.

## Parameters

### `visualize_vector_fields` Function

- **`magnetic_field`**: Tensor/array of shape (3, W, H) or (2, W, H) representing magnetic field vectors
- **`current_distribution`**: Tensor/array of shape (2, W, H) representing current distribution vectors (optional)
- **`z1`**: Z-coordinate for magnetic field visualization (optional, enables 3D mode)
- **`z0`**: Z-coordinate for current distribution visualization (optional, enables 3D mode)
- **`num_arrows`**: Approximate number of arrows to display (default: 20)
- **`show_inset`**: Whether to show an inset zoom region (default: False)
- **`inset_region`**: Tuple of ((x0, y0), (x1, y1)) for inset region coordinates (default: None)
- **`window_size`**: Tuple of (width, height) for the window (default: (800, 600))
- **`interactive`**: Whether to show interactive window (default: True)

## Installation

The Vedo library is included in the project dependencies. Install it with:

```bash
pip install -r requirements.txt
```

Or install Vedo directly:

```bash
pip install vedo
```

**Note**: Make sure to activate your conda environment (e.g., `conda activate currec`) before running the visualization scripts.

## Technical Details

### Arrow Generation

The function creates arrows by:
1. Dividing the field into a grid based on `num_arrows`
2. Averaging vector components within each grid cell
3. Scaling arrows to fit within grid cells
4. Positioning arrows at grid cell centers

### Color Mapping

- **Magnetic Field**: Uses plasma colormap for magnitude visualization
- **Current Distribution**: Uses viridis colormap for magnitude visualization
- **Arrows**: Black for magnetic field, red for current distribution

### 3D vs 2D Mode

- **3D Mode**: Activated when `z1` or `z0` parameters are provided
- **2D Mode**: Default mode when no Z-coordinates are specified

## Comparison with Original Function

| Feature | Original (Plotly) | New (Vedo) |
|---------|-------------------|------------|
| Visualization Type | Cone-based vectors | Color map + arrows |
| Interactivity | Limited | Full 3D controls |
| Magnitude Display | Hover text only | Color-coded surfaces |
| Arrow Scaling | Fixed | Adaptive to field strength |
| Performance | Slower with many vectors | Faster with grid sampling |
| Inset Support | No | Yes |
| Standalone Windows | No | Yes |

## Examples

### Magnetic Field Only (3D)
```python
plotter = visualize_vector_fields(B, z1=1.0, num_arrows=15, interactive=True)
```

### Current Distribution Only (2D)
```python
plotter = visualize_vector_fields(None, current_distribution=J, num_arrows=25, interactive=True)
```

### Both Fields with Inset
```python
plotter = visualize_vector_fields(
    B, J, z1=1.0, z0=0.0, 
    show_inset=True, 
    inset_region=((30, 30), (50, 50)),
    window_size=(1000, 800),
    interactive=True
)
```

### Non-interactive Mode (for saving)
```python
plotter = visualize_vector_fields(B, J, z1=1.0, z0=0.0, interactive=False)
# Save screenshot
plotter.screenshot('vector_field.png')
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'vedo'**
   - Solution: Install vedo with `pip install vedo` or activate your conda environment

2. **Window not appearing**
   - Make sure you're running with `interactive=True` (default)
   - Check that your system supports GUI windows

3. **Performance issues with large datasets**
   - Reduce `num_arrows` parameter to decrease the number of arrows displayed
   - Consider downsampling your data before visualization
