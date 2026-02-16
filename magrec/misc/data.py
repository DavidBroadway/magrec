# Classes and functions for handling spatial magnetic field data.

import re
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from magrec.plot.plot import plot_n_components

import pyvista as pv


class MagneticFieldDataMixin:
    """Mixin class to add common magnetic field data functionality.
    
    Allows accessing field_data and point_data keys as attributes,
    automatically converting to torch.Tensor.
    """            
    
    def __getattr__(self, name):
        """Dynamically expose field_data and point_data keys as attributes."""
        # Don't intercept private/internal attributes (like PyVista's _obbTree)
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Avoid recursion by using `object.__getattribute__`
        try:
            # Check if the attribute is in `field_data`
            field_data = object.__getattribute__(self, 'field_data')
            if name in field_data:
                return torch.tensor(field_data[name])
            
            # Check if the attribute is in `point_data`
            point_data = object.__getattribute__(self, 'point_data')
            if name in point_data:
                return torch.tensor(point_data[name])
            
        except AttributeError:
            pass
        
        # If not found in field_data or point_data, raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def map(self, func: callable, name):
        """Map a function over all points, assign result to point_data."""
        self.point_data[name] = func(torch.tensor(self.points, dtype=torch.float32)).detach().numpy()
        return self


class UnstructuredData(MagneticFieldDataMixin, pv.PolyData):
    """Unstructured point cloud data with magnetic field functionality.
    
    Used internally by Dataset.from_dict() for irregular point data.
    Has same interface as Dataset but without structured grid methods.
    """
    
    def pts_as_list(self):
        """Return points as (N, 3) array."""
        return np.asarray(self.points)
    
    def pts_as_grid(self, component=None):
        """Not available for unstructured data."""
        raise ValueError("pts_as_grid() only works for structured grids. Use .resample() or Dataset.from_unstructured() to create a regular grid first.")


class DataBlock(pv.MultiBlock):
    """Container for multiple Datasets with different spatial structures.
    
    Use DataBlock when you need to combine multiple grids or meshes that have:
    - Different number of points
    - Different spatial regions
    - Different resolutions
    - Mix of structured (ImageData) and unstructured (PolyData) data
    
    Examples
    --------
    >>> block = DataBlock()
    >>> block.append(dataset1, "scan1")
    >>> block.append(dataset2, "scan2")
    >>> # Visualize all datasets together
    >>> block.plot()
    """
    
    def append(self, dataset, name=None):
        """Append a dataset with a unique name."""
        if name is None:
            name = f"block_{len(self)}"
        elif name in self.keys():
            raise ValueError(f"A block with the name '{name}' already exists.")
        return super().append(dataset, name)

    def __repr__(self):
        return self[0:len(self)].__repr__()
    

# TODO: move to magrec.prop 
class Scaler(object):
    """Scaler class to scale data to have values with mean 0 and std 1."""
    def __init__(self, data):
        self.data = data
        self.mean = data.mean()
        self.std = data.std()

    def scale(self, data):
        return (data - self.mean) / self.std
    
    def unscale(self, data):
        return data * self.std + self.mean


class Region2D:
    """
    A rectangular region in 2D (x, y) space. The z-coordinate is intentionally excluded since
    selection/containment is always in the x-y plane. For points at a specific z, use region.at(z).
    
    Example:
        roi = Region(0.9e-5, 1.15e-5, 1.1e-5, 1.35e-5)
        sensor_pts, idx = roi.select(r_sensor)       # select by x,y, preserves original z
        
        sensor_layer = roi.at(z=0)                   # sensor plane at z=0
        source_layer = roi.at(z=-100e-9)             # source plane 100nm below
        sensor_grid = sensor_layer.grid(100, 100)    # (10000, 3) at z=0
        source_grid = source_layer.grid(50, 50)      # (2500, 3) at z=-100nm
    """
    
    def __init__(self, x_min, x_max, y_min, y_max, z=0):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z = z
    
    @classmethod
    def from_center(cls, cx, cy, dx, dy):
        """Construct from center (cx, cy) and half-widths (dx, dy)."""
        return cls(cx - dx, cx + dx, cy - dy, cy + dy)
    
    @classmethod
    def from_points(cls, pts, pad=0.0):
        """Construct bounding box around given points (ignores z) with optional padding."""
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        return cls(x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    
    def at(self, z):
        """Return a Layer: this region at a specific z-height. Use for grid generation."""
        return Region2D(self, z=z)
    
    @property
    def bounds(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max)
    
    @property
    def center(self):
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def size(self):
        return (self.x_max - self.x_min, self.y_max - self.y_min)
    
    def contains(self, pts):
        """Returns boolean mask for points inside region (x,y only, ignores z)."""
        if isinstance(pts, (torch.Tensor, np.ndarray)):
            x, y = pts[:, 0], pts[:, 1]
        elif isinstance(pts, (pv.PolyData, pv.UnstructuredGrid, pv.ImageData)):
            x, y = pts.points[:, 0], pts.points[:, 1]
        else:
            raise ValueError(f"Unsupported type: {type(pts)}")
            
        return (x > self.x_min) & (x < self.x_max) & (y > self.y_min) & (y < self.y_max)
    
    def select(self, pts):
        """Select points inside region by x,y. Returns (selected_points, indices). Original z preserved."""
        mask = self.contains(pts)
        indices = np.argwhere(mask).flatten()
        # Handle separately selection from PyVista PolyData
        if isinstance(pts, (pv.PolyData, pv.UnstructuredGrid)):
            return pts.extract_points(mask), indices
        
        return pts[mask], indices
    
    def grid(self, nx, ny, z=None):
        """Generate (nx * ny, 3) grid at height z. Shorthand for region.at(z).grid(nx, ny)."""
        if z is None:
            z = self.z
        return self.grid(nx, ny)
    
    def __mul__(self, factor):
        """Scale region bounds by a numeric factor."""
        if not isinstance(factor, (int, float)):
            return NotImplemented
        return Region2D(
            self.x_min * factor,
            self.x_max * factor,
            self.y_min * factor,
            self.y_max * factor,
            z=self.z * factor,
        )
    
    def __rmul__(self, factor):
        return self.__mul__(factor)
    
    def __repr__(self):
        return f"Region2D(x=[{self.x_min:.2e}, {self.x_max:.2e}], y=[{self.y_min:.2e}, {self.y_max:.2e}])"
    

class Pipeset(pv.MultiBlock, MagneticFieldDataMixin):
    """
    Pipeline + Dataset hybrid built on PyVista MultiBlock. Each named block is a point set,
    which can be a structured grid or an unstructured point cloud.
    
    Setting a PyTorch tensor with last dim 2 or 3 auto-creates a PolyData block.
    Scalars can be added to existing blocks via pipe['name.scalar'] = values.
    
    Usage:
        pipe = Pipeset()
        pipe['sensor'] = r_sensor          # (N, 3) tensor -> PolyData with N points
        pipe['sensor.B_NV'] = B_NV         # adds scalar to existing 'sensor' block
        pipe['source'] = r_source          # another point set
        
        pipe.region('sensor', 'roi', Region2D(...))  # creates 'sensor.roi' sub-block
    
    If any of the blocks is a structured grids, has the following methods:
    - expand_bounds_2d(), expand_bounds_3d()
    - pts_as_grid()
    
    Methods for any data:
    - from_dict(), from_unstructured()
    - pts_as_list()
    - add_dipole_locations()
    
    Examples
    --------
    Structured grid:
    >>> pipe = Pipeset()
    >>> pipe.dimensions = (50, 50, 1)
    >>> pipe.origin = (0, 0, 0)
    >>> pipe.spacing = (0.1, 0.1, 1.0)
    >>> pipe.point_data['B'] = field_values
    
    From dictionary:
    >>> data = {'xs': x_coords, 'ys': y_coords, 'B': field_values}
    >>> pipe = Pipeset.from_dict(data)
    
    From grid data:
    >>> data = {'xs': x_grid, 'ys': y_grid, 'B': field_values}
    >>> pipe = Pipeset.from_dict(data, x_grid=True, y_grid=True)
    
    Create Pipeset, add points and scalars
    >>> pipe = Pipeset()
    >>> pipe['sensor'] = r_sensor                    # creates PolyData from (N,3) tensor
    >>> pipe['sensor.B_NV'] = B_NV.flatten()         # adds scalar
    >>> pipe['source'] = r_source                    # another point set
    >>> pipe.region('sensor', 'roi', subregion)      # creates 'sensor.roi' sub-block

    After training (training is done by the pipeline)
    >>> pipe['sensor.roi.B_NV_pred'] = predicted     # add prediction to subregion
    
    Plotting the results
    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    >>> pipe.plot(ax=ax1, scalar='sensor.B_NV', cmap='viridis', s=1)
    >>> pipe.plot(ax=ax2, scalar='sensor.roi.B_NV', cmap='viridis', s=1)
    >>> pipe.plot(ax=ax3, scalar='sensor.roi.B_NV_pred', cmap='viridis', s=1)

    Print the pipeline
    >>> print(pipe)
    """
    
    def __init__(self, *args, **kwargs):
        pv.MultiBlock.__init__(self, *args, **kwargs)
        # Store plot objects for colorbar synchronization: {'name': {'sc': mappable, 'ax': ax, 'clim': (vmin, vmax)}}
        self._plots = {}
        # Steps connect blocks: propagators, trainers, etc.
        # Each step is a dict with 'source', 'target', 'fn', and optional extra state.
        self._steps = {}
    
    @classmethod
    def from_dict(cls, datadict, rename_map=None, x_grid=False, y_grid=False, 
                  as_regular_grid=False, nx=None, ny=None, nz=1):
        """Create a Dataset from a dictionary of coordinates and field data.
        
        Intelligently handles different coordinate formats:
        - (N,) or (N, 1): List of coordinates for N points
        - (N, M): Grid coordinates where xs[i, j] is x-coord at pixel (i,j)
        
        Parameters
        ----------
        datadict : dict
            Dictionary with coordinate and field data. Required: "xs", "ys".
            Optional: "zs"/"height"/"standoff" for z-coordinates.
            All other keys become point_data.
        rename_map : dict, list, tuple, optional
            Mapping to rename or split keys:
            - dict: {"old_name": "new_name"}
            - list/tuple: ["old->new", "B->[Bx, By, Bz]"]
            For splitting: use "[name1, name2, ...]" syntax
        x_grid, y_grid : bool
            If True, treat xs/ys as grid coordinates (N, M).
            If False (default), auto-detect or treat as point list.
        as_regular_grid : bool
            If True, resample to regular ImageData structure. Default: False
        nx, ny, nz : int, optional
            Grid dimensions when as_regular_grid=True
        
        Returns
        -------
        Dataset or pv.PolyData
            Dataset (ImageData) if as_regular_grid=True, otherwise PolyData
        
        Examples
        --------
        >>> # Point list coordinates
        >>> data = {'xs': [0, 1, 2], 'ys': [0, 1, 2], 'B': [...]}
        >>> ds = Dataset.from_dict(data)
        
        >>> # Grid coordinates
        >>> xx, yy = np.meshgrid(x, y)
        >>> data = {'xs': xx, 'ys': yy, 'B': field}
        >>> ds = Dataset.from_dict(data, x_grid=True, y_grid=True)
        
        >>> # With renaming
        >>> ds = Dataset.from_dict(data, rename_map=["BNV->B", "x->xs"])
        """
        # Create a copy to avoid modifying original
        data = datadict.copy()
        
        # Parse and apply rename_map
        if rename_map is not None:
            data = cls._apply_rename_map(data, rename_map)
        
        # Extract coordinates
        xs = data.pop("xs", None)
        ys = data.pop("ys", None)
        zs = data.pop("zs", None)
        height = data.pop("height", None)
        standoff = data.pop("standoff", None)
        
        if xs is None or ys is None:
            raise ValueError("datadict must contain 'xs' and 'ys' keys")
        
        # Convert to numpy arrays
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        
        # Determine coordinate format and create points
        points = cls._parse_coordinates(xs, ys, zs, height, standoff, x_grid, y_grid)
        n_points = len(points)
        
        # Create UnstructuredData with points
        unstruct = UnstructuredData(points)
        
        # Add all remaining keys as point_data
        for key, value in data.items():
            value_array = np.asarray(value)
            
            # Handle different shapes
            if value_array.ndim == 1:
                if len(value_array) != n_points:
                    raise ValueError(
                        f"Data array '{key}' has length {len(value_array)}, "
                        f"but expected {n_points} (number of points)"
                    )
                unstruct.point_data[key] = value_array
            elif value_array.ndim == 2:
                # For 2D arrays, check if either dimension matches n_points
                if value_array.shape[0] == n_points:
                    unstruct.point_data[key] = value_array
                elif value_array.shape[1] == n_points:
                    unstruct.point_data[key] = value_array.T
                else:
                    # Maybe it's grid data that needs flattening
                    if value_array.size == n_points:
                        unstruct.point_data[key] = value_array.ravel()
                    else:
                        raise ValueError(
                            f"2D array '{key}' with shape {value_array.shape} does not match "
                            f"number of points {n_points}"
                        )
            elif value_array.ndim == 3:
                # 3D array - flatten spatial dimensions
                if value_array.shape[0] * value_array.shape[1] == n_points:
                    # Shape is (nx, ny, n_components)
                    unstruct.point_data[key] = value_array.reshape(n_points, -1)
                else:
                    raise ValueError(f"Cannot match 3D array '{key}' shape {value_array.shape} to {n_points} points")
            else:
                raise ValueError(f"Data array '{key}' has unsupported dimensionality: {value_array.ndim}D")
        
        if as_regular_grid:
            # Resample to regular grid
            return cls.from_unstructured(unstruct, nx=nx, ny=ny, nz=nz)
        else:
            # Return UnstructuredData
            return unstruct
    
    @staticmethod
    def _parse_coordinates(xs, ys, zs=None, height=None, standoff=None, x_grid=False, y_grid=False):
        """Parse coordinates in various formats and return (N, 3) point array.
        
        Handles:
        - (N,) or (N, 1): Point list
        - (N, M): Grid coordinates
        """
        # Handle z-coordinate aliases
        if zs is None and height is None and standoff is None:
            z_val = 0.0
        elif zs is not None:
            z_val = zs
        elif height is not None:
            z_val = height
        elif standoff is not None:
            z_val = standoff
        else:
            raise ValueError("Multiple z-coordinate specifications provided")
        
        # Auto-detect grid vs list format
        if not x_grid and not y_grid:
            # Auto-detect: if 2D and shapes match, it's likely a grid
            if xs.ndim == 2 and ys.ndim == 2 and xs.shape == ys.shape:
                x_grid = y_grid = True
        
        if x_grid or y_grid:
            # Grid format: xs[i, j] and ys[i, j] are coordinates at pixel (i, j)
            if xs.shape != ys.shape:
                raise ValueError(f"Grid coordinates must have same shape, got xs:{xs.shape}, ys:{ys.shape}")
            
            # Flatten to point list
            xs_flat = xs.ravel()
            ys_flat = ys.ravel()
            
            # Handle z
            if isinstance(z_val, (int, float)):
                zs_flat = np.full_like(xs_flat, z_val)
            else:
                z_val = np.asarray(z_val)
                if z_val.shape == xs.shape:
                    zs_flat = z_val.ravel()
                elif z_val.ndim == 1 and len(z_val) == len(xs_flat):
                    zs_flat = z_val
                else:
                    raise ValueError(f"z-coordinate shape {z_val.shape} incompatible with grid shape {xs.shape}")
            
            points = np.column_stack([xs_flat, ys_flat, zs_flat])
        else:
            # Point list format: xs[i] and ys[i] are coordinates of point i
            xs_flat = xs.ravel()
            ys_flat = ys.ravel()
            
            if len(xs_flat) != len(ys_flat):
                raise ValueError(f"xs and ys must have same length, got {len(xs_flat)} and {len(ys_flat)}")
            
            # Handle z
            if isinstance(z_val, (int, float)):
                zs_flat = np.full_like(xs_flat, z_val)
            else:
                z_val = np.asarray(z_val).ravel()
                if len(z_val) == len(xs_flat):
                    zs_flat = z_val
                elif len(z_val) == 1:
                    zs_flat = np.full_like(xs_flat, z_val[0])
                else:
                    raise ValueError(f"z-coordinate length {len(z_val)} doesn't match point count {len(xs_flat)}")
            
            points = np.column_stack([xs_flat, ys_flat, zs_flat])
        
        return points
    
    @staticmethod
    def _apply_rename_map(data, rename_map):
        """Apply rename_map to transform dictionary keys."""
        # Convert rename_map to dict if in other formats
        if isinstance(rename_map, dict):
            pass
        elif isinstance(rename_map, (list, tuple)):
            if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in rename_map):
                rename_map = dict(rename_map)
            elif all(isinstance(item, str) for item in rename_map):
                new_map = {}
                for item in rename_map:
                    if '->' in item:
                        old, new = item.split('->', 1)
                        new_map[old.strip()] = new.strip()
                    elif '→' in item:
                        old, new = item.split('→', 1)
                        new_map[old.strip()] = new.strip()
                    else:
                        raise ValueError(f"Invalid rename format: {item}")
                rename_map = new_map
            else:
                raise ValueError(f"Invalid rename_map format")
        else:
            raise ValueError(f"rename_map must be dict, list, or tuple")
        
        # Parse for split operations
        split_operations = {}
        parsed_rename_map = {}
        
        for old_key, new_value in rename_map.items():
            if isinstance(new_value, (list, tuple)):
                split_operations[old_key] = list(new_value)
            elif isinstance(new_value, str):
                match = re.match(r'\[(.*?)\]', new_value)
                if match:
                    names_str = match.group(1)
                    split_names = [name.strip().strip('"').strip("'") for name in names_str.split(',')]
                    split_operations[old_key] = split_names
                else:
                    parsed_rename_map[old_key] = new_value
            else:
                raise ValueError(f"Invalid new_value in rename_map: {new_value}")
        
        # Apply simple renames
        for old_key, new_key in parsed_rename_map.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)
        
        # Apply split operations
        for old_key, split_names in split_operations.items():
            if old_key in data:
                array = np.asarray(data.pop(old_key))
                n_components = len(split_names)
                
                if array.ndim == 1:
                    raise ValueError(f"Cannot split 1D array '{old_key}' into {n_components} components")
                elif array.ndim == 2:
                    if array.shape[0] == n_components:
                        for i, name in enumerate(split_names):
                            data[name] = array[i]
                    elif array.shape[1] == n_components:
                        for i, name in enumerate(split_names):
                            data[name] = array[:, i]
                    else:
                        raise ValueError(
                            f"Array '{old_key}' shape {array.shape} cannot be split into {n_components} components"
                        )
                else:
                    raise ValueError(f"Cannot split array '{old_key}' with {array.ndim} dimensions")
        
        return data
    
    def pts_as_list(self):
        """Return points as (N, 3) array.
        
        Works for both structured (ImageData) and unstructured (PolyData) data.
        
        Returns
        -------
        ndarray, shape (N, 3)
            Point coordinates
        
        Examples
        --------
        >>> pts = ds.pts_as_list()
        >>> print(pts.shape)  # (N, 3)
        """
        return np.asarray(self.points)
    
    def pts_as_grid(self, component=None):
        """Return points reshaped as grid (nx, ny, nz, 3) or (nx, ny, 3) for 2D.
        
        Only works for structured grids (ImageData with dimensions set).
        
        Parameters
        ----------
        component : int, optional
            If specified, return only that component (0=x, 1=y, 2=z)
        
        Returns
        -------
        ndarray
            Grid of points, shape (nx, ny, nz, 3) or (nx, ny, 3) for 2D
        
        Raises
        ------
        ValueError
            If grid is not properly initialized with dimensions
        
        Examples
        --------
        >>> x_grid = ds.pts_as_grid(component=0)  # x-coordinates
        >>> pts_grid = ds.pts_as_grid()  # all coordinates
        """
        if self.dimensions is None or self.dimensions == (0, 0, 0):
            raise ValueError("Cannot reshape to grid: dimensions not set (unstructured data)")
        
        nx, ny, nz = self.dimensions
        points = self.pts_as_list()
        
        if nz == 1:
            # 2D grid
            grid = points.reshape(nx, ny, 3)
        else:
            # 3D grid
            grid = points.reshape(nx, ny, nz, 3)
        
        if component is not None:
            return grid[..., component]
        return grid
    
    @classmethod
    def from_unstructured(cls, unstructured_data, nx=None, ny=None, nz=1):
        """Create a regular Dataset (ImageData) from unstructured point data.
        
        Resamples arbitrary point data onto a regular grid via interpolation.
        
        Parameters
        ----------
        unstructured_data : pv.PolyData or similar
            Unstructured data to convert
        nx, ny, nz : int, optional
            Grid dimensions. Auto-determined if None.
        
        Returns
        -------
        Dataset
            Regular grid with interpolated data
        
        Examples
        --------
        >>> polydata = pv.PolyData(points)
        >>> ds = Dataset.from_unstructured(polydata, nx=50, ny=50)
        """
        bounds = unstructured_data.bounds
        if nx is None:
            nx = int(np.sqrt(unstructured_data.n_points))
        if ny is None:
            ny = int(np.sqrt(unstructured_data.n_points))
        
        # Create regular grid
        ds = cls()
        ds.dimensions = (nx, ny, nz)
        ds.origin = (bounds[0], bounds[2], bounds[4])
        ds.spacing = (
            (bounds[1] - bounds[0]) / max(nx - 1, 1),
            (bounds[3] - bounds[2]) / max(ny - 1, 1),
            (bounds[5] - bounds[4]) / max(nz - 1, 1)
        )
        
        # Interpolate data
        interpolated = ds.interpolate(unstructured_data)
        return cls(interpolated)
    
    def __sub__(self, other, threshold_distance=1e-2):
        """Subtract points of one dataset from another (set difference)."""
        if not hasattr(other, 'points'):
            raise TypeError(f"Cannot subtract {type(other)} from Dataset")
        
        points1 = np.array(self.points)
        points2 = np.array(other.points)
        
        # Build KDTree for fast nearest-neighbor search
        tree = scipy.spatial.KDTree(points2)
        distances, _ = tree.query(points1)
        
        # Keep points NOT close to any points in other
        mask = (distances > threshold_distance)
        filtered_points = points1[mask]
        
        # Create new PolyData with filtered points
        filtered_data = pv.PolyData(filtered_points)
        
        # Copy point data
        for key, value in self.point_data.items():
            filtered_data.point_data[key] = value[mask]
        
        return filtered_data
    
    def expand_bounds_3d(self, factor=1, name=None):
        """Expand bounds of regular grid in x, y, z directions.
        
        Only works for structured grids (ImageData with dimensions set).
        
        Parameters
        ----------
        factor : float, int, list, or tuple
            Expansion factor(s):
            - Single number: symmetric expansion all directions
            - [fx, fy, fz]: symmetric expansion per axis
            - [(fx_min, fx_max), ...]: asymmetric expansion
        name : str, optional
            Name for expanded grid in returned DataBlock
        
        Returns
        -------
        DataBlock
            MultiBlock containing original and expanded grids
        
        Examples
        --------
        >>> expanded = ds.expand_bounds_3d(1.5)
        >>> expanded = ds.expand_bounds_3d([2, 2, 1])
        """
        if self.dimensions is None or self.dimensions == (0, 0, 0):
            raise ValueError("Cannot expand bounds: grid dimensions not set (unstructured data)")
        
        # Parse factor
        if isinstance(factor, (list, tuple)):
            if all((isinstance(f, (list, tuple)) and len(f) == 2) for f in factor) and len(factor) == 3:
                a, b = factor[0]
                c, d = factor[1]
                e, f = factor[2]
            elif all(isinstance(f, (float, int)) for f in factor) and len(factor) == 3:
                a = b = factor[0]
                c = d = factor[1]
                e = f = factor[2]
            else:
                raise TypeError("Invalid factor specification")
        elif isinstance(factor, (int, float)):
            a = b = c = d = e = f = factor
        else:
            raise TypeError("factor must be number, list, or tuple")
        
        dims = self.dimensions
        spacing = self.spacing
        origin = self.origin
        
        # Calculate expanded dimensions and origin
        expanded_dims = (int(dims[0] * (1 + a + b)), 
                        int(dims[1] * (1 + c + d)), 
                        int(dims[2] * (1 + e + f)))
        expanded_origin = (
            origin[0] - spacing[0] * dims[0] * a,
            origin[1] - spacing[1] * dims[1] * c,
            origin[2] - spacing[2] * dims[2] * e
        )
        
        expanded_grid = Dataset()
        expanded_grid.dimensions = expanded_dims
        expanded_grid.origin = expanded_origin
        expanded_grid.spacing = spacing
        
        blocks = DataBlock()
        blocks.append(self, name or "original")
        blocks.append(expanded_grid, "expanded")
        return blocks
    
    def expand_bounds_2d(self, factor=1, name=None):
        """Expand bounds of regular grid in x, y directions (z unchanged).
        
        Only works for structured grids (ImageData with dimensions set).
        
        Parameters
        ----------
        factor : float, int, list, or tuple
            Expansion factor(s):
            - Single number: symmetric expansion in x and y
            - [fx, fy]: symmetric expansion per axis
            - [(fx_min, fx_max), (fy_min, fy_max)]: asymmetric
        name : str, optional
            Name for expanded grid
        
        Returns
        -------
        Dataset
            Expanded grid with original z dimension
        
        Examples
        --------
        >>> expanded = ds.expand_bounds_2d(2)
        >>> expanded = ds.expand_bounds_2d([1.5, 2])
        """
        if self.dimensions is None or self.dimensions == (0, 0, 0):
            raise ValueError("Cannot expand bounds: grid dimensions not set (unstructured data)")
        
        # Parse factor
        if isinstance(factor, (list, tuple)):
            if all([(isinstance(f, (list, tuple)) and len(f) == 2) for f in factor]) and len(factor) == 2:
                a, b = factor[0]
                c, d = factor[1]
            elif all(isinstance(f, (float, int)) for f in factor) and len(factor) == 2:
                a = b = factor[0]
                c = d = factor[1]
            else:
                raise TypeError("Invalid factor specification")
        elif isinstance(factor, (int, float)):
            a = b = c = d = factor
        else:
            raise TypeError("factor must be number, list, or tuple")
        
        dims = self.dimensions
        spacing = self.spacing
        origin = self.origin
        
        # Calculate expanded dimensions and origin
        expanded_dims = (int(dims[0] * (1 + a + b)), 
                        int(dims[1] * (1 + c + d)), 
                        dims[2])
        expanded_origin = (
            origin[0] - spacing[0] * dims[0] * a,
            origin[1] - spacing[1] * dims[1] * c,
            origin[2]
        )
        
        expanded_grid = Dataset()
        expanded_grid.dimensions = expanded_dims
        expanded_grid.origin = expanded_origin
        expanded_grid.spacing = spacing
        
        return expanded_grid
    
    @classmethod
    def _get_as_grid(cls, grid, point_data_name):
        """Return point data reshaped to grid structure (class method helper)."""
        data = grid[point_data_name]
        nx, ny, nz = grid.dimensions
        shape = (nz, ny, nx, 3) if nz > 1 else (ny, nx, 3)
        if nz > 1:
            return torch.tensor(data.reshape(*shape)).permute(2, 1, 0, 3)
        elif nz == 1:
            return torch.tensor(data.reshape(*shape)).permute(1, 0, 2)
        else:
            raise ValueError("Invalid dimensions for the grid.")
    
    def get_as_grid(self, point_data_name):
        """Return point data reshaped to grid matching mesh structure.
        
        Only available for ImageData (structured grids).
        """
        data = self.__getattr__(point_data_name)
        nx, ny, nz = self.dimensions
        shape = (nz, ny, nx, 3) if nz > 1 else (ny, nx, 3)
        if nz > 1:
            return data.reshape(*shape).permute(2, 1, 0, 3)
        elif nz == 1:
            return data.reshape(*shape).permute(1, 0, 2)
        else:
            raise ValueError("Invalid dimensions for the grid.")
    
    def interpolate(self, *args, **kwargs):
        """Interpolate data onto this grid, ensuring output is Dataset."""
        interpolated = super().interpolate(*args, **kwargs)
        return Dataset(interpolated)
    
    def plot_n_components(self, point_data_name):
        """Plot field components using plot_n_components function."""
        grid_data = self.get_as_grid(point_data_name=point_data_name)
        return plot_n_components(data=grid_data)
    
    def _looks_like_points(self, value):
        """Check if value is array-like with last dimension 2 or 3 (coordinates)."""
        if isinstance(value, torch.Tensor):
            return value.ndim >= 1 and value.shape[-1] in (2, 3)
        if isinstance(value, np.ndarray):
            return value.ndim >= 1 and value.shape[-1] in (2, 3)
        if isinstance(value, (pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid)):
            # Assume that PolyData is always a point set
            return True
        return False
    
    def _to_polydata(self, value):
        """Convert points array to PolyData, padding 2D to 3D if needed."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        pts = value.reshape(-1, value.shape[-1])
        if pts.shape[-1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        return pv.PolyData(pts)
    
    def __setitem__(self, key, value):
        # PyVista internally uses integer indices during append, pass through
        if isinstance(key, int):
            return super().__setitem__(key, value)
        
        # Convert torch tensors to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, (np.ndarray, pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid, pv.ImageData)):
            value = np.asarray(value)
        
        # If value is already a block with points, it must be set directly to the multiblock, 
        # without checking what are the values from the points. 
        if isinstance(value, (pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid, pv.ImageData)):
            super().__setitem__(key, value)
            return
        
        # For dotted keys like 'sensor.roi.B_NV', we need to figure out what's the block name
        # and what's the scalar name. Could be: block='sensor.roi', scalar='B_NV' (flat naming),
        # or block='sensor' with nested block 'roi' with scalar 'B_NV'. We prefer flat naming:
        # first check if 'sensor.roi' exists as a block, if yes, 'B_NV' is the scalar.
        # If 'sensor.roi' doesn't exist but 'sensor' does and is a MultiBlock containing 'roi',
        # that's nesting. If both exist, warn and use flat (the literal block name wins).
        
        elif '.' in key:
            parts = key.split('.')
            # Try flat naming first: all-but-last is block name, last is scalar
            flat_block_name = '.'.join(parts[:-1])
            attr_name = parts[-1]
            flat_exists = flat_block_name in self.keys()
            
            # Check for nested ambiguity: does parts[0] exist and could contain parts[1]?
            nested_exists = False
            if len(parts) >= 2 and parts[0] in self.keys():
                first_block = super().__getitem__(parts[0])
                if isinstance(first_block, pv.MultiBlock):
                    nested_exists = True
            
            if flat_exists and nested_exists:
                import warnings
                warnings.warn(
                    f"Ambiguous key '{key}': both '{flat_block_name}' exists as block and "
                    f"'{parts[0]}' is a MultiBlock. Using flat naming ('{flat_block_name}' + '{attr_name}')."
                )
            
            # Prefer flat naming if that block exists
            if flat_exists:
                parent_block = self[flat_block_name]
            elif nested_exists:
                # Fall back to nested: recurse into parts[0], let it handle the rest
                first_block = super().__getitem__(parts[0])
                first_block['.'.join(parts[1:])] = value
                return
            else:
                # Neither exists, fall through to create new block
                parent_block = None
            
            if parent_block is not None and hasattr(parent_block, 'n_points'):
                # Value can be (N,), (N, n), (W, H), or (W, H, n). n is components, usually <= 3.
                shape = value.shape
                if len(shape) == 1:
                    if shape[0] == parent_block.n_points:
                        parent_block.point_data[attr_name] = value
                        return
                elif len(shape) == 2:
                    if shape[0] == parent_block.n_points:
                        parent_block.point_data[attr_name] = value.reshape(parent_block.n_points, -1).squeeze()
                        return
                    else:
                        W, H = shape
                        if W * H == parent_block.n_points:
                            # Ambiguous (nx, ny) vs (ny, nx). Assume 'yx' (image convention).
                            # Use set_point_data() for explicit control.
                            parent_block.point_data[attr_name] = value.T.flatten('F')
                            return
                        else:
                            raise ValueError(f"Unsupported shape {shape} for assignment to {parent_block.n_points} points of {parent_block}")
                elif len(shape) == 3:
                    n, W, H = shape
                    if n > 3: 
                        W, H, n = shape
                        value = value.transpose(1, 2, 0)
                    if W * H == parent_block.n_points:
                        parent_block.point_data[attr_name] = value.reshape(W * H, n)
                        return
                else: 
                    raise ValueError(f"Unsupported shape {shape} for assignment to {parent_block.n_points} points of {parent_block}")
        
        # No valid parent block or length mismatch: treat as new point set or literal value
        if self._looks_like_points(value):
            parts = key.split('.', 1)
            first_part = parts[0]
            # Check if first_part is an existing MultiBlock we should recurse into
            if first_part in self.keys() and len(parts) > 1:
                existing = super().__getitem__(first_part)
                if isinstance(existing, pv.MultiBlock):
                    existing[parts[1]] = value
                    return
            # Use full key as literal block name
            if not isinstance(value, (pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid)):
                block = self._to_polydata(value)
            else:
                block = value
            super().__setitem__(key, block)
        else:
            # Non-points, non-matching value: store directly
            super().__setitem__(key, value)
            
    def set_point_data(self, name, attr_or_value, value_or_order=None, order='xy'):
        """
        Set point data on a block, handling 2D array reshaping for ImageData grids.
        
        Accepts either dot notation or separate arguments:
            pipe.set_point_data('sensor.B_NV', data)
            pipe.set_point_data('sensor.B_NV', data, order='yx')
            pipe.set_point_data('sensor', 'B_NV', data)
            pipe.set_point_data('sensor', 'B_NV', data, order='yx')
        
        For ImageData, points are ordered with x varying fastest: flat_idx = ix + iy * nx.
        
        order='xy': value shape is (nx, ny) where value[ix, iy] -> point at (x[ix], y[iy])
        order='yx': value shape is (ny, nx) where value[iy, ix] -> point at (x[ix], y[iy])
                    This is typical image/matrix convention (rows=y, cols=x).
        """
        # Parse arguments: either (name_with_dot, value, [order]) or (block, attr, value, [order])
        if isinstance(attr_or_value, str):
            # Two-string form: (block_name, attr_name, value, [order])
            block_name, attr_name = name, attr_or_value
            value = value_or_order
            # order stays as default or was passed as 4th arg
        else:
            # Dot notation form: (name.attr, value, [order])
            if '.' not in name:
                raise ValueError(f"Name must contain '.' for dot notation, got '{name}'")
            *parts, attr_name = name.split('.')
            block_name = '.'.join(parts)
            value = attr_or_value
            if value_or_order is not None:
                order = value_or_order
        
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        
        block = self[block_name]
        
        # If order='xy', first index is x, second is y: arr[ix, iy].
        # ImageData wants x to vary fastest: iterate all x before incrementing y.
        # So arr[0,1] has index nx in the flat array. That's Fortran order ('F').
        # If order='yx', arr[iy, ix], transpose first to get arr.T[ix, iy], then same.
        
        if value.ndim == 1:
            # Already flat, just assign
            block.point_data[attr_name] = value
        elif value.ndim == 2:
            # 2D grid data: need to flatten in correct order for ImageData
            if order == 'xy':
                # Shape is (nx, ny), flatten Fortran-order so x varies fastest
                flat = value.flatten('F')
            elif order == 'yx':
                # Shape is (ny, nx), transpose to (nx, ny) then flatten Fortran-order
                flat = value.T.flatten('F')
            else:
                raise ValueError(f"Invalid order: {order}. Use 'xy' or 'yx'.")
            block.point_data[attr_name] = flat
        elif value.ndim == 3:
            # Multi-component data, e.g. (nx, ny, 3) or (ny, nx, 3)
            if order == 'xy':
                # (nx, ny, n_components)
                nx, ny, nc = value.shape
                flat = value.reshape(nx * ny, nc, order='F')
            elif order == 'yx':
                # (ny, nx, n_components)
                ny, nx, nc = value.shape
                flat = value.transpose(1, 0, 2).reshape(nx * ny, nc, order='F')
            else:
                raise ValueError(f"Invalid order: {order}. Use 'xy' or 'yx'.")
            block.point_data[attr_name] = flat
        else:
            raise ValueError(f"Unsupported ndim={value.ndim}, expected 1, 2, or 3")
        
        
    def __getitem__(self, key):
        # PyVista uses integer indices internally
        if isinstance(key, int):
            return super().__getitem__(key)
        
        # Full key as literal block name (e.g., 'sensor.roi' is a block)
        if key in self.keys():
            return super().__getitem__(key)
        
        if '.' not in key:
            return super().__getitem__(key)  # will raise KeyError
        
        # For 'sensor.roi.B_NV': could be block='sensor.roi' with scalar='B_NV' (flat),
        # or block='sensor' containing 'roi.B_NV' (nested). Prefer flat naming.
        parts = key.split('.')
        flat_block_name = '.'.join(parts[:-1])
        scalar_name = parts[-1]
        flat_exists = flat_block_name in self.keys()
        
        # Check nested: is parts[0] a MultiBlock?
        nested_exists = False
        if parts[0] in self.keys():
            first_block = super().__getitem__(parts[0])
            if isinstance(first_block, pv.MultiBlock):
                nested_exists = True
        
        if flat_exists and nested_exists:
            import warnings
            warnings.warn(
                f"Ambiguous key '{key}': '{flat_block_name}' exists as block and "
                f"'{parts[0]}' is a MultiBlock. Using flat naming."
            )
        
        if flat_exists:
            block = self[flat_block_name]
            if scalar_name in block.point_data:
                return torch.tensor(block.point_data[scalar_name])
            else:
                raise KeyError(f"'{scalar_name}' not found in block '{flat_block_name}'")
        elif nested_exists:
            # Recurse into the MultiBlock
            first_block = super().__getitem__(parts[0])
            return first_block['.'.join(parts[1:])]
        else:
            raise KeyError(f"No block '{flat_block_name}' or MultiBlock '{parts[0]}' found for key '{key}'")
    
    def add_region(self, region, inp, name=None):
        """Create a sub-block from points in parent that fall within region."""
        if inp is None:
            raise ValueError("`inp` is required for adding Region2D")
        
        if isinstance(inp, str):
            pts = region.select(self[inp])[0]
            self[inp + "." + name] = pts
        elif isinstance(inp, tuple):
            for i in inp:
                self[i + "." + name] = region.select(self[i])[0]

    
    def plot(self, scalar, ax=None, name=None, clim=None, sync=True, 
             colorbar=False, symmetric=False, norm_type=None,
             cbar_width=0.05, cbar_pad=0.02, wspace=None,
             method='auto', labels=None, **kwargs):
        """
        Plot a scalar from a block. scalar is 'block.scalar_name'.
        
        name: store this plot under a name for later access via pipe.plots['name']
        clim: (vmin, vmax) to set color limits
        sync: if True, sync color limits with the plot with the same name, if False, do not sync, 
            if a string, it is the name of the plot to sync with,
        colorbar: if True, add a colorbar to the axis
        method: plotting method
            'auto' (default): imshow for ImageData, scatter for PolyData
            'scatter': scatter plot with circles at each point
            'imshow': pixel grid, requires regular grid (ImageData or convertible)
            'pcolormesh': cell-based grid, shows cell boundaries
        labels: for multi-component data, custom labels for each component.
            These become both the plot names (for sync_clim) and axis titles.
            E.g., labels=['B_x', 'B_y', 'B_z'] creates plots named 'B_x', 'B_y', 'B_z'.
        
        After plotting, call pipe.sync_clim('p1', 'p2', 'p3') to unify limits.
        """
        
        # If scalar is a tuple/list of strings, plot each as a separate "row" in a grid.
        # Each string may itself be multi-component (e.g., B with shape (N,3)), giving columns.
        # So ('sensor.roi.B', 'sensor.roi.B_NV') with B being 3-component and B_NV being scalar
        # gives a 2-row layout: first row has 3 axes, second row has 1 axis.
        if isinstance(scalar, (list, tuple)):
            # First pass: figure out how many columns each scalar needs
            col_counts = []
            for s in scalar:
                p = s.rsplit('.', 1)
                if len(p) == 2 and p[0] in self._steps:
                    col_counts.append(1)
                    continue
                block = self[p[0]]
                vals = block.point_data[p[1]]
                n_comp = vals.shape[1] if vals.ndim > 1 and vals.shape[1] > 1 else 1
                col_counts.append(n_comp)
            
            total_axes = sum(col_counts)
            
            if ax is None:
                # Create a single row with total_axes columns
                fig, flat_axs = plt.subplots(1, total_axes, figsize=(3.5 * total_axes + 1, 3.5),
                                              squeeze=False)
                flat_axs = flat_axs.flatten()
            else:
                # Flatten whatever axes array was passed
                flat_axs = np.array(ax).flatten()
                if len(flat_axs) < total_axes:
                    raise ValueError(f"Need {total_axes} axes for {scalar}, got {len(flat_axs)}")
                fig = flat_axs[0].figure
            
            # Slice labels and norm_type into per-scalar chunks, consumed sequentially.
            # labels=['B_x','B_y','B_z','B_NV'] with col_counts=[3,1] -> ['B_x','B_y','B_z'] and ['B_NV']
            label_idx = 0
            norm_idx = 0
            
            all_mappables = []
            all_plot_names = []
            ax_idx = 0
            for row_i, s in enumerate(scalar):
                n_comp = col_counts[row_i]
                
                # Slice labels for this scalar
                sub_labels = None
                if labels is not None:
                    sub_labels = labels[label_idx:label_idx + n_comp]
                    label_idx += n_comp
                
                if n_comp == 1:
                    # For single-component, use the label as the plot name
                    row_name = sub_labels[0] if sub_labels else (f"{name}_{row_i}" if name else None)
                    m = self.plot(s, ax=flat_axs[ax_idx], name=row_name, clim=clim, sync=sync,
                                 colorbar=colorbar, method=method, **kwargs)
                    if sub_labels:
                        title = f'${sub_labels[0]}$' if '$' not in sub_labels[0] else sub_labels[0]
                        flat_axs[ax_idx].set_title(title)
                    all_mappables.append(m)
                    if row_name:
                        all_plot_names.append(row_name)
                    ax_idx += 1
                else:
                    row_axes = [flat_axs[ax_idx + j] for j in range(n_comp)]
                    row_name = f"{name}_{row_i}" if name else None
                    m = self.plot(s, ax=row_axes, name=row_name, clim=clim, sync=sync,
                                 colorbar=colorbar, method=method, labels=sub_labels, **kwargs)
                    all_mappables.append(m)
                    # Collect the plot names that were created
                    if sub_labels:
                        all_plot_names.extend(sub_labels)
                    ax_idx += n_comp
            
            # Auto-sync if norm_type was given at this level. sync_clim handles
            # colorbar sizing and subplot spacing (wspace) precisely, so we skip
            # tight_layout when it runs -- tight_layout would override subplots_adjust
            # and make axes unequal when colorbar tick labels differ in width.
            synced = False
            if norm_type and all_plot_names:
                if len(norm_type) != len(all_plot_names):
                    raise ValueError(f"norm_type length ({len(norm_type)}) must match total plots ({len(all_plot_names)}): {all_plot_names}")
                self.sync_clim(*all_plot_names, norm_type=norm_type, symmetric=symmetric,
                               colorbar=colorbar, cbar_width=cbar_width, cbar_pad=cbar_pad, wspace=wspace)
                synced = True
            
            # Hide any leftover axes
            for j in range(ax_idx, len(flat_axs)):
                flat_axs[j].set_visible(False)
            
            if not synced:
                fig.tight_layout()
            return all_mappables
        
        parts = scalar.rsplit('.', 1)
        if len(parts) != 2:
            raise ValueError(f"scalar must be 'block.scalar_name', got '{scalar}'")
        block_name, scalar_name = parts[0], parts[1]
        
        # Check if block_name is a step (e.g., 'fit.loss')
        if block_name in self._steps:
            step = self._steps[block_name]
            if scalar_name == 'loss' and 'loss' in step:
                if ax is None:
                    fig, ax = plt.subplots()
                losses = step['loss']
                ax.semilogy(losses)
                ax.set_xlabel('iteration')
                ax.set_ylabel('loss')
                ax.set_title(f'{block_name} loss')
                ax.grid(True, alpha=0.3)
                if name:
                    self._plots[name] = {'sc': None, 'ax': ax, 'clim': None, 'scalar': scalar, 'cbar': None}
                return ax
            elif scalar_name == 'm' and 'm' in step:
                # Plot trainable parameters as spatial data on source block
                source_name = step['source']
                m = step['m'].detach().cpu().numpy()
                self[f'{source_name}._plot_m'] = m
                result = self.plot(f'{source_name}._plot_m', ax=ax, name=name, clim=clim,
                                   sync=sync, colorbar=colorbar, method=method, labels=labels, **kwargs)
                del self[source_name].point_data['_plot_m']
                return result
            else:
                raise KeyError(f"Step '{block_name}' has no plottable attribute '{scalar_name}'")
        
        block = self[block_name]
        pts = np.asarray(block.points)
        values = block.point_data[scalar_name]
        
        if values.ndim > 1 and values.shape[1] > 1:
            # Multi-component data (e.g., Bx, By, Bz). Create horizontal subplot layout like plot_n_components.
            n_comp = values.shape[1]
            
            # Labels: if provided, use directly as plot names and titles.
            # Otherwise generate defaults like 'name_x', 'name_y', 'name_z'.
            if labels is not None:
                if len(labels) != n_comp:
                    raise ValueError(f"labels length ({len(labels)}) must match components ({n_comp})")
                comp_names = labels  # Use labels directly as plot names
                # Wrap in math mode: 'B_x' → '$B_x$', unless already has '$'
                comp_titles = [f'${l}$' if '$' not in l else l for l in labels]
            else:
                suffixes = ['x', 'y', 'z', 'w', 'u', 'v'][:n_comp] if n_comp <= 6 else [f'c{i}' for i in range(n_comp)]
                base_name = name if name else scalar_name
                comp_names = [f"{base_name}_{s}" for s in suffixes]
                comp_titles = [f'{scalar_name}$_{s}$' for s in suffixes]
            
            if ax is None:
                # Figsize: ~4 inches per component plus padding
                fig, axs = plt.subplots(1, n_comp, figsize=(3.5 * n_comp + 1, 3.5))
            elif isinstance(ax, (list, np.ndarray)):
                axs = ax
                fig = axs[0].figure
            else:
                raise ValueError(f"For multi-component data, ax must be a list of {n_comp} axes or None")
            
            # Plot each component, storing with indexed names
            mappables = []
            for i in range(n_comp):
                comp_values = values[:, i]
                
                # Temporarily replace values in point_data to reuse single-component logic
                block.point_data[f'_temp_comp_{i}'] = comp_values
                m = self.plot(f'{block_name}._temp_comp_{i}', ax=axs[i], name=comp_names[i],
                              symmetric=symmetric, clim=clim, sync=sync, colorbar=colorbar, method=method, **kwargs)
                del block.point_data[f'_temp_comp_{i}']
                
                axs[i].set_title(comp_titles[i])
                mappables.append(m)
            
            synced = False
            if sync:
                self.sync_clim(*comp_names, symmetric=symmetric, norm_type=norm_type, 
                               colorbar=colorbar, cbar_width=cbar_width, 
                               cbar_pad=cbar_pad, wspace=wspace)
                synced = True
                
            if not synced:
                fig.tight_layout()
            return mappables
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Handle color limits
        if sync and sync in self._plots:
            clim = self._plots[sync]['clim']
        if clim is None:
            vmin, vmax = float(values.min()), float(values.max())
        else:
            vmin, vmax = clim
        
        if symmetric:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
        
        # Determine method
        is_image_data = isinstance(block, pv.ImageData)
        if method == 'auto':
            method = 'imshow' if is_image_data else 'scatter'
        
        if method == 'scatter':
            mappable = ax.scatter(pts[:, 0], pts[:, 1], c=values, vmin=vmin, vmax=vmax, **kwargs)
            
        elif method in ('imshow', 'pcolormesh'):
            # Need grid structure. For ImageData we have dimensions, for PolyData we infer from points.
            if is_image_data:
                nx, ny, nz = block.dimensions
                x0, y0, z0 = block.origin
                dx, dy, dz = block.spacing
            else:
                # Infer grid from unique x, y values
                xs = np.unique(pts[:, 0])
                ys = np.unique(pts[:, 1])
                nx, ny = len(xs), len(ys)
                if nx * ny != len(pts):
                    raise ValueError(f"Cannot use {method} on non-rectangular grid ({nx}x{ny} != {len(pts)} points)")
                x0, y0 = xs.min(), ys.min()
                dx = xs[1] - xs[0] if nx > 1 else 1.0
                dy = ys[1] - ys[0] if ny > 1 else 1.0
            
            grid = values.reshape((ny, nx), order='C')  # C-order: first index (nx) varies fastest in flat values
            
            # Extent for imshow: [x_min, x_max, y_min, y_max]. For pixel-centered data, extend by half-pixel.
            extent = [x0 - dx/2, x0 + (nx - 0.5)*dx, y0 - dy/2, y0 + (ny - 0.5)*dy]
            
            if method == 'imshow':
                # origin='lower' so y increases upward
                mappable = ax.imshow(grid, extent=extent, origin='lower', vmin=vmin, vmax=vmax, 
                                     aspect='equal', **kwargs)
            else:  # pcolormesh
                mappable = ax.pcolormesh(xs, ys, grid, shading='nearest', vmin=vmin, vmax=vmax, **kwargs)
                ax.set_aspect('equal')
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'auto', 'scatter', 'imshow', or 'pcolormesh'.")
        
        ax.set_aspect('equal')
        
        cbar = None
        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="2%")
            cbar = ax.figure.colorbar(mappable, cax=cax)
        
        # Store plot if named
        if name:
            self._plots[name] = {'sc': mappable, 'ax': ax, 'clim': (vmin, vmax), 'scalar': scalar, 'cbar': cbar}
        
        return mappable
    
    def sync_clim(self, *names, norm_type=None, symmetric=False, colorbar=False, 
                  cbar_width=0.05, cbar_pad=0.02, wspace=None, cbar_format='scientific'):
        """
        Synchronize color limits across named plots, with optional grouping like plot_n_components.
        
        names: plot names to sync. If empty, uses all stored plots.
        norm_type: grouping pattern like 'AAB' - plots with same letter share limits.
                   Length must match number of names. If None, all share same limits ('AAA...').
        symmetric: if True, center on 0: clim = (-max_abs, +max_abs)
        colorbar: if True, add colorbars to each plot
        cbar_width: colorbar width as fraction of plot width (default 0.05 = 5%)
        cbar_pad: padding between plot and colorbar as fraction (default 0.02 = 2%)
        wspace: spacing between subplots as fraction of plot width (default None = no change).
        cbar_format: colorbar tick format. Options:
            'scientific' (default): shows "2" with "×10⁻⁴" on top (ScalarFormatter with powerlimits)
            '%.2e': printf-style format string for scientific notation
            '%.3f': printf-style for fixed decimal
            None: use matplotlib default
            or pass a matplotlib.ticker.Formatter instance
        
        Usage:
            pipe.sync_clim('target', 'pred', 'error', symmetric=True, colorbar=True, wspace=0.4)
            pipe.sync_clim('target', 'pred', 'error', norm_type='AAB', colorbar=True, cbar_format='%.1e')
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        if not names:
            names = list(self._plots.keys())
        
        if norm_type is None:
            norm_type = 'A' * len(names)
        if len(norm_type) != len(names):
            raise ValueError(f"norm_type length ({len(norm_type)}) must match names count ({len(names)})")
        
        # Build groups: {'A': ['target', 'pred'], 'B': ['error']}
        groups = {}
        for name, group_key in zip(names, norm_type):
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(name)
        
        # Compute clim per group from original data, not stored clim (which may be stale)
        group_clims = {}
        for group_key, group_names in groups.items():
            vmins, vmaxs = [], []
            for n in group_names:
                info = self._plots[n]
                # Get actual data range from the scatter's array
                arr = info['sc'].get_array()
                vmins.append(float(arr.min()))
                vmaxs.append(float(arr.max()))
            vmin, vmax = min(vmins), max(vmaxs)
            if symmetric:
                bound = max(abs(vmin), abs(vmax))
                vmin, vmax = -bound, bound
            group_clims[group_key] = (vmin, vmax)
        
        # Apply clim to each plot
        for name, group_key in zip(names, norm_type):
            info = self._plots[name]
            vmin, vmax = group_clims[group_key]
            info['sc'].set_clim(vmin, vmax)
            info['clim'] = (vmin, vmax)
            
            # Add or replace colorbar with controlled sizing
            if colorbar:
                import matplotlib.ticker as ticker
                ax = info['ax']
                # Remove old colorbar if present
                if info.get('cbar') is not None:
                    info['cbar'].remove()
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=f"{cbar_width*100:.0f}%", pad=f"{cbar_pad*100:.0f}%")
                cbar = ax.figure.colorbar(info['sc'], cax=cax)
                
                # Apply tick format
                if cbar_format == 'scientific':
                    # ScalarFormatter with exponent on top, ticks show mantissa only
                    fmt = ticker.ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((-2, 2))  # use scientific for values outside 0.01-100
                    cbar.ax.yaxis.set_major_formatter(fmt)
                    cbar.ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
                elif isinstance(cbar_format, str):
                    # Printf-style format string like '%.2e' or '%.3f'
                    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(cbar_format))
                elif cbar_format is not None:
                    # Assume it's a Formatter instance
                    cbar.ax.yaxis.set_major_formatter(cbar_format)
                
                info['cbar'] = cbar
        
        # Adjust spacing between subplots if requested
        if wspace is not None and names:
            fig = self._plots[names[0]]['ax'].figure
            fig.subplots_adjust(wspace=wspace)
        
        return group_clims
    
    @property
    def plots(self):
        """Access stored plots by name."""
        return self._plots
    
    def to_image_data(self, block_name, tol=1e-5):
        """
        Convert a PolyData block to ImageData if points form a regular rectangular grid.
        
        Checks: uniform spacing in x and y (within tol * range), and n_x * n_y == n_points.
        If valid, replaces the block with ImageData and copies all scalars.
        """
        block = self[block_name]
        pts = np.asarray(block.points)
        
        x_unique = np.unique(pts[:, 0])
        y_unique = np.unique(pts[:, 1])
        n_x, n_y = len(x_unique), len(y_unique)
        
        # Check completeness: grid should have exactly n_x * n_y points
        if n_x * n_y != len(pts):
            raise ValueError(f"Not a complete grid: {n_x} x {n_y} = {n_x * n_y} != {len(pts)} points")
        
        # Check uniform spacing in x
        if n_x > 1:
            dx = np.diff(x_unique)
            x_range = x_unique[-1] - x_unique[0]
            if x_range > 0 and np.max(np.abs(dx - dx[0])) > tol * x_range:
                raise ValueError(f"Non-uniform x spacing: max deviation {np.max(np.abs(dx - dx[0])):.2e}")
            spacing_x = dx[0] if len(dx) > 0 else 1.0
        else:
            spacing_x = 1.0
        
        # Check uniform spacing in y
        if n_y > 1:
            dy = np.diff(y_unique)
            y_range = y_unique[-1] - y_unique[0]
            if y_range > 0 and np.max(np.abs(dy - dy[0])) > tol * y_range:
                raise ValueError(f"Non-uniform y spacing: max deviation {np.max(np.abs(dy - dy[0])):.2e}")
            spacing_y = dy[0] if len(dy) > 0 else 1.0
        else:
            spacing_y = 1.0
        
        # Create ImageData. Origin is the min corner, dimensions are n_x, n_y, 1
        z_val = pts[0, 2] if pts.shape[1] > 2 else 0.0
        origin = (x_unique[0], y_unique[0], z_val)
        
        img = pv.ImageData(dimensions=(n_x, n_y, 1), spacing=(spacing_x, spacing_y, 1.0), origin=origin)
        
        # Map scalars: need to reorder from arbitrary point order to grid order (x varies fastest in ImageData)
        # Build index map: for each point, find its (ix, iy) and compute flat index ix + iy * n_x
        x_to_ix = {v: i for i, v in enumerate(x_unique)}
        y_to_iy = {v: i for i, v in enumerate(y_unique)}
        
        # Find closest match for each point (handles floating point)
        def find_idx(val, unique_arr):
            return np.argmin(np.abs(unique_arr - val))
        
        reorder = np.array([find_idx(pts[i, 0], x_unique) + find_idx(pts[i, 1], y_unique) * n_x 
                           for i in range(len(pts))])
        
        # Copy scalars with reordering
        for name in block.point_data.keys():
            data = block.point_data[name]
            reordered = np.empty_like(data)
            reordered[reorder] = data
            img.point_data[name] = reordered
        
        # Replace block
        super().__setitem__(block_name, img)
        return self
    
    def __repr__(self):
        lines = ["Pipeset:"]
        for name in self.keys():
            block = super().__getitem__(name)
            n_pts = block.n_points
            scalars = list(block.point_data.keys())
            btype = type(block).__name__
            lines.append(f"  '{name}': {btype}, {n_pts} pts, scalars={scalars}")
        return "\n".join(lines)
    
    def add(self, obj, name=None, **kwargs):
        if name is None:
            name = obj.__class__.__name__
            
        if isinstance(obj, Region2D):
            inp = kwargs.get("inp", None)
            if inp is None:
                raise ValueError("`inp` is required for adding Region2D")
            self.add_region(obj, inp, name)
                    
        elif isinstance(obj, str):
            pts = kwargs.get("pts", None)
            pts = kwargs.get("points", None)
            if pts is None:
                raise ValueError(f"`pts` or `points` argument is required for adding a named block with name {name}")
            self[name] = pts
            
            # Check if rest of kwargs is a valid scalar
            for k, v in kwargs.items():
                if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim == 1:
                    self[name + "." + k] = v


# Backwards compatibility aliases
Dataset = Pipeset
MagneticFieldImageData = Dataset  # old name for the ImageData-based class
MagneticFieldUnstructuredGrid = UnstructuredData
