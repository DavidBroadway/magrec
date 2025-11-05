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


class Dataset(MagneticFieldDataMixin, pv.ImageData):
    """Primary interface for spatial magnetic field data.
    
    Dataset is a flexible container that handles:
    - **Structured grids**: Regular ImageData with dimensions, origin, spacing
    - **Unstructured data**: Arbitrary point clouds (internally uses PolyData)
    - **Point data**: Field values at each point (e.g., magnetic field B)
    - **Field data**: Metadata (e.g., dipole locations, grid parameters)
    
    Key Differences from DataBlock:
    - **Dataset**: Single spatial structure (one grid or point cloud)
    - **DataBlock**: Multiple Datasets combined (different regions/resolutions)
    
    Methods for structured grids only:
    - expand_bounds_2d(), expand_bounds_3d()
    - pts_as_grid()
    
    Methods for any data:
    - from_dict(), from_unstructured()
    - pts_as_list()
    - add_dipole_locations()
    
    Examples
    --------
    Structured grid:
    >>> ds = Dataset()
    >>> ds.dimensions = (50, 50, 1)
    >>> ds.origin = (0, 0, 0)
    >>> ds.spacing = (0.1, 0.1, 1.0)
    >>> ds.point_data['B'] = field_values
    
    From dictionary:
    >>> data = {'xs': x_coords, 'ys': y_coords, 'B': field_values}
    >>> ds = Dataset.from_dict(data)
    
    From grid data:
    >>> data = {'xs': x_grid, 'ys': y_grid, 'B': field_values}
    >>> ds = Dataset.from_dict(data, x_grid=True, y_grid=True)
    """
    
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
    
    def add_dipole_locations(self, points, name='dipole_locations'):
        """Store dipole locations in field_data for later use.
        
        Parameters
        ----------
        points : array_like, shape (n, 3)
            Dipole locations as (x, y, z) coordinates
        name : str, optional
            Storage key name
        
        Returns
        -------
        self : Dataset
            For method chaining
        
        Examples
        --------
        >>> ds.add_dipole_locations(np.array([[0, 0, 0], [1, 1, 1]]))
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be shape (n, 3), got {points.shape}")
        self.field_data[name] = points
        return self


# Backwards compatibility aliases
MagneticFieldImageData = Dataset
MagneticFieldUnstructuredGrid = UnstructuredData
