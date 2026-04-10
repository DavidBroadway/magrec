# Classes and functions for handling spatial magnetic field data.

import functools
import html
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
                return torch.as_tensor(field_data[name])
            
            # Check if the attribute is in `point_data`
            point_data = object.__getattribute__(self, 'point_data')
            if name in point_data:
                return torch.as_tensor(point_data[name])
            
        except AttributeError:
            pass
        
        # If not found in field_data or point_data, raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def map(self, func: callable, name):
        """Map a function over all points, assign result to point_data."""
        self.point_data[name] = func(torch.as_tensor(self.points, dtype=torch.float32)).detach().numpy()
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
    
    
def wrap_as_pipeset(method):
    """
    Decorator for Pipeset methods that may return bare VTK datasets.
    If the wrapped method returns an ImageData / PolyData / UnstructuredGrid /
    StructuredGrid, it is wrapped into a new Pipeset so that methods like
    get_as_grid() remain available on the returned object.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        return self._wrap_result_as_pipeset(result, default_name=method.__name__)
    return wrapper
    

class Pipeset(pv.MultiBlock, MagneticFieldDataMixin):
    """
    Pipeline + Dataset hybrid built on PyVista MultiBlock. Each block is a point set
    (structured or unstructured). When there is only one block, .points, .point_data,
    etc. delegate to it so you use the Pipeset like the block (no shallow wrap).
    
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
        self._plots = {}
        self._steps = {}
        self._assigned = set()
    
    def __repr__(self):
        if len(self) == 0:
            return "Pipeset (empty)"
        if len(self) == 1:
            return self[0].__repr__()
        lines = [f"Pipeset ({len(self)} blocks):"]
        for name in self.keys():
            blk = super().__getitem__(name)
            npts = getattr(blk, "n_points", "?")
            btype = type(blk).__name__
            scalars = list(blk.point_data.keys()) if hasattr(blk, "point_data") else []
            lines.append(f"  '{name}': {btype}, {npts} pts, scalars={scalars}")
        return "\n".join(lines)

    def _repr_html_(self):
        """Rich HTML representation for Jupyter notebooks.
        
        Single-block: custom table with header + data arrays where field string
        values (e.g. *_units) are shown directly in the arrays section.
        Multi-block: enhanced block table showing
        each block's type, point count, arrays, and units at a glance.
        """
        units = self.units

        if len(self) == 1:
            blk = super().__getitem__(0)
            return self._dataset_repr_with_field_values(blk)

        # Multi-block: two-column layout (info | blocks with details)
        fmt = "<table style='width: 100%;'>"
        fmt += '<tr><th>Pipeset</th><th>Blocks</th></tr>'

        # Left column: summary attributes
        fmt += '<tr><td><table>\n'
        fmt += f'<tr><td>N Blocks</td><td>{len(self)}</td></tr>\n'
        bds = self._aggregate_bounds()
        ff = '{:.3e}'
        fmt += f'<tr><td>X Bounds</td><td>{ff.format(bds[0])}, {ff.format(bds[1])}</td></tr>\n'
        fmt += f'<tr><td>Y Bounds</td><td>{ff.format(bds[2])}, {ff.format(bds[3])}</td></tr>\n'
        fmt += f'<tr><td>Z Bounds</td><td>{ff.format(bds[4])}, {ff.format(bds[5])}</td></tr>\n'
        if units:
            fmt += '<tr><td colspan="2"><b>Units</b></td></tr>\n'
            for name, unit in units.items():
                fmt += f'<tr><td style="padding-left:10px">{name}</td><td>{unit}</td></tr>\n'
        fmt += '</table></td>\n'

        # Right column: block details
        fmt += '<td><table>\n'
        fmt += '<tr><th>#</th><th>Name</th><th>Type</th><th>N Points</th><th>Arrays</th></tr>\n'
        for i in range(len(self)):
            blk = super().__getitem__(i)
            bname = self.get_block_name(i) or ''
            btype = type(blk).__name__
            npts = blk.n_points if hasattr(blk, 'n_points') else '—'

            # Collect array names with units annotation
            arr_parts = []
            if hasattr(blk, 'point_data'):
                for aname in blk.point_data.keys():
                    u_key = f'{aname}_units'
                    if hasattr(blk, 'field_data') and u_key in blk.field_data:
                        u = str(blk.field_data[u_key][0])
                        arr_parts.append(f'{aname} <i>[{u}]</i>')
                    else:
                        arr_parts.append(aname)
            arrays_str = ', '.join(arr_parts) if arr_parts else '—'
            fmt += f'<tr><td>{i}</td><td>{bname}</td><td>{btype}</td><td>{npts}</td><td>{arrays_str}</td></tr>\n'
        fmt += '</table></td></tr></table>'
        return fmt

    def _aggregate_bounds(self):
        """Axis-aligned union of each block's bounds; nested Pipeset/MultiBlock recurse via .bounds."""
        xs, ys, zs = [], [], []
        for i in range(len(self)):
            blk = super(Pipeset, self).__getitem__(i)
            b = getattr(blk, "bounds", None)
            if b is None or len(b) < 6:
                continue
            xs.extend((b[0], b[1]))
            ys.extend((b[2], b[3]))
            zs.extend((b[4], b[5]))
        if not xs:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    @staticmethod
    def _field_value_to_string(value):
        """Convert a field_data value to a readable string."""
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item()
            return item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
        flat = arr.ravel()
        if flat.size == 0:
            return ""
        if flat.size == 1:
            item = flat[0]
            return item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
        # Compact representation for short vectors
        if flat.size <= 4:
            items = []
            for item in flat:
                items.append(item.decode() if isinstance(item, (bytes, bytearray)) else str(item))
            return "[" + ", ".join(items) + "]"
        return f"array(shape={arr.shape})"

    def _dataset_repr_with_field_values(self, blk):
        """Single-block HTML repr with explicit field-data values."""
        if not hasattr(blk, "head"):
            return f"<pre>{blk!r}</pre>"

        fmt = "<table style='width: 100%;'>"
        fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
        fmt += "<tr><td>"
        fmt += blk.head(display=False, html=True)
        fmt += "</td><td>"
        fmt += "<table style='width: 100%;'>\n"
        titles = ["Name", "Field", "Type", "N Comp", "Min", "Max"]
        fmt += "<tr>" + "".join([f"<th>{t}</th>" for t in titles]) + "</tr>\n"
        row = "<tr>" + "".join(["<td>{}</td>" for _ in titles]) + "</tr>\n"

        def add_numeric_row(name, arr, field):
            dl, dh = blk.get_data_range(arr)
            dl = pv.FLOAT_FORMAT.format(dl)
            dh = pv.FLOAT_FORMAT.format(dh)
            ncomp = arr.shape[1] if arr.ndim > 1 else 1
            return row.format(html.escape(str(name)), field, arr.dtype, ncomp, dl, dh)

        for key, arr in blk.point_data.items():
            fmt += add_numeric_row(key, arr, "Points")
        for key, arr in blk.cell_data.items():
            fmt += add_numeric_row(key, arr, "Cells")
        for key, arr in blk.field_data.items():
            value = self._field_value_to_string(arr)
            # For field-data strings like *_units, use one merged cell for readability.
            fmt += (
                "<tr>"
                f"<td>{html.escape(str(key))}</td>"
                "<td>Fields</td>"
                f"<td colspan='4' style='text-align:left;'>Value: {html.escape(value)}</td>"
                "</tr>\n"
            )

        fmt += "</table></td></tr></table>"
        return fmt

    def __getattr__(self, name):
        if len(self) == 1:
            block = self[0]
            try:
                return getattr(block, name)
            except AttributeError:
                if hasattr(block, "point_data") and name in block.point_data:
                    return torch.as_tensor(block.point_data[name])
                raise
        if name == "bounds":
            return self._aggregate_bounds()
        # Multi-block: delegate other names to PyVista __getattr__ chain
        for cls in type(self).__mro__[1:]:
            ga = cls.__dict__.get("__getattr__")
            if ga is not None:
                try:
                    return ga(self, name)
                except AttributeError:
                    continue
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' "
            f"(multiple blocks: use dotted key e.g. pipe['block_name.{name}'])"
        )

    def _geometry_block(self, create_if_missing=False):
        """Return the single ImageData block used for geometry operations."""
        if len(self) == 0:
            if not create_if_missing:
                raise AttributeError("Pipeset has no blocks.")
            grid = pv.ImageData()
            super().__setitem__("grid", grid)
            return grid
        if len(self) != 1:
            block_names = [self.get_block_name(i) or str(i) for i in range(len(self))]
            raise AttributeError(
                "Geometry assignment is ambiguous for Pipeset with multiple blocks: "
                f"{block_names}"
            )
        blk = super().__getitem__(0)
        if not isinstance(blk, pv.ImageData):
            raise TypeError(
                f"Geometry assignment expects a single ImageData block, got {type(blk).__name__}"
            )
        return blk

    @property
    def dimensions(self):
        return self._geometry_block(create_if_missing=False).dimensions

    @dimensions.setter
    def dimensions(self, dims):
        blk = self._geometry_block(create_if_missing=True)
        old_spacing = tuple(blk.spacing)
        old_origin = tuple(blk.origin)
        blk.dimensions = tuple(int(v) for v in dims)
        # dimensions update keeps spacing constant; bounds expand/contract.
        blk.spacing = old_spacing
        blk.origin = old_origin
        self._assigned.add("dimensions")

    @property
    def origin(self):
        return self._geometry_block(create_if_missing=False).origin

    @origin.setter
    def origin(self, value):
        blk = self._geometry_block(create_if_missing=True)
        old_spacing = tuple(blk.spacing)
        blk.origin = tuple(float(v) for v in value)
        # keep spacing constant when origin changes.
        blk.spacing = old_spacing
        self._assigned.add("origin")

    @property
    def spacing(self):
        return self._geometry_block(create_if_missing=False).spacing

    @spacing.setter
    def spacing(self, value):
        blk = self._geometry_block(create_if_missing=True)
        sx, sy, sz = (float(v) for v in value)
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError("spacing components must be positive")
        # spacing has highest constancy priority:
        # keep current bounds, recompute dimensions.
        xmin, xmax, ymin, ymax, zmin, zmax = blk.bounds
        nx = int(round((xmax - xmin) / sx)) + 1
        ny = int(round((ymax - ymin) / sy)) + 1
        nz = int(round((zmax - zmin) / sz)) + 1
        blk.origin = (xmin, ymin, zmin)
        blk.spacing = (sx, sy, sz)
        blk.dimensions = (max(nx, 1), max(ny, 1), max(nz, 1))
        self._assigned.add("spacing")

    @property
    def bounds(self):
        return self._geometry_block(create_if_missing=False).bounds

    @bounds.setter
    def bounds(self, value):
        if len(value) != 6:
            raise ValueError("bounds must be (xmin, xmax, ymin, ymax, zmin, zmax)")
        blk = self._geometry_block(create_if_missing=True)
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in value)
        if xmax < xmin or ymax < ymin or zmax < zmin:
            raise ValueError("Invalid bounds ordering")
        # Assignment-aware precedence:
        # - if spacing was explicitly assigned, keep spacing and recompute dimensions
        # - else if dimensions were explicitly assigned, keep dimensions and recompute spacing
        # - else fallback to keep spacing and recompute dimensions
        if "spacing" in self._assigned:
            sx, sy, sz = blk.spacing
            nx = int(round((xmax - xmin) / sx)) + 1
            ny = int(round((ymax - ymin) / sy)) + 1
            nz = int(round((zmax - zmin) / sz)) + 1
            blk.origin = (xmin, ymin, zmin)
            blk.dimensions = (max(nx, 1), max(ny, 1), max(nz, 1))
            blk.spacing = (sx, sy, sz)
        elif "dimensions" in self._assigned:
            nx, ny, nz = blk.dimensions
            sx = (xmax - xmin) / max(nx - 1, 1)
            sy = (ymax - ymin) / max(ny - 1, 1)
            sz = (zmax - zmin) / max(nz - 1, 1)
            blk.origin = (xmin, ymin, zmin)
            blk.spacing = (sx, sy, sz)
            blk.dimensions = (max(int(nx), 1), max(int(ny), 1), max(int(nz), 1))
        else:
            sx, sy, sz = blk.spacing
            nx = int(round((xmax - xmin) / sx)) + 1
            ny = int(round((ymax - ymin) / sy)) + 1
            nz = int(round((zmax - zmin) / sz)) + 1
            blk.origin = (xmin, ymin, zmin)
            blk.dimensions = (max(nx, 1), max(ny, 1), max(nz, 1))
            blk.spacing = (sx, sy, sz)
        self._assigned.add("bounds")

    @property
    def n_points(self):
        """Number of points for single-block pipes.

        For multi-block pipes this is ambiguous, so an explicit error is raised
        with the available block names.
        """
        if len(self) == 1:
            blk = super().__getitem__(0)
            return int(getattr(blk, "n_points", 0))
        block_names = []
        for i in range(len(self)):
            name = self.get_block_name(i)
            block_names.append(name if name else str(i))
        raise AttributeError(
            "n_points is ambiguous for Pipeset with multiple blocks; "
            f"it has multiple sets of points in blocks: {block_names}"
        )
    
    # ── Units & Scaling ─────────────────────────────────────────────────
    #
    # Units are stored as field_data on each block following the convention:
    #   block.field_data['<array_name>_units'] = ['<units>']
    # Coordinate units use the reserved key 'coordinates_units'.
    #
    # pipe.set_units(B_NV='T', coordinates='m')   → stores on the single block
    # pipe.units                                   → {'B_NV': 'T', 'coordinates': 'm'}
    # pipe.scale('B_NV', factor=1e6)               → multiply values, record in _steps
    # pipe.scale('B_NV', to_units='uT')            → auto-compute factor from current units
    # pipe.scale(coordinates=True, to_units='um')   → rescale geometry + spacing/origin

    def set_units(self, block=None, **units):
        """Store unit strings as field_data['<name>_units'] on a block.
        
        For single-block Pipesets, block can be omitted. For multi-block, pass the
        block name or index. The special key 'coordinates' sets geometry units.
        Only accepts names that correspond to existing point_data arrays or
        the reserved key 'coordinates'.
        
            pipe.set_units(B_NV='T', coordinates='m')
            pipe.set_units(block='sensor', B_NV='T')
        """
        target = self.resolve_name(block)
        if not hasattr(target, "point_data"):
            raise TypeError(f"set_units() expected a leaf block; got {type(target).__name__}")
        valid_names = set(target.point_data.keys()) if hasattr(target, 'point_data') else set()
        valid_names.add('coordinates')
        for name in units:
            if name not in valid_names:
                raise KeyError(
                    f"'{name}' is not a point_data array on this block. "
                    f"Available: {sorted(valid_names)}")
        for name, unit in units.items():
            target.field_data[f'{name}_units'] = [str(unit)]
        return self

    @property
    def units(self):
        """Collect all '<name>_units' field_data across blocks into a dict.
        
        Returns dict like {'B_NV': 'T', 'coordinates': 'm'} for single block, or
        {'block_name.B_NV': 'T', ...} for multi-block.
        """
        result = {}
        suffix = '_units'
        for i in range(len(self)):
            blk = super().__getitem__(i)
            if not hasattr(blk, 'field_data'):
                continue
            prefix = '' if len(self) == 1 else f'{self.keys()[i]}.'
            for key in blk.field_data.keys():
                if key.endswith(suffix):
                    array_name = key[:-len(suffix)]
                    val = blk.field_data[key]
                    # field_data stores arrays; unit string is the first element
                    result[f'{prefix}{array_name}'] = str(val[0]) if hasattr(val, '__len__') else str(val)
        return result

    def scale(self, source=None, *, factor=None, to_units=None, coordinates=False,
              absolute=False, block=None, **array_factors):
        """Scale point data or coordinates, with optional unit tracking.
        
        Scaling a point_data array:
            pipe.scale('B_NV', factor=1e6)
            pipe.scale('B_NV', to_units='uT')       # requires current units set
        
        Scaling coordinates (points, and spacing/origin for ImageData):
            pipe.scale(coordinates=True, factor=1e6)
            pipe.scale(coordinates=True, to_units='um')  # default from-units is 'm'
        
        The scaling factor is recorded in _steps['scale.<name>'] so it can be
        reversed later via pipe.unscale(). The unit metadata is updated accordingly.
        """
        # Convenience form:
        #   pipe.scale(B_NV=1e4, Bz=1e3)
        # Applies per-array factors on the resolved block.
        if array_factors:
            if source is not None or factor is not None or to_units is not None or coordinates:
                raise ValueError(
                    "When using keyword factors (e.g. scale(B_NV=1e4)), do not pass "
                    "source/factor/to_units/coordinates."
                )
            target = self.resolve_name(block)
            if not hasattr(target, "point_data"):
                raise TypeError(f"scale() expected a leaf block; got {type(target).__name__}")
            for arr_name, arr_factor in array_factors.items():
                self._scale_array(
                    target, arr_name, arr_factor, to_units=None, block_ref=block, absolute=absolute
                )
            return self

        # Shorthand forms for coordinate scaling:
        #   scale(coordinates=1e7)   -> factor=1e7
        #   scale(coordinates='um')  -> to_units='um'
        coordinates_enabled = False
        if isinstance(coordinates, bool):
            coordinates_enabled = coordinates
        elif isinstance(coordinates, (int, float, np.number)):
            coordinates_enabled = True
            if factor is not None:
                raise ValueError("Pass either coordinates=<factor> or factor=, not both.")
            factor = float(coordinates)
        elif isinstance(coordinates, str):
            coordinates_enabled = True
            if to_units is not None:
                raise ValueError("Pass either coordinates=<unit> or to_units=, not both.")
            to_units = coordinates
        elif coordinates is not None:
            raise TypeError(
                f"coordinates must be bool, numeric factor, unit string, or None; got {type(coordinates).__name__}"
            )

        if coordinates_enabled and source is not None:
            raise ValueError("Pass either source=<array_name> or coordinates=True, not both.")
        if factor is not None and to_units is not None:
            raise ValueError("Pass either factor or to_units, not both.")
        if absolute and to_units is not None:
            raise ValueError("absolute=True is only valid with factor-based scaling.")
        
        target = self.resolve_name(block)
        if not hasattr(target, "point_data"):
            raise TypeError(f"scale() expected a leaf block; got {type(target).__name__}")
        
        if coordinates_enabled:
            return self._scale_coordinates(target, factor, to_units, block, absolute=absolute)
        
        if source is None:
            raise ValueError("Provide source=<array_name> or coordinates=True.")
        return self._scale_array(target, source, factor, to_units, block, absolute=absolute)

    def resolve_name(self, name):
        """Resolve a name to either a leaf block or a point_data array.

        Rules:
          - ``None``: single-block default (requires exactly one leaf block)
          - ``int``: block by index
          - ``str``:
              - No dots (``len(parts)==1``): return ``point_data[parts[0]]`` if it exists
                (unique across all leaves). Otherwise return the leaf block whose local
                name or full leaf path matches ``parts[0]``.
              - Dotted (``len(parts)>1``): traverse the MultiBlock tree using all parts
                except the last. If the traversal ends on a leaf dataset, the last part
                is interpreted as ``point_data``. If it ends on a MultiBlock container,
                the last part is interpreted as a child leaf block name, with a fallback
                to a unique ``point_data`` name across all leaves under that container.
        """
        if name is None:
            leaves = [path for path, _, _ in self._iter_leaf_blocks()]
            if len(leaves) != 1:
                raise ValueError(
                    "Resolving a block without a name requires exactly one block; "
                    "requires a name to specify which block to take from: "
                    f"{leaves}"
                )
            return super().__getitem__(0)

        if isinstance(name, int):
            return super().__getitem__(name)

        if not isinstance(name, str):
            raise TypeError(f"name must be int, str, or None; got {type(name).__name__}")

        def _get_child_by_local_name(container, local_name):
            """Return container[child] matching local_name, or None."""
            if not isinstance(container, pv.MultiBlock):
                return None

            for i in range(len(container)):
                candidate = container.get_block_name(i) or str(i)
                if local_name == candidate:
                    return container[i]

            if str(local_name).isdigit():
                idx = int(local_name)
                if 0 <= idx < len(container):
                    return container[idx]

            return None

        def _point_data_tensor(blk, array_name):
            return torch.as_tensor(np.asarray(blk.point_data[array_name]))

        parts = name.split(".")

        if len(parts) == 1:
            arr_name = parts[0]

            # Fast path for single-block Pipesets.
            if len(self) == 1:
                blk0 = super().__getitem__(0)
                if hasattr(blk0, "point_data") and arr_name in blk0.point_data:
                    return _point_data_tensor(blk0, arr_name)

            # Search for point_data across all leaves.
            point_matches = []
            for path, _, blk in self._iter_leaf_blocks():
                if hasattr(blk, "point_data") and arr_name in blk.point_data:
                    point_matches.append((path, blk))

            if len(point_matches) == 1:
                return _point_data_tensor(point_matches[0][1], arr_name)
            if len(point_matches) > 1:
                paths = [p for p, _ in point_matches]
                raise ValueError(f"Ambiguous point_data name '{arr_name}', matches: {paths}")

            # If point_data doesn't exist, resolve to a leaf block name/path.
            block_matches = []
            for path, local_name, blk in self._iter_leaf_blocks():
                if arr_name == local_name or arr_name == path:
                    block_matches.append((path, blk))

            if len(block_matches) == 1:
                return block_matches[0][1]
            if len(block_matches) > 1:
                paths = [p for p, _ in block_matches]
                raise ValueError(f"Ambiguous block name '{arr_name}', matches: {paths}")

            raise KeyError(f"Name '{name}' not found as point_data or leaf block")

        # Dotted resolution: traverse parts[:-1] down the tree, then resolve parts[-1].
        try:
            current = self
            for seg in parts[:-1]:
                if not isinstance(current, pv.MultiBlock):
                    raise KeyError(f"Cannot traverse into non-MultiBlock while resolving '{name}'")
                child = _get_child_by_local_name(current, seg)
                if child is None:
                    raise KeyError(f"Missing block segment '{seg}' while resolving '{name}'")
                current = child

            prefix_path = ".".join(parts[:-1])
            last = parts[-1]

            if isinstance(current, pv.MultiBlock):
                # Prefer an immediate leaf-block child match.
                child = _get_child_by_local_name(current, last)
                if child is not None:
                    if isinstance(child, pv.MultiBlock):
                        leaves = list(self._iter_leaf_blocks(container=child, parent_path=prefix_path + "." + last))
                        if len(leaves) == 1:
                            return leaves[0][2]
                        raise ValueError(
                            f"Name '{name}' matched a MultiBlock container; "
                            f"matched {len(leaves)} leaf blocks. Specify a leaf block name/path."
                        )
                    return child

                # Fallback: interpret last as point_data name within leaves under this container.
                point_matches = []
                for path, _, blk in self._iter_leaf_blocks(container=current, parent_path=prefix_path):
                    if hasattr(blk, "point_data") and last in blk.point_data:
                        point_matches.append((path, blk))

                if len(point_matches) == 1:
                    return _point_data_tensor(point_matches[0][1], last)
                if len(point_matches) > 1:
                    paths = [p for p, _ in point_matches]
                    raise ValueError(
                        f"Ambiguous point_data name '{last}' under '{prefix_path}', matches: {paths}"
                    )

                # Last attempt: resolve leaf block by local name under this container.
                block_matches = []
                for path, local_name, blk in self._iter_leaf_blocks(container=current, parent_path=prefix_path):
                    if last == local_name or last == path:
                        block_matches.append((path, blk))
                if len(block_matches) == 1:
                    return block_matches[0][1]
                if len(block_matches) > 1:
                    paths = [p for p, _ in block_matches]
                    raise ValueError(
                        f"Ambiguous block name '{last}' under '{prefix_path}', matches: {paths}"
                    )

                raise KeyError(f"Name '{name}' not found under '{prefix_path}'")

            # We ended on a leaf dataset: last part must be point_data.
            if hasattr(current, "point_data") and last in current.point_data:
                return _point_data_tensor(current, last)

            raise KeyError(f"'{last}' not found as point_data on leaf '{prefix_path}'")

        except KeyError:
            # Fallback for flattened leaf blocks (top-level keys may already include dots).
            block_matches = self._find_leaf_blocks_by_name(name)
            if len(block_matches) == 1:
                return block_matches[0][1]
            if len(block_matches) > 1:
                paths = [p for p, _ in block_matches]
                raise ValueError(f"Ambiguous block name '{name}', matches: {paths}")

            block_path = ".".join(parts[:-1])
            arr_name = parts[-1]
            prefix_matches = self._find_leaf_blocks_by_name(block_path)
            if len(prefix_matches) == 1:
                blk = prefix_matches[0][1]
                if hasattr(blk, "point_data") and arr_name in blk.point_data:
                    return _point_data_tensor(blk, arr_name)
                raise KeyError(f"'{arr_name}' not found in point_data of '{block_path}'")
            if len(prefix_matches) > 1:
                paths = [p for p, _ in prefix_matches]
                raise ValueError(f"Ambiguous block path '{block_path}', matches: {paths}")

            raise KeyError(f"Name '{name}' not found as leaf block or point_data")

    def _scale_array(self, target, source, factor, to_units, block_ref, absolute=False):
        """Scale a single point_data array on target block."""
        if source not in target.point_data:
            raise KeyError(f"'{source}' not in point_data of block")
        
        units_key = f'{source}_units'
        current_unit = None
        has_explicit_unit = units_key in target.field_data
        if has_explicit_unit:
            current_unit = str(target.field_data[units_key][0])
        
        if to_units is not None:
            if current_unit is None:
                raise ValueError(
                    f"Cannot convert to '{to_units}': no current units set for '{source}'. "
                    f"Call pipe.set_units({source}='<unit>') first.")
            # Both units must share the same base dimension (e.g., both end in 'T')
            # so the conversion is purely a prefix exponent difference
            from magrec.prop.constants import get_exponent_from_unit
            from_exp = get_exponent_from_unit(current_unit)
            to_exp = get_exponent_from_unit(to_units)
            factor = 10.0 ** (from_exp - to_exp)
        
        if factor is None:
            raise ValueError("Provide factor= or to_units=.")

        if absolute and to_units is None:
            # 'factor' is desired final cumulative data scale.
            # Convert to the incremental factor needed from current state.
            prev_scale, _ = self._parse_scaling_unit_string(current_unit) if current_unit else (1.0, None)
            current_total_factor = 1.0 / prev_scale
            factor = float(factor) / current_total_factor
        
        target.point_data[source] = np.asarray(target.point_data[source]) * factor
        
        # Update unit metadata
        if to_units is not None:
            target.field_data[units_key] = [str(to_units)]
        else:
            # Track manual scaling as inverse factor so physical values stay interpretable:
            # scaled_value * (1/factor) [unit] == original physical value.
            if has_explicit_unit and current_unit:
                target.field_data[units_key] = [self._compose_scaled_unit_string(current_unit, factor)]
            else:
                target.field_data[units_key] = [self._format_inverse_factor(factor)]
        
        # Record in steps for reversibility
        step_name = f'scale.{source}'
        self._steps[step_name] = {
            'type': 'scale', 'source': source, 'factor': factor,
            'from_units': current_unit, 'to_units': to_units,
        }
        return self

    @staticmethod
    def _format_inverse_factor(factor):
        """Format inverse scaling factor as compact scientific notation."""
        inv = 1.0 / float(factor)
        base, exp = f"{inv:.0e}".split("e")
        return f"{base}e{int(exp)}"

    @staticmethod
    def _parse_scaling_unit_string(unit_string):
        """Parse '<scale> <unit>' or legacy 'xN' scaling notation."""
        s = str(unit_string).strip()
        if not s:
            return 1.0, None

        parts = s.split()
        first = parts[0]
        rest = " ".join(parts[1:]).strip() if len(parts) > 1 else None

        try:
            return float(first), rest
        except ValueError:
            pass

        if first.startswith("x"):
            try:
                return 1.0 / float(first[1:]), rest
            except ValueError:
                pass

        return 1.0, s

    def _compose_scaled_unit_string(self, existing_unit_string, factor):
        """Compose cumulative inverse factor with optional base unit."""
        prev_scale, base_unit = self._parse_scaling_unit_string(existing_unit_string)
        new_scale = prev_scale * (1.0 / float(factor))
        base, exp = f"{new_scale:.0e}".split("e")
        scale_str = f"{base}e{int(exp)}"
        return f"{scale_str} {base_unit}" if base_unit else scale_str

    def _scale_coordinates(self, target, factor, to_units, block_ref, absolute=False):
        """Scale point coordinates (and spacing/origin for ImageData)."""
        units_key = 'coordinates_units'
        current_unit = None
        has_explicit_unit = units_key in target.field_data
        if has_explicit_unit:
            current_unit = str(target.field_data[units_key][0])
        else:
            current_unit = 'm'  # SI default for coordinates
        
        if to_units is not None:
            from magrec.prop.constants import get_exponent_from_unit
            from_exp = get_exponent_from_unit(current_unit)
            to_exp = get_exponent_from_unit(to_units)
            factor = 10.0 ** (from_exp - to_exp)
        
        if factor is None:
            raise ValueError("Provide factor= or to_units=.")

        if absolute and to_units is None:
            # Same absolute-mode semantics as for point_data arrays.
            prev_scale, _ = self._parse_scaling_unit_string(current_unit) if current_unit else (1.0, None)
            current_total_factor = 1.0 / prev_scale
            factor = float(factor) / current_total_factor
        
        # Scale point positions. For ImageData we must adjust origin and spacing
        # (points are computed from those); for PolyData we scale points directly.
        if isinstance(target, pv.ImageData):
            ox, oy, oz = target.origin
            sx, sy, sz = target.spacing
            target.origin = (ox * factor, oy * factor, oz * factor)
            target.spacing = (sx * factor, sy * factor, sz * factor)
        else:
            target.points = np.asarray(target.points) * factor
        
        if to_units is not None:
            new_unit = str(to_units)
        else:
            new_unit = (
                self._compose_scaled_unit_string(current_unit, factor)
                if has_explicit_unit
                else self._format_inverse_factor(factor)
            )
        target.field_data[units_key] = [new_unit]
        
        step_name = 'scale.coordinates'
        self._steps[step_name] = {
            'type': 'scale_coordinates', 'factor': factor,
            'from_units': current_unit, 'to_units': new_unit,
        }
        return self

    # TODO: replace unscale() with a proper step-referencing approach. The problem
    # with a standalone unscale('B_NV') is ambiguity: there could be multiple
    # successive scalings of the same array (e.g. first unit conversion T→uT,
    # then normalization by std). Instead, each scale() call should return a step
    # handle (or be named), and unscale should reference that handle:
    #
    #     s1 = pipe.scale('B_NV', to_units='uT')        # step handle
    #     s2 = pipe.scale('B_NV', factor=1/std)          # another step
    #     pipe.unscale(s2)                                # undo just s2
    #     pipe.unscale(s1)                                # undo s1
    #
    # The steps are already recorded in _steps with enough info to invert.
    # A cleaner design would store them as an ordered list per array (not
    # overwriting 'scale.B_NV'), pop the last one on unscale, and compose
    # factors when queried. For now this naive version only handles the last
    # scaling per array.

    def unscale(self, source=None, *, coordinates=False, block=None):
        """Reverse the most recent scale() call for a given source.
        
        Only handles one scaling per array — see TODO above for the proper
        step-referencing design that supports stacked scalings.
        """
        if coordinates:
            step_name = 'scale.coordinates'
        elif source is not None:
            step_name = f'scale.{source}'
        else:
            raise ValueError("Provide source=<array_name> or coordinates=True.")
        
        if step_name not in self._steps:
            raise KeyError(f"No recorded scaling step '{step_name}' to reverse.")
        
        step = self._steps[step_name]
        inv_factor = 1.0 / step['factor']
        target = self.resolve_name(block)
        if not hasattr(target, "point_data"):
            raise TypeError(f"unscale() expected a leaf block; got {type(target).__name__}")
        
        if coordinates:
            if isinstance(target, pv.ImageData):
                ox, oy, oz = target.origin
                sx, sy, sz = target.spacing
                target.origin = (ox * inv_factor, oy * inv_factor, oz * inv_factor)
                target.spacing = (sx * inv_factor, sy * inv_factor, sz * inv_factor)
            else:
                target.points = np.asarray(target.points) * inv_factor
            if step['from_units']:
                target.field_data['coordinates_units'] = [step['from_units']]
        else:
            name = step['source']
            target.point_data[name] = np.asarray(target.point_data[name]) * inv_factor
            if step['from_units']:
                target.field_data[f'{name}_units'] = [step['from_units']]
        
        del self._steps[step_name]
        return self

    @classmethod
    def from_dict(cls, datadict, rename_map=None, x_grid=False, y_grid=False, 
                  as_regular_grid=False, nx=None, ny=None, nz=1):
        """Create a Dataset from a dictionary of coordinates and field data.
        
        Intelligently handles different coordinate formats:
        - (N,) or (N, 1): List of coordinates for N points
        - (N, M): Grid coordinates where xs[i, j] is x-coord at pixel (i,j)
        - (nx,), (ny,), and (nz,): Coordinates for each dimension of the 
            grid to make a regular grid, so that nx * ny * nz = N points.
        
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
        Pipeset
            Pipeset with one block (index 0). When only one block, .points, .point_data
            etc. delegate to that block so you use the pipe like the block directly.
        
        Examples
        --------
        >>> data = {'xs': [0, 1, 2], 'ys': [0, 1, 2], 'B': [...]}
        >>> pipe = Pipeset.from_dict(data)
        >>> pipe.point_data['B']   # single block: direct access
        
        >>> # Grid coordinates
        >>> xx, yy = np.meshgrid(x, y)
        >>> data = {'xs': xx, 'ys': yy, 'B': field}
        >>> pipe = Pipeset.from_dict(data, x_grid=True, y_grid=True)
        
        >>> # With renaming
        >>> pipe = Pipeset.from_dict(data, rename_map=["BNV->B", "x->xs"])
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
        
        # Determine coordinate format and create points.
        # Special-case axis-vector regular grids:
        # xs:(nx,), ys:(ny,), zs:(nz,) with data arrays like (nx, ny, nz).
        inferred_dims = None
        z_candidate = zs if zs is not None else (height if height is not None else standoff)
        if not x_grid and not y_grid and xs.ndim == 1 and ys.ndim == 1 and z_candidate is not None:
            z_arr = np.asarray(z_candidate)
            if z_arr.ndim == 1 and z_arr.size > 1:
                candidate_dims = (xs.size, ys.size, z_arr.size)
                for value in data.values():
                    arr = np.asarray(value)
                    if arr.ndim >= 3 and arr.shape[:3] == candidate_dims:
                        inferred_dims = candidate_dims
                        break
                if inferred_dims is not None:
                    xx, yy, zz = np.meshgrid(xs, ys, z_arr, indexing='ij')
                    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
                else:
                    points = cls._parse_coordinates(xs, ys, zs, height, standoff, x_grid, y_grid)
            else:
                points = cls._parse_coordinates(xs, ys, zs, height, standoff, x_grid, y_grid)
        else:
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
                # 3D arrays can be either:
                # - scalar volume (nx, ny, nz) where size == n_points
                # - vector-like grid (nx, ny, n_components) where nx*ny == n_points
                if value_array.size == n_points:
                    unstruct.point_data[key] = value_array.ravel()
                elif value_array.shape[0] * value_array.shape[1] == n_points:
                    unstruct.point_data[key] = value_array.reshape(n_points, -1)
                else:
                    raise ValueError(f"Cannot match 3D array '{key}' shape {value_array.shape} to {n_points} points")
            else:
                raise ValueError(f"Data array '{key}' has unsupported dimensionality: {value_array.ndim}D")
        
        if as_regular_grid:
            if inferred_dims is not None:
                if nx is None:
                    nx = inferred_dims[0]
                if ny is None:
                    ny = inferred_dims[1]
                if nz in (None, 1):
                    nz = inferred_dims[2]
            return cls.from_unstructured(unstruct, nx=nx, ny=ny, nz=nz)
        pipe = cls()
        pipe.append(unstruct)
        return pipe
    
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
                # Else z is a list of coordinates and there can be multiple of z'em
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
        """Apply rename_map to transform dictionary keys. Used in renaming 
        passed dictionary keys to names of arrays in the dataset."""
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
        
        # Parse for split/merge operations
        split_operations = {}
        merge_operations = {}
        parsed_rename_map = {}
        
        for old_key, new_value in rename_map.items():
            if isinstance(old_key, str) and ',' in old_key:
                if not isinstance(new_value, str):
                    raise ValueError(
                        f"Merge target for '{old_key}' must be a string key, got {type(new_value).__name__}"
                    )
                source_keys = [k.strip() for k in old_key.split(',') if k.strip()]
                if len(source_keys) < 2:
                    raise ValueError(f"Invalid merge source specification: {old_key}")
                merge_operations[tuple(source_keys)] = new_value.strip()
            elif isinstance(new_value, (list, tuple)):
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

        # Apply merge operations (e.g. "Jx, Jy, Jz->J")
        for source_keys, target_key in merge_operations.items():
            missing = [k for k in source_keys if k not in data]
            if missing:
                raise ValueError(
                    f"Cannot merge into '{target_key}': missing source keys {missing}"
                )
            arrays = [np.asarray(data[k]) for k in source_keys]
            base_shape = arrays[0].shape
            for key, arr in zip(source_keys, arrays):
                if arr.shape != base_shape:
                    raise ValueError(
                        f"Cannot merge keys {list(source_keys)} into '{target_key}': "
                        f"shape mismatch at '{key}' ({arr.shape}) vs {base_shape}"
                    )
            data[target_key] = np.stack(arrays, axis=-1)
            for key in source_keys:
                data.pop(key, None)
        
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
    
    def _wrap_result_as_pipeset(self, result, default_name="block"):
        """
        Wrap common VTK dataset outputs into a new Pipeset.
        - If result is already a Pipeset, return it unchanged.
        - If result is a single PyVista dataset, wrap it as a one-block Pipeset.
        - If result is a list/tuple of datasets, wrap them as multiple blocks.
        - Otherwise, return result unchanged.
        """
        # Already a Pipeset
        if isinstance(result, Pipeset):
            return result
        vtk_types = (pv.ImageData, pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid)
        # Single dataset
        if isinstance(result, vtk_types):
            pipe = Pipeset()
            pipe[default_name] = result
            return pipe
        # List/tuple of datasets
        if isinstance(result, (list, tuple)) and result and all(isinstance(r, vtk_types) for r in result):
            pipe = Pipeset()
            for i, r in enumerate(result):
                pipe[f"{default_name}_{i}"] = r
            return pipe
        # Anything else: leave as is (numbers, tensors, dicts, etc.)
        return result
    
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

    def _iter_leaf_blocks(self, container=None, parent_path="", block_type=None):
        """Yield leaf blocks recursively as (path, local_name, block).

        A leaf is any block that is not a MultiBlock. Paths are dot-joined names,
        with numeric indices used for unnamed blocks.
        """
        if container is None:
            container = self
        if not isinstance(container, pv.MultiBlock):
            return

        for i in range(len(container)):
            child = container[i]
            local_name = container.get_block_name(i) or str(i)
            path = f"{parent_path}.{local_name}" if parent_path else local_name
            if isinstance(child, pv.MultiBlock):
                yield from self._iter_leaf_blocks(child, path, block_type=block_type)
            else:
                if block_type is None or isinstance(child, block_type):
                    yield path, local_name, child

    def _find_leaf_blocks_by_name(self, name, block_type=None):
        """Return all leaf blocks matching name by local name or full path."""
        matches = []
        for path, local_name, block in self._iter_leaf_blocks(block_type=block_type):
            if name == local_name or name == path:
                matches.append((path, block))
        return matches

    def _find_by_name(self, name, block_type=None):
        """Find all leaves and arrays matching a name.

        Returns a list of dicts with:
            kind: 'block' or 'array'
            path: full leaf path
            block: leaf object
            name: local matched name (for arrays: array name)
        """
        matches = []
        for path, local_name, block in self._iter_leaf_blocks(block_type=block_type):
            if name == local_name or name == path:
                matches.append({"kind": "block", "path": path, "block": block, "name": local_name})
            if hasattr(block, "point_data") and name in block.point_data:
                matches.append({"kind": "array", "path": path, "block": block, "name": name})
        return matches
    
    def downsample(self, new_shape, out=None, name=None, source=None, interpolation='linear',
                   anti_aliasing=False, as_tensor=False, as_grid=False, inplace=False):
        """Downsample ImageData blocks to new dimensions.

        Source resolution:
        - None            -> unique ImageData leaf block, all arrays
        - int             -> top-level block index, all arrays
        - str             -> leaf block name/path OR array name
        - list/tuple[str] -> array names on a unique ImageData leaf block

        If a string matches both block name and array name, raises ValueError.
        For block sources (or None), all arrays are resampled together.
        
        If requested `new_shape` is already found in an array of any block, it is
        considered as a cached result and returned directly.

        Parameters
        ----------
        out : str, optional
            Output block name. If provided, behaves like ``inplace=True`` and stores
            the result under this name.
        inplace : bool, default False
            If True, store the downsampled result into this Pipeset under ``out``
            (or ``downsampled`` when ``out`` is not provided) and return ``self``.
            If False, return a new one-block Pipeset with the downsampled result.
        """
        if out is None and name is not None:
            out = name

        dims = tuple(int(d) for d in new_shape)
        if len(dims) == 2:
            dims = (dims[0], dims[1], 1)

        # Resolve source -> (block, selected_arrays)
        # selected_arrays is None for whole-block resampling.
        if source is None:
            image_leaves = list(self._iter_leaf_blocks(block_type=pv.ImageData))
            if len(image_leaves) != 1:
                raise ValueError(
                    "downsample(..., source=None) requires exactly one ImageData leaf block; "
                    "pass source=<block_name>, source=<index>, or source=<array_name>."
                )
            block, selected_arrays = image_leaves[0][2], None
        elif isinstance(source, int):
            block = self[source]
            if not isinstance(block, pv.ImageData):
                raise TypeError(f"downsample only supports ImageData; got {type(block).__name__}")
            selected_arrays = None
        elif isinstance(source, (list, tuple)):
            selected_arrays = [str(s) for s in source]
            if len(selected_arrays) == 0:
                raise ValueError("source iterable cannot be empty")
            block_candidates = []
            for _, _, b in self._iter_leaf_blocks(block_type=pv.ImageData):
                if all(a in b.point_data for a in selected_arrays):
                    block_candidates.append(b)
            if len(block_candidates) == 0:
                raise KeyError(f"no ImageData leaf block contains all arrays {selected_arrays}")
            if len(block_candidates) > 1:
                raise ValueError(
                    f"arrays {selected_arrays} match multiple ImageData leaf blocks; "
                    "disambiguate by downsampling a block first."
                )
            block = block_candidates[0]
        elif isinstance(source, str):
            matches = self._find_by_name(source, block_type=pv.ImageData)
            if len(matches) == 0:
                raise KeyError(
                    f"'{source}' is neither a leaf block name/path nor a point_data array "
                    "on any ImageData leaf block"
                )
            if len(matches) > 1:
                # Prefer a unique cached hit already at requested output dimensions.
                dim_matches = []
                for m in matches:
                    b = m["block"]
                    if isinstance(b, pv.ImageData) and tuple(b.dimensions) == tuple(dims):
                        # For array-kind matches, ensure the array exists on that block.
                        if m["kind"] == "array" and source in b.point_data:
                            dim_matches.append(m)
                        elif m["kind"] == "block":
                            dim_matches.append(m)

                # De-duplicate by block identity in case both block+array match same block.
                uniq_by_block = {}
                for m in dim_matches:
                    uniq_by_block[id(m["block"])] = m
                dim_unique = list(uniq_by_block.values())

                if len(dim_unique) == 1:
                    m = dim_unique[0]
                    block = m["block"]
                    # Prefer array interpretation when available so as_grid/as_tensor works.
                    if source in block.point_data:
                        selected_arrays = [source]
                    else:
                        selected_arrays = None
                else:
                    kinds = {m["kind"] for m in matches}
                    if len(kinds) > 1:
                        raise ValueError(
                            f"'{source}' matches both a block name and a point_data array; "
                            "disambiguate with source=<int index> or unique names."
                        )
                    raise ValueError(
                        f"'{source}' matches multiple {next(iter(kinds))} entries; use unique names or paths."
                    )
            else:
                match = matches[0]
                block = match["block"]
                selected_arrays = None if match["kind"] == "block" else [source]
        else:
            raise TypeError(f"source must be str, int, list, tuple, or None; got {type(source).__name__}")

        base_kw = dict(
            dimensions=dims,
            interpolation=interpolation,
            anti_aliasing=anti_aliasing,
            inplace=False,
        )

        # Reuse an already stored downsampled grid when possible, especially for
        # fast as_grid/as_tensor access. Prefer explicit cache name, otherwise
        # search leaf blocks for a matching ImageData with same dimensions/array.
        cache_name = out or "downsampled"
        if selected_arrays is not None and len(selected_arrays) == 1:
            arr0 = selected_arrays[0]
            cached_block = None

            # 1) Preferred: named cache block
            if isinstance(cache_name, str) and cache_name in self.keys():
                maybe = super().__getitem__(cache_name)
                if isinstance(maybe, Pipeset) and len(maybe) == 1:
                    maybe = maybe[0]
                elif isinstance(maybe, pv.MultiBlock) and len(maybe) == 1:
                    maybe = maybe[0]
                if (
                    isinstance(maybe, pv.ImageData)
                    and tuple(maybe.dimensions) == tuple(dims)
                    and arr0 in maybe.point_data
                ):
                    cached_block = maybe

            # 2) Fallback: search any matching ImageData leaf
            if cached_block is None:
                for _, _, b in self._iter_leaf_blocks(block_type=pv.ImageData):
                    if tuple(b.dimensions) == tuple(dims) and arr0 in b.point_data:
                        cached_block = b
                        break

            if cached_block is not None:
                wrapped_cached = Pipeset()
                wrapped_cached.append(cached_block)
                if as_grid:
                    return wrapped_cached.get_as_grid(arr0, keep_dims=True)
                if as_tensor:
                    return torch.as_tensor(np.asarray(cached_block.point_data[arr0]))

        # Always resample explicitly per-array. This avoids backend-dependent behavior
        # where ImageData.resample() can keep only active scalars in whole-block mode.
        arrays_to_resample = (
            list(block.point_data.keys()) if selected_arrays is None else list(selected_arrays)
        )
        if len(arrays_to_resample) == 0:
            # No arrays to carry over; keep geometry-only result.
            resampled = block.resample(**base_kw)
        else:
            resampled = None
            for arr_name in arrays_to_resample:
                # PyVista's ImageData.resample() can ignore the explicit ``scalars=``
                # argument and sample whichever point array is currently active. 
                # Work on a copy with the requested array marked active so each pass
                # really samples the intended data instead of relabeling active scalars.
                block_copy = block.copy(deep=True)
                block_copy.set_active_scalars(arr_name, preference="point")
                r = block_copy.resample(**{**base_kw, "scalars": arr_name, "preference": "point"})
                if resampled is None:
                    resampled = r
                else:
                    resampled.point_data[arr_name] = r.point_data[arr_name]

        # Preserve field_data metadata (e.g. *_units) from source block.
        for k in block.field_data.keys():
            resampled.field_data[k] = block.field_data[k]

        wrapped = Pipeset()
        wrapped.append(resampled)

        if as_grid:
            if selected_arrays is None or len(selected_arrays) != 1:
                raise ValueError("as_grid requires a single array source")
            return wrapped.get_as_grid(selected_arrays[0], keep_dims=True)

        if as_tensor:
            if selected_arrays is None or len(selected_arrays) != 1:
                raise ValueError("as_tensor requires a single array source")
            return torch.as_tensor(np.asarray(resampled.point_data[selected_arrays[0]]))

        if inplace or out is not None:
            out_name = out or "downsampled"
            self.append(wrapped, out_name)
            return self
        return wrapped
    
    @classmethod
    def _get_as_grid(cls, grid, point_data_name):
        """Return point data reshaped to grid structure (class method helper)."""
        data = grid[point_data_name]
        nx, ny, nz = grid.dimensions
        shape = (nz, ny, nx, 3) if nz > 1 else (ny, nx, 3)
        if nz > 1:
            return torch.as_tensor(data.reshape(*shape)).permute(2, 1, 0, 3)
        elif nz == 1:
            return torch.as_tensor(data.reshape(*shape)).permute(1, 0, 2)
        else:
            raise ValueError("Invalid dimensions for the grid.")
    
    def get_as_grid(self, point_data_name, keep_dims=False):
        """Return point data reshaped to (nx, ny, nz[, ncomp]) to match grid. ImageData only."""
        block = self.resolve_name(None)
        if not hasattr(block, "point_data") or point_data_name not in block.point_data:
            available = sorted(block.point_data.keys()) if hasattr(block, "point_data") else []
            raise KeyError(
                f"'{point_data_name}' not found in point_data of the resolved block. "
                f"Available: {available}"
            )

        # Read the array explicitly from point_data instead of delegating through
        # attribute lookup on the wrapped PyVista dataset. That path can resolve to
        # the active scalar/vector rather than the requested named array.
        data = torch.as_tensor(np.asarray(block.point_data[point_data_name]))
        nx, ny, nz = block.dimensions
        ncomp = data.shape[1] if data.ndim == 2 else 1
        has_comp_dim = ncomp > 1 or keep_dims  # determine if there's a dimension for the component of the point data
        # Reshape flat to VTK order (z, y, x) or (z, y, x, ncomp); then permute to (x, y, z[, ncomp])
        if nz > 1:
            shape = (nz, ny, nx, ncomp) if has_comp_dim else (nz, ny, nx)
            perm = (2, 1, 0, 3) if has_comp_dim else (2, 1, 0)
        else:
            shape = (ny, nx, ncomp) if has_comp_dim else (ny, nx)
            perm = (1, 0, 2) if has_comp_dim else (1, 0)
        return data.reshape(*shape).permute(*perm)
    
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
    
    def _looks_like_grid(self, value):
        """Check if value is given in a grid format of shape (N, M[, ncomp]) 
        where N and M are the number of points in the x and y directions, 
        and ncomp is the number of components."""
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and value.ndim <= 3:
                # Check if N and M correspond to the number of points in the x and y directions
                raise NotImplementedError("Not implemented yet.")
        if isinstance(value, np.ndarray):
            return value.ndim >= 2 and value.shape[-1] in (2, 3)
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
        if self._looks_like_grid(value):
            
            super().__setitem__(key, value)
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
            blk = super().__getitem__(key)
            # Expose leaf datasets through a one-block Pipeset wrapper so callers
            # get dotted/scalar helpers and methods like .scale(), .get_as_grid().
            if isinstance(blk, (pv.ImageData, pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid)):
                wrapped = Pipeset()
                wrapped.append(blk)
                return wrapped
            return blk
        
        if '.' not in key:
            # Plain key fallback:
            # 1) try direct block lookup first (PyVista behavior)
            # 2) if not found, resolve as a unique leaf array name across nested blocks
            try:
                return super().__getitem__(key)
            except KeyError:
                pass

            def iter_leaf_blocks(container, prefix=""):
                """Yield (full_block_name, leaf_dataset) recursively."""
                for child_name in container.keys():
                    child = super(Pipeset, container).__getitem__(child_name)
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    if isinstance(child, pv.MultiBlock):
                        yield from iter_leaf_blocks(child, full_name)
                    elif isinstance(child, (pv.ImageData, pv.PolyData, pv.UnstructuredGrid, pv.StructuredGrid)):
                        yield full_name, child

            matches = []
            for block_name, block in iter_leaf_blocks(self):
                if key in block.point_data:
                    matches.append((block_name, block))

            if len(matches) == 1:
                _, block = matches[0]
                return torch.as_tensor(block.point_data[key])
            if len(matches) > 1:
                match_blocks = ", ".join(name for name, _ in matches)
                raise KeyError(
                    f"Array name '{key}' is ambiguous. Found in blocks: {match_blocks}. "
                    f"Use a full key like 'block.{key}'."
                )
            raise KeyError(f"Block name ({key}) not found, and no leaf array named '{key}' was found.")
        
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
                return torch.as_tensor(block.point_data[scalar_name])
            else:
                raise KeyError(f"'{scalar_name}' not found in block '{flat_block_name}'")
        elif nested_exists:
            # Recurse into the MultiBlock
            first_block = super().__getitem__(parts[0])
            return first_block['.'.join(parts[1:])]
        else:
            raise KeyError(f"No block '{flat_block_name}' or MultiBlock '{parts[0]}' found for key '{key}'")

    def select(self, *, axis=None, value=None, tol=None, x=None, y=None, z=None):
        """
        Select points near one coordinate plane and return them as a new Pipeset.

        Supported call styles:
            pipe.select(z=0)
            pipe.select(axis='z', value=0)
            pipe.select(axis=2, value=0, tol=1e-9)

        Notes:
        - tol is an absolute tolerance in coordinate units.
        - If tol is not given, a small data-driven tolerance is chosen automatically.
        """
        shorthand = {"x": x, "y": y, "z": z}
        shorthand_used = [name for name, val in shorthand.items() if val is not None]

        if len(shorthand_used) > 1:
            raise ValueError("Use only one shorthand axis at a time: x=..., y=..., or z=....")

        if shorthand_used:
            if axis is not None or value is not None:
                raise ValueError("Use either shorthand (x/y/z) or explicit (axis, value), not both.")
            axis = shorthand_used[0]
            value = shorthand[axis]
        else:
            if axis is None or value is None:
                raise ValueError("Pass either z=<value> (or x/y) or pass both axis=<x|y|z> and value=<number>.")

        axis_map = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}
        if axis not in axis_map:
            raise ValueError("axis must be one of: 'x', 'y', 'z', 0, 1, 2.")
        axis_idx = axis_map[axis]
        value = float(value)

        if tol is not None:
            tol = float(tol)
            if tol < 0:
                raise ValueError("tol must be non-negative.")

        out = Pipeset()
        for path, _, block in self._iter_leaf_blocks():
            pts = np.asarray(block.points)
            if pts.size == 0:
                continue

            # Auto tolerance: half of the smallest non-zero spacing along selected axis.
            # If spacing cannot be inferred, use a tiny fallback.
            local_tol = tol
            if local_tol is None:
                axis_vals = np.unique(pts[:, axis_idx])
                if axis_vals.size > 1:
                    diffs = np.diff(np.sort(axis_vals))
                    positive_diffs = diffs[diffs > 0]
                    if positive_diffs.size > 0:
                        local_tol = 0.5 * positive_diffs.min()
                    else:
                        local_tol = 1e-12
                else:
                    local_tol = 1e-12

            mask = np.isclose(pts[:, axis_idx], value, atol=local_tol, rtol=0.0)
            if not np.any(mask):
                continue

            selected = block.extract_points(mask)
            out[path] = selected

        if len(out) == 0:
            axis_name = {0: "x", 1: "y", 2: "z"}[axis_idx]
            raise ValueError(
                f"No points found for {axis_name}={value} within tol={tol if tol is not None else 'auto'}."
            )
        return out
    
    def add_region(self, region, inp, name=None):
        """Create a sub-block from points in parent that fall within region."""
        if isinstance(inp, str):
            pts = region.select(self[inp])[0]
            self[inp + "." + name] = pts
        elif isinstance(inp, tuple):
            for i in inp:
                self[i + "." + name] = region.select(self[i])[0]

    
    def plot(self, scalar, ax=None, name=None, clim=None, sync=True, 
             colorbar=True, symmetric=True, norm_type=None,
             cbar_width=0.05, cbar_pad=0.02, wspace=None,
             method='auto', labels=None, **kwargs):
        """
        Plot a scalar from a block. scalar is 'block.scalar_name'.
        
        name: store this plot under a name for later access via pipe.plots['name']
        clim: (vmin, vmax) to set color limits
        sync: if True, sync color limits with the plot with the same name, if False, do not sync, 
            if a string, it is the name of the plot to sync with,
        colorbar, cbar: if True, add a colorbar to the axis
        cbar_width: width of the colorbar as a fraction of the plot width
        cbar_pad: padding between the colorbar and the plot
        wspace: horizontal space between subplots
        symmetric: if True, symmetrize the colorbar so that 0 is in the middle of the colormap and lower and upper data limits
            have the same absolute values.
        norm_type: type of the normalization: per map ('map'), common to all ('all'), or by row/column with grouping,
            e.g. 'AAB' for 1st and 2nd maps to have the same norm, 3rd map to have its own norm. Default: 'map'
            Allows to have common normalization for different maps, e.g. when they both show similar fields to compare.
            
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
        # Update kwargs with default values
        kwargs.setdefault('cmap', 'bwr')
        
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
                    # Give every subplot a stable name so grouped norm_type sync works
                    # even when the caller only passed raw scalar paths like 'Block-00.B_NV'.
                    row_name = sub_labels[0] if sub_labels else (f"{name}_{row_i}" if name else s)
                    m = self.plot(s, ax=flat_axs[ax_idx], name=row_name, clim=clim, sync=sync,
                                 colorbar=colorbar, symmetric=symmetric, method=method, **kwargs)
                    if sub_labels:
                        title = f'${sub_labels[0]}$' if '$' not in sub_labels[0] else sub_labels[0]
                        flat_axs[ax_idx].set_title(title)
                    all_mappables.append(m)
                    if row_name:
                        all_plot_names.append(row_name)
                    ax_idx += 1
                else:
                    row_axes = [flat_axs[ax_idx + j] for j in range(n_comp)]
                    row_name = f"{name}_{row_i}" if name else s
                    m = self.plot(s, ax=row_axes, name=row_name, clim=clim, sync=sync,
                                 colorbar=colorbar, symmetric=symmetric, method=method, labels=sub_labels, **kwargs)
                    all_mappables.append(m)
                    # Mirror the inner auto-naming so top-level norm_type can address
                    # component plots without the caller having to pass labels manually.
                    if sub_labels:
                        all_plot_names.extend(sub_labels)
                    else:
                        suffixes = ['x', 'y', 'z', 'w', 'u', 'v'][:n_comp] if n_comp <= 6 else [f'c{i}' for i in range(n_comp)]
                        all_plot_names.extend([f"{row_name}_{suffix}" for suffix in suffixes])
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
        if len(parts) == 1:
            block_name = 0
            scalar_name = scalar
        elif len(parts) == 2:
            block_name, scalar_name = parts[0], parts[1]
        else:
            block_name, rest = parts[0], parts[1]
            return self[block_name].plot(rest, ax=ax, name=name, 
                                         clim=clim, sync=sync,
                                         colorbar=colorbar, symmetric=symmetric, 
                                         method=method, **kwargs)
        
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
        
        # Map string group labels like 'AABC' to integer ids so we can store
        # per-group metadata and limits in a compact dict.
        group_ids = {}
        plot_group_ids = []
        norm_groups = {}
        for plot_name, group_label in zip(names, norm_type):
            if group_label not in group_ids:
                group_id = len(group_ids)
                group_ids[group_label] = group_id
                norm_groups[group_id] = {
                    'name': group_label,
                    'lims': [0.0, 0.0] if symmetric else [np.inf, -np.inf],
                    'plot_names': [],
                }
            group_id = group_ids[group_label]
            norm_groups[group_id]['plot_names'].append(plot_name)
            plot_group_ids.append(group_id)
        
        # Compute clim per group from the actual plotted arrays, not any stored
        # clim, so repeated sync operations always reflect current data.
        for plot_name, group_id in zip(names, plot_group_ids):
            info = self._plots[plot_name]
            arr = np.ma.asarray(info['sc'].get_array())
            arr = arr.compressed() if np.ma.isMaskedArray(arr) else np.asarray(arr).ravel()
            
            if symmetric:
                bound = float(np.abs(arr).max())
                current_bound = max(abs(norm_groups[group_id]['lims'][0]),
                                    abs(norm_groups[group_id]['lims'][1]),
                                    bound)
                norm_groups[group_id]['lims'] = [-current_bound, current_bound]
            else:
                norm_groups[group_id]['lims'][0] = min(norm_groups[group_id]['lims'][0], float(arr.min()))
                norm_groups[group_id]['lims'][1] = max(norm_groups[group_id]['lims'][1], float(arr.max()))
        
        group_clims = {
            group['name']: tuple(group['lims'])
            for group in norm_groups.values()
        }
        
        # Apply clim to each plot
        for name, group_id in zip(names, plot_group_ids):
            info = self._plots[name]
            vmin, vmax = norm_groups[group_id]['lims']
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
    
    def to_image_data(self, block_name=None, tol=1e-5):
        """
        Convert a PolyData block to ImageData if points form a regular rectangular grid.
        
        Checks: uniform spacing in x and y (within tol * range), and n_x * n_y == n_points.
        If valid, replaces the block with ImageData and copies all scalars.
        """
        block = self.resolve_name(block_name)
        if not hasattr(block, "points"):
            raise TypeError(f"to_image_data expects a leaf point-set block, got {type(block).__name__}")
        pts = np.asarray(block.points)
        
        x_unique = np.unique(pts[:, 0])
        y_unique = np.unique(pts[:, 1])
        z_unique = np.unique(pts[:, 2])
        n_x, n_y, n_z = len(x_unique), len(y_unique), len(z_unique)
        
        # Check completeness: grid should have exactly n_x * n_y points
        if n_x * n_y * n_z != len(pts):
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
            
        # Check uniform spacing in z
        if n_z > 1:
            dz = np.diff(z_unique)
            z_range = z_unique[-1] - z_unique[0]
            if z_range > 0 and np.max(np.abs(dz - dz[0])) > tol * z_range:
                raise ValueError(f"Non-uniform z spacing: max deviation {np.max(np.abs(dz - dz[0])):.2e}")
            spacing_z = dz[0] if len(dz) > 0 else 1.0
        else:
            spacing_z = 1.0
        
        # Create ImageData. Origin is the min corner, dimensions are n_x, n_y, 1
        z_val = pts[0, 2] if pts.shape[1] > 2 else 0.0
        origin = (x_unique[0], y_unique[0], z_val)
        
        img = pv.ImageData(dimensions=(n_x, n_y, n_z), spacing=(spacing_x, spacing_y, spacing_z), origin=origin)
        
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
        
        # Replace block. If block_name is omitted, use single-block replacement.
        if block_name is None:
            super().__setitem__(0, img)
        else:
            super().__setitem__(block_name, img)
        return self


# Backwards compatibility aliases
Dataset = Pipeset
MagneticFieldImageData = Dataset  # old name for the ImageData-based class
MagneticFieldUnstructuredGrid = UnstructuredData
