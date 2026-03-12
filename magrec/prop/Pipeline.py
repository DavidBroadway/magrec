"""
Pipeline system for modular magnetic field reconstruction.

Data-oriented architecture with immutable dict flow and mutable Dataset reference.
Implements the `_last` convention for step chaining and `out=` keyword for explicit naming.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

from magrec.misc.data import Dataset
from magrec.prop.Propagator import MagneticDipolePropagator, AxisProjectionPropagator

import numpy as np
import torch
import torch.nn as nn

# Optional imports for visualization
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn("PyVista not found. Visualization steps will not work.")

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Step:
    """Base class for pipeline steps.
    
    Steps transform data by:
    1. Extracting inputs using `in` if set
    2. Computing results
    3. Appending to datadict: {**datadict, specific_keys..., '_last': result}
    4. Optionally storing result in `out` key
    
    Parameters
    ----------
    in : str or tuple, optional
        Input key(s) from datadict. If None, uses '_last' or step-specific defaults.
        Note: Stored internally as `in_` since `in` is a Python keyword.
    out : str or tuple, optional
        Output key(s) to store result. If None, only updates '_last'.
    """
    
    def __init__(self, **kwargs):
        # Accept 'in' as keyword but store as 'in_' since 'in' is reserved
        self.in_ = kwargs.get('in', None)
        self.out = kwargs.get('out', None)
        self._step_name = None  # Set by Pipeline
        
    def run(self, datadict: Dict[str, Any], dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Execute the step.
        
        Parameters
        ----------
        datadict : dict
            Current pipeline data dictionary
        dataset : Dataset, optional
            Mutable dataset reference
            
        Returns
        -------
        dict
            Updated datadict with new keys and '_last' set
        """
        # Default: pass through
        return datadict
    
    # Alias for backwards compatibility
    def transform(self, datadict: Dict[str, Any], dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Alias for run()."""
        return self.run(datadict, dataset)
    
    def fit(self, datadict: Dict[str, Any] = None, dataset: Optional[Dataset] = None):
        """Prepare step (optional, for steps that need initialization)."""
        return self
    
    def _get_input(self, datadict: Dict[str, Any]) -> Any:
        """Extract input from datadict based on self.in_."""
        if self.in_ is None:
            return None  # No input
        elif isinstance(self.in_, str):
            return datadict.get(self.in_)
        elif isinstance(self.in_, (tuple, list)):
            return tuple(datadict.get(k) for k in self.in_)
        else:
            raise ValueError(f"Invalid in_ type: {type(self.in_)}")
    
    def _set_output(self, datadict: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Set output in datadict based on self.out."""
        new_dict = {**datadict, '_last': result}
        
        if self.out is not None:
            if isinstance(self.out, str):
                new_dict[self.out] = result
            elif isinstance(self.out, (tuple, list)):
                # Unpack result into multiple keys
                if not isinstance(result, (tuple, list)):
                    raise ValueError(f"Cannot unpack {type(result)} into {len(self.out)} keys")
                if len(result) != len(self.out):
                    raise ValueError(f"Result length {len(result)} != output keys {len(self.out)}")
                for key, val in zip(self.out, result):
                    new_dict[key] = val
        
        return new_dict


class ValueStep(Step):
    """Wrapper that turns any value into a step that outputs itself.
    
    This allows Pipeline to accept raw values like:
    - Dataset objects
    - Dicts
    - Numbers
    - Any other Python object
    """
    
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Return the wrapped value."""
        return self._set_output(datadict, self.value)


class Pipeline:
    """Modular pipeline for data processing and optimization.
    
    Passes immutable dictionaries between steps while maintaining a mutable
    Dataset reference. Supports interactive dipole placement, optimization,
    and artifact tracking.
    
    Parameters
    ----------
    steps : list
        List of steps. Can be:
        - Step instances (auto-named as step_0, step_1, ...)
        - (name, Step) tuples
        - Any value (Dataset, dict, int, etc.) - wrapped in ValueStep
        - Dataset objects are automatically set as pipeline._dataset
        Tuple syntax ('name', Step(...)) sets step.out='name'
        
    Examples
    --------
    >>> pipe = Pipeline([
    ...     dataset,  # Dataset object directly
    ...     Show("B_NV"),
    ...     DipoleLocator(field_name="B_NV", n_dipoles=10),
    ... ])
    >>> pipe.run()
    
    >>> pipe = Pipeline([
    ...     5,  # Just a number
    ...     ('double', Function(lambda x: x * 2)),
    ... ])
    >>> result = pipe.run()  # result['double'] == 10
    """
    
    def __init__(self, steps: List[Union[Step, Tuple[str, Step], Any]]):
        self._steps = OrderedDict()
        self._dataset = None
        self._log = []  # (step_name, step_ref, output_keys, location)
        self._datadict = {}
        
        # Parse and add steps
        step_counter = 0
        for item in steps:
            if isinstance(item, tuple) and len(item) == 2:
                name, step = item
                # Wrap non-Step objects
                if not isinstance(step, Step):
                    step = ValueStep(step)
                # Tuple syntax: set out= on the step
                if not hasattr(step, 'out') or step.out is None:
                    step.out = name
                self._steps[name] = step
                step._step_name = name
            elif isinstance(item, Step):
                name = f"step_{step_counter}"
                step_counter += 1
                self._steps[name] = item
                item._step_name = name
            else:
                # Wrap any other value as a step
                name = f"step_{step_counter}"
                step_counter += 1
                wrapped = ValueStep(item)
                wrapped._step_name = name
                self._steps[name] = wrapped
    
    def add_step(self, step: Step, name: Optional[str] = None):
        """Add a step to the pipeline.
        
        Parameters
        ----------
        step : Step
            Step to add
        name : str, optional
            Name for the step. If None, auto-generated.
        """
        if name is None:
            name = f"step_{len(self._steps)}"
        
        self._steps[name] = step
        step._step_name = name
        return self
    
    def run(self, X=None) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Parameters
        ----------
        X : any, optional
            Initial input (typically not used, data comes from Dataset step)
            
        Returns
        -------
        dict
            Final datadict after all steps
        """
        datadict = {}
        
        for name, step in self._steps.items():
            # Resolve string references in step parameters
            resolved_step = self._resolve_string_refs(step, datadict)
            
            # Run the step
            old_keys = set(datadict.keys())
            datadict = step.run(datadict, self._dataset)
            new_keys = set(datadict.keys()) - old_keys
            
            # Check if this step produced a Dataset and set it as pipeline dataset
            if self._dataset is None and '_last' in datadict:
                last_value = datadict['_last']
                if hasattr(last_value, 'points') and hasattr(last_value, 'point_data'):
                    self._dataset = last_value
            
            # If this step produced dipole locations (3D points), add to dataset
            if self._dataset is not None and step.out is not None:
                out_key = step.out if isinstance(step.out, str) else None
                if out_key and out_key in datadict:
                    data = datadict[out_key]
                    # Check if it's a numpy array of 3D points
                    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
                        # Add to dataset
                        if hasattr(self._dataset, 'add_dipole_locations'):
                            self._dataset.add_dipole_locations(data, name=out_key)
            
            # Log what was created
            self._log.append((name, step, list(new_keys), 'dict'))
        
        self._datadict = datadict
        return datadict
    
    # Alias
    transform = run
    
    def fit(self, X=None):
        """Prepare and execute pipeline with validation.
        
        1. Track optimizable parameters
        2. Validate tensor shapes
        3. Dry run (if applicable)
        4. Execute pipeline
        
        Parameters
        ----------
        X : any, optional
            Initial input
            
        Returns
        -------
        Pipeline
            Self for chaining
        """
        # Collect optimizable parameters
        self._collect_optimizable_params()
        
        # TODO: Add validation and dry run logic
        
        # Run pipeline
        self.run(X)
        
        return self
    
    def _resolve_string_refs(self, step: Step, datadict: Dict[str, Any]) -> Step:
        """Resolve string references in step parameters.
        
        For example, if a step has Dipoles(pts="pts"), this resolves "pts"
        to the actual value from datadict['pts'].
        
        Special attributes like 'in_' and 'out' are NOT resolved - they should
        remain as key names.
        
        Parameters
        ----------
        step : Step
            Step to process
        datadict : dict
            Current datadict
            
        Returns
        -------
        Step
            Step with resolved references (modifies in place, returns for convenience)
        """
        # Attributes to skip (should remain as key names, not be resolved)
        skip_attrs = {'in_', 'out', 'func', '_step_name', '_propagator'}
        
        # Iterate through step attributes
        for attr_name in dir(step):
            if attr_name.startswith('_') or attr_name in skip_attrs:
                continue
            
            try:
                attr_value = getattr(step, attr_name)
                
                # If attribute is a string and exists as a key in datadict, resolve it
                if isinstance(attr_value, str) and attr_value in datadict:
                    setattr(step, attr_name, datadict[attr_value])
            except AttributeError:
                continue
        
        return step
    
    def _collect_optimizable_params(self) -> List[nn.Parameter]:
        """Collect all optimizable parameters from the pipeline.
        
        Scans all steps for Optimizable wrappers and collects their parameters.
        
        Returns
        -------
        list
            List of torch.nn.Parameter objects
        """
        params = []
        for name, step in self._steps.items():
            if isinstance(step, Optimizable):
                params.append(step.param)
        return params
    
    def __getitem__(self, key: Union[str, int]) -> Step:
        """Access steps by name or index."""
        if isinstance(key, int):
            return list(self._steps.values())[key]
        else:
            return self._steps[key]
    
    def __repr__(self):
        steps_str = '\n  '.join(f"{name}: {step.__class__.__name__}" 
                                for name, step in self._steps.items())
        return f"Pipeline(\n  {steps_str}\n)"


# ============================================================================
# Core Steps
# ============================================================================


class DatasetStep(Step):
    """Load or create a Dataset from a dictionary.
    
    Wraps Dataset.from_dict() and sets the pipeline's _dataset reference.
    
    Parameters
    ----------
    datadict : dict, optional
        Data dictionary to load
    units : dict, optional
        Unit specifications for fields
    rename_map : dict or list, optional
        Renaming specifications
    **kwargs
        Additional arguments for Dataset.from_dict()
    """
    
    def __init__(self, datadict=None, units=None, rename_map=None, **kwargs):
        super().__init__(**kwargs)
        self.datadict = datadict
        self.units = units
        self.rename_map = rename_map
        self.extra_kwargs = {k: v for k, v in kwargs.items() if k not in ['in', 'out']}
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Create Dataset and set pipeline reference."""
        
        # Create Dataset
        if self.datadict is not None:
            ds = Dataset.from_dict(self.datadict, rename_map=self.rename_map)
        else:
            ds = Dataset()
        
        # Store as pipeline's dataset (will be set by Pipeline)
        # This is a bit of a hack - we need the pipeline reference
        # For now, just create and return it
        
        result = ds
        return self._set_output(datadict, result)


class Show(Step):
    """Visualize data using PyVista, Vedo, or matplotlib.
    
    Parameters
    ----------
    field_name : str, optional
        Name of field to visualize. If None, visualizes '_last'
    backend : str
        Visualization backend ('pyvista', 'vedo', 'matplotlib')
    component : int, optional
        Component index for vector fields
    plot_type : str
        Type of plot: 'mesh', 'points', 'field'
    **kwargs
        Additional plotting arguments
    """
    
    def __init__(self, field_name=None, backend='pyvista', component=None, 
                 plot_type='mesh', **kwargs):
        super().__init__(**kwargs)
        self.field_name = field_name
        self.backend = backend
        self.component = component
        self.plot_type = plot_type
        self.plot_kwargs = {k: v for k, v in kwargs.items() if k not in ['in', 'out']}
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Create visualization."""
        if self.field_name is not None:
            if self.field_name in datadict:
                data = datadict[self.field_name]
            elif dataset is not None and self.field_name in dataset.point_data:
                data = dataset.point_data[self.field_name]
            else:
                raise KeyError(f"Field '{self.field_name}' not found")
        else:
            data = self._get_input(datadict)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
            plotter = self._visualize_points(data, dataset)
        elif dataset is not None:
            plotter = self._visualize_dataset(dataset, self.field_name)
        elif isinstance(data, np.ndarray) and data.ndim in [2, 3]:
            # 2D or 3D array - use plot_n_components
            plotter = self._visualize_array(data)
        else:
            warnings.warn(f"Don't know how to visualize {type(data)}")
            return self._set_output(datadict, None)
        
        return self._set_output(datadict, plotter)
    
    def _visualize_dataset(self, dataset, field_name):
        """Visualize a Dataset object."""
        if not HAS_PYVISTA:
            warnings.warn("PyVista not available")
            return None
        
        if self.backend == 'pyvista':
            plotter = pv.Plotter(notebook=True)
            
            # Add mesh with scalar field
            if field_name and field_name in dataset.point_data:
                scalar_data = dataset.point_data[field_name]
                # Handle vector fields
                if scalar_data.ndim > 1 and self.component is not None:
                    scalar_data = scalar_data[:, self.component]
                elif scalar_data.ndim > 1:
                    # Use magnitude
                    scalar_data = np.linalg.norm(scalar_data, axis=1)
                
                plotter.add_mesh(dataset, scalars=scalar_data, **self.plot_kwargs)
            else:
                plotter.add_mesh(dataset, **self.plot_kwargs)
            
            # Show inline if in notebook
            try:
                plotter.show(jupyter_backend='static', return_viewer=True)
            except:
                plotter.show()
            
            return plotter
        
        elif self.backend == 'vedo':
            warnings.warn("Vedo backend not yet implemented")
            return None
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _visualize_points(self, points, dataset=None):
        """Visualize a point cloud."""
        if not HAS_PYVISTA:
            warnings.warn("PyVista not available")
            return None
        
        if self.backend == 'pyvista':
            plotter = pv.Plotter(notebook=True)
            
            # Create point cloud
            point_cloud = pv.PolyData(points)
            
            # Add to plotter
            plotter.add_mesh(point_cloud, color='red', point_size=10, 
                           render_points_as_spheres=True, **self.plot_kwargs)
            
            # If dataset exists, add it as background
            if dataset is not None:
                plotter.add_mesh(dataset, opacity=0.3, color='lightgray')
            
            # Show
            try:
                plotter.show(jupyter_backend='static', return_viewer=True)
            except:
                plotter.show()
            
            return plotter
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _visualize_array(self, data):
        """Visualize 2D/3D array using plot_n_components."""
        from magrec.plot.plot import plot_n_components
        
        # Use existing plot_n_components
        fig = plot_n_components(data)
        
        return fig


class Dipoles(Step):
    """Represent magnetic dipoles.
    
    Parameters
    ----------
    pts : array-like or str
        Dipole positions. If str, resolved from datadict.
    directions : list
        Dipole moment directions, e.g. ['z'] for z-oriented
    moments : array-like, optional
        Dipole moment magnitudes
    """
    
    def __init__(self, pts, directions=['z'], moments=None, **kwargs):
        super().__init__(**kwargs)
        self.pts = pts
        self.directions = directions
        self.moments = moments
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Create dipole representation."""
        # Get points (may be string reference)
        pts = self.pts
        if isinstance(pts, str) and pts in datadict:
            pts = datadict[pts]
        
        # Convert to tensor if needed
        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts, dtype=torch.float32)
        
        # Create moments if not provided
        if self.moments is None:
            n_dipoles = pts.shape[0]
            moments = torch.ones(n_dipoles, dtype=torch.float32)
        else:
            moments = self.moments if isinstance(self.moments, torch.Tensor) else torch.tensor(self.moments)
        
        # Store in datadict
        dipole_dict = {
            'dipole_pts': pts,
            'dipole_dirs': self.directions,
            'dipole_moments': moments
        }
        
        # Append to datadict
        new_datadict = {**datadict, **dipole_dict}
        return self._set_output(new_datadict, dipole_dict)


class DipoleLocator(Step):
    """Interactive dipole placement tool.
    
    Uses PyVista or matplotlib to allow user to draw polygons and place
    dipoles via Delaunay triangulation.
    
    Parameters
    ----------
    field_name : str
        Field to visualize for placement (default: 'B')
    n_dipoles : int
        Number of dipoles to place
    z_offset : float
        Height offset for dipoles (default: 0)
    component : int, optional
        Component to show (default: 2 for z-component)
    interactive : bool
        If True, show interactive widget. If False, place uniformly (default: True)
    method : str
        Placement method ('polygon' for interactive polygon drawing)
    """
    
    def __init__(self, field_name='B', n_dipoles=10, z_offset=0.0, 
                 component=2, interactive=True, method='polygon', **kwargs):
        super().__init__(**kwargs)
        self.field_name = field_name
        self.n_dipoles = n_dipoles
        self.z_offset = z_offset
        self.component = component
        self.interactive = interactive
        self.method = method
        self.placed_pts = None  # Store for inspection
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Launch interactive placement and return dipole positions."""
        if dataset is None:
            raise ValueError("DipoleLocator requires a Dataset")
        
        if not self.interactive:
            # Place dipoles uniformly
            pts = self._uniform_placement(dataset)
        else:
            # Interactive placement
            pts = self._interactive_placement(dataset)
        
        self.placed_pts = pts
        return self._set_output(datadict, pts)
    
    def _uniform_placement(self, dataset):
        """Place dipoles uniformly over the dataset bounds."""
        bounds = dataset.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        x = np.linspace(bounds[0], bounds[1], int(np.sqrt(self.n_dipoles)))
        y = np.linspace(bounds[2], bounds[3], int(np.sqrt(self.n_dipoles)))
        xx, yy = np.meshgrid(x, y)
        pts = np.stack([xx.ravel(), yy.ravel(), 
                       np.full(xx.ravel().shape, self.z_offset)], axis=1)
        return pts[:self.n_dipoles]
    
    def _interactive_placement(self, dataset):
        """Interactive placement using matplotlib polygon selector."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector
        from matplotlib.path import Path
        
        # Get field data for visualization
        if self.field_name not in dataset.point_data:
            raise ValueError(f"Field '{self.field_name}' not found in dataset")
        
        field_data = dataset.point_data[self.field_name]
        
        # Extract component
        if field_data.ndim > 1:
            field_data = field_data[:, self.component]
        
        # Get coordinates
        points = dataset.points
        x = points[:, 0]
        y = points[:, 1]
        
        # Create 2D grid for imshow
        # Assume structured grid
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        
        if len(x_unique) * len(y_unique) == len(x):
            # Structured grid
            nx, ny = len(x_unique), len(y_unique)
            field_2d = field_data.reshape(ny, nx)
            extent = [x_unique.min(), x_unique.max(), 
                     y_unique.min(), y_unique.max()]
        else:
            # Unstructured - use griddata
            from scipy.interpolate import griddata
            nx, ny = 100, 100
            xi = np.linspace(x.min(), x.max(), nx)
            yi = np.linspace(y.min(), y.max(), ny)
            xx, yy = np.meshgrid(xi, yi)
            field_2d = griddata((x, y), field_data, (xx, yy), method='linear')
            extent = [x.min(), x.max(), y.min(), y.max()]
        
        # Create interactive figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(field_2d, origin='lower', extent=extent, 
                      cmap='RdBu_r', aspect='auto')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Draw polygon to place {self.n_dipoles} dipoles (close polygon when done)')
        plt.colorbar(im, ax=ax, label=f'{self.field_name}[{self.component}]')
        
        # Store polygon vertices
        vertices = []
        
        def onselect(verts):
            """Called when polygon is complete."""
            vertices.clear()
            vertices.extend(verts)
            # Draw the polygon
            ax.plot([v[0] for v in verts] + [verts[0][0]], 
                   [v[1] for v in verts] + [verts[0][1]], 
                   'r-', linewidth=2)
            fig.canvas.draw_idle()
        
        # Create polygon selector
        selector = PolygonSelector(ax, onselect, 
                                  props=dict(color='r', linestyle='-', 
                                           linewidth=2, alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # After polygon is drawn, generate points using Delaunay
        if len(vertices) < 3:
            print("No valid polygon drawn, using uniform placement")
            return self._uniform_placement(dataset)
        
        # Generate points within polygon
        pts = self._generate_points_in_polygon(vertices, self.n_dipoles, self.z_offset)
        
        return pts
    
    def _generate_points_in_polygon(self, vertices, n_points, z_offset):
        """Generate points within polygon using Delaunay triangulation.
        
        Parameters
        ----------
        vertices : list of tuples
            Polygon vertices [(x1, y1), (x2, y2), ...]
        n_points : int
            Number of points to generate
        z_offset : float
            Z coordinate for points
        
        Returns
        -------
        points : ndarray, shape (n_points, 3)
            Generated 3D points
        """
        from scipy.spatial import Delaunay
        from matplotlib.path import Path
        
        # Convert vertices to numpy array
        vertices = np.array(vertices)
        
        # Get bounding box
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        
        # Create path for point-in-polygon test
        path = Path(vertices)
        
        # Generate candidate points using random sampling
        # Use oversampling to ensure we get enough points
        n_candidates = n_points * 10
        x_cand = np.random.uniform(x_min, x_max, n_candidates)
        y_cand = np.random.uniform(y_min, y_max, n_candidates)
        candidates = np.column_stack([x_cand, y_cand])
        
        # Filter points inside polygon
        inside = path.contains_points(candidates)
        points_inside = candidates[inside]
        
        # If not enough points, retry with more candidates
        max_retries = 5
        retry = 0
        while len(points_inside) < n_points and retry < max_retries:
            n_additional = (n_points - len(points_inside)) * 10
            x_cand = np.random.uniform(x_min, x_max, n_additional)
            y_cand = np.random.uniform(y_min, y_max, n_additional)
            candidates = np.column_stack([x_cand, y_cand])
            inside = path.contains_points(candidates)
            points_inside = np.vstack([points_inside, candidates[inside]])
            retry += 1
        
        # Subsample to desired number
        if len(points_inside) > n_points:
            # Use farthest point sampling for better distribution
            indices = self._farthest_point_sampling(points_inside, n_points)
            points_inside = points_inside[indices]
        
        # Add z coordinate
        points_3d = np.column_stack([points_inside, 
                                    np.full(len(points_inside), z_offset)])
        
        return points_3d
    
    def _farthest_point_sampling(self, points, n_samples):
        """Farthest point sampling for better point distribution.
        
        Parameters
        ----------
        points : ndarray, shape (n, 2)
            Input points
        n_samples : int
            Number of samples to select
        
        Returns
        -------
        indices : ndarray
            Indices of selected points
        """
        n = len(points)
        if n <= n_samples:
            return np.arange(n)
        
        # Start with random point
        indices = [np.random.randint(0, n)]
        distances = np.full(n, np.inf)
        
        for _ in range(n_samples - 1):
            # Update distances to closest selected point
            last_idx = indices[-1]
            dist_to_last = np.linalg.norm(points - points[last_idx], axis=1)
            distances = np.minimum(distances, dist_to_last)
            
            # Select farthest point
            next_idx = np.argmax(distances)
            indices.append(next_idx)
        
        return np.array(indices)


class Propagator(Step):
    """Propagate magnetic dipoles to field using Biot-Savart law.
    
    Wraps MagneticDipolePropagator.
    
    Parameters
    ----------
    height : float or str, optional
        Sensor height above dipoles. If str, resolved from datadict.
    in_ : str
        Input key for dipole moments (default: 'dipole_moments')
    """
    
    def __init__(self, height=None, **kwargs):
        # Default input is 'dipole_moments'
        if 'in' not in kwargs:
            kwargs['in'] = 'dipole_moments'
        super().__init__(**kwargs)
        self.height = height
        self._propagator = None
    
    def fit(self, datadict: Dict[str, Any] = None, dataset=None):
        """Create propagator from dipole locations and sensor positions."""
        
        if datadict is None or dataset is None:
            return self # TODO: raise error
        
        # Get dipole positions
        r_source = datadict.get('dipole_pts')
        if r_source is None:
            warnings.warn("No dipole_pts in datadict, cannot initialize propagator")
            return self
        
        # Get sensor positions from dataset
        r_sensor = dataset.points
        
        # Create propagator
        self._propagator = MagneticDipolePropagator(r_source, r_sensor, use_torch=True)
        
        return self
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Compute magnetic field from dipole moments."""
        # Get moments
        moments = self._get_input(datadict)
        
        if self._propagator is None:
            # Initialize on first run
            self.fit(datadict, dataset)
        
        # Compute field
        B_field = self._propagator(moments)
        
        # Store in both 'B' and _last
        new_datadict = {**datadict, 'B': B_field}
        return self._set_output(new_datadict, B_field)


class Projection(Step):
    """Project field onto NV axis direction.
    
    Parameters
    ----------
    theta : float or str
        Polar angle (degrees)
    phi : float or str
        Azimuthal angle (degrees)
    in_ : str
        Input field key
    """
    
    def __init__(self, theta, phi, **kwargs):
        # Default input is 'B'
        if 'in' not in kwargs:
            kwargs['in'] = 'B'
        super().__init__(**kwargs)
        self.theta = theta
        self.phi = phi
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Project field to NV axis."""
        
        # Get field
        B_field = self._get_input(datadict)
        
        # Resolve theta, phi if strings
        theta = self.theta if not isinstance(self.theta, str) else datadict[self.theta]
        phi = self.phi if not isinstance(self.phi, str) else datadict[self.phi]
        
        # Use AxisProjectionPropagator
        projector = AxisProjectionPropagator(theta=theta, phi=phi)
        B_projected = projector(B_field)
        
        return self._set_output(datadict, B_projected)


class Put(Step):
    """Move data from datadict to Dataset.
    
    Parameters
    ----------
    name : str or tuple
        Key(s) to store in dataset
    source_key : str
        Source key in datadict (default: '_last')
    data_type : str
        Where to store: 'point_data' or 'field_data'
    """
    
    def __init__(self, name, source_key='_last', data_type='point_data', **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.source_key = source_key
        self.data_type = data_type
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Move value from datadict to dataset."""
        if dataset is None:
            raise ValueError("Put step requires dataset to be set")
        
        # Get value
        value = datadict.get(self.source_key)
        
        # Store in dataset
        if isinstance(self.name, str):
            if self.data_type == 'point_data':
                dataset.point_data[self.name] = value
            elif self.data_type == 'field_data':
                dataset.field_data[self.name] = value
        elif isinstance(self.name, (tuple, list)):
            # Unpack multiple values
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"Cannot unpack {type(value)} into {len(self.name)} keys")
            for n, v in zip(self.name, value):
                if self.data_type == 'point_data':
                    dataset.point_data[n] = v
                elif self.data_type == 'field_data':
                    dataset.field_data[n] = v
        
        return self._set_output(datadict, value)


class Function(Step):
    """Apply arbitrary function to data.
    
    Parameters
    ----------
    func : callable
        Function to apply
    in_ : str or tuple
        Input key(s) from datadict
    """
    
    def __init__(self, func: Callable, **kwargs):
        # Default input is '_last' if not specified
        if 'in' not in kwargs:
            kwargs = {**kwargs, 'in': '_last'}
        super().__init__(**kwargs)
        self.func = func
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Apply function."""
        # Get inputs
        inputs = self._get_input(datadict)
        
        # Apply function
        if self.in_ is None:
            # No inputs - call function with no arguments
            result = self.func()
        elif isinstance(inputs, tuple):
            result = self.func(*inputs)
        else:
            result = self.func(inputs)
        
        return self._set_output(datadict, result)


class Optimizable(Step):
    """Wrapper to make step parameters optimizable.
    
    Parameters
    ----------
    wrapped_step : Step
        Step to wrap
    param_name : str
        Name of parameter to optimize
    init_value : tensor or array
        Initial value for parameter
    requires_grad : bool
        Whether parameter requires gradients
    """
    
    def __init__(self, wrapped_step: Step, param_name: str, init_value, requires_grad=True):
        # Inherit in/out from wrapped step  
        super().__init__(**{'in': wrapped_step.in_, 'out': wrapped_step.out})
        self.wrapped_step = wrapped_step
        self.param_name = param_name
        
        # Create parameter
        if not isinstance(init_value, torch.Tensor):
            init_value = torch.tensor(init_value, dtype=torch.float32)
        self.param = nn.Parameter(init_value, requires_grad=requires_grad)
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Run wrapped step with optimizable parameter."""
        # Inject parameter into wrapped step
        setattr(self.wrapped_step, self.param_name, self.param)
        
        # Run wrapped step
        return self.wrapped_step.run(datadict, dataset)
    
    def fit(self, datadict: Dict[str, Any] = None, dataset=None):
        """Fit wrapped step."""
        if hasattr(self.wrapped_step, 'fit'):
            self.wrapped_step.fit(datadict, dataset)
        return self


class Optimize(Step):
    """Optimization step to train parameters.
    
    Parameters
    ----------
    criterion_key : str
        Key in datadict containing loss value
    optimizer_cls : torch.optim.Optimizer
        Optimizer class
    lr : float
        Learning rate
    epochs : int
        Number of training epochs
    **optimizer_kwargs
        Additional optimizer arguments
    """
    
    def __init__(self, criterion_key='criterion', optimizer_cls=torch.optim.Adam, 
                 lr=0.01, epochs=100, **kwargs):
        # Extract optimizer kwargs (everything except 'in' and 'out')
        optimizer_kwargs = {k: v for k, v in kwargs.items() if k not in ['in', 'out']}
        super().__init__(**{k: v for k, v in kwargs.items() if k in ['in', 'out']})
        self.criterion_key = criterion_key
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.epochs = epochs
        self.optimizer_kwargs = optimizer_kwargs
        self._pipeline_ref = None  # Set by Pipeline
    
    def run(self, datadict: Dict[str, Any], dataset=None) -> Dict[str, Any]:
        """Run optimization loop."""
        # This would need access to the full pipeline to re-run forward passes
        # For now, just pass through
        # TODO: Implement full optimization loop
        warnings.warn("Optimize step not fully implemented yet")
        
        return self._set_output(datadict, datadict.get(self.criterion_key))
