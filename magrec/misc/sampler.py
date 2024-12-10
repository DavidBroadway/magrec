import torch
from typing import Tuple

# needs to implement sample_points

class NDGridPoints(object):
    
    @staticmethod
    def get_grid_pts(*n_points: int, origin: Tuple[float, ...] = None, diagonal: Tuple[float, ...] = None) -> torch.Tensor:
        """
        Generate points in an N-dimensional space, arranged in a regular n-dimensional grid within a rectangular parallelepiped, 
        represented as a PyTorch tensor. tensor's size will be determined by the product of the n_points integers 
        and the length of the origin (or `diagonal`, as `diagonal` and origin should be of the same length)::
        
        shape = (n_points[0] * n_points[1] * ... * n_points[N-1], N)
        
        The order of the returned points is such that the first dimension is the fastest changing and the last dimension
        is the slowest changing, i.e. pts[:, 0] will be the fastest changing (with period of n_points[0]), pts[:, 1] the 
        second fastest (with period n_points[0] * n_points[1]), etc.

        Args:
            n_points (int):                 A variable number of arguments, each specifying the number of points in each dimension. If keyword
                                            arguments for `origin` and `diagonal` are not provided, use the last two elements of `*n_points` as the
                                            origin and diagonal coordinates, respectively.
            origin (Tuple[float, ...]):     The origin coordinates of the rectangular parallelepiped. If None, assume that the last two 
                                            elements of `*n_points` are the origin and diagonal coordinates, respectively. This allows to 
                                            call the function without keyword arguments.
            diagonal (Tuple[float, ...]):   The end coordinates (diagonal opposite to the origin) of the rectangular parallelepiped. If None,
                                            attempt to extract the origin and diagonal coordinates from the last two elements of `*n_points`.

        Returns:
        torch.Tensor:   A tensor of points in the N-dimensional space. 
                        The shape of the tensor is (product_of_n_points, len(origin)).

        Example:
        >>> get_grid_pts(2, 2, origin=(0.0, 0.0), diagonal=(1.0, 1.0))
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        """
        
        if origin is None and diagonal is None:
            # If origin and diagonal are none, then the last two elements of n_points are the origin and diagonal, 
            # and the rest are the number of points in each dimension. Then len(n_points) - 2 is the number of dimensions,
            # and len(origin) and len(diagonal) should be len(n_points) - 2.
            if len(n_points) <= 2:
                raise ValueError("Not enough arguments to specify number of points, origin, and diagonal")
            else: 
                num_dims = len(n_points) - 2
                if len(origin) != num_dims:
                    raise ValueError("Number of dimensions specified by origin {} does not match number of dimensions \
                                     for which number of points is given.".format(origin, len(n_points-2)))
                if len(diagonal) != num_dims:
                    raise ValueError("Number of dimensions specified by diagonal {} does not match number of dimensions \
                                     for which number of points is given.".format(diagonal, len(n_points-2)))
            
            origin = n_points[-2]
            diagonal = n_points[-1]
            n_points = n_points[:-2]
        elif origin is not None and diagonal is not None:
            pass
        else:
            raise ValueError("Must provide either both origin and diagonal, or neither. But {}".format(
                "origin is None, and diagonal is not" if diagonal is not None else "diagonal is None, and origin is not"
            ))
        
        if len(origin) != len(diagonal):
            raise ValueError(("Origin and diagonal must have the same number of dimensions, " + \
                              "got origin {} of length {} and diagonal {} of length {}.").format(origin, len(origin), diagonal, len(diagonal)))
        # Check that number of points specified is the same as the number of dimensions
        if len(n_points) != len(origin):
            raise ValueError(("Number of points specified must be the same as the number of dimensions, " + \
                              "got {} point number specification and {} dimensions set by origin and diagonal" + \
                              "provided.").format(len(n_points), len(origin)))

        # Generate grid points
        grids = [torch.linspace(origin[dim], diagonal[dim], n_points[dim]) for dim in range(len(n_points))]
        # meshgrid returns the fastest changing dimension last in 'ij' indexing, but we want x, y, z, etc. 
        # to be the changing faster from left to right, in that order
        mesh = torch.meshgrid(*grids[::-1], indexing='ij')                       # run [::-1] in reverse order here and below 
        flattened_mesh = torch.stack([m.flatten() for m in mesh[::-1]], dim=-1)  # to get the fastest changing dimension first

        return flattened_mesh
    
    @staticmethod
    def get_random_pts(n_points: int, origin: Tuple[float, ...], diagonal: Tuple[float, ...]) -> torch.Tensor:
        num_dims = len(origin)
        if len(diagonal) != num_dims:
            raise ValueError("Origin and diagonal must have the same number of dimensions.")
        coords = [torch.empty(n_points).uniform_(origin[dim], diagonal[dim]) for dim in range(num_dims)]
        return torch.stack(coords, dim=1)
        

class Sampler(object):
    
    def __init__(self, sample_fn, batch_n_points, cached_n_points, regenerate_cache=False):
        """
        Parameters
        ----------
        sample_fn : callable
            A function which returns a set of points.
        batch_n_points : int
            The number of points to be sampled at once on sampler.sample_points().
        cached_n_points : int
            The number of points to be pregenerated and stored in memory.
        regenerate_cache : bool
            If True, the cache points are regenerated every time cache is exhausted.
        """
        self.sample_fn = sample_fn
        self.batch_n_points = batch_n_points
        self.cached_n_points = cached_n_points
        self.regenerate_cache = regenerate_cache
        
        self.cached_points = sample_fn(cached_n_points)
        self.points_generator = self.get_points_generator()
        
    def get_points_generator(self):
        # Split cached_points into batches of size batch_n_points
        batches = torch.split(self.cached_points, self.batch_n_points)

        # Infinite loop to support regenerate_cache functionality
        while True:
            for batch in batches:
                yield batch

            # Once all batches are exhausted:
            if self.regenerate_cache:
                # If regenerate_cache is True, regenerate the cached_points
                self.cached_points = self.sample_fn(self.cached_n_points)
                # Re-split the cached_points for the next iteration
                batches = torch.split(self.cached_points, self.batch_n_points)
            else:
                # If regenerate_cache is False, just reset the batches generator 
                # (i.e., start yielding from the beginning of cached_points)
                pass
        
    def sample_points(self, device):
        return next(self.points_generator).to(device)

    
class StaticSampler(Sampler):
    """A sampler which returns the same set of points every time."""
    
    def __init__(self, sample_fn):
        super().__init__(sample_fn=sample_fn, batch_n_points=None, cached_n_points=None, regenerate_cache=False)
    
    def sample_points(self, device):
        return self.sample_fn().to(device)
    
    
class RectangleSampler(Sampler):
    
    def __init__(self):
        super().__init__(sample_fn=self.sample_rectangular_region)
    
    @staticmethod
    def sample_rectangular_region(n_points, origin, diagonal):
        x1, y1 = origin  # Coordinates of the origin vertex
        x2, y2 = diagonal  # Coordinates of the opposite, diagonal vertex

        # Sample x and y coordinates independently
        x_coords = x1 + (x2 - x1) * torch.rand(n_points)
        y_coords = y1 + (y2 - y1) * torch.rand(n_points)

        # Combine x and y coordinates into a single tensor of shape (n_points, 2)
        sampled_points = torch.stack([x_coords, y_coords], dim=1)

        return sampled_points
    
    
class GridSampler(Sampler):
    
    @staticmethod
    def sample_grid(nx_points, ny_points, origin, diagonal, z=None):
        """
        Generate a regular rectangular grid in a rectangle as a list of points.
        
        Parameters:
            nx_points (int): Number of grid points in the x-direction, per unit rectangle.
            ny_points (int): Number of grid points in the y-direction, per unit rectangle.
            origin (Tuple[float, float]): Coordinates of the origin vertex of the rectangle.
            diagonal (Tuple[float, float]): Coordinates of the opposite, diagonal vertex of the rectangle.
            z (float): Optional, the z-coordinate of the points. If None, the points are 2D. Otherwise, the points are 3D that
                       lie in the plane z = z. Default is None. Note that with z != None, static method pts_to_grid will not work.

        Returns:
            torch.Tensor: A tensor containing the grid points, shape (ny_points * nx_points, 2).
        """
        x1, y1 = origin    # Coordinates of the origin vertex
        x2, y2 = diagonal  # Coordinates of the opposite, diagonal vertex
        
        # Generate one-dimensional linspaces for x and y
        x = torch.linspace(x1, x2, nx_points)
        y = torch.linspace(y1, y2, ny_points)

        # Create a grid using meshgrid
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Flatten and concatenate to get a list of coordinates
        flat_x = grid_x.reshape(-1, 1)
        flat_y = grid_y.reshape(-1, 1)
        grid_points = torch.cat((flat_x, flat_y), 1)
        
        if z is not None:
            # If z is not None, the points are 3D, and the z-coordinate is set to z
            grid_points = torch.cat((grid_points, z * torch.ones((nx_points * ny_points, 1))), 1)
        
        return grid_points
    
    @staticmethod
    def pts_to_grid(pts, nx_points, ny_points):
        """Given a tensor of batched points of shape (n_points, n), return a tensor of shape 
        (n, nx_points, ny_points) where the first dimension are the components of the grid values, 
        
        n is inferred automatically. 
        
        Parameters
            nx_points: number of points in the grid in x direction
            ny_points: number of points in the grid in y direction
        
        Returns:
            torch.Tensor, 
        """
        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts)   
    
        pts = pts.reshape(ny_points, nx_points, -1)
        pts = pts.permute(2, 1, 0)
        return pts
    
    @staticmethod
    def grid_to_pts(grid):
        """Given a tensor with a grid of points of shape (n, nx_points, ny_points), return a tensor
        of shape (nx_points * ny_points, n) where first index iterates through points in the original
        grid in the given indexing order to match torch.meshgrid(..., 'xy')
        """
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid)
            
        pts = grid.reshape(grid.shape[0], -1).permute(1, 0)
        return pts

