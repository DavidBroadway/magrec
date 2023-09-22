import torch

# needs to implement sample_points


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
    def sample_grid(nx_points, ny_points, origin, diagonal):
        """
        Generate a regular rectangular grid in a rectangle.
        
        Parameters:
            nx_points (int): Number of grid points in the x-direction, per unit rectangle.
            ny_points (int): Number of grid points in the y-direction, per unit rectangle.

        Returns:
            torch.Tensor: A tensor containing the grid points, shape (ny_points * nx_points, 2).
        """
        x1, y1 = origin  # Coordinates of the origin vertex
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
        pts = pts.reshape(ny_points, nx_points, -1)
        pts = pts.permute(2, 1, 0)
        return pts

