import torch
import numpy as np
from magrec.prop.constants import twopi
import pyvista as pv

import deepxde as dde


class GaussianFourierFeaturesTransform(torch.nn.Module):
    """
    An implementation a random Fourier mapping as described in the paper:
    Wang (2021) https://doi.org/10.1016/j.cma.2021.113938, eq. (3.9)

    Random Fourier features are sampled from a Gaussian distribution with mean 0 and
    standard deviation std = sigma^2, that is B ~ N(0, sigma^2).

    B can be thought of as a tensor of random Fourier wave-vectors. B[c, f] where c is
    a component corresponding to the x or y direction and f, for example, and f is the
    value of the Fourier wave-vector. This definition is consistent with the code in
    Fourier module.
    """

    def __init__(self, in_features: int, out_features: int, sigma: float) -> None:
        super().__init__()
        # Register tensor as buffers so that they are saved with state_dict of the model
        # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/19
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("B", twopi * torch.normal(mean=0, std=sigma**2, size=(in_features, out_features)))
        self.register_buffer("m", torch.tensor(out_features))
        self.short_name = "GauFF"
        # Below they are also accessible as object attributes: self.B, self.m, self.sigma

    def forward(self, x):
        x = torch.einsum("...c,cj->...j", x, self.B.to(x.device))
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) / torch.sqrt(self.m)
    
    
class UniformFourierFeaturesTransform(torch.nn.Module):
    
    def __init__(self, in_features: int, out_features: int, K: float) -> None:
        """Fourier features with uniform sampling in the ball of radius 
        K and `out_features` number of features."""
        super().__init__()
        self.register_buffer("K", torch.tensor(K))
        self.register_buffer("B", uniform_sample_ball_nd(in_features, out_features, K))
        self.register_buffer("m", torch.tensor(out_features))
        self.short_name = "UniFF"

    def forward(self, x):
        x = torch.einsum("...c,cj->...j", x, self.B.to(x.device))
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) / torch.sqrt(self.m)
    
class RegularFourierFeaturesTransform(torch.nn.Module):
    
    def __init__(self, in_features: int, r_in, r_out, r_res, c_res) -> None:
        """Fourier features with regular sampling in the ball or a disc of radius 
        K and `out_features` number of features. Can be used only with """
        super().__init__()
        self.register_buffer("r_in", torch.tensor(r_in))
        self.register_buffer("r_out", torch.tensor(r_out))
        
        if in_features == 2:
            s = pv.DiscSource(inner=r_in, outer=r_out, r_res=r_res, c_res=c_res)
            B = s.output.points[:, 0:2]  # get the x, y coordinates of the points on the disc
            B = torch.tensor(B, dtype=torch.float32).T
        elif in_features == 3:
            if len(c_res) != 2:
                raise ValueError("c_res must be a tuple of two integers to specify resolution in two angles in 3d case.")
            s = pv.SolidSphere(inner_radius=r_in, outer_radius=r_out, radius_resolution=r_res, 
                               theta_resolution=c_res[0], 
                               phi_resolution=c_res[1])
            B = s.points[:, 0:3]
            B = torch.tensor(B, dtype=torch.float32).T
        
        self.register_buffer("B", B)
        self.register_buffer("m", torch.tensor(B.shape[1]))
        self.short_name = "RegFF"

    def forward(self, x):
        x = torch.einsum("...c,cj->...j", x, self.B.to(x.device))
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) / torch.sqrt(self.m)
    
    
class LogarithmicFourierFeaturesTransform(torch.nn.Module):
    
    def __init__(self, in_features: int, out_features: int, K: float) -> None:
        super().__init__()
        self.register_buffer("K", torch.tensor(K))
        self.register_buffer("B", logarithmic_sample_ball_nd(in_features, out_features, K))
        self.register_buffer("m", torch.tensor(out_features))
        self.short_name = "LogFF"

    def forward(self, x):
        x = torch.einsum("...c,cj->...j", x, self.B.to(x.device))
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) / torch.sqrt(self.m)
    
    
class DivergenceFreeTransform2d(torch.nn.Module):
    """
    Obtains a 2d divergence-free vector field y(x) from a scalar function f(x):

    y(x) = (∂f/∂y, -∂f/∂x)

    The result is a vector field that is divergence-free by construction.
    """

    def __init__(self):
        super().__init__()

    def forward(self, f, x):
        # Calculate the curl of the field (f, 0):
        df_dx = dde.grad.jacobian(f, x, i=0, j=0)
        df_dy = dde.grad.jacobian(f, x, i=0, j=1)
        return torch.cat([df_dy, -df_dx], dim=1)
    

def uniform_sample_ball_nd(n_samples, n_dim, K, device='cpu'):
    """
    Sample `n_samples` vectors uniformly from within an n-dimensional ball of radius K.

    Parameters:
    - n_samples: Number of samples to generate.
    - n_dim: Dimensionality of the space.
    - K: Radius of the n-dimensional ball.
    - device: 'cpu' or 'cuda' for computation.

    Returns:
    - k: Tensor of shape (n_dim, n_samples) containing the sampled vectors.
    """
    device = torch.device(device)
    
    # Step 1: Generate random directions
    # Sample from a standard normal distribution
    k = torch.randn(n_dim, n_samples, device=device)
    
    # Normalize the vectors to have unit length
    k_unit = k / torch.norm(k, p=2, dim=0, keepdim=True)
    
    # Step 2: Generate radii with the appropriate distribution
    # Sample uniform random numbers between 0 and 1
    u = torch.rand(n_samples, device=device)
    
    # Compute radii to ensure uniform sampling within the ball
    radii = K * u ** (1 / n_dim)
    
    # Step 3: Multiply unit vectors by the radii
    k = k_unit * radii
    k.transpose_(0, 1)
    
    return k


def logarithmic_sample_ball_nd(n_samples, n_dim, K_max, K_min, device='cpu'):
    """
    Sample `n_samples` vectors with logarithmic radial distribution within an n-dimensional ball.

    Parameters:
    - n_samples: Number of samples to generate.
    - n_dim: Dimensionality of the space.
    - K_max: Maximum radius of the n-dimensional ball.
    - K_min: Minimum radius (must be > 0).
    - device: 'cpu' or 'cuda' for computation.

    Returns:
    - k: Tensor of shape (n_dim, n_samples) containing the sampled vectors.
    """
    device = torch.device(device)
    
    # Step 1: Generate random directions
    # Sample from a standard normal distribution
    k = torch.randn(n_dim, n_samples, device=device)
    
    # Normalize the vectors to have unit length
    k_unit = k / torch.norm(k, p=2, dim=0, keepdim=True)
    
    # Step 2: Generate radii with logarithmic distribution
    # Sample uniform random numbers between 0 and 1
    u = torch.rand(n_samples, device=device)
    
    # Compute radii to ensure logarithmic sampling within the ball
    # r = K_min * (K_max / K_min) ** u
    radii = K_min * (K_max / K_min) ** u
    
    # Step 3: Multiply unit vectors by the radii
    k = k_unit * radii
    
    return k