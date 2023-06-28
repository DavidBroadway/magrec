import torch
import numpy as np
from magrec.prop.constants import twopi

import deepxde as dde

class GaussianFourierFeatureTransform(torch.nn.Module):
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
        self._B = twopi * torch.normal(mean=0, std=sigma ** 2, size=(in_features, out_features))
        self._m = out_features
        
    def forward(self, x):
        x = torch.einsum('...c,cj->...j', x, self._B.to(x.device))
        return torch.cat([torch.cos(x), torch.sin(x)], dim=1) / np.sqrt(self._m)


class ZeroDivTransform(torch.nn.Module):
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