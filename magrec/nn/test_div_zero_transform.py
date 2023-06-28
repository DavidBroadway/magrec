from magrec.nn.modules import ZeroDivTransform
import deepxde as dde
import torch

def test_div_zero_shape():
    # Instantiate the DivZeroTransform module
    div_zero_transform = ZeroDivTransform()

    # Define the scalar function f(x)
    def scalar_function(x):
        return torch.sin(x[:, [0]]) * torch.cos(x[:, [1]])

    # Generate the coordinates x
    x = torch.rand(20, 2, requires_grad=True)
    
    # Evaluate the scalar function f(x)
    f = scalar_function(x)

    # Calculate the vector field y(x) using DivZeroTransform
    y = div_zero_transform(f, x)

    # Check shape, we expect a 2d vector field y
    assert y.shape == (20, 2)
    

def test_div_zero_transform():
    # Instantiate the DivZeroTransform module
    div_zero_transform = ZeroDivTransform()

    # Define the scalar function f(x)
    def scalar_function(x):
        # see comment before the assertion, this function has a divergence of 1e-6 
        # which can be due to some numerical precision error in the gradient calculation
        return torch.sin(x[:, [0]]) * torch.cos(x[:, [1]]) ** 3

    # Generate the coordinates x
    x = torch.rand(100, 2, requires_grad=True)

    # Evaluate the scalar function f(x)
    f = scalar_function(x)

    # Calculate the vector field y(x) using DivZeroTransform
    y = div_zero_transform(f, x)
    
    div_y = dde.grad.jacobian(y, x, i=0, j=0) + dde.grad.jacobian(y, x, i=1, j=1)

    # Check that ∇•y = ∂y_1/∂x_1 + ∂y_2/∂x_2 = 0
    # TODO: Check numerical precision of differentiation in deepxde and pytorch. The error is currently 1e-7 for some function, 
    # it might be because of the float precision of the gradient calculation. Setting the absolute tolerance to 1e-6 for now.
    assert torch.allclose(div_y, torch.zeros_like(div_y), atol=1e-6)