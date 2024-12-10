import pytest
import torch
from magrec.nn.utils import batched_curl, batched_div, batched_grad

@pytest.fixture
def test_points():
    return torch.randn(10, 3, requires_grad=True)

def test_curl_free_field(test_points):
    """Test curl of gradient field (should be zero)"""
    def gradient_field(x):
        # phi = x^2 + y^2 + z^2, gradient is curl-free
        return 2 * x
    
    curl_grad = batched_curl(gradient_field)(test_points)
    assert torch.allclose(curl_grad, torch.zeros_like(curl_grad), atol=1e-6), \
        "Curl of gradient field should be zero"

def test_divergence_free_field(test_points):
    """Test divergence of rotational field (should be zero)"""
    def rot_field(x):
        # A = (-y, x, 0), curl is divergence-free
        return torch.stack([-x[..., 1], x[..., 0], torch.zeros_like(x[..., 0])], dim=-1)
    
    div_rot = batched_div(rot_field)(test_points)
    assert torch.allclose(div_rot, torch.zeros_like(div_rot), atol=1e-6), \
        "Divergence of curl field should be zero"

def test_analytical_vector_field(test_points):
    """Test vector field with known analytical curl and divergence"""
    def test_field(x):
        # Field: (yx, yz, zx)
        return torch.stack([x[..., 1] * x[..., 0],
                          x[..., 1] * x[..., 2],
                          x[..., 2] * x[..., 0]], dim=-1)
    
    def analytical_curl(x):
        # curl = (-y, -z, -x)
        return torch.stack([-x[..., 1],
                            -x[..., 2],
                            -x[..., 0]], dim=-1)
    
    def analytical_div(x):
        # div = y + x + z
        return x[..., 1] + x[..., 0] + x[..., 2]
    
    curl_test = batched_curl(test_field)(test_points)
    div_test = batched_div(test_field)(test_points)
    
    assert torch.allclose(curl_test, analytical_curl(test_points), atol=1e-6), \
        "Curl computation incorrect for test field"
    assert torch.allclose(div_test, analytical_div(test_points), atol=1e-6), \
        "Divergence computation incorrect for test field"
        
def test_grad(test_points):
    def test_scalar_fn(x):
        # Function: (x, y, z) -> (x**2 + y**2 + z**2)
        return torch.sum(x**2, dim=-1)
    
    grad_test = batched_grad(test_scalar_fn)(test_points)
    grad_target = 2 * test_points

    expected_shape = (test_points.shape[0],) + (3,)    
    assert grad_test.shape == expected_shape, \
        "Wrong shape for gradient computation, should be {}, but found {} instead".format(expected_shape, grad_test.shape)
    assert torch.allclose(grad_test, grad_target, atol=1e-6), \
        "Gradient computation incorrect for test field"