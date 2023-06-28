import pytest
from magrec.nn.modules import GaussianFourierFeatureTransform
import torch

def test_basic_tensor():
    """Test that the output of the module is the correct shape.
    
    Given an input with 50 batches and 2 features, the output should be 
    50 batches and 40 Fourier features. 
    
    An example: 50 points with 2 coordinates (x, y) are transformed into 50 points 
    wtih 40 Fourier features coordinates (20 cosines and 20 sines).
    """
    x = torch.randn(50, 2)
    y = GaussianFourierFeatureTransform(2, 20, 1)(x)
    assert y.shape == (50, 40)
