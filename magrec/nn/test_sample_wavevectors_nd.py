from magrec.nn.modules import uniform_sample_ball_nd, logarithmic_sample_ball_nd
import pytest
import matplotlib.pyplot as plt

def test_sample_wavevectors_nd():
    """Check sampled wavevectors shape."""
    n_samples = 100
    n_dim = 3
    K = 10
    wavevectors = uniform_sample_ball_nd(n_samples, n_dim, K)
    assert wavevectors.shape == (n_dim, n_samples)
    
def test_plot_uniform_sample_ball_nd():
    # For 2D visualization (n_dim = 2)
    K = 2  # Radius of the disk
    k_np = uniform_sample_ball_nd(200, K, 1).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(k_np[0], k_np[1], s=1)
    plt.title('Uniform Sampling in a 2D Disk')
    plt.xlabel('$k_1$')
    plt.ylabel('$k_2$')
    plt.axis('equal')
    plt.xlim(-2*K, 2*K)
    plt.ylim(-2*K, 2*K)
    plt.show()
    
    
def test_plot_logarithmic_sample_ball_nd():
    # For 2D visualization (n_dim = 2)
    K = 2  # Radius of the disk
    k_np = logarithmic_sample_ball_nd(200, K_max=K, K_min=0.1, n_dim=2).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(k_np[0], k_np[1], s=1)
    plt.title('Logarithmic Sampling in a 2D Disk')
    plt.xlabel('$k_1$')
    plt.ylabel('$k_2$')
    plt.axis('equal')
    plt.xlim(-2*K, 2*K)
    plt.ylim(-2*K, 2*K)
    plt.show()