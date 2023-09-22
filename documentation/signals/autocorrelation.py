import numpy as np
import matplotlib.pyplot as plt

"""
The autocorrelation provides a measure of how a signal correlates with itself at
various shifts. Essentially, it quantifies the degree of similarity between a
function and a shifted version of itself. Here's a breakdown of how the
autocorrelation behaves and what to expect:

1. **Pure Noise**: - For purely random noise, the autocorrelation will
   essentially be a delta function, which means it'll have a peak at `(0,0)` and
   will be close to zero elsewhere. - This is because noise is, by definition,
   uncorrelated. The only place where it correlates perfectly is with zero shift
   (i.e., with itself).
   
2. **Signal with Noise**: - The autocorrelation will still have a peak at
   `(0,0)`, since any signal perfectly correlates with itself at zero shift. -
   Away from `(0,0)`, the structure will depend on the signal's shape. A
   periodic signal will have repeated peaks in its autocorrelation, while
   non-periodic signals will have a more complex structure. - The noisy
   component will add perturbations to the autocorrelation, making it less
   pristine than the autocorrelation of a pure signal.
   
3. **Importance of the (0,0) Value**: - The value at `(0,0)` in the
   autocorrelation is always the highest because it represents the signal's
   power. This is where the signal is being compared to an un-shifted version of
   itself, hence maximum correlation. - For normalized autocorrelation, this
   value will be 1, since a signal perfectly correlates with itself. - This
   `(0,0)` value can be a useful diagnostic. In the context of noise, if the
   signal is purely noise, the `(0,0)` value will be high (signal power), and
   everything else will be close to zero. If there's any structure in the signal
   (like a periodic component), the autocorrelation will have non-zero values
   away from `(0,0)`.

In summary, analyzing the 2D autocorrelation provides insights into the
structure and patterns present in the signal. The presence of patterns or
structures will manifest as distinct shapes in the autocorrelation away from the
`(0,0)` point, while purely random noise will mostly only have a significant
value at the `(0,0)` point.
"""

def compute_2d_autocorrelation(array):
    """
    Compute 2D autocorrelation of the given array.
    """
    # Fourier transform the array
    f = np.fft.fft2(array)
    
    # Compute power spectrum (magnitude squared)
    ps = np.abs(f) ** 2
    
    # Inverse Fourier transform the power spectrum to get autocorrelation
    autocorr = np.fft.ifft2(ps)
    
    # Shift the zero-frequency component to the center of the spectrum
    autocorr = np.fft.fftshift(autocorr)
    
    # Take real part (imaginary part should be negligible)
    return np.real(autocorr)

def rotate_image(image, angle):
    """
    Rotate the given image by a specified angle.
    """
    from scipy.ndimage import rotate
    return rotate(image, angle, reshape=False)

# Generate 2D Gaussian noise (50x50)
noise = np.random.randn(50, 50)

# Compute 2D autocorrelation of noise
autocorr_noise = compute_2d_autocorrelation(noise)

# Create a 2D skewed and asymmetrical signal with fading periodicity
x = np.linspace(0, 4 * np.pi, 50)
y = np.linspace(0, 4 * np.pi, 50)
x, y = np.meshgrid(x, y)

decay = np.exp(-x / 10)  # exponential decay
phase_shift_x = np.pi / 3
phase_shift_y = np.pi / 6

signal = decay * np.sin(0.5 * x + phase_shift_x) * np.sin(0.5 * y + phase_shift_y)  # fading periodicity with phase shift

# Rotate the signal for asymmetry
signal = rotate_image(signal, 30)  # rotate by 30 degrees

# Add Gaussian noise to the signal
noisy_signal = signal + noise * 0.5  # scale the noise to ensure signal is more dominant

# Compute 2D autocorrelation of the noisy signal
autocorr_signal = compute_2d_autocorrelation(noisy_signal)

# Visualize
fig = plt.figure(figsize=(7, 7))

# 2D plots
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(noise, cmap='gray')
ax1.set_title("2D Gaussian Noise")
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(autocorr_noise, cmap='gray')
ax2.set_title("2D Autocorrelation of Noise")
ax2.axis('off')

ax3 = fig.add_subplot(2, 3, 4)
ax3.imshow(noisy_signal, cmap='gray')
ax3.set_title("Noisy Signal")
ax3.axis('off')

ax4 = fig.add_subplot(2, 3, 5)
ax4.imshow(autocorr_signal, cmap='gray')
ax4.set_title("2D Autocorrelation of Noisy Signal")
ax4.axis('off')

# 3D plots
x_vals, y_vals = np.meshgrid(np.arange(autocorr_noise.shape[0]), np.arange(autocorr_noise.shape[1]))

ax5 = fig.add_subplot(2, 3, 3, projection='3d')
ax5.plot_surface(x_vals, y_vals, autocorr_noise, cmap='viridis')
ax5.set_title("Autocorrelation of Noise")
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Autocorr Value')

ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.plot_surface(x_vals, y_vals, autocorr_signal, cmap='viridis')
ax6.set_title("Autocorrelation of Noisy Signal")
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Autocorr Value')

plt.show()