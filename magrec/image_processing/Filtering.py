"""
This module implements filtering of 2D arrays
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from magrec.transformation.Fourier import FourierTransform2d


class HannFilter():

    def __init__(self, real_signal=True):
        super().__init__()
        self.real_signal = real_signal

    def fit(self, X, y=None, **fit_params):
        # Proper definition of the Hann filter does not depend on dx and dy, for it cancels out in the 
        # filter defintion by multiplying the k_vector by dx. This is how it should be, and equivalent call
        # without the additional parameter would be to hardcode dx = 1, dy = 1, in the units of image pixels. 
        # However it raises a question of what to do when image is padded or modified?
        self.ft = FourierTransform2d(grid_shape=X.shape, dx=1, dy=1, real_signal=self.real_signal)
        if self.real_signal:
            self.filter = 0.5 * (1 + np.cos(self.ft.kx_vector / 2))[:, None] * 0.5 * (1 + np.cos(self.ft.ky_vector / 4))[None, :]
        else: 
            self.filter = 0.5 * (1 + np.cos(self.ft.kx_vector / 2))[:, None] * 0.5 * (1 + np.cos(self.ft.ky_vector / 2))[None, :]
        return self

    def transform(self, X, y=None):
        return self.ft.backward(self.ft.forward(X, dim=(-2, -1)) * self.filter, dim=(-2, -1)).real

    def show_filter(self, centered_zero_frequency=False):
        """
        Plots the filter in Fourier space.

        Args:
            centered (bool):    whether to center zero frequency in the middle of the plot. By default,
                                the zero frequency is at the index [0, 0] of the filter. If centered, the
                                zero frequency is at the index [N//2, N//2] where N is the size of the filter.
                                The standard behavior is to have the zero frequency at the index [0, 0], positive
                                frequencies to be at [i, j] and negative frequencies to be at [-i, -j] where
                                i, j ∈ [1, N//2].
        """
        from magrec.misc.plot import plot_n_components

        filter = self.filter
        kx_vector = self.ft.kx_vector
        ky_vector = self.ft.ky_vector

        if centered_zero_frequency:
            if self.real_signal:
                filter = np.fft.fftshift(filter, axes=(-2,))
            else:
                filter = np.fft.fftshift(filter, axes=(-2,-1))
                ky_vector = np.fft.fftshift(ky_vector)
            
            kx_vector = np.fft.fftshift(kx_vector)
            
        
        fig = plot_n_components(filter, show_coordinate_system=False, climits=(0, 1), labels='no_labels')
        ax: plt.Axes = fig.axes[0]
        ax.set_xlabel(r'$k_x$ (radians per unit length)')
        ax.set_ylabel(r'$k_y$ (radians per unit length)')
        ax.set_title('Hann filter')

        plt.show(fig)
        plt.close()
        xticks_locs = ax.get_xticks()
        yticks_locs = ax.get_yticks()

        # set tick labels to correspond to the kx, ky values
        ax.set_xticklabels(['{:.2f}'.format(l.item()) for l in kx_vector[xticks_locs]])
        ax.set_yticklabels(['{:.2f}'.format(l.item()) for l in ky_vector[yticks_locs]])

        return fig


class GaussianFilter(object):

    def __init__(self, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0, radius=None):
        super().__init__()

        import scipy
        self.gaussian_filter = scipy.ndimage.gaussian_filter1d

        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.cval = cval
        self.truncate = truncate
        self.radius = radius

    def fit(self, X, y=None, **fit_params):
        # Overwrite the default values of the parameters with the ones provided in fit_params
        for key, value in fit_params.items():
            setattr(self, key, value)

        return self

    def transform(self, X, y=None):
        res = self.gaussian_filter(X, axis=-1, sigma=self.sigma, order=self.order, mode=self.mode, cval=self.cval, truncate=self.truncate)
        res = self.gaussian_filter(res, axis=-2, sigma=self.sigma, order=self.order, mode=self.mode, cval=self.cval, truncate=self.truncate)
        return torch.tensor(res, device=X.device)


class DataFiltering(object):

    def __init__(self, data, dx, dy):
        """
        Args:
            data:               2D array of data
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
        """
        self.data = data
        self.data_filtered = data
        self.ft = FourierTransform2d(grid_shape=data.size(), dx=dx, dy=dy, real_signal=False)

    def remove_DC_background(self, data=None):
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        fourier_data = self.ft.forward(data, dim=(-2, -1))
        # Remove zero element which represents the DC field
        fourier_data[0,0] = 0
        # Transform back to real space
        self.data_filtered = self.ft.backward(fourier_data, dim=(-2, -1))
        return self.data_filtered 


    def apply_hanning_filter(self, wavelength, data=None, plot_results=False):
        """
        Takes a data set, transforms it into Fourier space, applies a hanning filter, and the transforms back into real space
        """
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        fourier_data = self.ft.forward(data, dim=(-2, -1))
        # Define the hanning filter
        han2d = 0.5*(1 + np.cos(self.ft.k_matrix * wavelength/2 ))
        # Apply the filter
        new_data = fourier_data * han2d
        # Transform back to real space
        self.data_filtered = self.ft.backward(new_data, dim=(-2, -1))
        if plot_results:
            self.plot_filter_result(han2d, data, self.data_filtered)

        return self.data_filtered.real

    def apply_short_wavelength_filter(self, wavelength, data=None, plot_results=False):
        """
        Takes a data set and applies a hard short wavelength filter in Fourier space.
        """

        print(f"Applied a high frequency filter, removing all components smaller than {wavelength} um")
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        fourier_data = self.ft.forward(data, dim=(-2, -1) )
        # Define a filter full of ones
        filter = torch.ones(fourier_data.shape)
        # Remove wavelengths that are shorter than the given wavelength
        filter[(self.ft.k_matrix > (2*np.pi/(wavelength)))] = 0
        # Apply the filter
        new_data = fourier_data*filter
        # Transform back to real space
        self.data_filtered = self.ft.backward(new_data, dim=(-2, -1))
        if plot_results:
            self.plot_filter_result(filter, data, self.data_filtered)
        return self.data_filtered.real

    def apply_long_wavelength_filter(self, wavelength, data=None, plot_results=False):
        """
        Takes a data set and applies a hard short wavelength filter in Fourier space.
        """

        print(f"Applied a high frequency filter, removing all components larger than {wavelength} um")
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        fourier_data = self.ft.forward(data, dim=(-2, -1))
        # Define a filter full of ones
        filter = torch.ones(data.shape())
        # Remove wavelengths that are shorter than the given wavelength
        filter[(self.ft.k_matrix < (2* np.pi / wavelength))] = 0
        # Apply the filter
        new_data = fourier_data*filter
        # Transform back to real space
        self.data_filtered = self.ft.backward(new_data, dim=(-2, -1))
        if plot_results:
            self.plot_filter_result(filter, data, self.data_filtered)

        return self.data_filtered


    @staticmethod
    def plot_filter_result(filter, data, data_filtered):
        fig = plt.figure()
        fig.set_figheight(10)

        plt.subplot(5,1,1)
        plt.imshow((filter.real), cmap='bwr')
        plt.title('Filter')
        plt.colorbar()

        plt.subplot(5,1,2)
        plt.imshow(torch.rot90(data.real), cmap='bwr')
        plt.title('array real component')
        plt.colorbar()
        
        plt.subplot(5,1,3)
        plt.imshow(torch.rot90(data_filtered.real), cmap='bwr')
        plt.title('Filtered array real component')
        plt.colorbar()

        # plt.subplot(5,1,4)
        # plt.imshow(torch.rot90(data.imag), cmap='bwr')
        # plt.title('array imaginary component')
        # plt.colorbar()

        # plt.subplot(5,1,5)
        # plt.imshow(torch.rot90(data_filtered.imag), cmap='bwr')
        # plt.title('Filtered array imaginary component')
        # plt.colorbar()