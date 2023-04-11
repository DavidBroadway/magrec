"""
This module implements filtering of 2D arrays
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from magrec.transformation.Fourier import FourierTransform2d

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
        return self.data_filtered.real


    def apply_hanning_filter(self, wavelength, data=None, plot_results=False, in_fourier_space=False):
        """
        Takes a data set, transforms it into Fourier space, applies a hanning filter, and the transforms back into real space
        """
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        if in_fourier_space:
            fourier_data = data
        else:
            fourier_data = self.ft.forward(data, dim=(-2, -1) )
        # Define the hanning filter
        han2d = 0.5*(1 + np.cos(0.5*self.ft.k_matrix *wavelength))
        # Apply the filter
        new_data = fourier_data * han2d
        # Transform back to real space
        if in_fourier_space:
            self.data_filtered = new_data
        else:
            self.data_filtered = self.ft.backward(new_data, dim=(-2, -1))
        if plot_results:
            self.plot_filter_result(han2d, data, self.data_filtered)

        return self.data_filtered.real

    def apply_short_wavelength_filter(self, wavelength, data=None, plot_results=False, in_fourier_space=False, print_action=True):
        """
        Takes a data set and applies a hard short wavelength filter in Fourier space.
        """
        if print_action:
            print(f"Applied a high frequency filter, removing all components smaller than {wavelength} um")
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        if in_fourier_space:
            fourier_data = data
        else:
            fourier_data = self.ft.forward(data, dim=(-2, -1) )
        # Define a filter full of ones
        filter = torch.ones(fourier_data.shape)
        # Remove wavelengths that are shorter than the given wavelength
        filter[..., (torch.abs(self.ft.k_matrix)> (2*np.pi/(wavelength)))] = 0
        # filter[(self.ft.k_matrix > (1/(wavelength)))] = 0
        # Apply the filter
        new_data = fourier_data*filter
        # Transform back to real space
        if in_fourier_space:
            self.data_filtered = new_data
        else:
            self.data_filtered = self.ft.backward(new_data, dim=(-2, -1))
        if plot_results:
            self.plot_filter_result(filter, data, self.data_filtered)
        return self.data_filtered.real

    def apply_long_wavelength_filter(self, wavelength, data=None, plot_results=False, print_action=True):
        """
        Takes a data set and applies a hard short wavelength filter in Fourier space.
        """
        if print_action:
            print(f"Applied a high frequency filter, removing all components larger than {wavelength} um")
        if data is None:    
            data = self.data_filtered
        # Transform into fourier space
        fourier_data = self.ft.forward(data, dim=(-2, -1))
        # Define a filter full of ones
        filter = torch.ones(data.shape())
        # Remove wavelengths that are shorter than the given wavelength
        filter[(torch.abs(self.ft.k_matrix) < (2* np.pi / wavelength))] = 0
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