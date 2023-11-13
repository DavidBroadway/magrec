"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
import torch
import numpy as np 
from magrec.transformation.Fourier import FourierTransform2d
from magrec.transformation.Kernel import HarmonicFunctionComponentsKernel

class GenericTranformation(object):
     # Super class that other transformation can be based off.
    def __init__(self, dataset):
        """
        Args:
            data:   data class
        """
        # Define the dataset
        self.dataset = dataset
        # Define the Fourier class
        self.ft = FourierTransform2d(
            grid_shape=self.dataset.target.size(),
            dx=self.dataset.dx,
            dy=self.dataset.dy,
            real_signal=False
        )


    def get_cartesian_dir(self, theta, phi):
        # check if theta and phi are ints, if so convert to a tensor
        if isinstance(theta, int):
            theta = torch.tensor(theta)
        if isinstance(phi, int):
            phi = torch.tensor(phi)

        theta = torch.deg2rad(theta)
        phi = torch.deg2rad(phi)

        cart_dir = torch.tensor([ \
            torch.cos(phi)*torch.sin( theta ), \
            torch.sin(phi)*torch.sin( theta ), \
            torch.cos(theta)], dtype=torch.complex64)
        return cart_dir
    
    def transform(self):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        raise NotImplementedError("transform must be overridden in a child class.")
    
    
    def inverse_transform(self):    
        raise NotImplementedError("inverse_transform must be overridden in a child class.")
    
    def plot_transform(self):
        raise NotImplementedError("plot_transform must be overridden in a child class.")
    
