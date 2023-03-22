"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
import torch
import numpy as np

from magrec.transformation.generic import GenericTranformation
from magrec.transformation.Kernel import MagnetizationFourierKernel2d


class Bsensor2Mxy(GenericTranformation):

    def __init__(self, dataset, m_theta = 0, m_phi = 0):
        """
        Create a propagator for a 2d magnetization distribution that computes the magnetic field at `height` above
        the 2d magnetization layer of finite thickness `layer_thickness`, that has potentially 3 components of the magnetization.

        Assumes uniform magnetization across the layers thickness and uses the integration factor to account for the finite thickness.

        Args:
            source_shape:       shape of the magnetization distribution, shape (3, n_x, n_y)
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
            height:             height above the magnetization layer at which to evaluate the magnetic field, in [mm]
            layer_thickness:    thickness of the magnetization layer, in [mm]
        """
        super().__init__(dataset)

        # Define the magnetization direction
        self.mag_dir = self.get_cartesian_dir(m_theta, m_phi)

        # Define the sensor direction
        self.sensor_dir = self.get_cartesian_dir(dataset.sensor_theta, dataset.sensor_phi)

        # Define the kernal for the transformation
        self.m_to_b_matrix = MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, dataset.height, dataset.layer_thickness)
        
        # sum over the magnetisation direction
        self.m_to_b_matrix = torch.einsum("...ijkl,i->...jkl", self.m_to_b_matrix, self.mag_dir)

        # sum over the sensor direction
        self.m_to_b_matrix = torch.einsum("...jkl,j->...kl", self.m_to_b_matrix, self.sensor_dir)

    def get_m_from_b(self, b):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # b = torch.einsum("ijkl,bjkl->bikl", self.m_to_b_matrix, m)

        # Define the finally transformation
        transform =  1/ self.m_to_b_matrix

        # remove the 0 componenet
        transform[0,0] = 0
        # If there exists any nans set them to zero
        transform[transform != transform] = 0

        m = b * transform
        m[0,0] = 0 # remove DC componenet
        return m


    def transform(self):

        B = self.dataset.target

        b = self.ft.forward(B, dim=(-2, -1))
        b[0,0] = 0
        m = self.get_m_from_b(b)
        M = self.ft.backward(m, dim=(-2, -1))
        # convert from A/m to uB/nm^2
        unit_conversion = 1e-18 / 9.27e-24
        return M * unit_conversion
    