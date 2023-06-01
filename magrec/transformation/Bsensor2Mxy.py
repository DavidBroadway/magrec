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

        # convert from A/m to (uB/nm^2) * conversion from um k vectors 
        unit_conversion = (1e-18 / 9.27e-24) * 1e-6

        # Define the kernal for the transformation
        self.m_to_b_matrix = (1/unit_conversion) * MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, 
                                  self.ft.ky_vector, 
                                  dataset.height,
                                  dataset.layer_thickness,
                                  dx = dataset.dx,
                                  dy = dataset.dy,
                                  add_filter = True)
        
        # sum over the magnetisation direction
        self.m_to_b_matrix = torch.einsum("...ijkl,i->...jkl", self.m_to_b_matrix, self.mag_dir)

        # sum over the sensor direction
        self.m_to_b_matrix = torch.einsum("...jkl,j->...kl", self.m_to_b_matrix, self.sensor_dir)
        
        # Define the finally transformation
        self.b_to_m_matrix = 1/ self.m_to_b_matrix

        # If there exists any nans set them to zero
        self.b_to_m_matrix[self.b_to_m_matrix != self.b_to_m_matrix] = 0


    def transform(self, B : torch.Tensor = None):
        if B is None:
            print("no input provided, using the dataset target")
            B = self.dataset.target
        b = self.ft.forward(B, dim=(-2, -1))
        # get the magnetisation in F space
        m = b  * self.b_to_m_matrix
        
        M = self.ft.backward(m, dim=(-2, -1)).real
        
        return M 
    