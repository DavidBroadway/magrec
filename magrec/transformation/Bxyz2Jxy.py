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

from magrec.transformation.Kernel import CurrentLayerFourierKernel2d, MagneticFieldToCurrentInversion2d
from magrec.transformation.Fourier import FourierTransform2d

from magrec.image_processing.Padding import Padder


class Bxyz2Jxy(GenericTranformation):
    def __init__(self, dataset):
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
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=False)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)
        self.Padder = Padder()

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        self.dataset = dataset

        # self.j_to_b_matrix = CurrentLayerFourierKernel2d\
        #     .define_kernel_matrix(
        #         self.ft.kx_vector, 
        #         self.ft.ky_vector, 
        #         dataset.height, 
        #         dataset.layer_thickness)

        self.b_to_j_matrix = MagneticFieldToCurrentInversion2d.define_kernel_matrix(
            self.ft.kx_vector, 
            self.ft.ky_vector, 
            height=dataset.height, 
            layer_thickness=dataset.layer_thickness,
            dx=dataset.dx,
            dy=dataset.dy
        )
        # set the Bz term to zero as we don't need it
        # self.b_to_j_matrix[2, 0, :, :] = 0 
        # self.b_to_j_matrix[2, 1, :, :] = 0 

        # set the DC component to zero
        # self.b_to_j_matrix[0,0] = 0

        # # If there exists any nans set them to zero
        self.b_to_j_matrix[self.b_to_j_matrix != self.b_to_j_matrix] = 0


    def get_j_from_b(self, b):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # j = torch.einsum("...ijkl,...ikl->...jkl", self.b_to_j_matrix, b)
        return torch.einsum("...ijkl,...ikl->...jkl", self.b_to_j_matrix, b)


    def transform(self):
        # B = self.Padder.pad_zeros2d(self.dataset.target[0:2,...])
        B = self.dataset.target[0:2,...]

        b = self.ft.forward(B, dim=(-2, -1))
        # b[0,0] = 0
        j = self.get_j_from_b(b)
        J = self.ft.backward(j, dim=(-2, -1))
        # J = self.Padder.remove_padding2d(J)
        return J.real