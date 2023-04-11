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
from magrec.transformation.Kernel import CurrentLayerFourierKernel2d
from magrec.transformation.Fourier import FourierTransform2d
from magrec.image_processing.Padding import Padder


class Jxy2Bsensor(GenericTranformation):

    def __init__(self, dataset, target = None, pad = True, fourier_target = False):
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
        self.dataset = dataset

        self.Padder = Padder()
        if pad:
            if target is None:
                grid = self.Padder.pad_zeros2d(dataset.target)
            else:
                grid = self.Padder.pad_zeros2d(target)
        else:
            if target is None:
                grid= dataset.target
            else:
                grid = target

        if fourier_target:
            grid_shape = grid[...,0:-1].size()
        else:
            grid_shape = grid.size()
 

        self.ft = FourierTransform2d(grid_shape=grid_shape, dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        
        self.j_to_b_matrix = CurrentLayerFourierKernel2d\
            .define_kernel_matrix(
                self.ft.kx_vector, 
                self.ft.ky_vector, 
                height= dataset.height, 
                layer_thickness=dataset.layer_thickness, 
                dx=dataset.dx,
                dy=dataset.dy)


        # remove the 0 componenet
        # self.j_to_b_matrix[0,0] = 0
        # If there exists any nans set them to zero
        self.j_to_b_matrix[self.j_to_b_matrix != self.j_to_b_matrix] = 0

        # Multiply the kernel matrix by the sensor direction
        self.j_to_b_matrix = torch.einsum("ijkl,i->jkl", self.j_to_b_matrix, self.sensor_dir)

        self.transformation = self.j_to_b_matrix
        # self.transformation[0,0] = 0
        # If there exists any nans set them to zero
        self.transformation[self.transformation != self.transformation] = 0

    

    def transform(self, J = None):
        if J is None:
            J = self.dataset.target
        J = self.Padder.pad_zeros2d(J)
        
        # Get the current density from the magnetization
        j = self.ft.forward(J, dim=(-2, -1))
        # Get the magnetic field from the current density
        b = torch.einsum("...jkl,...jkl->...kl", self.j_to_b_matrix, j)
        # b[0,0] = 0 # remove DC componenet
        B = self.ft.backward(b, dim=(-2, -1))
        B = self.Padder.remove_padding2d(B)


        return B