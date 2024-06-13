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
from magrec.transformation.Fourier import FourierTransform2d


class MxandMy2Bxyz(GenericTranformation):

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
        super().__init__(dataset)

        # Define the magnetization direction
        self.mag_dir_x = self.get_cartesian_dir(90, 0)
        self.mag_dir_y = self.get_cartesian_dir(90, 90)

        # Define the sensor direction
        self.sensor_x = self.get_cartesian_dir(90, 0)
        self.sensor_y = self.get_cartesian_dir(90, 90)
        self.sensor_z = self.get_cartesian_dir(0, 0)

        # Define the kernal for the transformation
        # convert from A/M to uB/nm^2
        unit_conversion = (1e-18 / 9.27e-24) * 1e-6

        # Define the kernal for the transformation
        self.m_to_b_matrix = (1/unit_conversion) * MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, dataset.height, dataset.layer_thickness, 
                                  dataset.dx, dataset.dy, add_filter = True)
        
        # sum over the magnetisation direction
        self.mx_to_b = torch.einsum("...ijkl,i->...jkl", self.m_to_b_matrix, self.mag_dir_x)
        self.my_to_b = torch.einsum("...ijkl,i->...jkl", self.m_to_b_matrix, self.mag_dir_y)

        # sum over the sensor direction
        self.mx_to_bx = torch.einsum("...jkl,j->...kl", self.mx_to_b, self.sensor_x)
        self.my_to_bx = torch.einsum("...jkl,j->...kl", self.my_to_b, self.sensor_x)

        self.mx_to_by = torch.einsum("...jkl,j->...kl", self.mx_to_b, self.sensor_y)
        self.my_to_by = torch.einsum("...jkl,j->...kl", self.my_to_b, self.sensor_y)

        self.mx_to_bz = torch.einsum("...jkl,j->...kl", self.mx_to_b, self.sensor_z)
        self.my_to_bz = torch.einsum("...jkl,j->...kl", self.my_to_b, self.sensor_z)

        # Define the finally transformation
        self.trans_mx_2_bx =  self.clean_transformation(self.mx_to_bx)
        self.trans_my_2_bx =  self.clean_transformation(self.my_to_bx)

        self.trans_mx_2_by =  self.clean_transformation(self.mx_to_by)
        self.trans_my_2_by =  self.clean_transformation(self.my_to_by)

        self.trans_mx_2_bz =  self.clean_transformation(self.mx_to_bz)
        self.trans_my_2_bz =  self.clean_transformation(self.my_to_bz)


    def clean_transformation(self, transformation):
        transformation[0, 0] = 0
        transformation[transformation != transformation] = 0
        return transformation


    def transform(self, M = None):

        if M is None:
            print("no input provided, using the dataset target")
            M = self.dataset.target
        
        # Transform into Fourier space
        m = self.ft.forward(M, dim=(-2, -1))

        if len(M.size())<4:
            # Transform the magnetization to the magnetic field
            bx = m[0,:,:] * self.trans_mx_2_bx + m[1,:,:]*self.trans_my_2_bx
            by = m[0,:,:] * self.trans_mx_2_by + m[1,:,:]*self.trans_my_2_by
            bz = m[0,:,:] * self.trans_mx_2_bz + m[1,:,:]*self.trans_my_2_bz

        else:
            # Transform the magnetization to the magnetic field
            bx = m[0,0,:,:] * self.trans_mx_2_bx + m[0,1,:,:]*self.trans_my_2_bx
            by = m[0,0,:,:] * self.trans_mx_2_by + m[0,1,:,:]*self.trans_my_2_by
            bz = m[0,0,:,:] * self.trans_mx_2_bz + m[0,1,:,:]*self.trans_my_2_bz

        # Transform back into real space
        Bx = self.ft.backward(bx, dim=(-2, -1))
        By = self.ft.backward(by, dim=(-2, -1))
        Bz = self.ft.backward(bz, dim=(-2, -1))

        B = torch.stack([Bx, By, Bz], dim = 0)
        
        return B.real
    