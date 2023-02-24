"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
from abc import abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.Kernel import HarmonicFunctionComponentsKernel
from magrec.prop.Kernel import MagnetizationFourierKernel2d
from magrec.prop.Kernel import CurrentLayerFourierKernel2d

class MagneticField(object):
    def __init__(self, dataset):
        """
        Args:
            data:   data class
        """
        self.dataset = dataset

        self.ft = FourierTransform2d(
            grid_shape=self.dataset.target.size(),
            dx=self.dataset.dx,
            dy=self.dataset.dy,
        )

        self.kernel = HarmonicFunctionComponentsKernel.define_kernel_matrix(
            self.ft.kx_vector,
            self.ft.ky_vector,
            self.dataset.sensor_theta,
            self.dataset.sensor_phi
        )
    
    def transform(self):
        """
        Transform the source to the sensor domain.

        Args:
            source:     source, shape (3, n_x, n_y)

        Returns:
            sensor:     sensor, shape (3, n_sensor_theta, n_sensor_phi)
        """
        b_fourier = self.ft.forward(self.dataset.target, dim=(-2, -1))
        b = torch.einsum("jkl,kl-> jkl", self.kernel, b_fourier)
        self.bxyz = self.ft.backward(b, dim=(-2, -1))

        return self.bxyz


class MagneticField2Magnetisation(object):

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
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.m_theta = np.deg2rad(m_theta)
        self.m_phi = np.deg2rad(m_phi)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)

        self.mag_dir = torch.tensor([ \
            np.cos(self.m_phi)*np.sin( self.m_theta ), \
            np.sin(self.m_phi)*np.sin( self.m_theta ), \
            np.cos(self.m_theta)], dtype=torch.complex64)

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        self.dataset = dataset

        self.m_to_b_matrix = MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, dataset.height, dataset.layer_thickness)
        

    def get_m_from_b(self, b):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # b = torch.einsum("ijkl,bjkl->bikl", self.m_to_b_matrix, m)

        #b = torch.tensor(b, dtype=torch.complex64)

        # sum over the magnetisation direction
        m_to_b_matrix = torch.einsum("...ijkl,i->...jkl", self.m_to_b_matrix, self.mag_dir)

        # sum over the sensor direction
        m_to_b_matrix = torch.einsum("...jkl,j->...kl", m_to_b_matrix, self.sensor_dir)

        # Define the finally transformation
        b_to_m_matrix =  1/ m_to_b_matrix

        # remove the 0 componenet
        b_to_m_matrix[0,0] = 0
        # If there exists any nans set them to zero
        b_to_m_matrix[b_to_m_matrix != b_to_m_matrix] = 0

        m = b * b_to_m_matrix
        m[0,0] = 0 # remove DC componenet
        return m


    def transform(self):

        B = self.dataset.target

        b = self.ft.forward(B, dim=(-2, -1))
        b[0,0] = 0
        m = self.get_m_from_b(b)
        M = self.ft.backward(m, dim=(-2, -1))
        # convert from A/M to uB/nm^2
        unit_conversion = 1e-18 / 9.27e-24
        return M * unit_conversion


class Magnetisation2MagneticField(object):
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
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.m_theta = np.deg2rad(m_theta)
        self.m_phi = np.deg2rad(m_phi)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)

        self.mag_dir = torch.tensor([ \
            np.cos(self.m_phi)*np.sin( self.m_theta ), \
            np.sin(self.m_phi)*np.sin( self.m_theta ), \
            np.cos(self.m_theta)], dtype=torch.complex64)

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        self.dataset = dataset

        self.m_to_b_matrix = MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, dataset.height, dataset.layer_thickness)
        
        # sum over the magnetisation direction
        self.m_to_b_matrix = torch.einsum("ijkl,i->jkl", self.m_to_b_matrix, self.mag_dir)

        # sum over the sensor direction
        self.m_to_b_matrix = torch.einsum("jkl,j->kl", self.m_to_b_matrix, self.sensor_dir)

        # remove the 0 componenet
        self.m_to_b_matrix[0,0] = 0
        # If there exists any nans set them to zero
        self.m_to_b_matrix[self.m_to_b_matrix != self.m_to_b_matrix] = 0

    def get_b_from_m(self, m):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # b = torch.einsum("ijkl,bjkl->bikl", self.m_to_b_matrix, m)

        #b = torch.tensor(b, dtype=torch.complex64)
        b = m * self.m_to_b_matrix
        b[0,0] = 0 # remove DC componenet
        return b


    def transform(self, nn_output):
        # convert from A/M to uB/nm^2
        unit_conversion = 1e-18 / 9.27e-24
        M = nn_output/ unit_conversion

        m = self.ft.forward(M, dim=(-2, -1))
        m[0,0] = 0
        b= self.get_b_from_m(m)
        B = self.ft.backward(b, dim=(-2, -1))

        return B 


class Bsensor2Jxy(object):

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
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        self.dataset = dataset

        self.j_to_b_matrix = CurrentLayerFourierKernel2d\
            .define_kernel_matrix(
                self.ft.kx_vector, 
                self.ft.ky_vector, 
                -dataset.height, 
                -dataset.layer_thickness)

        # Define the finally transformation
        self.b_to_j_matrix =  1/ self.j_to_b_matrix
        # remove the 0 componenet
        self.b_to_j_matrix[0,0] = 0
        # If there exists any nans set them to zero
        self.b_to_j_matrix[self.b_to_j_matrix != self.b_to_j_matrix] = 0

    def get_j_from_b(self, b):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # torch.einsum("...ijkl,...ikl->...jkl", self.b_to_j_matrix, b)

        # sum over the sensor direction
        b_to_j_matrix = torch.einsum("ijkl,i->jkl", self.b_to_j_matrix, self.sensor_dir)

        j = b * b_to_j_matrix
        j[0,0] = 0 # remove DC componenet
        return j 


    def transform(self):

        B = self.dataset.target

        b = self.ft.forward(B, dim=(-2, -1))
        b[0,0] = 0
        j = self.get_j_from_b(b)
        J = self.ft.backward(j, dim=(-2, -1))
        return J


class Bxyz2Jxy(object):
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
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.s_theta = np.deg2rad(dataset.sensor_theta)
        self.s_phi = np.deg2rad(dataset.sensor_phi)

        self.sensor_dir = torch.tensor([ \
            np.cos(self.s_phi)*np.sin(self.s_theta), \
            np.sin(self.s_phi)*np.sin(self.s_theta), \
            np.cos(self.s_theta)], dtype=torch.complex64)

        self.dataset = dataset

        self.j_to_b_matrix = CurrentLayerFourierKernel2d\
            .define_kernel_matrix(
                self.ft.kx_vector, 
                self.ft.ky_vector, 
                dataset.height, 
                dataset.layer_thickness)
        
        self.b_to_j_matrix = 1/self.j_to_b_matrix
        # set the Bz term to zero as we don't need it
        self.b_to_j_matrix[2, 0, :, :] = 0 
        self.b_to_j_matrix[2, 1, :, :] = 0 

        # set the DC component to zero
        self.b_to_j_matrix[0,0] = 0

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
        B = self.dataset.target

        b = self.ft.forward(B, dim=(-2, -1))
        b[0,0] = 0
        j = self.get_j_from_b(b)
        J = self.ft.backward(j, dim=(-2, -1))
        return J


