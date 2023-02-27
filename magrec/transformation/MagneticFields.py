"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
import torch

from magrec.transformation.generic import GenericTranformation
from magrec.transformation.Kernel import HarmonicFunctionComponentsKernel


class MagneticFields(GenericTranformation):
    def __init__(self, dataset):
        """
        Args:
            data:   data class
        """
        super().__init__(dataset)

        # Define the kernal for the transformation
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