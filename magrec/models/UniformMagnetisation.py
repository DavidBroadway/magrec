

import torch
import numpy as np
import matplotlib.pyplot as plt
import magrec.models.GenericModel as GenericModel

class UniformMagnetisation(GenericModel):
    def __init__(self, data, dx, dy, height, layer_thickness):
        super().__init__(data)

        # Define the propagator so that this isn't performed during a loop.
        self.define_propagtor(data, dx, dy, height, layer_thickness)

    def define_propagtor(self, data, dx, dy, height, layer_thickness):
        from magrec.prop.Propagator import MagnetizationPropagator2d as Propagator
        self.propagator = Propagator(data.shape, dx, dy, height, layer_thickness)


    def transform(self, nn_output):
        """
        Args:
            nn_output:  The output of the neural network which is a be a 2D array of
                        magnetisation along a single direction

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        return self.propagator.get_B(nn_output)

    def loss_function(self, nn_output, target):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """
        b = self.transform(nn_output)
        return torch.mean((b - target)**2)

    def unpack_results(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        return self.transform(nn_output)

    def plot_results(self, nn_output, target):  
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            None
        """
        b = self.unpack_results(nn_output)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(b)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(target)
        plt.colorbar()
        plt.show()

    
