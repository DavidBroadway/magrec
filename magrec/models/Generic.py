

import torch
import numpy as np


class GenericModel(object):
    # Super class that other models can be based off.

    def __init__(self, data):
        """
        Args:
            data:   class object that contains the data and the parameters of the data.

        """
        self.data = data


    def transform(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        raise NotImplementedError("transform must be overridden in a child class.")

    def loss_function(self, nn_output, target):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """
        raise NotImplementedError("loss_function must be overridden in a child class.")


    def unpack_results(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        raise NotImplementedError("unpack_results must be overridden in a child class.")

    def plot_results(self, nn_output, target):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            None
        """
        raise NotImplementedError("plot_results must be overridden in a child class.")



class UniformDirectionMagnetisation(GenericModel):
    def __init__(self, data, dx, dy, height, layer_thickness):
        super().__init__(data)

        # Define the propagator so that this isn't performed during a loop.
        self.define_propagtor(data, dx, dy, height, layer_thickness)

    def define_propagtor(self, data, dx, dy, height, layer_thickness):
        from magrec.prop.Propagator import MagnetizationPropagator2d as Propagator
        self.propagator = Propagator(data.shape, dx, dy, height, layer_thickness)



    def model(self, nn_output):
        """
        Args:
            nn_output:  The output of the neural network which is a be a 2D array of
                        magnetisation along a single direction

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        b = self.propagator.get_B(nn_output)
        return b
