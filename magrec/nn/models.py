

import torch
import numpy as np
from abc import ABC, abstractmethod


class GenericModel(ABC):
    # Super class that other models can be based off.

    def __init__(self, data):
        """
        Args:
            data:   The 2D array that will be used in the training.
                    This is passed to automatically define the size of the network.

        """
        # Define the default values fro the architexure for this model.
        self.arch = dict()
        self.arch["n_channels_in"] = 1
        self.arch["n_channels_out"] = 1

        self.arch["kernel"] = 5
        self.arch["stride"] = 2
        self.arch["padding"] = 2

        # Pad the data to define the size of the neural network.
        padded_data = self.pad_2d_array(data)
        self.arch["size"] = np.shape(padded_data)[0]

    @abstractmethod
    def model(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        raise NotImplementedError("model must be overridden in a child class.")

    def pad_2d_array(self, arr: np.ndarray) -> np.ndarray:
        rows, cols = arr.shape
        new_rows = self.next_power_of_two(rows)
        new_cols = self.next_power_of_two(cols)
        padded_arr = np.pad(arr, [(0, new_rows - rows), (0, new_cols - cols)], mode='constant')
        return padded_arr

    def next_power_of_two(self, x: int) -> int:
        return 1 if x == 0 else 2**(x - 1).bit_length()




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
