

import torch
import torch.nn as nn
import numpy as np


class GenericModel(object):
    # Super class that other models can be based off.

    def __init__(self, dataset, loss_type):
        """
        Args:
            data:   class object that contains the data and the parameters of the data.

        """
        self.dataset = dataset
        self.define_loss_function(loss_type)
        # Add addtional requirements of the model here.
        self.requirements()


    def define_loss_function(self, loss_type):
        
        if loss_type == "L1":
            self.loss_function = nn.L1Loss()
        elif loss_type == "MSE":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError("ERROR: Loss type not recognised. Options are: L1 and MSE (L2)")


    def requirements(self):
        """
        Define requirements for the model.
        Args:
            None
        """ 
        pass


    def transform(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        raise NotImplementedError("transform must be overridden in a child class.")

    def calculate_loss(self, nn_output, target):
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
