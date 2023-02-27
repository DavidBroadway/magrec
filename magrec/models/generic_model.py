

import torch
import torch.nn as nn
import numpy as np
from magrec.image_processing.Padding import Padder

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

    def remove_padding_from_results(self):
        # Remove the padding from the results.
        
        padding = Padder()
        print('Removed the padding that was applied to the data')

        for idx in range(len(self.dataset.actions)):
            if self.dataset.actions.loc[len(self.dataset.actions) - 1-idx].reverseable:
                    roi = self.dataset.reverse_parameters[-1-idx]
                    for key in self.results.keys():
                        self.results[key] = padding.crop_data(self.results[key], roi)

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


    def extract_results(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        raise NotImplementedError("extract_results must be overridden in a child class.")

    def plot_results(self, nn_output, target):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            None
        """
        raise NotImplementedError("plot_results must be overridden in a child class.")
