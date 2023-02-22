

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.prop.Transformations import Magnetisation2MagneticField as M2B

class UniformMagnetisation(GenericModel):
    def __init__(self, data, loss_type,  m_theta, m_phi,):
        super().__init__(data, loss_type)

        # Define the propagator so that this isn't performed during a loop.
        self.magClass = M2B(data, m_theta = 0, m_phi = 0)

        self.requirements()

    def requirements(self):
        """
        Define requirements for the model.
        Args:
            None
        """ 
        # Define the number of targets and sources for the network. 
        self.require = dict()
        self.require["num_targets"] = 1
        self.require["num_sources"] = 1
        
    def transform(self, nn_output):
        return self.magClass(nn_output)

    def calculate_loss(self, nn_output, target):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """
        b = self.transform(nn_output)
    

        return self.loss_function(b, target)

    def unpack_results(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        self.results = dict()
        self.results["Magnetisation"] = nn_output.detach().numpy()
        self.results["Reconstructed Magnetic Field"] = self.transform(nn_output).detach().numpy()
        return self.results


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
        plt.subplot(1, 3, 1)
        plt.imshow(self.data.target)
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(self.results["Reconstructed Magnetic Field"])
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(self.results["Magnetisation"])
        plt.colorbar()
        plt.show()


