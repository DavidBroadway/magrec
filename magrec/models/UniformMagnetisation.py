

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.transformation.Mxy2Bsensor import Mxy2Bsensor

class UniformMagnetisation(GenericModel):
    def __init__(self, dataset, loss_type,  m_theta, m_phi):
        super().__init__(dataset, loss_type)

        # Define the propagator so that this isn't performed during a loop.
        self.magClass = Mxy2Bsensor(dataset, m_theta = m_theta, m_phi = m_phi)

        # define the requirements for the model that may change the fitting method
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
        return self.magClass.transform(nn_output)

    def calculate_loss(self, b, target, loss_weight=None):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """
        if loss_weight is not None:
            b = torch.einsum("...kl,kl->...kl", b, loss_weight)
            target = torch.einsum("...kl,kl->...kl", target, loss_weight)
    
        return self.loss_function(b, target)

    def extract_results(self, final_output, final_b, remove_padding = True):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        self.results = dict()
        self.results["Magnetisation"] = final_output[0,0,::] / self.scaling_factor
        self.results["Recon B"] = final_b[0,0, ::] / self.scaling_factor
        self.results["original B"] = self.original_target

        if remove_padding:
            self.remove_padding_from_results()
        return self.results
    

    def plot_results(self, results):  
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            None
        """
        
        plt.figure()
        plt.subplot(2, 2, 1)
        plot_data = 1e3*results["original B"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="bwr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('original B')
        cb.set_label("B (mT)")


        plt.subplot(2, 2, 2)
        plot_data = 1e3*results["Recon B"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="bwr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed B')
        cb.set_label("B (mT)")

        plt.subplot(2, 2, 3)
        plot_data = 1e3*results["original B"] - 1e3*results["Recon B"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="bwr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('difference $\Delta B$')
        cb.set_label("B (mT)")

        plt.subplot(2,2,4)
        plot_data = results["Magnetisation"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed M')
        cb.set_label("M ($\mu_b/nm^2$)")
