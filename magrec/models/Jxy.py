

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.transformation.Jxy2Bsensor import Jxy2Bsensor 
from magrec.image_processing.Padding import Padder

class Jxy(GenericModel):
    def __init__(self, data, loss_type):
        super().__init__(data, loss_type)

        # Define the transformation so that this isn't performed during a loop.
        self.magClass = Jxy2Bsensor(data)

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
        self.require["num_sources"] = 2
        
    def transform(self, nn_output):
        return self.magClass.transform(nn_output)

    def calculate_loss(self, b, target, nn_output = None, loss_weight = None):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """
        # a scaling
        alpha = 0

        if loss_weight is not None:
            # b = b* loss_weight
            b = torch.einsum("...kl,kl->...kl", b, loss_weight)
            target = torch.einsum("...kl,kl->...kl", target, loss_weight)
            if nn_output is not None:
                # use the std of the outputs as an additional loss function
                loss_std = alpha * torch.std(
                    torch.einsum("...kl,kl->...kl", nn_output, loss_weight), dim=(-2, -1)).sum()
            else:
                loss_std = 0
        else:
            if nn_output is not None:
                loss_std = alpha * torch.std(nn_output, dim=(-2, -1)).sum()
            else:
                loss_std = 0

        return self.loss_function(b, target) + loss_std

    def extract_results(self, final_output, final_b, remove_padding = True):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """

        self.results = dict()
        self.results["Jx"] = final_output[0,0,::] / self.scaling_factor
        self.results["Jy"] = final_output[0,1,::] / self.scaling_factor
        self.results["Recon B"] = final_b[0,::] / self.scaling_factor
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
        
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(10)

        plt.subplot(3, 2, 1)
        plot_data = results["original B"] * 1e3
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('original B')
        cb.set_label("B (mT)")


        plt.subplot(3, 2, 2)
        plot_data = results["Recon B"] * 1e3
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed B')
        cb.set_label("B (mT)")

        plt.subplot(3, 2, 3)
        plot_data = (results["original B"] - results["Recon B"])* 1e3
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed difference')
        cb.set_label("B (mT)")

        plt.subplot(3, 2, 5)
        plot_data = results["Jx"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("Jx (A/m)")

        plt.subplot(3, 2, 6)
        plot_data = results["Jy"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("Jy (A/m)")

