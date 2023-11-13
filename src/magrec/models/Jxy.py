

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from magrec.models.generic_model import GenericModel
from magrec.transformation.Jxy2Bsensor import Jxy2Bsensor 
from magrec.image_processing.Padding import Padder



class Jxy(GenericModel):
    def __init__(self, 
                 dataset : object, 
                 loss_type : str = "MSE", 
                 scaling_factor: float = 1, 
                 std_loss_scaling : float = 0, 
                 loss_weight: torch.Tensor = None,
                 source_weight: torch.Tensor = None,
                 spatial_filter: bool = False,
                 spatial_filter_width: float = 0.5):
        super().__init__(dataset, loss_type, scaling_factor)

        """
        Args:
            dataset: The dataset to be fitted.
            loss_type: The type of loss function to be used.
            scaling_factor: The scaling factor to be applied to the target data to obtain better 
                gradients. This is automatically removed from the results. 
            std_loss_scaling: The scaling factor to be applied to the standard deviation loss function. 
                If this is set to 0 then the standard deviation loss function is not used.
            loss_weight: The weight of the loss function.
            source_weight: The weight of the sources.
            spatial_filter: Whether to apply a spatial filter to the output of the network.
            spatial_filter_width: The width of the spatial filter.
        """


        # Define the transformation so that this isn't performed during a loop.
        self.magClass = Jxy2Bsensor(dataset)
        self.std_loss_scaling = std_loss_scaling
        self.loss_weight = loss_weight
        self.source_weight = source_weight
        self.spatial_filter = spatial_filter
        self.spatial_filter_width = spatial_filter_width
        self.requirements()

        if self.spatial_filter:
            # Blur the output of the NN based off the standoff distance compared to the pixel size
            # From Nyquists theorem the minimum frequency that can be resolved is 1/2 the pixel size 
            # or in our case 1/2 the standoff distance. Therefore FWHM = 1/2 the standoff distance 
            # relative to the pixel size 
            sigma = [self.spatial_filter_width, self.spatial_filter_width]
            self.blurrer = T.GaussianBlur(kernel_size=(51, 51), sigma=(sigma))

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
        # Apply the weight matrix to the output of the NN
        if self.source_weight is not None:
            nn_output = nn_output*self.source_weight
        
        # Apply a spatial filter to the output of the NN
        if self.spatial_filter:
            nn_output = self.blurrer(nn_output)

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
        alpha = self.std_loss_scaling

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
        self.results["J"] = np.sqrt(self.results["Jx"]**2 + self.results["Jy"]**2)
        sp = [self.dataset.dx, self.dataset.dy]
        div_j = self.divergence(self.results["Jx"], self.results["Jy"], sp)
        self.results["divJ"] = div_j

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


    def divergence(self, fx, fy,sp):
        """ Computes divergence of vector field 
        f: array -> vector field components [Fx,Fy,Fz,...]
        sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
        """
        return torch.gradient(fx, spacing = sp[0], dim = 0)[0] + torch.gradient(fy, spacing = sp[1], dim = 1)[0]