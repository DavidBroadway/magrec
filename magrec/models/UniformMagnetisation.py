

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from magrec.models.generic_model import GenericModel
from magrec.transformation.Mxy2Bsensor import Mxy2Bsensor

class UniformMagnetisation(GenericModel):
    def __init__(self,  
                dataset : object, 
                loss_type : str = "MSE", 
                m_theta : float = 0,
                fit_m_theta: bool = False, 
                m_phi: float = 0,
                fit_m_phi: bool = False,
                scaling_factor: float = 1, 
                std_loss_scaling : float = 0, 
                loss_weight: torch.Tensor = None,
                source_weight: torch.Tensor = None,
                spatial_filter: bool = False,
                spatial_filter_width: float = 0.5):
        super().__init__(dataset, loss_type, scaling_factor)

        # Define the propagator so that this isn't performed during a loop.
        self.magClass = Mxy2Bsensor(dataset, m_theta = m_theta, m_phi = m_phi)
        self.std_loss_scaling = std_loss_scaling
        self.loss_weight = loss_weight
        self.source_weight = source_weight
        self.spatial_filter = spatial_filter
        self.spatial_filter_width = spatial_filter_width

        self.m_theta = m_theta
        self.m_phi = m_phi
        self.fit_m_theta = fit_m_theta
        self.fit_m_phi = fit_m_phi

        # define the requirements for the model that may change the fitting method
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
        self.require["num_sources"] = 1
        self.require["source_angles"] = True

    def transform(self, nn_output, m_theta = None, m_phi = None):
        if self.fit_m_theta:
            self.m_theta = m_theta
        if self.fit_m_phi:
            self.m_phi = m_phi

        if self.m_theta or self.m_phi:
            self.magClass = Mxy2Bsensor(self.dataset, m_theta = self.m_theta, m_phi = self.m_phi)

        # Apply the weight matrix to the output of the NN
        if self.source_weight is not None:
            nn_output = nn_output*self.source_weight
        
        # Apply a spatial filter to the output of the NN
        if self.spatial_filter:
            nn_output = self.blurrer(nn_output)
        return self.magClass.transform(nn_output)

    def calculate_loss(self, b, target,  nn_output = None):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """

        # a scaling
        alpha = self.std_loss_scaling

        if self.loss_weight is not None:
            # b = b* loss_weight
            b = torch.einsum("...kl,kl->...kl", b, self.loss_weight)
            target = torch.einsum("...kl,kl->...kl", target, self.loss_weight)
            if nn_output is not None:
                # use the std of the outputs as an additional loss function
                loss_std = alpha * torch.std(
                    torch.einsum("...kl,kl->...kl", nn_output, self.loss_weight), dim=(-2, -1)).sum()
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
