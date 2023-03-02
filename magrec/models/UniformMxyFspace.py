

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.transformation.Mxy2Bsensor import Mxy2Bsensor

class UniformMxyFspace(GenericModel):
    def __init__(self, dataset, loss_type,  m_theta, m_phi,):
        super().__init__(dataset, loss_type)

        # Define the propagator so that this isn't performed during a loop.
        self.magClass = Mxy2Bsensor(dataset, m_theta = m_theta, m_phi = m_phi)

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
    
    def prepareTargetData(self):
        self.dataset.original_target = self.dataset.target
        self.dataset.target_fourier = self.ft.forward(self.dataset.target, dim=(-2, -1))
        self.dataset.target = torch.cat((torch.real(self.dataset.target_fourier), 
                               torch.imag(self.dataset.target_fourier)), dim=-1)
        
    def transform(self, nn_output):
        nn_shape = nn_output.shape
        real_component =  nn_output[..., 0:int(0.5*nn_shape[-1])]
        imag_component =  nn_output[..., int(0.5*nn_shape[-1]):nn_shape[-1]]
        complex_form = torch.complex(real_component, imag_component)

        transformed = torch.einsum("kl,...kl->...kl", self.magClass.transformation, complex_form)
        # transformed = complex_form * self.magClass.transformation
        b = torch.cat((torch.real(transformed), torch.imag(transformed)), dim=-1)

        return  b

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

        complex_out = torch.complex(torch.from_numpy(final_output[0,0,::].real), 
                                    torch.from_numpy(final_output[0,0,::].imag))
        complex_b = torch.complex(torch.from_numpy(final_b[0,0,::].real), 
                                   torch.from_numpy(final_b[0,0,::].imag))

        self.results = dict()
        self.results["Magnetisation"] = self.ft.backward(complex_out,  dim=(-2, -1))
        self.results["Recon B"] = self.ft.backward(complex_b,  dim=(-2, -1))
        self.results["original B"] = self.dataset.original_target

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
        plt.title('original magnetic field')
        cb.set_label("Magnetic Field (mT)")


        plt.subplot(2, 2, 2)
        plot_data = 1e3*results["Recon B"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="bwr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed magnetic field')
        cb.set_label("Magnetic Field (mT)")

        plt.subplot(2, 2, 3)
        plot_data = 1e3*results["original B"] - 1e3*results["Recon B"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="bwr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('difference $\Delta B$')
        cb.set_label("Magnetic Field (mT)")

        plt.subplot(2,2,4)
        plot_data = results["Magnetisation"]
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed magnetisation')
        cb.set_label("magnetisation ($\mu_b/nm^2$)")
