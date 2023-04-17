

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.transformation.Jxy2Bsensor import Jxy2Bsensor 
from magrec.image_processing.Padding import Padder
from magrec.transformation.Fourier import FourierTransform2d


class JxyFspace(GenericModel):
    def __init__(self, data, loss_type, scaling_factor = None):
        super().__init__(data, loss_type, scaling_factor)


        self.Padder = Padder()

        # Define the transformation so that this isn't performed during a loop.
        self.prepareTargetData()

        self.magClass = Jxy2Bsensor(data, target = self.training_target, pad = False, fourier_target=True)
        # self.magClass = Jxy2Bsensor(data, pad = False)

        grid= self.dataset.target
        grid_shape = grid[...,0:-1].size()

        self.ft = FourierTransform2d(grid_shape=grid_shape, dx=self.dataset.dx, dy=self.dataset.dy, real_signal=True)


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
        

    def prepareTargetData(self):
        # Add a scalling factor to the target data to help the network learn   
        # 
        
        self.original_target = self.dataset.target 
        self.training_target = self.dataset.target * self.scaling_factor
        self.training_target = self.Padder.pad_zeros2d(self.training_target)

        # transform into fourier space
        self.dataset.target_fourier = self.ft.forward(self.training_target, dim=(-2, -1))
        # concatenate real and imaginary components to form a single tensor that is real
        # as the network requires real inputs
        self.training_target = torch.cat((torch.real(self.dataset.target_fourier), 
                               torch.imag(self.dataset.target_fourier)), dim=-1)


    def transform(self, nn_output):
        nn_shape = nn_output.shape
        real_component =  nn_output[..., 0:int(0.5*nn_shape[-1])]
        imag_component =  nn_output[..., int(0.5*nn_shape[-1]):nn_shape[-1]]
        complex_form = torch.complex(real_component, imag_component) 

        transformed = torch.einsum("jkl,...jkl->...kl", self.magClass.transformation, complex_form)
        b = torch.cat((torch.real(transformed), torch.imag(transformed)), dim=-1)

        return  b


    def calculate_loss(self, b, target, loss_weight = None, nn_output=None):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """

        if loss_weight is not None:
            b = self.multiply_loss_mask_in_real_space(b, loss_weight)
            target = self.multiply_loss_mask_in_real_space(target, loss_weight)

        return self.loss_function(b, target)
    
    def multiply_loss_mask_in_real_space(self, image, loss_weight):

        nn_shape = image.shape
        real_component =  image[..., 0:int(0.5*nn_shape[-1])]
        imag_component =  image[..., int(0.5*nn_shape[-1]):nn_shape[-1]]
        
        grid= real_component
        grid_shape = grid[...,0:-1].size()
        ft = FourierTransform2d(grid_shape=grid_shape, dx=self.dataset.dx, dy=self.dataset.dy, real_signal=False)

        loss_weight = self.Padder.pad_zeros2d(loss_weight)

        RS_real_component =  ft.backward(real_component, dim=(-2, -1))

        
        # weight = torch.cat((loss_weight[::,0:-2], loss_weight[::,0:-2]), dim=-1)
        weight = loss_weight[::,0:-1]

        # print(nn_shape, real_component.shape, imag_component.shape)
        # print(RS_real_component.shape, loss_weight.shape, weight.shape)

        RS_real_component = RS_real_component * weight
        real_component = ft.forward(RS_real_component, dim=(-2, -1))

        # RS_imag_component =  ft.backward(imag_component, dim=(-2, -1))
        # RS_imag_component = RS_imag_component * loss_weight[...,0:-1]
        # imag_component = ft.forward(RS_imag_component, dim=(-2, -1))

        image[..., 0:int(0.5*nn_shape[-1])] = real_component
        image[..., int(0.5*nn_shape[-1]):nn_shape[-1]] = imag_component

        # image[..., 0:int(0.5*nn_shape[-1])] = torch.real(complex_form)
        # image[..., int(0.5*nn_shape[-1]):nn_shape[-1]] = torch.imag(complex_form)
        return image
        


    def extract_results(self, final_output, final_b, remove_padding = True):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            results: The results of the neural network
        """
        nn_shape = final_output.shape
        real_component =  final_output[0,0, :, 0:int(0.5*nn_shape[-1])]
        imag_component =  final_output[0,0, :, int(0.5*nn_shape[-1]):nn_shape[-1]]
        complex_jx = torch.complex(real_component, imag_component)

        real_component =  final_output[0,1, :, 0:int(0.5*nn_shape[-1])]
        imag_component =  final_output[0,1, :, int(0.5*nn_shape[-1]):nn_shape[-1]]
        complex_jy = torch.complex(real_component, imag_component)

        b_shape = final_b.shape
        real_component =  final_b[0, :, 0:int(0.5*b_shape[-1])]
        imag_component =  final_b[0, :, int(0.5*b_shape[-1]):b_shape[-1]]
        complex_b = torch.complex(real_component, imag_component)


        self.results = dict()
        self.results["Jx"] = self.ft.backward(complex_jx,  dim=(-2, -1)) / self.scaling_factor
        self.results["Jy"] = self.ft.backward(complex_jy,  dim=(-2, -1)) / self.scaling_factor
        self.results["Recon B"] = self.ft.backward(complex_b,  dim=(-2, -1)) / self.scaling_factor
        self.results["original B"] = self.original_target


        self.results["Jx"] = self.Padder.remove_padding2d(self.results["Jx"])
        self.results["Jy"] = self.Padder.remove_padding2d(self.results["Jy"])
        self.results["Recon B"] = self.Padder.remove_padding2d(self.results["Recon B"])

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

