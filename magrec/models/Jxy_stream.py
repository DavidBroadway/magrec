

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch.nn.functional import conv1d, conv2d, conv3d
from torch.distributions import Normal, Cauchy

from magrec.models.generic_model import GenericModel
from magrec.transformation.Jxy2Bsensor import Jxy2Bsensor 
from magrec.image_processing.Filtering import DataFiltering


class Jxy(GenericModel):
    def __init__(self, 
                 dataset : object, 
                 loss_type : str = "MSE", 
                 scaling_factor: float = 1, 
                 std_loss_scaling : float = 0, 
                 loss_weight: torch.Tensor = None,
                 source_weight: torch.Tensor = None,
                 spatial_filter: bool = False,
                 spatial_filter_type: str = "Gaussian",
                 spatial_filter_kernal_size: int = 3,
                 spatial_filter_width: list = [0.5, 0.5]):
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
            spatial_filter_kernal_size: The multiplication factor that defines the size of the spatial filter kernal 
            spatial_filter: Whether to apply a spatial filter to the output of the network.
            spatial_filter_width: The width of the spatial filter.
        """


        # Define the transformation so that this isn't performed during a loop.
        self.magClass = Jxy2Bsensor(dataset)
        self.std_loss_scaling = std_loss_scaling
        self.loss_weight = loss_weight
        self.source_weight = source_weight
        self.spatial_filter = spatial_filter
        self.spatial_filter_type = spatial_filter_type
        self.spatial_filter_width = spatial_filter_width
        self.spatial_filter_kernal_size = spatial_filter_kernal_size
        self.requirements()

        self.Filtering = DataFiltering(dataset.target, dataset.dx, dataset.dy)



        if self.spatial_filter:
            '''
            Define the spatial filter to be used.
            Currently there are three options:
                - Gaussian
                - Lorentzian
                - Hanning
            The Hanning filter is a simple windowing function that is applied to the output of the network, 
            this does not require a kernal and is not implemented through the self.blurrer class..
            '''

            # define the sigma values for the spatial filter
            sigma_x = self.spatial_filter_width[0]/ self.dataset.dy
            sigma_y = self.spatial_filter_width[1]/ self.dataset.dx
            # define the kernal size based off the spatial filter width, 
            # The default value of 3 is a good approximation and doesn't add significant computational time. 
            # This value can be increased if you see artifacts in the output of the network.
            kernal_size = max(sigma_x, sigma_y)*self.spatial_filter_kernal_size
            # make the kernal size odd
            if kernal_size % 2 == 0:
                kernal_size += 1
            if spatial_filter_type == "Gaussian":
                self.blurrer = GuassianBlur(kernel_size=kernal_size, sigma_x=sigma_x, sigma_y=sigma_y)
                print("Using a Gaussian filter")
            elif spatial_filter_type == "Lorentzian":
                self.blurrer = LorentzianBlur(kernel_size=kernal_size, sigma_x=sigma_x, sigma_y=sigma_y)
                print("Using a Lorentzian filter")
            

            print(f"Spatial filter implemented into the model with a width of"
                  + f" {self.spatial_filter_width[0]:0.2f} and {self.spatial_filter_width[1]:0.2f} pixels"
                  + f" or {self.spatial_filter_width[0]*self.dataset.dx:0.2f} um.")


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
            if self.spatial_filter_type == "Hanning":                
                nn_output = self.Filtering.apply_hanning_filter(self.spatial_filter_width[0], data=nn_output, plot_results=False)
                nn_output = self.Filtering.apply_short_wavelength_filter(self.spatial_filter_width[0], data=nn_output, plot_results=False, print_action=False)
            else: 
                nn_output = self.blurrer(nn_output)


        # Calculate the diveregence of the output of the NN
        sp = [self.dataset.dx, self.dataset.dy]
        nn_output[0,1,::] = torch.gradient(nn_output[0,0,::], spacing = sp[0], dim = 0)[0]
        nn_output[0,0,::] = -torch.gradient(nn_output[0,0,::], spacing = sp[1], dim = 1)[0]


        return self.magClass.transform(nn_output), nn_output

    def calculate_loss(self, b, target, nn_output = None,):
        """
        Args:
            nn_output: The output of the neural network
            target: The target magnetic field

        Returns:
            loss: The loss function
        """

        if self.loss_weight is not None:
            b = torch.einsum("...kl,kl->...kl", b, self.loss_weight)
            target = torch.einsum("...kl,kl->...kl", target, self.loss_weight)


        return self.loss_function(b, target) 

    def extract_results(self, final_output,  final_Jxy, final_b, remove_padding = True,  additional_roi=None):
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
            self.remove_padding_from_results(additional_roi= additional_roi)

        return self.results

    def plot_results(self, results, x_positions = None, y_positions = None,):  
        """
        Args:
            results: A dictionary containing the following keys:
                "original B": The original magnetic field
                "Recon B": The reconstructed magnetic field
                "Jx": The x-component of the current density
                "Jy": The y-component of the current density
                "J": The magnitude of the current density
                "divJ": The divergence of the current density

        Returns:
            None
        """

        dx = self.dataset.dx
        dy = self.dataset.dy
        x_size = results["original B"].shape[0]
        y_size = results["original B"].shape[1]
        real_x = np.linspace(-dx*x_size/2, dx*x_size/2, x_size)
        real_y = np.linspace(-dy*y_size/2, dy*y_size/2, y_size)
        extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]

        fig = plt.figure(figsize=(10, 8))

        plt.subplot(3, 3, 1)
        plot_data = results["original B"].T * 1e6
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range, extent=extent)
        # plt.xticks([])
        # plt.yticks([])
        plt.title('original B')
        plt.ylabel("y ($\mu$m)")
        plt.xlabel("x ($\mu$m)")
        cb = plt.colorbar()
        cb.set_label("B ($\mu$T)")

        plt.subplot(3, 3, 2)
        plot_data = results["Recon B"].T * 1e6
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range, extent=extent)
        # plt.xticks([])
        # plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed B')
        cb.set_label("B ($\mu$T)")

        plt.subplot(3, 3, 3)
        plot_data = (results["original B"] - results["Recon B"]).T* 1e6
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap='bwr', vmin=-plot_range, vmax=plot_range, extent=extent)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        plt.title('reconstructed difference')
        cb.set_label("B ($\mu$T)")

        plt.subplot(3, 3, 4)
        plot_data = results["Jx"].T
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range, extent=extent)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("Jx (A/m)")

        plt.subplot(3, 3, 5)
        plot_data = results["Jy"].T
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range, extent=extent)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("Jy (A/m)")

        plt.subplot(3, 3, 6)
        plot_data = results["J"].T
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="viridis", vmin=0, vmax=plot_range, extent=extent)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("J (A/m)")

        plt.subplot(3, 3, 8)
        plot_data = results["divJ"].T
        plot_range = abs(plot_data).max()
        plt.imshow(plot_data, cmap="PuOr", vmin=-plot_range, vmax=plot_range, extent=extent)
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label("div(J)")

        plt.show()

    def divergence(self, fx, fy,sp):
        """ Computes divergence of vector field 
        f: array -> vector field components [Fx,Fy,Fz,...]
        sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
        """
        return torch.gradient(fx, spacing = sp[0], dim = 0)[0] + torch.gradient(fy, spacing = sp[1], dim = 1)[0]
    

class LorentzianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_x: float, sigma_y: float):
        super(LorentzianBlur, self).__init__()
        """
        Class for applying a Lorentzian blur to a 2D image. The kernel is generated using the formula for a 2D Lorentzian
        function and then applied using a convolution.
        Args:
            kernel_size (int): The size of the kernel to use
            sigma_x (float): The standard deviation of the kernel in the x direction
            sigma_y (float): The standard deviation of the kernel in the y direction
        """
        self.kernel_size = kernel_size
        self.kernel = self.get_2d_lorentzian_kernel(kernel_size, sigma_x=sigma_x, sigma_y=sigma_y)

    def forward(self, input_tensor):
        """
        Applies the Lorentzian blur to the input tensor.
        Args:
            input_tensor (torch.Tensor): The input tensor to apply the blur to
        Returns:
            torch.Tensor: The tensor after the Lorentzian blur has been applied
        """
        batch_size, channels, height, width = input_tensor.size()
        kernel = self.kernel.unsqueeze(0).repeat(channels, 1, 1, 1).to(input_tensor.device)
        return torch.nn.functional.conv2d(input_tensor, weight=kernel, groups=channels, stride=1, padding="same")

    def get_2d_lorentzian_kernel(self, size, sigma_x: float, sigma_y: float):
        """
        Returns a 2D Lorentzian kernel with the given standard deviation in the x and y directions
        (sigma_x and sigma_y) and size (size x size).

        Args:
            sigma_x (float): standard deviation in the x direction
            sigma_y (float): standard deviation in the y direction
            size (int): size of the kernel (must be odd)

        Returns:
            torch.Tensor: a 2D tensor representing the Lorentzian kernel
        """
        if size % 2 == 0:
            raise ValueError("Size must be odd")

        center = size // 2
        x, y = torch.meshgrid(torch.arange(size), torch.arange(size))
        x = x - center
        y = y - center
        kernel = 1 / (1 + (x**2 / sigma_x**2 + y**2 / sigma_y**2))
        kernel /= kernel.sum()

        return kernel
    

class GuassianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma_x: float, sigma_y: float):
        super(GuassianBlur, self).__init__()
        """
        Class for applying a Lorentzian blur to a 2D image. The kernel is generated using the formula for a 2D Gaussian
        function and then applied using a convolution.
        Args:
            kernel_size (int): The size of the kernel to use
            sigma_x (float): The standard deviation of the kernel in the x direction
            sigma_y (float): The standard deviation of the kernel in the y direction
        """
        self.kernel_size = kernel_size
        self.kernel = self.get_2d_gaussian_kernel(kernel_size, sigma_x=sigma_x, sigma_y=sigma_y)

    def forward(self, input_tensor):
        """
        Applies the Lorentzian blur to the input tensor.
        Args:
            input_tensor (torch.Tensor): The input tensor to apply the blur to
        Returns:
            torch.Tensor: The tensor after the Lorentzian blur has been applied
        """
        batch_size, channels, height, width = input_tensor.size()
        kernel = self.kernel.unsqueeze(0).repeat(channels, 1, 1, 1).to(input_tensor.device)
        return torch.nn.functional.conv2d(input_tensor, weight=kernel, groups=channels, stride=1, padding="same")

    def get_2d_gaussian_kernel(self, size, sigma_x: float, sigma_y: float):
        """
        Returns a 2D Gaussian kernel with the given standard deviation in the x and y directions
        (sigma_x and sigma_y) and size (size x size).

        Args:
            sigma_x (float): standard deviation in the x direction
            sigma_y (float): standard deviation in the y direction
            size (int): size of the kernel (must be odd)

        Returns:
            torch.Tensor: a 2D tensor representing the Gaussian kernel
        """
        if size % 2 == 0:
            raise ValueError("Size must be odd")

        center = size // 2
        x, y = torch.meshgrid(torch.arange(size), torch.arange(size))
        x = x - center
        y = y - center
        kernel = torch.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
        kernel /= kernel.sum()

        return kernel
