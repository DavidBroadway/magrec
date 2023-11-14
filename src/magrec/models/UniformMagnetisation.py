

import torch
import matplotlib.pyplot as plt
from magrec.models.generic_model import GenericModel
from magrec.transformation.Mxy2Bsensor import Mxy2Bsensor
from magrec.transformation.Fourier import FourierTransform2d
from magrec.image_processing.Filtering import DataFiltering

class UniformMagnetisation(GenericModel):
    def __init__(self,  
                dataset : object, 
                loss_type : str = "MSE",
                positive_magnetisation : bool = False, 
                m_theta : float = 0,
                fit_m_theta: bool = False, 
                m_phi: float = 0,
                fit_m_phi: bool = False,
                scaling_factor: float = 1, 
                std_loss_scaling : float = 0, 
                loss_weight: torch.Tensor = None,
                source_weight: torch.Tensor = None,
                spatial_filter: bool = False,
                spatial_filter_type: str = "Gaussian",
                spatial_filter_kernal_size: int = 3,
                spatial_filter_width: float = 0.5):
        super().__init__(dataset, loss_type, scaling_factor)

        # Define the propagator so that this isn't performed during a loop.
        self.magClass = Mxy2Bsensor(dataset, m_theta = m_theta, m_phi = m_phi)
        self.std_loss_scaling = std_loss_scaling
        self.loss_weight = loss_weight
        self.source_weight = source_weight
        self.spatial_filter = spatial_filter
        self.spatial_filter_width = spatial_filter_width
        self.spatial_filter_type = spatial_filter_type
        self.spatial_filter_kernal_size = spatial_filter_kernal_size

        self.positive_magnetisation = positive_magnetisation

        self.m_theta = m_theta
        self.m_phi = m_phi
        self.fit_m_theta = fit_m_theta
        self.fit_m_phi = fit_m_phi


        self.Filtering = DataFiltering(dataset.target, dataset.dx, dataset.dy)

        self.ft = FourierTransform2d(
            grid_shape=self.dataset.target.shape,
            dx=self.dataset.dx,
            dy=self.dataset.dy,
            real_signal=True,
                )

        # define the requirements for the model that may change the fitting method
        self.requirements()

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
                  + f" {sigma_x:0.2f} and {sigma_y:0.2f} pixels"
                  + f" or {self.spatial_filter_width[0]:0.3f} um.")
        
        

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
        if self.fit_m_theta or self.fit_m_phi:
            self.require["source_angles"] = True
        else:
            self.require["source_angles"] = False

    def transform(self, nn_output, m_theta = None, m_phi = None):
        if self.fit_m_theta:
            self.m_theta = m_theta
        if self.fit_m_phi:
            self.m_phi = m_phi

        if self.m_theta or self.m_phi:
            self.magClass = Mxy2Bsensor(self.dataset, m_theta = self.m_theta, m_phi = self.m_phi)

        # Apply a spatial filter to the output of the NN
        if self.spatial_filter:
            if self.spatial_filter_type == "Hanning":                
                nn_output = self.Filtering.apply_hanning_filter(self.spatial_filter_width[0], data=nn_output, plot_results=False)
                nn_output = self.Filtering.apply_short_wavelength_filter(self.spatial_filter_width[0], data=nn_output, plot_results=False, print_action=False)
            else: 
                nn_output = self.blurrer(nn_output)

        # Apply the weight matrix to the output of the NN
        if self.source_weight is not None:
            nn_output = nn_output*self.source_weight

        # if requested apply a positive magnetisation constraint
        if self.positive_magnetisation:
            nn_output = nn_output.abs()

        return self.magClass.transform(M = nn_output), nn_output

    def calculate_loss(self, b, target,  nn_output = None):
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