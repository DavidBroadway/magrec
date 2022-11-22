"""
Module docstring here

"""

__author__ = "Adrien Dubois"

# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from currec.core.Propagator import Propagator

import numpy as np

# Variotional estimator is used to calculate uncertainties on the reconstructed parameters. So far we do not do this.
# @variational_estimator
class GeneratorCNN(nn.Module):
    r"""
    Architecture for 2d → 2d image reconstruction, which learns to reconstruct 2d image from another 2d image.
    """

    def __init__(self, size=2, kernel=5, stride=2, padding=2):
        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super(GeneratorCNN, self).__init__()

        M = size

        self.convi = nn.Conv2d(in_channels=3, out_channels=8 * M, kernel_size=kernel, stride=1, padding=padding)
        # This was in the original architecture given by Adrien, but it is not used anywhere AFAIS
        # self.conv_r0 = nn.Conv2d(3, 8 * M, kernel, 1, padding)
        self.conv1 = nn.Conv2d(8 * M, 8 * M, kernel, stride, padding)
        self.bn1 = nn.BatchNorm2d(8 * M)
        self.conv2 = nn.Conv2d(8 * M, 16 * M, kernel, stride, padding)
        self.bn2 = nn.BatchNorm2d(16 * M)
        self.conv3 = nn.Conv2d(16 * M, 32 * M, kernel, stride, padding)
        self.bn3 = nn.BatchNorm2d(32 * M)
        self.conv4 = nn.Conv2d(32 * M, 64 * M, kernel, stride, padding)
        self.bn4 = nn.BatchNorm2d(64 * M)

        self.conv5 = nn.Conv2d(64 * M, 128 * M, 5, 1, 2)
        self.bn5 = nn.BatchNorm2d(128 * M)

        self.trans1 = nn.ConvTranspose2d(128 * M, 64 * M, kernel, stride, padding, 1)
        self.bn4t = nn.BatchNorm2d(64 * M)
        self.trans2 = nn.ConvTranspose2d(64 * M + 32 * M, 32 * M, kernel, stride, padding, 1)
        self.bn3t = nn.BatchNorm2d(32 * M)
        self.trans3 = nn.ConvTranspose2d(32 * M + 16 * M, 16 * M, kernel, stride, padding, 1)
        self.bn2t = nn.BatchNorm2d(16 * M)
        self.trans4 = nn.ConvTranspose2d(16 * M + 8 * M, 8 * M, kernel, stride, padding, 1)
        self.bn1t = nn.BatchNorm2d(8 * M)
        self.conv6 = nn.Conv2d(8 * M, 3, 5, 1, 2)
        self.conv7 = nn.Conv2d(3, 3, kernel, 1, padding)

    def forward(self, input):

        conv0 = self.convi(input)
        conv0 = F.leaky_relu(conv0, 0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4t(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(self.bn3t(self.trans2(torch.cat([conv3, trans1], dim=1))), 0.2)
        trans3 = F.leaky_relu(self.bn2t(self.trans3(torch.cat([conv2, trans2], dim=1))), 0.2)
        trans4 = F.leaky_relu(self.bn1t(self.trans4(torch.cat([conv1, trans3], dim=1))), 0.2)

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)

        return conv7
    

class GeneratorCNN2d3dLong(nn.Module):
    r"""
    A variant of architecture for 2d → 3d image reconstruction, which learns to reconstruct 3d image of current
    distribution from a 2d magnetic field measurement.
    """

    def __init__(self, depth, size=2, kernel=5, stride=2, padding=2, downsampling_coef=1):
        """
        Create the net that takes an image of magnetic field of size (3, W, H) and reconstructs an image of current
        distribution (3, W * D, H). 3 corresponds to the number of channels in the input and output images, and represents
        the components of B and J fields (e.g. B_x, B_y, B_z)

        The 2d → 3d conversion happens by inflating the input image by repeating it D times along z-axis, which is a 3rd
        axis. This increases the number of parameters by D compared to the 2d → 2d convnet.

        Args:
            depth:              depth D of the resulting 3d current reconstruction
            size:               kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:             kernel size
            stride:             step in which to do the convolution
            padding:            whether to pad input image for convolution and by how much
            downsampling_coef:  int of tuple = (W_in / W_out, H_in / H_out)

        Returns:
            GeneratorCNN2d3d:   the net
        """
        super(GeneratorCNN2d3dLong, self).__init__()

        M = size

        self.depth = depth
        self.scale = int(np.sqrt(downsampling_coef))

        self.convi = nn.Conv2d(in_channels=3, out_channels=8 * M, kernel_size=kernel, stride=1, padding=padding)
        self.conv1 = nn.Conv2d(8 * M, 8 * M, kernel, stride, padding)
        self.bn1 = nn.BatchNorm2d(8 * M)
        self.maxpool1 = nn.MaxPool2d(self.scale, self.scale)
        self.conv2 = nn.Conv2d(8 * M, 16 * M, kernel, stride, padding)
        self.bn2 = nn.BatchNorm2d(16 * M)
        self.maxpool2 = nn.MaxPool2d(self.scale, self.scale)
        self.conv3 = nn.Conv2d(16 * M, 32 * M, kernel, stride, padding)
        self.bn3 = nn.BatchNorm2d(32 * M)
        self.conv4 = nn.Conv2d(32 * M, 64 * M, kernel, stride, padding)
        self.bn4 = nn.BatchNorm2d(64 * M)

        self.conv5 = nn.Conv2d(64 * M, 128 * M, 5, 1, 2)
        self.bn5 = nn.BatchNorm2d(128 * M)

        self.trans1 = nn.ConvTranspose2d(128 * M, 64 * M, kernel, stride, padding, 1)
        self.bn4t = nn.BatchNorm2d(64 * M)
        self.trans2 = nn.ConvTranspose2d(64 * M + 32 * M, 32 * M, kernel, stride, padding, 1)
        self.bn3t = nn.BatchNorm2d(32 * M)
        self.trans3 = nn.ConvTranspose2d(32 * M + 16 * M, 16 * M, kernel, stride, padding, 1)
        self.bn2t = nn.BatchNorm2d(16 * M)
        self.trans4 = nn.ConvTranspose2d(16 * M + 8 * M, 8 * M, kernel, stride, padding, 1)
        self.bn1t = nn.BatchNorm2d(8 * M)

        self.conv6 = nn.Conv2d(8 * M, 3, 5, 1, 2)
        self.conv7 = nn.Conv2d(3, 3, kernel, 1, padding)

    def forward(self, input):

        # Generate an inflated 3d image from the input image by repeating the input D
        # times along a 3rd axis. Takes (N, 3, W, H) and returns (N, 3, W * D, H) tensor
        input = torch.cat([input] * self.depth, dim=-2)

        conv0 = F.leaky_relu(self.convi(input), 0.2)
        # maxpooling downsamples by a factor of
        conv1 = F.leaky_relu(self.maxpool1(self.bn1(self.conv1(conv0))), 0.2)  # (256 × 256 * 16 → 64 × 64 * 16)
        # maxpooling downsamples by a factor of 4 again, to the total of the factor of 16
        conv2 = F.leaky_relu(self.maxpool2(self.bn2(self.conv2(conv1))), 0.2)  # (64 × 64 * 16 → 16 × 16 * 16)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4t(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(self.bn3t(self.trans2(torch.cat([conv3, trans1], dim=1))), 0.2)
        trans3 = F.leaky_relu(self.bn2t(self.trans3(torch.cat([conv2, trans2], dim=1))), 0.2)
        trans4 = F.leaky_relu(self.bn1t(self.trans4(torch.cat([self.maxpool1(conv1), trans3], dim=1))), 0.2)

        conv7 = self.conv6(trans4)

        return conv7


class GeneratorMultipleCNN(nn.Module):
    r"""
    Architecture for 2d → 3d image reconstruction, which learns to reconstruct 3d current image from a 2d magnetic field image.
    """

    def __init__(self, zs, size=2, kernel=5, stride=2, padding=2):
        """

        :param size:
        :param kernel:
        :param stride:
        :param padding:
        """
        super(GeneratorMultipleCNN, self).__init__()

        # for each horizontal layer, create a CNN that will learn to reconstruct
        # the current field at that depth
        self.layers = nn.ModuleList([GeneratorCNN(size, kernel, stride, padding) for z in zs])
        # for layer in self.layers:
        #     layer.type(torch.complex64)
        pass

    def forward(self, x):
        layer_outputs = [layer(x) for layer in self.layers]
        return torch.stack(layer_outputs, dim=2)

