"""This module implements padding
"""
# used for base class methods that need to be implemented
from abc import abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt


class Padder(object):
    """
    Class to deal with padding tensors before converting them to Fourier space.

    This is necessary because of the important
    assumptions introduced when using `HarmonicFunctionComponentsKernel` to get the connection between different field components,
    and the assumption intrinsic to the Fourier transform that the signal is periodic. Namely, just doing the FFT introduces and
    error when using the representation of the field components connection in the Fourier domain.
    """

    def crop_data(self, x, ROI):
        """
        Crops the data to the region of interest (ROI).
        """
        x = x.numpy()
        x = x[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        return torch.from_numpy(x)

    def pad_to_next_power_of_2(self, x):
        """
        Pads the input with zeros to the next power of 2 such that the array is 
        now square.
        """
        x = x.numpy()
        rows, cols = x.shape
        original_roi = [0, rows, 0, cols]
        # find the next power of 2
        new_rows = self.next_power_of_two(rows)
        new_cols = self.next_power_of_two(cols)
        new_size = max(new_rows, new_cols)
        # pad the array with zeros
        padded_x = np.pad(x, ((0,new_size - rows),(0,new_size - cols)), mode='constant')
    
        return torch.from_numpy(padded_x), original_roi

    def next_power_of_two(self, x: int) -> int:
        return 1 if x == 0 else 2**(x - 1).bit_length()


    def pad_reflective2d(x: torch.Tensor) -> torch.Tensor:
        """
        Pads the input with the reflection of the input along two dimensions.

        Args:
            x (torch.Tensor):      input tensor
            dim (tuple):           two dimensions along which to pad, for example for a 2d tensor, dim=(0, 1) will pad along the x and y dimensions,
                                   for a higher dimensional tensor, where there are (batch_n, component_n, x_n, y_n, z_n) dimensions, dims=(-3, -2) will pad
                                   along the x and y dimensions, as is expected for the Fourier transform in FourierTransform2d.

        Returns:
            torch.Tensor:          padded tensor

        Notes:
            It works specifically along 2 dimensions, because it is not so trivial to implement reflection in more dimensions, and
            in this library it is not needed, actually.

            TODO: When doing backpropagation, check how the padding gradient is calculated. In principle, it should matter, so I need to check
            math how it is properly done.
        """
        # size along each dimension remains the same unless dimension is in dim, in which case it is doubled for padding
        replication = torch.nn.ReplicationPad1d((0, 1, 0, 1))  # to pad by 1 along x, y dimensions
        height, width = x.shape[-2:]
        reflection = torch.nn.ReflectionPad2d((0, width - 1, 0, height - 1))

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            x = reflection(replication(x))
            x = x.squeeze(0)
        elif len(x.shape) > 2:
            x = reflection(replication(x))

        # a nice bonus: now the tensor size is divisible by 2
        return x


    def pad_zeros2d(x: torch.Tensor) -> torch.Tensor:
        """
        Pads the input with zeros along two dimensions.

        Args:
            x (torch.Tensor):      input tensor

        Returns:
            torch.Tensor:          padded tensor

        """
        height, width = x.shape[-2:]
        zeropad = torch.nn.ZeroPad2d((0, width, 0, height))  # that's the order of padding specificatin: (left, right, top, bottom)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            x = zeropad(x)
            x = x.squeeze(0)
        elif len(x.shape) > 2:
            x = zeropad(x)

        # a nice bonus: now the tensor size is divisible by 2
        return x

    def pad_2d(x: torch.Tensor, pad_width: int, mode: str, plot: bool = False) -> torch.Tensor:
        """
        Pads using numpy. Converts the torch tensor to a numpy array, performs the padding, and then converts back.

        Args:
            x (torch.Tensor):      input tensor

        Returns:
            torch.Tensor:          padded tensor

        """
        npArray = x.numpy()
        paddedArray = np.pad(npArray, pad_width, mode=mode)
        x = torch.from_numpy(paddedArray)

        if plot:
            plt.figure()
            plt.imshow(paddedArray, cmap='bwr')
            plt.title('Padded array')
            plt.colorbar()
        return x


