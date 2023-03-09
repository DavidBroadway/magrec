"""
Module docstring here

"""

__author__ = "Adrien Dubois"

# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Variotional estimator is used to calculate uncertainties on the reconstructed parameters. So far we do not do this.
# @variational_estimator
class GeneratorCNN(nn.Module):
    r"""
    Architecture for 2d → 2d image reconstruction, which learns to reconstruct 2d image from another 2d image.
    """

    def __init__(
        self, n_channels_in=3, n_channels_out=3, size=2, kernel=5, stride=2, padding=2
    ):
        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            n_channels_in:  number of channels in input image (number of components)
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super().__init__()

        M = size

        self.convi = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=8 * M,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        )
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
        self.trans2 = nn.ConvTranspose2d(
            64 * M + 32 * M, 32 * M, kernel, stride, padding, 1
        )
        self.bn3t = nn.BatchNorm2d(32 * M)
        self.trans3 = nn.ConvTranspose2d(
            32 * M + 16 * M, 16 * M, kernel, stride, padding, 1
        )
        self.bn2t = nn.BatchNorm2d(16 * M)
        self.trans4 = nn.ConvTranspose2d(
            16 * M + 8 * M, 8 * M, kernel, stride, padding, 1
        )
        self.bn1t = nn.BatchNorm2d(8 * M)
        self.conv6 = nn.Conv2d(8 * M, n_channels_out, 5, 1, 2)
        self.conv7 = nn.Conv2d(n_channels_out, n_channels_out, kernel, 1, padding)

    def forward(self, input):

        conv0 = self.convi(input)
        conv0 = F.leaky_relu(conv0, 0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4t(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(
            self.bn3t(self.trans2(torch.cat([conv3, trans1], dim=1))), 0.2
        )
        trans3 = F.leaky_relu(
            self.bn2t(self.trans3(torch.cat([conv2, trans2], dim=1))), 0.2
        )
        trans4 = F.leaky_relu(
            self.bn1t(self.trans4(torch.cat([conv1, trans3], dim=1))), 0.2
        )

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)

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
        self.layers = nn.ModuleList(
            [GeneratorCNN(size, kernel, stride, padding) for z in zs]
        )
        pass

    def forward(self, x):
        layer_outputs = [layer(x) for layer in self.layers]
        return torch.stack(layer_outputs, dim=2)


class UNet(nn.Module):
    r"""
    Architecture for 2d → 2d image reconstruction, which learns to reconstruct 2d image from another 2d image.
    """

    def __init__(
        self, n_channels_in=3, n_channels_out=3, size=2, kernel=5, stride=2, padding=2
    ):
        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            n_channels_in:  number of channels in input image (number of components)
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super().__init__()

        M = size

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        self.convi = nn.Conv2d(
            in_channels=n_channels_in,
            out_channels=8 * M,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        )
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
        self.trans2 = nn.ConvTranspose2d(64 * M, 32 * M, kernel, stride, padding, 1)
        self.bn3t = nn.BatchNorm2d(32 * M)
        self.trans3 = nn.ConvTranspose2d(32 * M, 16 * M, kernel, stride, padding, 1)
        self.bn2t = nn.BatchNorm2d(16 * M)
        self.trans4 = nn.ConvTranspose2d(16 * M, 8 * M, kernel, stride, padding, 1)
        self.bn1t = nn.BatchNorm2d(8 * M)
        self.conv6 = nn.Conv2d(8 * M, n_channels_out, 5, 1, 2)
        self.conv7 = nn.Conv2d(n_channels_out, n_channels_out, kernel, 1, padding)

    def forward(self, input):

        conv0 = self.convi(input)
        conv0 = F.leaky_relu(conv0, 0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4t(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(self.bn3t(self.trans2(trans1)), 0.2)
        trans3 = F.leaky_relu(self.bn2t(self.trans3(trans2)), 0.2)
        trans4 = F.leaky_relu(self.bn1t(self.trans4(trans3)), 0.2)

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)

        return conv7


class BnCNN(torch.nn.Module):
    def __init__(
        self,
        Size=1,
        ImageSize=256,
        kernel=5,
        stride=2,
        padding=2,
        n_channels_in=1,
        n_channels_out=1,
    ):

        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            n_channels_in:  number of channels in input image (number of components)
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super().__init__()

        M = Size

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        if ImageSize == 512:
            ConvolutionSize = 32
        elif ImageSize == 256:
            ConvolutionSize = 16
        else:  # size is 128
            ConvolutionSize = 8
        # first index is the number of channels
        self.convi = nn.Conv2d(n_channels_in, 8 * M, kernel, 1, padding)
        self.conv_r0 = nn.Conv2d(1, 8 * M, kernel, 1, padding)
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
        self.trans2 = nn.ConvTranspose2d(
            64 * M + 32 * M, 32 * M, kernel, stride, padding, 1
        )
        self.trans3 = nn.ConvTranspose2d(
            32 * M + 16 * M, 16 * M, kernel, stride, padding, 1
        )
        self.trans4 = nn.ConvTranspose2d(
            16 * M + 8 * M, 8 * M, kernel, stride, padding, 1
        )
        self.conv6 = nn.Conv2d(8 * M, n_channels_out, kernel, 1, padding)
        self.conv7 = nn.Conv2d(n_channels_out, n_channels_out, kernel, 1, padding)

        self.fc11 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.fc12 = nn.Linear(120, 84)
        self.fc13 = nn.Linear(84, 1)

        self.fc21 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.fc22 = nn.Linear(120, 84)
        self.fc23 = nn.Linear(84, 1)

        self.fc31 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.fc32 = nn.Linear(120, 84)
        self.fc33 = nn.Linear(84, 1)

        self.fc41 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.fc42 = nn.Linear(120, 84)
        self.fc43 = nn.Linear(84, 1)

        self.fc51 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.fc52 = nn.Linear(120, 84)
        self.fc53 = nn.Linear(84, 1)

        self.transfc1 = nn.Linear(64 * M * ConvolutionSize * ConvolutionSize, 120)
        self.transfc2 = nn.Linear(120, 256)
        self.transfc3 = nn.Linear(256, 65536)

    def forward(self, input):

        conv0 = self.convi(input)
        conv0 = F.leaky_relu(conv0, 0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(
            self.bn3(self.trans2(torch.cat([conv3, trans1], dim=1))), 0.2
        )
        trans3 = F.leaky_relu(
            self.bn2(self.trans3(torch.cat([conv2, trans2], dim=1))), 0.2
        )
        trans4 = F.leaky_relu(
            self.bn1(self.trans4(torch.cat([conv1, trans3], dim=1))), 0.2
        )

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)

        return conv7


class FCCNN(torch.nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.conv1 = nn.Conv2d(
            in_channels=n_channels_in, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=n_channels_out, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=n_channels_out,
            out_channels=n_channels_out,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class GioCNN(torch.nn.Module):
    def __init__(
        self,
        kernel=5,
        stride=2,
        padding=2,
        n_channels_in=1,
        n_channels_out=1,
        size_out=None,
    ):

        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            n_channels_in:  number of channels in input image (number of components)
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super().__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.size_out = size_out

        self.relu = nn.LeakyReLU(0.2)

        self.convi = nn.Conv2d(n_channels_in, 8, kernel, 1, padding)
        self.conv1 = nn.Conv2d(8, 8, kernel, stride, padding)
        self.conv2 = nn.Conv2d(8, 16, kernel, stride, padding)
        self.conv3 = nn.Conv2d(16, 32, kernel, stride, padding)
        self.conv4 = nn.Conv2d(32, 64, kernel, stride, padding)

        self.conv5 = nn.Conv2d(64, 128, 5, 1, 2)

        self.trans1 = nn.ConvTranspose2d(128, 64, kernel, stride, padding, 1)
        self.trans2 = nn.ConvTranspose2d(64, 32, kernel, stride, padding, 1)
        self.trans3 = nn.ConvTranspose2d(32, 16, kernel, stride, padding, 1)
        self.trans4 = nn.ConvTranspose2d(16, 8, kernel, stride, padding, 1)
        self.conv6 = nn.Conv2d(8, n_channels_out, kernel, 1, padding)
        self.conv7 = nn.Conv2d(n_channels_out, n_channels_out, kernel, 1, padding)

    def forward(self, x):

        conv0 = self.convi(x)
        conv0 = self.relu(conv0)
        conv1 = self.relu(self.conv1(conv0))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        conv4 = self.relu(self.conv4(conv3))

        conv5 = self.relu(self.conv5(conv4))

        trans1 = self.relu(self.trans1(conv5))
        trans2 = self.relu(self.trans2(trans1))
        trans3 = self.relu(self.trans3(trans2))
        trans4 = self.relu(self.trans4(trans3))

        conv6 = self.conv6(trans4)
        out = self.conv7(conv6)

        # force to be of `self.size_out` if it is not None
        if self.size_out:
            out = F.interpolate(out, size=self.size_out, mode="bicubic")

        return out


# Classes Block, Encoder, Decoder and UNet are taken from https://amaarora.github.io/posts/2020-09-13-unet.html
# and from communication with Giovanni P.
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class GioUNet(nn.Module):
    def __init__(
        self,
        n_channels_in=1,
        n_channels_out=2,
        enc_chs=(16, 32, 64),
        dec_chs=(64, 32, 16),
        size_out=None,
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        enc_chs = (n_channels_in,) + enc_chs
        dec_chs = dec_chs
        self.encoder = Encoder(chs=enc_chs)
        self.decoder = Decoder(chs=dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], n_channels_out, 1)
        self.size_out = size_out

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.size_out:
            out = F.interpolate(out, self.size_out, mode="bicubic")
        return out
