"""This module implements the physical Fourier Transform (FT) by
approximating it with a Discrete Fourier Transform (DFT) and .

Writing a documentation like this was insired by `pypret` package,
which enlightened me with its discussion of DFT and FFT:
https://github.com/ncgeib/pypret/blob/master/pypret/fourier.py.

To quote from `pypret` documentation:

    The following code approximates the continuous Fourier transform (FT) on
    equidistantly spaced grids. While this is usually associated with
    'just doing a fast Fourier transform (FFT)', surprisingly, much can be done
    wrong.

Unsurprisingly, much was done wrong.

FFT Conventions
---------------

The convention for the physical Fourier transform that we use is the following:

.. math::
    g(k) =      ∫ G(x) exp(-ikx) dx

.. math::
    G(x) = 1/2π ∫ g(k) exp(ikx)  dk

where g(k) is called the fourier transform of a function G(x), and G(x) is
the inverse Fourier transform of g(k). k will be called the wave-vector.

Note that this definition is different from both `pypret` choice and the
Tetienne et al. (2018) definition. Luckily, it coincides with most
definitions I have found in physics literature, as well as in `Numpy` and
`PyTorch`, almost. By almost I mean that at least the sign in the exponential
is the same here and in the libraries.

However, whereas I use the wave-vector k, in DFT the cyclic frequency f is
used instead (sometimes ω symbol is used for the same quantity f). The conversion
between k and f is the following:

.. math::
    k = 2πf

Notice that using f makes (1) and (2) almost symmetrical, at least gets rid
of the 1/2π factor. That is g(f) = ∫ G(x) exp(-i2πfx) dx and G(x) = ∫ g(f) exp(i2πfx) df.

Consequences of the choice of the sign convention
-------------------------------------------------

In particular, this sign convention leads to the following result for the derivative:

.. math::
    F[dG(x)/dx] = i k_x g(k)

where there is no “−” in front, contrary to the Tetienne et al. (2018).


DFT in Numpy and PyTorch
------------------------

Now consider the definition of FFT in `Numpy` and `PyTorch`. Given a sequence
x_n (n = 1, ..., N-1) of length N, the FFT of x_n is defined as:

.. math::
     FFT[x_n] -> X_m =     Σ_m x_n e^{-2\pi i n m / N}
    iFFT[X_m] -> x_n = 1/N Σ_k X_m e^{+2\pi i n m / N}

1/N is the normalization factor, and its appearance is controlled by the `norm`
parameter in `Numpy` and `PyTorch`. With `norm='backward'` (default), the backward
direction is scaled by 1/N (i.e. the iFFT has the 1/N factor), as above. When `norm` is
the same between forward and backward calls, the FFT ∘ iFFT pair is identity,
see :doc:`numpy/fft` and :doc:`torch/fft`.
"""

import torch
from magrec.misc.constants import twopi

_norm = "backward"

# TODO: Add cyclic and linear Fourier transform switches, the latter padding the signal
# with zeros obligatory.


class FourierTransform2d(object):
    def __init__(
        self,
        grid_shape: tuple,
        dx: float,
        dy: float,
        real_signal: bool = True,
        type="cyclic",
    ):
        """Creates conjugate grids in the physical and Fourier domains and
        calculates the Fourier transform of a 3d tensor along the first 2 spatial dimensions,
        that is does the transformation (…, x, y, z) → (…, k_x, k_y, z).

        Args:
            grid_shape (tuple):      shape of the sampling grid in the physical domain
            dx (float):              grid spacing in units of length in x-direction
            dy (float):              grid spacing in units of length in y-direction
            real_signal (bool):      whether the signal is real-valued (default True)
                                     if True, Fourier transform is computed only on positive k_y, since
                                     for real signals, FFT[f](k) is the conjugate of FFT[f](-k)
            type (str):              type of the Fourier transform, either `cyclic` or `linear`. DFT is
                                     inherently cyclic, meaning that the signal is considered to be periodic
                                     with the period equal to the grid size. See Yazhdanian et al (2020), p. 3
                                     for the discussion. If the transform is `linear`, the grid is doubled in size
                                     before computing the DFT by padding with zeros.

        """

        self.dx = dx
        self.dy = dy
        self.real_signal = real_signal
        self.type = type

        if type == "linear":
            # change the size of the grid by doubling it in each direction
            grid_shape = tuple(2 * n for n in grid_shape)
            # prepare a padder that would perform the padding the signal with zeros
            # it is required to make the inherently cyclic DFT perform as a linear FT
            padding = tuple()
            for n in grid_shape[::-1]:
                a, remainder = divmod(n, 2)
                padding += (n, n + remainder)
            
            self.pad = torch.nn.ConstantPad2d(padding=padding, value=0)

        self.kx_vector, self.ky_vector = self.define_kx_ky_vectors(
            grid_shape=grid_shape, dx=dx, dy=dy, real_signal=real_signal
        )

        self.k_matrix = self.define_k_matrix(
            kx_vector=self.kx_vector, ky_vector=self.ky_vector
        )

    def to(self, device):
        """
        Puts all instance attributes to the specified device if they are torch.Tensors

        Args:
            device (torch.device | str): device on which to put all attributes
        """
        if isinstance(device, str):
            if torch.cuda.is_available():
                device = torch.device(device)
            else:
                raise ValueError(
                    "Device {} was requested, which is not available.".format(device)
                )

        for attr, name in [
            (self.__getattribute__(a), a) for a in dir(self) if not a.startswith("__")
        ]:
            if isinstance(attr, torch.Tensor):
                attr: torch.Tensor
                attr = attr.to(device=device)
                setattr(self, name, attr)

        return self

    @staticmethod
    def define_kx_ky_vectors(
        grid_shape: tuple = None, dx: float = 1.0, dy: float = 1.0, real_signal=True
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the kx and ky vectors in the Fourier space for a grid with shape
        `shape` and grid spacing `dx` and `dy`, in inverse units of length.

        That is, if dx is in [mm], kx is in [1/mm], etc.

        Cyclic frequencies of DFT are defined as:

            f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

        Where n is the length of the dimension of the input. To obtain the
        wave-vector k, we need to multiply f by 2π.

            k = 2π * [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)

        This correctly enforces the reciprocity condition:

            Δk Δx = 2π / n.

        Args:
            grid_shape:     shape of the grid in the physical domain
            dx:             in [mm]. The sampling length scale in x-direction.
                            The spacing between individual samples of the FFT input.
                            The default assumes unit spacing, dividing that
                            result by the actual spacing gives the result in physical
                            frequency units.
            dy:             in [mm]. Same as above but for y-direction.
            real_signal:    If True, the input is assumed to be real-valued,
                            that means that FFT can contain only n_f // 2 + 1 frequencies
                            in one of the dimensions (last, y-dimension).

        Returns:
            (kx_vector, ky_vector): vectors with spacial wave-vectors in x- and
                                    y-direction, shape (n_kx,) and (n_ky,)

        """
        # the ouput of fftfreq is a list of cyclic frequencies f, we need k instead
        kx_vector = twopi * torch.fft.fftfreq(grid_shape[-2], dx)

        if real_signal:
            ky_vector = twopi * torch.fft.rfftfreq(grid_shape[-1], dy)
        else:
            ky_vector = twopi * torch.fft.fftfreq(grid_shape[-1], dy)

        # return kx_vector +  torch.finfo().eps, ky_vector +  torch.finfo().eps
        return kx_vector, ky_vector

    @staticmethod
    def define_k_matrix(kx_vector, ky_vector):
        return torch.sqrt(kx_vector[:, None] ** 2 + ky_vector[None, :] ** 2)

    def forward(self, x, dim):
        """Computes a 2d-Fourier transform of an at-least-a-3d-tensor along
        -3rd, -2nd dimensions.

        Args:
            x:     torch.Tensor with shape (…, n_x, n_y, n_z), in units A
            dim:   tuple of dimensions along which to compute the Fourier transform, e.g. for x, y
                   dim = (-3, -2)

        Returns:
            torch.Tensor with shape (…, n_kx, n_ky, n_z), in units A * [dx * dy]

            For example, a 2d-Fourier image of a current density field in units
            of [A/mm^2] has units [A] (yes it does!)
        """
        dx = self.dx
        dy = self.dy

        if self.type == "linear":
            a, reminder_a = divmod(x.shape[-1], 2)
            b, reminder_b = divmod(x.shape[-2], 2)
            x = torch.nn.functional.pad(
                x, (a, a + reminder_a, b, b + reminder_b), mode="constant", value=0.0
            )

        if self.real_signal:
            # multiply by dx and dy to get the correct physical units
            return torch.fft.rfft2(x, dim=dim, norm=_norm) * (dx * dy)
        else:
            return torch.fft.fft2(x, dim=dim, norm=_norm) * (dx * dy)

    def backward(self, x, dim):
        """
        Computes an inverse 2d-Fourier transform of an at-least-a-3d-tensor along -3, -2 dimensions.

        Args:
            x:  torch.Tensor with shape (…, n_kx, n_ky, n_z), in units A

        Returns:
            torch.Tensor with shape (…, n_x, n_y, n_z), in units A / [dx * dy]

            For example, an inverse 2d-Fourier image of a Fourier-image of the
            current density field in units of [A] has units [A/mm^2] (as expected)
        """
        dx = self.dx
        dy = self.dy

        if self.real_signal:
            # divide by dx and dy to get the correct physical units
            Y = torch.fft.irfft2(x, dim=dim, norm=_norm) / (dx * dy)
        else:
            Y = torch.fft.ifft2(x, dim=dim, norm=_norm) / (dx * dy)

        # if FFT is linear, then the backward transformation will be a larger space
        # then the original input tensor, so we need to crop it
        if self.type == "linear":
            # guaranteed to be even if Y is obtained from a `linear` FFT
            #                         |
            #                     {   V            }
            a, reminder_a = divmod(Y.shape[-2] // 2, 2)
            b, reminder_b = divmod(Y.shape[-1] // 2, 2)
            Y = Y[..., a : -a - reminder_a, b : -b - reminder_b]

        return Y

    @staticmethod
    def test_fourier_transform2d():
        x = torch.rand(2, 3, 4, 5)
        ft = FourierTransform2d(
            grid_shape=(4, 5),
            dx=1.0,
            dy=1.0,
            real_signal=True,
            type="linear",
        )
        xf = ft.forward(x, dim=(-2, -1))
        if torch.allclose(x, ft.backward(xf, dim=(-2, -1))):
            print("Cyclic Fourier transform is working correctly")
        else:
            raise ValueError("Cyclic Fourier transform is not working correctly")
