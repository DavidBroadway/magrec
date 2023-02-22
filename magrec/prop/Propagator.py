"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
from abc import abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.Kernel import HarmonicFunctionComponentsKernel, UniformLayerFactor2d, MagnetizationFourierKernel2d, CurrentFourierKernel2d, CurrentLayerFourierKernel2d, SphericalUnitVectorKernel


# CurrentFourierPropagtor3d
class CurrentFourierPropagator3d(object):
    def __init__(
        self,
        shape,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        height=10.0,
        width=10.0,
        depth=10.0,
        D=1.0,
        rule="trapezoid",
    ):
        """
        CurrentFourierPropagator3d to evaluate the magnetic field from the current density distribution using Fourier transform representation
        of the Biot-Savart law. Evaluates the field at elevation D from the highest point of the current distribution.

        Call it like propagator(J) to get the magnetic field B, where propagator is an instance initialized with the desired parameters.

        Args:
            shape (tuple):      shape of the current field grid (n_x, n_y, n_z) where n_x, n_y, n_z are the number of grid points in x, y, z direction
            dx (float):         current distribution grid spacing in x direction, in [mm]
            dy (float):         current distribution grid spacing in y direction, in [mm]
            height (float):     current distribution grid height, in [mm]
            width (float):      current distribution grid width, in [mm]
            depth (float):      current distribution grid depth, in [mm]
            D (float):          elevation above the current distribution at which to evaluate the magnetic field, in [mm]
            rule (str):         integration rule to use for the Biot-Savart integral, either 'trapezoid' or 'simpson'
        """
        self.device = "cpu"
        self.shape = shape

        # either dx, dy, dz or height, width, depth must be provided together with shape
        if (dx is not None) and (dy is not None) and (dz is not None):
            self.dx = dx
            self.dy = dy
            self.dz = dz
        elif (height is not None) and (width is not None) and (depth is not None):
            self.dx = height / (shape[0] - 1)
            self.dy = width / (shape[1] - 1)
            self.dz = depth / (shape[2] - 1)
        else:
            raise ValueError("Either dx, dy, dz or height, width, depth must be provided together with shape")

        # define a spatial and conjugate grids and a Fourier transform on them for
        # quantities sampled on that grid
        self.ft = FourierTransform2d(
            grid_shape=shape,
            dx=dx,
            dy=dy,
            real_signal=True,
        )

        self.j_to_b_z_matrix = CurrentFourierKernel2d.\
            define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, self.ft.k_matrix)

        # torch.linspace includes endpoint and contains exactly `shape[0]` elements
        self.zs = torch.linspace(0, height, shape[0], device=self.device)
        self.z0 = height + D
        """z-coordinate of the plane where the magnetic field is evaluated"""

        self.exp_matrix = self.get_exp_matrix(self.zs, self.ft.k_matrix, self.z0)

        assert (
            rule == "trapezoid" or rule == "rectangle"
        ), "Rule must be either `trapezoid` or `rectangle`"
        self.rule = rule

        return

    def __call__(self, J):
        return self.get_B_from_J(J)

    def to(self, device: torch.device | str):
        """
        Puts all instance attributes to the specified device if they are torch.Tensors or
        currec.FourierTransform

        Args:
            device (torch.device | str): device on which to put all attributes
        """
        self.device = device

        if isinstance(device, str):
            if torch.cuda.is_available():
                device = torch.device(device)
            else:
                raise ValueError(
                    "Device {} was requested, which is not available.".format(device)
                )

        self.j_to_b_z_matrix = self.j_to_b_z_matrix.to(device)
        self.exp_matrix = self.exp_matrix.to(device)
        self.zs = self.zs.to(device)
        self.ft = self.ft.to(device)

        return self

    @staticmethod
    def get_exp_matrix(zs: torch.Tensor, k_matrix, z0: float = None, D: float = None):
        """
        Returns a matrix of the exponential factors exp(-k [z0 - z']) for each k in k_matrix and each z' in zs

        Args:
            zs:         z-coordinates of the current layers, shape (n_z,)
            k_matrix:   matrix with all possible k = sqrt(k_x ** 2 + k_y ** 2), shape (n_kx, n_ky)
            z0:         z-coordinate of the plane where the magnetic field is measured (z > z' for each z' in zs)
            D:          distance between the plane where the magnetic field is measured and the top layer where current field is
                        specified, i.e. z0 = z.max + D. If z0 is given, D is ignored.

        Returns:
            exp_matrix (torch.Tensor): matrix of the exponential factors exp(-k [z0 - z']) for each k in k_matrix and each z' in zs
        """
        if z0 and D:
            if zs.max() + D != z0:
                raise RuntimeError(
                    "Are you sure? 'Cause you specified both z0 and D, but z0 != zs.max() + D."
                )

        if not z0:
            assert D, "D must be specified if z0 is not"
            z0 = zs.max() + D

        assert z0 > zs.max(), "z0 must be greater than z.max"

        # This einsum actually performs an outer product (tensor product to obtain a higher dimensional tensor) zs[z] ⊗ k_matrix[ij]
        exp_matrix = (
            torch.exp(torch.einsum("z,ij->ijz", -(z0 - zs), k_matrix)) + 0j
        )  # add 0j to make it a complex tensor
        return exp_matrix

    @staticmethod
    def get_b_from_j(
        M: torch.Tensor,
        j: torch.Tensor,
        exp_matrix: torch.Tensor,
        zs: torch.Tensor,
        rule="trapezoid",
    ):
        """
        Calculates the magnetic field b(k_x, k_y, z) from the current field j(k_x, k_y, z), given the transformation matrix M

        Args:
            M:              transformation matrix, shape (3, 3, n_kx, n_ky)
            j:              current field, 2d-Fourier transformed at each z-plane, shape (b_n, 3, n_kx, n_ky, z), where b_n is the number of samples
            exp_matrix:     matrix of exponential factors exp(-(z0 - z) k), shape (n_kx, n_ky, n_z)
            zs:             coordinates of the current layers, shape (n_z,)
            rule:           'trapezoid' or 'rectangle' for integration of the current field contributions, default is 'trapezoid'. See `integrate_b_z_contributions`

        Returns:
            b (torch.Tensor): 2d-Fourier image of the magnetic field, shape (3, n_kx, n_ky, z), b(k_x, k_y, z) for each sample in the batch


        Matrix multiplication:
            I know, indexing sucks. Anticipating named tensors from Pytorch, but they are not ready yet
            https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.refine_names
            For now Einstein notation is used to index the tensor: torch.einsum


        Fourier sign convention in Tetienne et al. (2018) and in torch.fft (
        as well as in numpy.fft):
            Note that in Tetienne et al. this matrix
            is derived using a different sign convention for the Fourier
            transform, namely, in Tetienne et al. F(k) = ∫ f(x) exp(+i k x) dx,
            where F(k) is the Fourier transform of f(x), and k is the wave-number.
            Note the + sign in the exponent. It leads to the matrix with the k_y
            -> -k_y and k_x -> -k_x in the matrix below.

            Here we use the definition of the Fourier transform,

                F(k) = ∫ f(x) exp(-i k x) dx,

            because it is the same used by torch.fft and numpy.fft. Indeed,
            in numpy the definition is with - sign as shown
            here: https://numpy.org/devdocs/reference/routines.fft.html.

            In Pytorch, I have not found a definition of the Fourier transform
            with the - sign, but implicitly I could see that in this
            documentation: https://pytorch.org/docs/1.7.0/generated/torch.stft.html
            and by manually transforming torch.fft.fft(x), where x
            = [0, 1, 2, 3]. The result is X = [6, -2+2j, -2, -2-2j], and X[1]
            = -2+2j implies that at 1st frequency the associated harmonics
            is exp(-2πi m / n), where m is … and n is the number of samples (4).
        """

        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the current field component, i.e. j_x, j_y, j_z
        # k, l — indices of k_x and k_y, respectively
        # z — index along the z-axis
        _b = torch.einsum("ijkl,bjklz->biklz", M, j)

        # Calculates the magnetic field contribution to b(k_x, k_y, z0) per the current field layer j(k_x, k_y, z)
        b_zs = torch.einsum("ijz,bcijz->bcijz", exp_matrix, _b)

        # Performs the integration ∫ exp(-k [z0 - z']) M j dz' according to the rule
        if rule == "trapezoid":
            b = torch.trapezoid(y=b_zs, x=zs, dim=-1)
        elif rule == "rectangle":
            # Multiply the contribution from the current in each k_x and k_y to the magnetic field by the exponential
            # factor and sum along z, assuming each contribution is scaled by dz (lower Riemann sum)
            dzs = zs[1:] - zs[:-1]
            b = torch.einsum("z,bcijz->bcij", dzs, b_zs[:-1])
        else:
            raise ValueError("Unknown integration rule: {}".format(rule))

        return b

    def get_B_from_J(self, J):
        j = self.ft.forward(J, dim=(-2, -1))
        b = self.get_b_from_j(M=self.j_to_b_z_matrix, j=j, exp_matrix=self.exp_matrix, zs=self.zs, rule=self.rule)
        B = self.ft.backward(b, dim=(-2, -1))


class DipoleCurrentFourierPropagator3d(CurrentFourierPropagator3d):
    """
    CurrentFourierPropagator3d for dipole current distributions
    """

    @staticmethod
    def get_cross_product_matrix(vect):
        """
        Produces a matrix A, such that A @ another_vect = another_vect × vect

        Args:
            vect: vector, shape (3,)

        Returns:
            A (torch.Tensor): matrix, shape (3, 3)
        """
        assert vect.shape[0] == 3, "vect must be a 3d vector"
        return torch.tensor(
            [[0, vect[2], -vect[1]], [-vect[2], 0, vect[0]], [vect[1], -vect[0], 0]],
            device=vect.device,
        )

    @staticmethod
    def get_ffm_matrix(source_locations, sensing_locations):
        source_locations = torch.Tensor(source_locations).refine_names(
            "source_component", "x", "y", "z"
        )
        source_locations.flatten(["x", "y", "z"], "source_index").align_to(
            "source_component", "source_index"
        )

        sensing_locations = torch.Tensor(sensing_locations).refine_names(
            "sensing_component", "x", "y", "z"
        )
        sensing_locations.flatten(["x", "y", "z"], "sensing_index").align_to(
            "sensing_component", "sensing_index"
        )

class MagnetizationPropagator2d(object):

    def __init__(self, source_shape, dx, dy, height, layer_thickness):
        """
        Create a propagator for a 2d magnetization distribution that computes the magnetic field at `height` above
        the 2d magnetization layer of finite thickness `layer_thickness`, that has potentially 3 components of the magnetization.

        Assumes uniform magnetization across the layers thickness and uses the integration factor to account for the finite thickness.

        Args:
            source_shape:       shape of the magnetization distribution, shape (3, n_x, n_y)
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
            height:             height above the magnetization layer at which to evaluate the magnetic field, in [mm]
            layer_thickness:    thickness of the magnetization layer, in [mm]
        """
        self.ft = FourierTransform2d(grid_shape=source_shape, dx=dx, dy=dy, real_signal=True)

        k_matrix = self.ft.k_matrix
        self.Filter = None

        self.depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness)

        self.m_to_b_matrix = MagnetizationFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, height, layer_thickness)
        """Forward field matrix that connects sources to the measured field"""

        pass

    def __call__(self, M):
        """Propagates planar magnetization M of shape (batch_size, 3, width, height) to the magnetic field
        this magnetization creates at distance `self.height` from the plane where this magnetization is present.
        """
        return self.B_from_M(M)

    def get_b_from_m(self, m, magnetisation_theta, magnetisation_phi):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # b = torch.einsum("ijkl,bjkl->bikl", self.m_to_b_matrix, m)

        m = torch.tensor(m, dtype=torch.complex64)

        magnetisation_phi = np.deg2rad(magnetisation_phi)
        magnetisation_theta = np.deg2rad(magnetisation_theta)
        magnetisation_direction = torch.tensor([ \
            np.cos(magnetisation_phi)*np.sin(magnetisation_theta), \
            np.sin(magnetisation_phi)*np.sin(magnetisation_theta), \
            np.cos(magnetisation_theta)], dtype=torch.complex64)

        if len(m.shape) == 2:
            m = torch.einsum("kl,j->jkl", m, magnetisation_direction)
            m = torch.tensor(m, dtype=torch.complex64)

        # m_to_b_matrix = self.m_to_b_matrix * magnetisation_direction

        # m_to_b_matrix[:,0] = self.m_to_b_matrix[:,0] * magnetisation_direction[0]
        # m_to_b_matrix[:,1] = self.m_to_b_matrix[:,1] * magnetisation_direction[1]
        # m_to_b_matrix[:,1] = self.m_to_b_matrix[:,2] * magnetisation_direction[2]

        # b = torch.einsum("ijkl,kl->ijkl", m_to_b_matrix, m)

        b = torch.einsum("ijkl,jkl->ikl", self.m_to_b_matrix, m)

        return b


    def B_from_M(self, M, magnetisation_theta, magnetisation_phi):
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M)

        m = self.ft.forward(M, dim=(-2, -1))
        b = self.get_b_from_m(m, magnetisation_theta, magnetisation_phi)
        B = self.ft.backward(b, dim=(-2, -1))
        return B



    def get_m_from_b(self, b, magnetisation_theta, magnetisation_phi, sensor_theta, sensor_phi):
        # Calculate the matrix product M @ j for each k_x, k_y, z
        # b — batch index
        # i — index of the magnetic field component, i.e. b_x, b_y, b_z,
        # j — index of the magnetization distribution component, i.e. m_x, m_y, m_z
        # k, l — indices of k_x and k_y, respectively
        # b = torch.einsum("ijkl,bjkl->bikl", self.m_to_b_matrix, m)

        #b = torch.tensor(b, dtype=torch.complex64)

        magnetisation_phi = np.deg2rad(magnetisation_phi)
        magnetisation_theta = np.deg2rad(magnetisation_theta)
        magnetisation_dir = torch.tensor([ \
            np.cos(magnetisation_phi)*np.sin(magnetisation_theta), \
            np.sin(magnetisation_phi)*np.sin(magnetisation_theta), \
            np.cos(magnetisation_theta)], dtype=torch.complex64)

        # sum over the magnetisation direction
        m_to_b_matrix = torch.einsum("ijkl,i->jkl", self.m_to_b_matrix, magnetisation_dir)

        sensor_phi = np.deg2rad(sensor_phi)
        sensor_theta = np.deg2rad(sensor_theta)
        sensor_dir = torch.tensor([ \
            np.cos(sensor_phi)*np.sin(sensor_theta), \
            np.sin(sensor_phi)*np.sin(sensor_theta), \
            np.cos(sensor_theta)], dtype=torch.complex64)

        # sum over the sensor direction
        m_to_b_matrix = torch.einsum("jkl,j->kl", m_to_b_matrix, sensor_dir)

        # Define the finally transformation
        b_to_m_matrix =  1/ m_to_b_matrix

        # remove the 0 componenet
        b_to_m_matrix[0,0] = 0
        # If there exists any nans set them to zero
        b_to_m_matrix[b_to_m_matrix != b_to_m_matrix] = 0

        # Apply Filter
        if self.Filter is not None:
            b_to_m_matrix = self.Filter*b_to_m_matrix

        m = b * b_to_m_matrix
        m[0,0] = 0 # remove DC componenet
        return m


    def M_from_B(self, B, magnetisation_theta, magnetisation_phi,  sensor_theta, sensor_phi):
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B)

        b = self.ft.forward(B, dim=(-2, -1))
        b[0,0] = 0
        m = self.get_m_from_b(b, magnetisation_theta, magnetisation_phi, sensor_theta, sensor_phi)
        M = self.ft.backward(m, dim=(-2, -1))
        return M

    def add_hanning_filter(self,
            HanningWavelength,
            short_wavelength_cutoff = None,
            long_wavelength_cutoff = None):
        # load the padder class
        Padder = FourierPadder()
        # get the filter.
        self.Filter = Padder.get_hanning(
            self.ft.k_matrix,
            HanningWavelength = HanningWavelength,
            short_wavelength_cutoff = short_wavelength_cutoff,
            long_wavelength_cutoff = long_wavelength_cutoff,
            plot = False)
        return filter



class CurrentPropagator2d(object):
    def __init__(self, source_shape, dx, dy, height, layer_thickness):
        """
        Create a propagator for a 2d current distribution that computes the magnetic field at `height` above
        the 2d current layer of finite thickness `layer_thickness`.

        Assumes uniform current distribution across the layer thickness and uses the integration factor to account for the finite thickness.

        Args:
            source_shape:       shape of the magnetization distribution, shape (3, n_x, n_y)
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
            height:             height above the magnetization layer at which to evaluate the magnetic field, in [mm]
            layer_thickness:    thickness of the magnetization layer, in [mm]
        """
        self.ft = FourierTransform2d(grid_shape=source_shape, dx=dx, dy=dy, real_signal=True)

        self.j_to_b_matrix = CurrentLayerFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, height, layer_thickness)
        """Forward field matrix that connects sources to the measured field"""
        pass

    def __call__(self, J):
        """Propagates planar magnetization M of shape (batch_size, 3, width, height) to the magnetic field
        this magnetization creates at distance `self.height` from the plane where this magnetization is present.
        """
        return self.B_from_J(J)

    def get_b_from_j(self, j):
        b = torch.einsum("...ijkl,...jkl->...ikl", self.j_to_b_matrix, j)
        return b

    def B_from_J(self, J):
        if isinstance(J, np.ndarray):
            J = torch.from_numpy(J)

        j = self.ft.forward(J, dim=(-2, -1))
        b = self.get_b_from_j(j)
        B = self.ft.backward(b, dim=(-2, -1))
        return B


class Padder(object):
    """Class to pad a torch.tensor according to the specified rule. The constructor method takes a
    torch.Tensor as it would appear in the input or a shape.
    """

    def __init__(self, shape=None, tensor=None):
        """Initialize the class with a shape or a tensor. Shape specifies the shape of the last two dimensions
        that are to be expaded, if tensor is provided, its last two dimensions are used to calculate the expected
        expanded shape."""
        if tensor is not None:
            self.shape = tensor.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Either `shape` or `tensor` must be specified.")

        self.expanded_shape = self.shape[:-2] + \
            (self.shape[-2] * 3, self.shape[-1] * 3,)

    @abstractmethod
    def pad(self, tensor):
        """Pad a tensor according to the rule."""
        raise NotImplementedError

    def unpad(self, x):
        """Unpad a tensor according to the rule."""
        W, H = self.shape[-2:]
        return x[..., W-1:2*W-1, H-1:2*H-1]

class HeightContinuationPadder(Padder):

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad a tensor according to the rule."""
        y = torch.zeros(size=self.expanded_shape)
        # take (W, H) size image X and turn it into (3*W, 3*H) obtained by appending 0s to X along W to get 0X0,
        # and repeating the last lines of X along H, which we call H, H times, to get HXH vertically, like so:
        # 0H0
        # 0X0
        # 0H0
        W, H = self.shape[-2:]
        y[..., W:2*W, H:2*H] = x
        # how to propagate shape (W, 1) into the proper (W, H) shape?
        y[..., W:2*W, 2*H:] = x[..., -1][..., None]
        y[..., W:2*W, :H] = x[..., 0][..., None]
        return y

class WidthContinuationPadder(Padder):

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad a tensor according to the rule."""
        y = torch.zeros(size=self.expanded_shape)
        # take (W, H) size image X and turn it into (3*W, 3*H) obtained by appending 0s to X along H to get 0X0,
        # and repeating the last lines of X along W, which we call W, W times, to get WXW horizontally, like so:
        # 000
        # WXW
        # 000
        W, H = self.shape[-2:]
        # TODO: Here is a problem that makes y to be of incorrect shape. y[..., W:2*W, H:2*H] gets implicitly
        # expanded to y[..., x.shape[-3], W:2*W, H:2*H] to fit the shape of x, instead of assigning the values
        # to the other axis of y.
        y[..., W:2*W, H:2*H] = x[..., :, :]
        # how to propagate shape (W, 1) into the proper (W, H) shape?
        y[..., 2*W:, H:2*H] = x[..., 1, :][..., None, :]
        y[..., :W, H:2*H] = x[..., 0, :][..., None, :]
        return y

class ZeroPadder(Padder):

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad a tensor with zeros symmetrically."""
        y = torch.zeros(size=self.expanded_shape)
        # 000
        # 0X0
        # 000
        W, H = self.shape[-2:]
        y[..., W:2*W, H:2*H] = x
        return y


class FourierPadder(object):
    """
    Class to deal with padding tensors before converting them to Fourier space.

    This is necessary because of the important
    assumptions introduced when using `HarmonicFunctionComponentsKernel` to get the connection between different field components,
    and the assumption intrinsic to the Fourier transform that the signal is periodic. Namely, just doing the FFT introduces and
    error when using the representation of the field components connection in the Fourier domain.
    """

    @staticmethod
    def pad_to_next_power_of_2(x):
        """
        Pads the input to the next power of 2 along each dimension.
        """
        return np.pad(x, FourierPadder.get_padding(x), "constant")

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


    def get_hanning(self, k_matrix, HanningWavelength, short_wavelength_cutoff = None, long_wavelength_cutoff = None, plot: bool = False) -> torch.Tensor:
        """
        Pads using numpy. Converts the torch tensor to a numpy array, performs the padding, and then converts back.

        Args:
            x (torch.Tensor):      input tensor

        Returns:
            torch.Tensor:          padded tensor

        """
        han2d = 0.5*(1 + np.cos(k_matrix * HanningWavelength/2 ))
        filter = han2d
        # apply frequency cutoffs
        if short_wavelength_cutoff:
            print(f"Applied a high frequency filter, removing all components smaller than {short_wavelength_cutoff} um")
            high_freq_cutoff = 2* np.pi / short_wavelength_cutoff
            filter[(k_matrix > high_freq_cutoff)] = 0
        if long_wavelength_cutoff:
            print(f"Applied a high frequency filter, removing all components larger than {long_wavelength_cutoff} um")
            low_freq_cutoff = 2* np.pi / long_wavelength_cutoff
            filter[(k_matrix < low_freq_cutoff)] = 0
        return filter

    def apply_hanning(self, x: torch.Tensor, k_matrix, HanningWavelength, short_wavelength_cutoff = None, long_wavelength_cutoff = None, plot: bool = False) -> torch.Tensor:
        """
        Pads using numpy. Converts the torch tensor to a numpy array, performs the padding, and then converts back.

        Args:
            x (torch.Tensor):      input tensor

        Returns:
            torch.Tensor:          padded tensor

        """

        img_filter = self.get_hanning(k_matrix, HanningWavelength, short_wavelength_cutoff = short_wavelength_cutoff, long_wavelength_cutoff = long_wavelength_cutoff , plot=plot)
        x_filtered = x * img_filter

        if plot:
            fig = plt.figure()
            fig.set_figheight(10)

            plt.subplot(5,1,1)
            plt.imshow(torch.rot90(img_filter.real), cmap='bwr')
            plt.title('Filter')
            plt.colorbar()

            plt.subplot(5,1,2)
            plt.imshow(torch.rot90(x.real), cmap='bwr')
            plt.title('array real component')
            plt.colorbar()

            plt.subplot(5,1,3)
            plt.imshow(torch.rot90(x_filtered.real), cmap='bwr')
            plt.title('Filtered array real component')
            plt.colorbar()

            plt.subplot(5,1,4)
            plt.imshow(torch.rot90(x.imag), cmap='bwr')
            plt.title('array imaginary component')
            plt.colorbar()

            plt.subplot(5,1,5)
            plt.imshow(torch.rot90(x_filtered.imag), cmap='bwr')
            plt.title('Filtered array imaginary component')
            plt.colorbar()

        return x_filtered


class AxisProjectionPropagator(object):

    def __init__(self, theta, phi):
        self.n = SphericalUnitVectorKernel.define_unit_vector(theta, phi)

    def project(self, x):
        return torch.einsum('...cij,c->...ij', x, self.n.type(x.type()))
