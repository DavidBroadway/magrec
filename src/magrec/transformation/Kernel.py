import torch
import numpy as np

from magrec.transformation.Fourier import FourierTransform2d
from magrec.misc.constants import MU0, twopi, convert_AperM2_to_ubpernm2
from magrec.image_processing.Filtering import DataFiltering


class CurrentLayerFourierKernel2d(object):

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness, dx, dy):
        """Defines a transformation matrix that connects a 2d current distribution that has 2 components of
        the current density to the magnetic field it creates, that has 3 components.
        ```
                                            ┌─                 ─┐
                                            │     0        1    │ ┌─   ─┐
                            μ0              │                   │ │ j_x │
           b(k_x, k_y, z) = -- depth_factor │    -1        0    │ │     │
                             2              │                   │ │ j_y │
                                            │-ik_y / k  ik_x / k│ └─   ─┘
                                            └─                 ─┘ └──┬──┘
                           └───────── M[:, :, k_x, k_y] ────────┘    │
                                                                 j[k_x, k_y]

        ```
        where `depth_factor` is a factor that accounts for the fact that the current layer has finite thickness and
        is at `height` standoff distance from the observation plane. The factor is defined in `UniformLayerFactor2d`.

        Example:

            .. code-block:: python
            M = CurrentLayerFourierKernel2d.define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness)
            b = torch.einsum('ijkl,jkl->ikl', M, j)
            # i — component of the magnetic field
            # j — component of the current density
            # k — spatial frequency in x direction
            # l — spatial frequency in y direction

        """
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)
        # components of ─┐  ┌─ components of
        # magnetic field │  │  current density
        #                V  V
        M = torch.zeros((3, 2,) + k_matrix.shape, dtype=torch.complex64,)

        M[0, 1, :, :] =  torch.ones_like(k_matrix)
        M[1, 0, :, :] = -torch.ones_like(k_matrix)

        M[2, 0, :, :] = -1j * ky_vector[None, :] / k_matrix
        M[2, 1, :, :] = 1j * kx_vector[:, None] / k_matrix

        # Deal with the case where k = 0 by setting the corresponding elements to 0
        M[[2, 2], [0, 1], [0, 0], [0, 0]] = 0

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness, dx, dy, add_filter=False)

        M = (MU0 / 2) / depth_factor * M
        
        return M 


class MagnetizationFourierKernel2d(object):

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness, dx, dy, add_filter: bool = True):
        """Defines a transformation matrix that connects a 2d magnetization distribution that has 3 components of
        the magnetization to the magnetic field it creates.
        """
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)

        _M = torch.zeros((3, 3,) + k_matrix.shape, dtype=torch.complex64,)

        # _M[0, 1, :, :] = kx_vector[:, None] * ky_vector[None, :] / k_matrix/2
        # _M[0, 2, :, :] = - 1j * kx_vector[:, None] / k_matrix /2
        # _M[1, 2, :, :] = - 1j * ky_vector[None, :] / k_matrix /2

        # _M[0, 0, :, :] = kx_vector[:, None] ** 2 / k_matrix /2
        # _M[1, 1, :, :] = ky_vector[None, :] ** 2 / k_matrix /2
        # _M[2, 2, :, :] = k_matrix /2

        # # Use the property of the M matrix that it is symmetric
        # # divide by 2 to get the proper quantity when _M + _M.T
        # M = _M + _M.transpose(0, 1)

        _M[0, 1, :, :] = kx_vector[:, None] * ky_vector[None, :] / k_matrix
        _M[0, 2, :, :] = 1j * kx_vector[:, None] 
        _M[1, 2, :, :] = 1j * ky_vector[None, :]  

        _M[1, 0, :, :] = kx_vector[:, None] * ky_vector[None, :] / k_matrix
        _M[2, 0, :, :] = 1j * kx_vector[:, None] 
        _M[2, 1, :, :] = 1j * ky_vector[None, :]  

        _M[0, 0, :, :] = kx_vector[:, None] ** 2 / k_matrix 
        _M[1, 1, :, :] = ky_vector[None, :] ** 2 / k_matrix 
        _M[2, 2, :, :] = -k_matrix 

        M = _M

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness, dx, dy, add_filter)
            
        M = (MU0 / 2) * depth_factor * M

        # Set the components of the kernel matrix to zero for k = 0, where the denominator is zero
        M[:, 0, 0] = 0
        # If there exists any nans set them to zero
        M[M != M] = 0

        return M


class UniformLayerFactor2d(object):
    """
    Defines a factor that appears after integration of a uniform distribution along the z axis.
    """

    @staticmethod
    def define_depth_factor(k_matrix: torch.Tensor, height: float, layer_thickness: float, dx: float, dy: float, add_filter: bool = True):
        """
        Returns a matrix that scales each k-vector by the factor that appears after integration of a uniform source distribution.

        Args:
            k_matrix:                   matrix with all possible k = sqrt(k_x ** 2 + k_y ** 2), shape (n_kx, n_ky)
            height (float):             height above the layer at which to evaluate the factor
            layer_thickness (float):    thickness of the layer
        """

        # TODO: Check when this condition is satisfied, currently layer_thickness = 0 case gives proper field, but the
        # expression below seems wrong.
        #
        # `layer_thickness` parameters does not play role and can be set to 0 in the limit of k * thickness ≫ 1,
        # in which case the factor is just -exp(-k * height) / k. Since that must be true for the smallest k which
        # is of order 1/L, the neccessary and satisfactory condition is that the L ≫ thickness, where L is the window size
        if layer_thickness == 0:
            depth_factor = (
                    torch.exp(k_matrix * height))
            depth_factor[0, 0] = 0
        else:
            depth_factor = -(
                    torch.exp(k_matrix * height) 
                    * (torch.exp(k_matrix * layer_thickness) - 1)
            )
            depth_factor[0, 0] = layer_thickness
        
        if add_filter:
            Filtering = DataFiltering(depth_factor, dx, dy)
            
            wavelength = np.abs(height + layer_thickness)
            # add a hanning filter to the depth factor
            depth_factor = Filtering.apply_hanning_filter(wavelength, data=depth_factor, plot_results=False, in_fourier_space=True)
            depth_factor = Filtering.apply_short_wavelength_filter(wavelength, data=depth_factor, plot_results=False,  in_fourier_space=True) 

        return depth_factor


class HarmonicFunctionComponentsKernel(object):
    """
    Defines a transform that maps a single component of the magnetic field, usually
    denoted by B_NV, to the three components of the magnetic field in the Cartesian coordinate system,
    if B_NV is a component defined along a unit vector n, given by the spherical angles `theta` and `phi`.

    It works for components of any harmonic function in the source-free region, not only the magnetic field,
    see Lima, Weiss (2009) and Casola, van der Sar, Yacoby (2018) [Box 1, eq. 1] for details.

    Let :math:`n` be a unit vector along the NV axis defined by the polar coordinates (theta, phi) on the unit sphere.

    .. math::
        n = [sin(θ) cos(φ), sin(θ) sin(φ), cos(θ)]

    where we use the convention as the one used in physics (ISO 80000-2:2019): polar angle θ (theta) (angle with
    respect to polar axis, z axis), and azimuthal angle φ (phi) (angle of rotation from the initial meridian plane,
    counterclockwise from x if looking from the top of z).

    Let :math:`u` be a vector in 2d Fourier space that represent the Hamilton operator ∇,

    .. math::
        u = [k_x, k_y, i k]

    Then for a single component b_NV(k_x, k_y, z) of the magnetic field, the three components of the magnetic field are given by

    .. math::
        b(k_x, k_y, z) = b_NV(k_x, k_y, z) u / (u \cdot n)

    That is true for all k except k = 0, where the denominator is zero. In general, constant components of the fields are
    not connected through the requirement of the divergence- and curl- free condition, since it is always possible to add
    a constant offset without violating these conditions. Therefore, it is important to set the constant offset of the magnetic
    field components to zero, by subtracting the mean of the field components (separete for each component).
    """

    def __init__(self):
        pass

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, theta, phi, height, dx, dy) -> torch.Tensor:
        """
        Defines a transform matrix that maps from a single component of the magnetic field, usually
        denoted by b_NV in 2d Fourier space, to the three components of the magnetic field in the Cartesian coordinate system,
        if B_NV is a component defined along a unit vector n, given by the spherical angles `theta` and `phi`.

        It works for components of any harmonic function in the source-free region, not only the magnetic field.

        Args:
            kx_vector (array-like):  1d vector of k_x values in 2d Fourier space where transform is computed
            ky_vector (array-like):  1d vector of k_y values
            theta (float):           a single value of the polar angle (angle with respect to polar axis, z axis) [deg]
            phi (float):             a single value of the azimuthal angle (angle of rotation from the initial meridian plane [deg]

        Returns: M matrix, shape (3, n_kx, n_ky). Doing an element-wise product with a map of a single component b_NV
        along the NV n-axis of shape (n_kx, n_ky) gives b of shape (3, n_kx, n_ky), where b is the 3d magnetic field with first
        dimension corresponds to the x, y, z components of the magnetic field.

        Usage example:
        --------------

        .. code-block:: python
            M = HarmonicFunctionComponentsKernel.define_kernel_matrix(kx_vector, ky_vector, theta, phi)
            b = torch.einsum('cjk,jk->cjk', M, b_NV)

        """
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, 0, dx, dy, add_filter=False)
        

        # Do type conversion
        theta = torch.deg2rad(torch.tensor(theta))
        phi = torch.deg2rad(torch.tensor(phi))

        # Define the unit vector along the NV axis
        n = torch.tensor(
            [torch.sin(theta) * torch.cos(phi),
             torch.sin(theta) * torch.sin(phi),
             torch.cos(theta)],
            dtype=torch.complex64)

        # Define the vector in 2d Fourier space that represent the Hamilton operator ∇
        u = torch.empty((3,) + k_matrix.shape, dtype=torch.complex64,)
        u[0, :, :] = kx_vector[:, None]
        u[1, :, :] = ky_vector[None, :]
        u[2, :, :] = 1j * k_matrix

        _denominator = torch.einsum('cjk,c->jk', u, n)

        # denominator is zero for k = 0, where u = (k_x, k_y, ik) = 0. We set it to 1 to avoid division by zero, and later
        # set the corresponding elements of the kernel matrix to zero to null those components.
        _denominator[0, 0] = 1

        M = u  * (1) / _denominator

        # Set the components of the kernel matrix to zero for k = 0, where the denominator is zero
        M[:, 0, 0] = 0

        # kx = kx_vector[:, None]
        # ky = ky_vector[None, :]
        # k = k_matrix

        # bnv2bx =  1/(n[0]                   + n[1] * ky / kx        + 1j * n[2] * k / kx)
        # bnv2by =  1/(n[0] * kx / ky         + n[1]                  + 1j * n[2] * k / ky)
        # bnv2bz = 1/(-1j * n[0] * kx / k     - 1j * n[1] * ky / k    + n[2])

        # bnv2bx[0,0] = 1
        # bnv2by[0,0] = 1
        # bnv2bz[0,0] = 1
        
        # M[0, :, :] =  bnv2bx
        # M[1, :, :] =  bnv2by
        # M[2, :, :] =  bnv2bz

        # M[:, 0, 0] = 1

        return M


class MagneticFieldToCurrentInversion2d(object):
    """
    Implmenets inversion in 2d Fourier space from the b_x, b_y magnetic field map to the current map j_x, j_y.
    In this case, the connection is invertable in k-space, below is the inverse transform:

    .. math::
                             ┌─        ─┐
                       2  1  │  0   -1  │
        j(k_x, k_y) = -- --- │          │ b(k_x, k_y)
                       μ0 D  │  1    0  │
                             └─        ─┘

    where :math:`\mu_0` is the permeability of free space, and D is the depth factor as in `MagnetizationFourierKernel2d`, defined by `UniformLayerFactor2d`.
    """

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness,  dx, dy):
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)

        M = torch.zeros((2, 2,) + k_matrix.shape, dtype=torch.complex64,)

        M[0, 1, :, :] = torch.ones_like(k_matrix)
        M[1, 0, :, :] = -torch.ones_like(k_matrix)

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness, dx, dy, add_filter=True)
        # Temporary set to avoid division by zero
        # depth_factor[0, 0] = 1

        M = (2 / MU0) * depth_factor * M
        # Deal with the case where k = 0 by setting the corresponding elements to 0
        M[0, 0] = 0

        return M 


class SphericalUnitVectorKernel(object):

    @staticmethod
    def define_unit_vector(theta, phi):
        """
        Defines a unit vector in the Cartesian coordinate system given by the spherical angles `theta` and `phi`.

        Args:
            theta (float):  a single value of the polar angle (angle with respect to polar axis, z axis) [deg]
            phi (float):    a single value of the azimuthal angle (angle of rotation from the initial meridian plane [deg]

        Returns: a unit vector of shape (3,)

        """
        # Do type conversion
        theta = torch.deg2rad(torch.tensor(theta))
        phi = torch.deg2rad(torch.tensor(phi))

        # Define the unit vector along the NV axis
        n = torch.tensor(
            [torch.sin(theta) * torch.cos(phi),
             torch.sin(theta) * torch.sin(phi),
             torch.cos(theta)],)

        return n
