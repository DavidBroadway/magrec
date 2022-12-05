import torch

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.constants import MU0, twopi


class CurrentFourierKernel2d(object):

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, k_matrix):
        """
        Defines matrix M that transforms the current field at z' to the magnetic field contribution
        exp(-k [z - z']) M j such that the total magnetic field at z is::

        ..math::
            b(k_x, k_y, z) = ∫ exp(-k [z - z']) M j dz' (z > z')

        in the Fourier space.

        Args:
            kx_vector:  vector with spacial frequencies in x direction, shape (n_kx,)
            ky_vector:  vector with spacial frequencies in y direction, shape (n_ky,)
            k_matrix:   matrix with all possible k = sqrt(k_x ** 2 + k_y ** 2), shape (n_kx, n_ky)

        Returns:
            M matrix, shape (3, 3, n_kx, n_ky). Contracting with j of shape (3, n_kx, n_ky) in the first dimension
            gives b of shape (3, n_kx, n_ky): b = M j, or more specifically [b_x b_y b_z] = M [j_x j_y j_z]
            Units: mT / (mm * A)

        Matrix definition
        -----------------
        If Tetienne et al. definition of the transform (and of the
        matrix) is used, the b_z component ends up with a wrong sign in
        the inverse Fourier transform, that is, in the real B(x, y, z) space::


                                                       ┌─                            ─┐ ┌─   ─┐
                                                       │     0        1      ik_y / k │ │ j_x │
                               ┌─                ─┐ μ0 │                              │ │     │
           b(k_x, k_y, z) =  ∫ │ exp(-k [z - z']) │ -- │    -1        0     -ik_x / k │ │ j_y │ dz'
                               └─                ─┘  2 │                              │ │     │
                               └── [k_x, k_y, z] ─┘    │-ik_y / k  ik_x / k      0    │ │ j_z │
                                    `exp_matrix`       └─                            ─┘ └─   ─┘
                                                   └───────── M[:, :, k_x, k_y] ──────┘ └──┬──┘
                                                                                            j[k_x, k_y, z']


        """
        _M = torch.zeros((3, 3, len(kx_vector), len(ky_vector)), dtype=torch.complex64)

        _M[0, 1, :, :] = torch.full_like(
            k_matrix, fill_value=1.0, dtype=torch.complex64
        )
        _M[0, 2, :, :] = 1j * ky_vector[None, :] / k_matrix
        _M[1, 2, :, :] = -1j * kx_vector[:, None] / k_matrix

        # Deal with the case where k = 0 by setting the corresponding elements to 0
        _M[[0, 1], [2, 2], [0, 0], [0, 0]] = 0

        # M has a nice property that is it antisymmetric, when viewed for specific values of k_x and k_y
        # above we defined its upper triangle, below we add the parts of the antisymmetric parts together.
        # Transpose to change dimensions pertaining to the components _x, _y, _z of the field
        M = _M - _M.transpose(0, 1)
        M = (MU0 / 2) * M  # scale by mu0 to get [mT * mm / A] units
        return M

    
class CurrentLayerFourierKernel2d(object):

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness):
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
        M[2, 1, :, :] =  1j * kx_vector[:, None] / k_matrix        

        # Deal with the case where k = 0 by setting the corresponding elements to 0
        M[[2, 2], [0, 1], [0, 0], [0, 0]] = 0

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness)

        M = (MU0 / 2) * depth_factor * M
        return M  
    

class MagnetizationFourierKernel2d(object):

    @staticmethod
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness):
        """Defines a transformation matrix that connects a 2d magnetization distribution that has 3 components of
        the magnetization to the magnetic field it creates.
        """
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)

        _M = torch.zeros((3, 3,) + k_matrix.shape, dtype=torch.complex64,)

        # divide by 2 to get the proper quantity when _M + _M.T
        _M[0, 1, :, :] = kx_vector[:, None] * ky_vector[None, :] / k_matrix / 2
        _M[0, 2, :, :] = - 1j * kx_vector[:, None] / k_matrix / 2
        _M[1, 2, :, :] = - 1j * ky_vector[None, :] / k_matrix / 2

        _M[0, 0, :, :] = kx_vector[:, None] ** 2 / k_matrix / 2
        _M[1, 1, :, :] = ky_vector[None, :] ** 2 / k_matrix / 2
        _M[2, 2, :, :] = k_matrix / 2

        # Deal with the case where k = 0 by setting the corresponding elements to 0
        _M[[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]] = 0

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness)

        # Use the property of the M matrix that it is symmetric (that's why we divide by 1/2 above, to get the proper diagonal terms)
        M = _M + _M.transpose(0, 1)
        M = -(MU0 / 2) * depth_factor * M
        return M
    

class UniformLayerFactor2d(object):
    """
    Defines a factor that appears after integration of a uniform distribution along the z axis.
    """

    @staticmethod
    def define_depth_factor(k_matrix: torch.Tensor, height: float, layer_thickness: float):
        """
        Returns a matrix that scales each k-vector by the factor that appears after integration of a uniform source distribution.

        Args:
            k_matrix:                   matrix with all possible k = sqrt(k_x ** 2 + k_y ** 2), shape (n_kx, n_ky)
            height (float):             height above the layer at which to evaluate the factor
            layer_thickness (float):    thickness of the layer
        """
        depth_factor = (
                torch.exp(-k_matrix * height)
                / k_matrix
                * (torch.exp(-k_matrix * layer_thickness)-1)
        )
        if layer_thickness == 0:
                 depth_factor = (
                torch.exp(-k_matrix * height))

        depth_factor[0, 0] = 0
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
    def define_kernel_matrix(kx_vector, ky_vector, theta, phi) -> torch.Tensor:
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

        M = u / _denominator

        # Set the components of the kernel matrix to zero for k = 0, where the denominator is zero
        M[:, 0, 0] = 0

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
    def define_kernel_matrix(kx_vector, ky_vector, height, layer_thickness):
        k_matrix = FourierTransform2d.define_k_matrix(kx_vector, ky_vector)

        M = torch.zeros((2, 2,) + k_matrix.shape, dtype=torch.complex64,)

        M[0, 1, :, :] = -torch.ones_like(k_matrix)
        M[1, 0, :, :] =  torch.ones_like(k_matrix)

        depth_factor = UniformLayerFactor2d.define_depth_factor(k_matrix, height, layer_thickness)
        # Temporary set to avoid division by zero
        depth_factor[0, 0] = 1

        M = (2 / MU0) / depth_factor * M
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