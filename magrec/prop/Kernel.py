import torch

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.constants import MU0


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
                               ─── [k_x, k_y, z] ──    │-ik_y / k  ik_x / k      0    │ │ j_z │
                                     `exp_matrix`      └─                            ─┘ └─   ─┘
                                                    ───────── M[k_x, k_y, :, :] ───────  ──┬──
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

        depth_factor = (
                torch.exp(-k_matrix * height)
                / k_matrix
                * (torch.exp(-k_matrix * layer_thickness) - 1)
        )

        depth_factor[0, 0] = 0  # k = 0 case

        # Use the property of the M matrix that it is symmetric (that's why we divide by 1/2 above, to get the proper diagonal terms)
        M = _M + _M.transpose(0, 1)
        M = -(MU0 / 2) * depth_factor * M
        return M


class SphericalCartesianKernel(object):
    """
    Defines a transform matrix that maps from a single component of the magnetic field, usually
    denoted by B_NV, to the three components of the magnetic field in the Cartesian coordinate system,
    if B_NV is a component defined along a unit vector n, given by the spherical angles `theta` and `phi`.

    It works for components of any harmonic function in the source-free region, not only the magnetic field.
    """

    def __init__(self):
        pass
