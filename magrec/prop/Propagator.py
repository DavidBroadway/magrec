"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by currec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J.
"""
import torch

from currec.core.Fourier import FourierTransform
from currec.core.constants import Constants

MU0 = Constants.MU0
_factor = 4


# CurrentFourierPropagtor3d
class Propagator(object):
    def __init__(
        self,
        J=None,
        shape=None,
        dx=1.0,
        dy=1.0,
        height=10.0,
        width=10.0,
        depth=10.0,
        D=1.0,
        padding: int = 0,
        rule="trapezoid",
    ):
        """
        Args:
            J (torch.Tensor):   current field, shape (b, 3, n_x, n_y, n_z). Eiter J or `shape` must be provided.
            shape (tuple):      shape of the current field grid (n_x, n_y, n_z) where n_x, n_y, n_z are the number of grid points in x, y, z direction
            dx (float):         current distribution grid spacing in x direction, in [mm]
            dy (float):         current distribution grid spacing in y direction, in [mm]
            height (float):     current distribution grid height, in [mm]
            width (float):      current distribution grid width, in [mm]
            depth (float):      current distribution grid depth, in [mm]
            D (float):          elevation above the current distribution at which to evaluate the magnetic field, in [mm]
            padding (int):      whether to apply padding and how. 0 — no padding, positive integer — pad the original
                                current distribution with that many times zeros on one side, negative integer — pad the original
                                tensor with zeros half the times on both sides
            rule (str):         integration rule to use for the Biot-Savart integral, either 'trapezoid' or 'simpson'

        Attributes:
            cshape (tuple):     computation shape of the current field grid (n_x, n_y, n_z) where n_x, n_y, n_z
                                are the number of grid points in x, y, z direction where the computation is performed
            pdom (list):        physical domain of the current distribution grid [[x1, x2], [y1, y2]] where x1, x2, y1, y2 are the indices
                                of the grid points in x, y direction which correspond to the physical domain where the current distribution is defined
                                It can be different from the input grid if padding is applied. Padding is needed to account for the Fourier transform
                                periodicity which might not necessarily exist.
        """
        self.device = "cpu"

        if J is None:
            assert shape is not None, "shape must be provided if J is not provided"
            # (1, 3,) — 1 stand for singleton batch dimension and 3 for the three components of the current
            J = torch.zeros(
                (
                    1,
                    3,
                )
                + shape
            )

        if shape is None:
            assert J is not None, "J must be provided if shape is not provided"
            shape = J.shape[-3:]

        # We handle padding here. If padding is present, then the physical domain
        # is different from the computational domain (in the sense defined by Yazdanian et. al. 2020). That is,
        # the physical domain: the domain where the current is defined and where the magnetic field is of interest,
        # is smaller than the computational domain. The computational domain is the domain where the values of the current
        # are assumed to be known, and where the magnetic field is evaluated.

        # To handle padding,
        # 1. we need to increase a vector J we work with by creating an empty tensor with 0s, in which we put the original J
        # 2. change the width and height of the domain, since now the wave-vectors are different
        # 3. change the assesrtion that J and j have the same shape, — not they do not
        # 4. when returning B, we need to crop the result to the original shape

        if padding >= 0:
            self.pdom = [[0, shape[0]], [0, shape[1]]]
        elif padding < 0:
            self.pdom = [
                [
                    (shape[0] * (abs(padding) + 1) - shape[0]) // 2,
                    (shape[0] * (abs(padding) + 1) + shape[0]) // 2,
                ],
                [
                    (shape[1] * (abs(padding) + 1) - shape[1]) // 2,
                    (shape[1] * (abs(padding) + 1) + shape[1]) // 2,
                ],
            ]

        self.cshape = (
            shape[0] * (abs(padding) + 1),
            shape[1] * (abs(padding) + 1),
            shape[2],
        )
        self.padding = padding

        # Define extents of the horizontal grid
        assert (dx is not None and dy is not None) or (
            width is not None and depth is None
        ), "Either dx and dy or `width` and `depth` must be provided"
        assert not (
            (dx is not None and dy is not None)
            and (width is not None and depth is None)
        ), "Only one pair: dx and dy or `width` and `depth` must be provided"

        if dx is not None and dy is not None:
            self._dx = dx
            self._dy = dy
            self._width = (shape[-3] - 1) * dx * (abs(padding) + 1)
            self._depth = (shape[-2] - 1) * dy * (abs(padding) + 1)

        if width is not None and depth is not None:
            self._dx = width / (shape[-3] - 1)
            self._dy = depth / (shape[-2] - 1)
            self._width = width * (abs(padding) + 1)
            self._depth = depth * (abs(padding) + 1)

        # define a spatial and conjugate grids and a Fourier transform on them for
        # quantities sampled on that grid
        self.ft = FourierTransform(
            grid_shape=(self.cshape[0], self.cshape[1]),
            dx=self.dx,
            dy=self.dy,
            real_signal=True,
        )

        # Capital B, J stand for real space values B = B(x, y, z), and small b, j for Fourier space in 2 frequency
        # dimensions and one space dimension values b = b(k_x, k_y, z)
        self.J = J

        # transformed fields (x, y, z) → (k_x, k_y, z)
        self._j = None

        self._height = height
        self._dz = height / (shape[-1] - 1)
        self._D = D
        self._xs = None
        self._ys = None
        self._zs = None
        self._z0 = None

        self.shape = shape

        self._kx_vector, self._ky_vector = self.ft.kx_vector, self.ft.ky_vector
        self._k_matrix = self.ft.k_matrix

        self._M = self.define_j_to_b_z_matrix(
            self.kx_vector, self.ky_vector, self.k_matrix
        )

        self._exp_matrix = self.get_exp_matrix(self.zs, self.k_matrix, self.z0)

        self._rule = rule

        self._b = None
        self._B = self.get_B(J)
        return

    def __call__(self, *args, **kwargs):
        return self.get_B(*args, **kwargs)

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

        self._M = self.M.to(device)
        self._j = self.j.to(device)
        self._exp_matrix = self.exp_matrix.to(device)
        self._zs = self._zs.to(device)

        self.ft = self.ft.to(device)

        return self

    @property
    def real_grid(self):
        xs = self.xs
        ys = self.ys
        zs = self.zs
        return torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=0)

    @staticmethod
    def define_j_to_b_z_matrix(kx_vector, ky_vector, k_matrix):
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
            b = torch.einsum("z,bcijz->bcij", zs, b_zs)
        else:
            raise ValueError("Unknown integration rule: {}".format(rule))

        return b

    @property
    def exp_matrix(self):
        """
        Matrix of the exponential factors exp(-k [z0 - z']) for each k in k_matrix and each z' in zs
        """
        if self._exp_matrix is None:
            self._exp_matrix = Propagator.get_exp_matrix(
                zs=self.zs, k_matrix=self.k_matrix, z0=self.z0, D=self.D
            )
        return self._exp_matrix

    @exp_matrix.setter
    def exp_matrix(self, exp_matrix: torch.Tensor):
        assert (
            exp_matrix.shape == self.k_matrix.shape + self.zs.shape
        ), "exp_matrix must have shape {}, got {}".format(
            self.k_matrix.shape + self.zs.shape, exp_matrix.shape
        )
        self._exp_matrix = exp_matrix
        self._b = None
        self._B = None
        return

    @property
    def kx_vector(self):
        """
        Vector of k_x values, shape (n_kx,)
        """
        if self._kx_vector is None:
            self._kx_vector = self.ft.kx_vector

        return self._kx_vector

    @property
    def ky_vector(self):
        """
        Vector of k_y values, shape (n_ky,)
        """
        if self._ky_vector is None:
            self._ky_vector = self.ft.ky_vector

        return self._ky_vector

    @property
    def k_matrix(self):
        """
        Matrix with all possible k = sqrt(k_x ** 2 + k_y ** 2), shape (n_kx, n_ky)
        """
        if self._k_matrix is None:
            self._k_matrix = self.ft.k_matrix

        return self._k_matrix

    @property
    def M(self) -> torch.Tensor:
        """
        Transformation matrix that connects b and j components before filtering in Fourier space by the exponent, shape (3, 3, n_kx, n_ky)
        """
        if self._M is None:
            self._M = Propagator.define_j_to_b_z_matrix(
                self.kx_vector, self.ky_vector, self.k_matrix
            )
        return self._M

    @property
    def xs(self) -> torch.Tensor:
        if self._xs is None:
            self._xs = torch.arange(
                0, self.dx * (self.pdom[0][1] - self.pdom[0][0]), self.dx
            )
        return self._xs

    @property
    def ys(self) -> torch.Tensor:
        if self._ys is None:
            self._ys = torch.arange(
                0, self.dy * (self.pdom[1][1] - self.pdom[1][0]), self.dy
            )
        return self._ys

    @property
    def zs(self) -> torch.Tensor:
        """
        Returns the z-coordinates of the current layers, shape (n_z,), in [mm]
        """
        if self._zs is None:
            self._zs = torch.arange(0, self.shape[-1]) * self.dz
        return self._zs

    @zs.setter
    def zs(self, zs):
        # Check that zs is a valid z-coordinate vector
        dz = zs[1] - zs[0]
        assert torch.diff(zs).allclose(
            dz
        ), "zs must be equally spaced, i.e. zs[i+1] - zs[i] = dz"
        self._dz = dz
        self._height = zs[0]
        self._zs = zs
        self._z0 = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def dz(self) -> float:
        """
        Spacing between z-coordinates of the current layers, in [mm]
        """
        return self._dz

    @dz.setter
    def dz(self, dz: float):
        assert dz > 0, "dz must be positive"
        self._dz = dz
        self._height = None
        self._zs = None  # next time zs is requested, it will be recalculated, because spacing between layers has changed
        self._z0 = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def height(self) -> float:
        """
        Height of the current layers, in [mm]
        """
        if self._height is None:
            self._height = self.zs[-1]
        return self._height

    @height.setter
    def height(self, height: float):
        assert height > 0, "height must be positive"
        shape = self.shape
        self._dz = height / (shape[-1] - 1)
        self._z0 = height + self._D
        self._zs = torch.linspace(0, height, shape[-1])
        self._height = height
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def dx(self) -> float:
        """
        Spacing between x-coordinates of the current layers, in [mm]
        """
        if self._dx is None:
            self._dx = self.width / (self.shape[0] - 1)
        return self._dx

    @dx.setter
    def dx(self, dx: float):
        self._dx = dx
        self._width = None
        self._kx_vector = None
        self._k_matrix = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def width(self) -> float:
        """
        Extent of the current layers in x-direction, in [mm], i.e. the distance between the first and last x-grid points
        """
        if self._width is None:
            self._width = (self.shape[0] - 1) * self.dx
        return self._width

    @width.setter
    def width(self, width: float):
        self._width = width
        self._dx = None
        self._kx_vector = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def dy(self):
        """
        Spacing between y-coordinates of the current layers, in [mm]
        """
        if self._dy is None:
            self._dy = self.depth / (self.shape[1] - 1)
        return self._dy

    @dy.setter
    def dy(self, dy):
        self._dy = dy
        self.ft.dy = dy
        self._ky_vector = None
        self._k_matrix = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def depth(self) -> float:
        """
        Extent of the current layers in y-direction, in [mm], i.e. the distance between the first and last y-grid points
        """
        if self._depth is None:
            self._depth = (self.shape[1] - 1) * self.dy
        return self._depth

    @depth.setter
    def depth(self, depth: float):
        self._depth = depth
        self._dy = None
        self._ky_vector = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    @property
    def z0(self) -> float:
        """
        z-coordinate of the plane where the magnetic field is measured, in [mm]
        """
        if self._z0 is None:
            self._z0 = self.zs.max() + self.D
        return self._z0

    @z0.setter
    def z0(self, z0: float):
        assert (
            z0 > self.zs.max()
        ), "z0 where the measurement is made must be higher than z-coordinates of the highest layer"
        self._z0 = z0
        self._D = None
        self._exp_matrix = None  # next time exp_matrix is requested, it will be recalculated, because z0 has changed
        self._b = None
        self._B = None
        return

    @property
    def D(self) -> float:
        """
        Distance between the plane where the magnetic field is measured and the top layer where current field is specified, in [mm]
        """
        if self._D is None:
            self._D = self.zs.max() - self.z0
        return self._D

    @D.setter
    def D(self, D: float):
        assert D > 0, "D must be positive"
        self._D = D
        self._z0 = None
        self._exp_matrix = None
        self._b = None
        self._B = None
        return

    def get_b(self, J=None, j=None, D=None) -> torch.Tensor:
        """
        2d-Fourier transformed magnetic field at elevation D above the current field, in [mT]

        Args:
            J (torch.Tensor): current field in real space, shape (b_n, 3, n_x, n_y, n_z), where b_n is the number of samples, J(x, y, z)
            j (torch.Tensor): 2d-Fourier transformed current field, shape (b_n, 3, n_kx, n_ky, n_z), where b_n is the number of samples, j(k_x, k_y, z)
            D (float):        elevation above the current distribution at which to evaluate the magnetic field, in [mm]

        Returns:
            b (torch.Tensor): 2d-Fourier image of the magnetic field, shape (3, n_kx, n_ky, z), where 3 is the number of components of the magnetic field, in [mT]
        """
        assert not (
            J is not None and j is not None
        ), "Only one of J or j must be specified"

        # if called with parameters, update them in the object
        if J is not None:
            self.J = J
            self._j = self.ft.forward(self._J, dim=(-3, -2))
            self._b = None

        if j is not None:
            self.j = j
            self._J = self.ft.backward(j, dim=(-2, -1))
            self._b = None

        if D is not None:
            self.D = D
            self._b = None

        if self._b is None:
            self._b = Propagator.get_b_from_j(
                M=self.M,
                j=self.j,
                exp_matrix=self.exp_matrix,
                zs=self.zs,
                rule=self.rule,
            )
        return self._b

    @property
    def b(self):
        """
        2d-Fourier transformed magnetic field at elevation D above the current field, in [mT], b(k_x, k_y), shape (3, n_kx, n_ky)
        """
        if self._b is None:
            self._b = self.get_b()
        return self._b

    def get_B(self, J=None, j=None, D=None) -> torch.Tensor:
        if self._b is None or J is not None or j is not None or D is not None:
            self._b = self.get_b(J=J, j=j, D=D)

        self._B = self.ft.backward(self.b, dim=(-2, -1))
        return self.B

    @property
    def B(self):
        """
        Magnetic field at elevation D above the current field, in [mT], B(x, y), shape (3, n_x, n_y)
        """
        pdom = self.pdom
        if self._B is None:
            self._B = self.get_B()
        return self._B[..., pdom[0][0] : pdom[0][1], pdom[1][0] : pdom[1][1]]

    @property
    def j(self):
        """
        2d-Fourier transformed current field, shape (b_n, 3, n_kx, n_ky, n_z), where b_n is the number of samples, j(k_x, k_y, z)
        """
        if self._j is None:
            self._j = self.ft.forward(self._J, dim=(-3, -2))
        return self._j

    @j.setter
    def j(self, j):
        self._j = j
        self._J = self.ft.backward(j, dim=(-2, -1))
        self._b = None
        self._B = None
        return

    @property
    def J(self):
        """
        Current field in real space, shape (b_n, 3, n_x, n_y, n_z), where b_n is the number of samples, J(x, y, z)
        """
        pdom = self.pdom
        if self._J is None:
            self._J = self.ft.backward(self.j, dim=(-2, -1))
        return self._J[..., pdom[0][0] : pdom[0][1], pdom[1][0] : pdom[1][1], :]

    @J.setter
    def J(self, J):
        # get physical domain indices
        pdom = self.pdom
        # obtain a copy of an empty input tensor with the correct size
        _J = torch.zeros(J.shape[0:-3] + self.cshape, device=self.device)
        # assign the input tensor to the indices of the empty tensor which correspond to the physical domain
        _J[..., pdom[0][0] : pdom[0][1], pdom[1][0] : pdom[1][1], :] = J

        self._J = _J
        self._j = self.ft.forward(_J, dim=(-3, -2))
        self._b = None
        self._B = None
        return

    @property
    def rule(self):
        """
        Rule for integrating the contribution to the magnetic field from each current layer
        """
        return self._rule

    @rule.setter
    def rule(self, rule):
        assert (
            rule == "trapezoid" or rule == "rectangle"
        ), "Rule must be either `trapezoid` or `rectangle`"
        self._rule = rule
        self._b = None
        self._B = None
        return


class DipolePropagator(Propagator):
    """
    Propagator for dipole current distributions
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

    @staticmethod
    def define_J_to_B_matrix():

        return _M

    def __init__(
        self,
        J=None,
        shape=None,
        dx=1.0,
        dy=1.0,
        height=10.0,
        width=10.0,
        depth=10.0,
        D=1.0,
        padding: int = 0,
        rule="trapezoid",
        device="cpu",
    ):
        """
        Args:
            D (float):        elevation above the dipole at which to evaluate the magnetic field, in [mm]
            rule (str):       rule for integrating the contribution to the magnetic field from each current layer
            device (str):     device on which to perform the computations
        """
        super().__init__(
            J=J,
            shape=shape,
            dx=dx,
            dy=dy,
            height=height,
            width=width,
            depth=depth,
            D=D,
            padding=padding,
            rule="trapezoid",
        )
        return

    @property
    def M(self) -> torch.Tensor:
        """
        Transformation matrix that connects the current distribution of dipoles with the magnetic field
        """
        if self._M is None:
            self._M = DipolePropagator.define_J_to_B_matrix(
                self.kx_vector, self.ky_vector, self.k_matrix
            )
        return self._M

    @M.setter
    def M(self, M):
        assert M.shape == (3, 1), "M must be a 3x1 tensor"
        self._M = M
        self._b = None
        self._B = None
        return

    def get_b(self, M=None, D=None) -> torch.Tensor:
        """
        2d-Fourier transformed magnetic field at elevation D above the current field, in [mT]

        Args:
            M (torch.Tensor): dipole moment, shape (3, 1)
            D (float):        elevation above the current distribution at which to evaluate the magnetic field, in [mm]

        Returns:
            b (torch.Tensor): 2d-Fourier image of the magnetic field, shape (3, n_kx, n_ky, z), where 3 is the number of components of the magnetic field, in [mT]
        """
        if M is not None:
            self.M = M
            self._b = None

        if D is not None:
            self.D = D
            self._b = None

        if self._b is None:
            self._b = DipolePropagator.get_b_from_M(
                M=self.M, exp_matrix=self.exp_matrix, zs=self.zs, rule=self.rule
            )
        return self._b


class MagnetizationPropagator2d(Propagator):
    def __init__(self, source_shape, dx, dy, abstand, layer_thickness):
        """Initiate propagator that gets a magnetization distribution and

        :param source_shape:
        :param abstand:
        :param layer_thickness:
        :param field_shape:
        :returns:

        """
        self.ft = FourierTransform(grid_shape=source_shape, dx=dx, dy=dy, real_signal=True)
        k_matrix = self.ft.k_matrix

        self.depth_factor = (
            torch.exp(-k_matrix * abstand)
            / k_matrix
            * (torch.exp(-k_matrix * layer_thickness) - 1)
        )

        self.m_to_b_matrix = self.define_m_to_b_matrix(ft=self.ft, depth_factor=self.depth_factor)
        """Forward field matrix that connects sources to the measured field"""

        pass

    def __call__(self, M):
        """Propagates planar magnetization M of shape (batch_size, 3, width, height) to the magnetic field
        this magnetization creates at distance `self.abstand` from the plane where this magnetization is present.
        """
        return self.B_from_M(M)

    @staticmethod
    def define_m_to_b_matrix(ft, depth_factor):
        # use .ft information about the Fourier-space grid to know the propagation properties
        kx_vector = ft.kx_vector
        ky_vector = ft.kx_vector
        k_matrix = ft.k_matrix

        _M = torch.zeros(
            (
                3,
                3,
            )
            + k_matrix.shape,
            dtype=torch.complex64,
        )

        _M[0, 0, :, :] = (
            kx_vector[:, None] ** 2 / k_matrix / 2
        )  # divide by 2 to get the proper quantity when _M + _M.T
        _M[1, 1, :, :] = ky_vector[None, :] ** 2 / k_matrix / 2
        _M[2, 2, :, :] = k_matrix / 2

        # Deal with the case where k = 0 by setting the corresponding elements to 0
        _M[[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]] = 0

        # Use the property of the M matrix that it is symmetric (that's why we divide by 1/2 above, to get the proper diagonal terms)
        M = _M + _M.transpose(0, 1)
        M = -(MU0 / 2) * depth_factor * M

        return M

    def get_b_from_m(self, m):
        b = torch.einsum("bijklmn,blmn->bijk", self.m_to_b_matrix, m)
        return b

    def B_from_M(self, M):
        m = self.ft.forward(M, dim=(-2, -1))
        b = self.get_b_from_m(m)
        B = self.ft.backward(b, dim=(-2, -1))
        return B
