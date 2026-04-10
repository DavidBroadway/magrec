"""This module implements evaluation of Biot-Savart law using FFT.

The following code uses Fourier transform and its approximation by FFT
provided by magrec.Fourier module, to evaluate Biot-Savart integral which
connects the magnetic field B with the current density distribution J or
magnetization distribution m.
"""
# used for base class methods that need to be implemented
from abc import abstractmethod
from types import MethodType
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.constants import DEFAULT_UNITS, MU0, get_exponent_from_unit
from magrec.prop.Kernel import (
    UniformLayerFactor2d, 
    MagnetizationFourierKernel2d, 
    CurrentFourierKernel2d, 
    CurrentLayerFourierKernel2d, 
    InverseCurrentLayerFourierKernel2d,
    SphericalUnitVectorKernel
    )

from magrec.misc.sampler import GridSampler

class Propagator(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        pass
    
    def to(self, device: torch.device | str):
        pass
    
    def set_units(self, *units):
        """Set the units of the propagator and rescale the unit-dependent matrix.
        
        Expects subclasses to define:
            self.units: dict with keys {"current", "length", "magnetic_field"}
            self._units_matrix_attr: name of matrix attribute to rescale
        """
        if not hasattr(self, "units"):
            self.units = DEFAULT_UNITS.copy()
        
        if not hasattr(self, "_units_matrix_attr"):
            raise RuntimeError("Units matrix attribute not defined on this propagator.")
        
        matrix = getattr(self, self._units_matrix_attr, None)
        if matrix is None:
            raise RuntimeError(f"Units matrix `{self._units_matrix_attr}` not found on this propagator.")
        
        updated_units = self.units.copy()
        
        if len(units) == 1 and isinstance(units[0], dict):
            updated_units.update(units[0])
        else:
            for unit in units:
                if isinstance(unit, dict):
                    raise ValueError("If dict is provided, it must be the only argument.")
                if not isinstance(unit, str):
                    raise ValueError(f"Units must be strings or a dict, got {type(unit)}.")
                if unit[-1] == "m":
                    updated_units["length"] = unit
                elif unit[-1] == "A":
                    updated_units["current"] = unit
                elif unit[-1] == "T":
                    updated_units["magnetic_field"] = unit
                else:
                    raise ValueError(f"Units must end with 'm', 'A', or 'T', got `{unit}`.")
        
        updated_exponents = {}
        default_exponents = {}
        for key in updated_units.keys():
            default_exponents[key] = get_exponent_from_unit(self.units[key])
            updated_exponents[key] = get_exponent_from_unit(updated_units[key])
        
        base_exponent = \
            (default_exponents["magnetic_field"] - updated_exponents["magnetic_field"]) + \
            (default_exponents["length"] - updated_exponents["length"]) - \
            (default_exponents["current"] - updated_exponents["current"])
        sign = getattr(self, "_units_prefactor_sign", 1)
        prefactor_exponent = sign * base_exponent
        
        matrix = matrix * (10 ** prefactor_exponent)
        setattr(self, self._units_matrix_attr, matrix)
        self.units = updated_units
        return self
    
    def get_units(self):
        return self.units
    
    def get_units_dict(self):
        return self.units.copy()


# CurrentFourierPropagtor3d
class CurrentFourierPropagator3d(Propagator):
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
        
        if len(shape) == 3:
            nx, ny, nz = shape
        else:
            raise ValueError(f"Wrong size of `shape` argument. For {__class__.name()} it has"
                             + f"define sizes of 3 dimensions: (nx, ny, nz), got shape {shape} instead.")
        

        # either dx, dy, dz or height, width, depth must be provided together with shape
        if (dx is not None) and (dy is not None) and (dz is not None):
            self.dx = dx
            self.dy = dy
            self.dz = dz
        elif (height is not None) and (width is not None) and (depth is not None):
            self.dx = height / (nx - 1)
            self.dy = width / (ny - 1)
            self.dz = depth / (nz - 1)
        else:
            raise ValueError("Either dx, dy, dz or height, width, depth must be provided together with shape")

        # define a spatial and conjugate grids and a Fourier transform on them for
        # quantities sampled on that grid
        self.ft = FourierTransform2d(
            grid_shape=(nx, ny),  # assume shape defines size of (n_x, n_y, n_z) in this order
            dx=dx,                   # and take out only (n_x, n_y) part for the Fourier transform
            dy=dy,
            real_signal=True,
        )

        self.j_to_b_z_matrix = CurrentFourierKernel2d.\
            define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, self.ft.k_matrix)

        # torch.linspace includes endpoint and contains exactly `shape[0]` elements
        self.zs = torch.linspace(0, height, nz, device=self.device)
        self.z0 = height + D
        """z-coordinate of the plane where the magnetic field is evaluated"""
        
        # --------> z0
        # .
        # .             } D
        # .
        # --------> height
        # ||||||||||  
        # |material|-> dz 
        # ||||||||||
        # --------> 0

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
        _b = torch.einsum("ijkl,...jklz->...iklz", M, j)

        # Calculates the magnetic field contribution to b(k_x, k_y, z0) per the current field layer j(k_x, k_y, z)
        b_zs = torch.einsum("ijz,...cijz->...cijz", exp_matrix, _b)

        # Performs the integration ∫ exp(-k [z0 - z']) M j dz' according to the rule `trapezoid` or `rectangle`,
        # i.e. integrates the contributions from the current field layers to the magnetic field 
        # at the observation plane.
        if rule == "trapezoid":
            b = torch.trapezoid(y=b_zs, x=zs, dim=-1)
        elif rule == "rectangle":
            # Multiply the contribution from the current in each k_x and k_y to the magnetic field by the exponential
            # factor and sum along z, assuming each contribution is scaled by dz (lower Riemann sum)
            dzs = zs[1:] - zs[:-1]
            b = torch.einsum("z,...cijz->...cij", dzs, b_zs[:-1])
        else:
            raise ValueError("Unknown integration rule: {}".format(rule))

        return b

    def get_B_from_J(self, J):
        """
        Calculates the magnetic field B from the volume current density J, 
        expected in the unit dimensions of [A/mm^2].

        Args:
            J: current density, shape ([batch_size, optional], 3, n_x, n_y, n_z)

        Returns:
            B: magnetic field, shape ([batch_size, optional], 3, n_x, n_y, 1) at z = z0, 
                where z0 is the z coordinate of the observation plane, self.D above the current slab.
        """
        j = self.ft.forward(J, dim=(-3, -2))
        b = self.get_b_from_j(M=self.j_to_b_z_matrix, j=j, exp_matrix=self.exp_matrix, zs=self.zs, rule=self.rule)
        B = self.ft.backward(b, dim=(-2, -1))  # here dim = (-2, -1) because b is of shape (3, n_kx, n_ky) already
                                                      # i.e. z direction is contracted
        return B


class MagnetizationPropagator2d(Propagator):

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


class CurrentPropagator2d(Propagator):
    def __init__(self, source_shape, dx, dy, height, layer_thickness, real_signal=True, units=None):
        """
        Create a propagator for a 2d current density that computes the magnetic field at `height` above
        the 2d current layer of finite thickness `layer_thickness` from a constant volume current density density.

        Assumes uniform volume current density across the layer thickness and uses the integration factor to account for the finite thickness.

        Args:
            source_shape:       shape of the volume current density, shape (2, n_x, n_y)
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
            height:             height above the current layer at which to evaluate the magnetic field, in [mm]
            layer_thickness:    thickness of the current layer, in [mm]
        """
        self.ft = FourierTransform2d(grid_shape=source_shape, dx=dx, dy=dy, real_signal=real_signal)

        self.j_to_b_matrix = CurrentLayerFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, height, layer_thickness)
        """Forward field matrix that connects sources to the measured field"""
        self._units_matrix_attr = "j_to_b_matrix"
        self._units_prefactor_sign = 1

        # Take into the account provided units, if any, otherwise use default `DEFAULT_UNITS`
        self.units = DEFAULT_UNITS.copy()
        if units is not None:
            # .set_units() needs to be called after .j_to_b_matrix is defined. I know it's not optimal
            # but that needs to do for now. I'd prefer for units setting to be order-independent. One
            # option is to have an attribue .prefactor that is multiplied by the j_to_b_matrix, but then  
            self.set_units(units)
        
        # Generate a grid of points of 
        nx_points, ny_points = source_shape[-2], source_shape[-1]
        
        pts = GridSampler.sample_grid(nx_points, ny_points, (0., 0.), (dx * nx_points, dy * ny_points))
        
        self.pts = pts
        self.grid = GridSampler.pts_to_grid(pts, nx_points, ny_points)
            
        pass

    def __call__(self, J):
        """Propagates planar 2d current density J of shape (batch_size, 2, width, height) to the magnetic field
        this current density creates at distance `self.height` from the plane where this current is present.
        
        Note that the current density is assumed to be volume density, in units A / mm^2, altough constant across z-layers. 
        Planar current density is obtained by integrating the volume current density along the z-axis, which is just a multiplication
        by the layer thickness in case of the constant in z current density. `CurrentPropagator2d` takes *volume current density* as input.
        
        ..math::
        
            surface current density = volume current density * layer thickness
            
        """
        return self.B_from_J(J)

    def get_b_from_j(self, j):
        """Calculate the Fourier image of the magnetic field b(k_x, k_y, z) from the Fourier image current field j(k_x, k_y, z)."""
        b = torch.einsum("...ijkl,...jkl->...ikl", self.j_to_b_matrix, j)
        return b
    
    def B_from_J(self, J):
        if isinstance(J, np.ndarray):
            J = torch.from_numpy(J)

        j = self.ft.forward(J, dim=(-2, -1))
        b = self.get_b_from_j(j)
        B = self.ft.backward(b, dim=(-2, -1))
        if not self.ft.real_signal:
            B = B.real
        return B
    
    
class InverseCurrentPropagator2d(Propagator):
    def __init__(self, source_shape, dx, dy, height, 
                 layer_thickness, real_signal=True, units=None,
                 filters=None
                 ):
        """
        Create a propagator for a 2d current density that computes the magnetic field at `height` above
        the 2d current layer of finite thickness `layer_thickness` from a constant volume current density density.

        Assumes uniform volume current density across the layer thickness and uses the integration factor to account for the finite thickness.

        Args:
            source_shape:       shape of the volume current density, shape (2, n_x, n_y)
            dx:                 pixel size in the x direction, in [mm]
            dy:                 pixel size in the y direction, in [mm]
            height:             height above the current layer at which to evaluate the magnetic field, in [mm]
            layer_thickness:    thickness of the current layer, in [mm]
        """
        self.ft = FourierTransform2d(grid_shape=source_shape, dx=dx, dy=dy, real_signal=real_signal)

        self.b_to_j_matrix = InverseCurrentLayerFourierKernel2d\
            .define_kernel_matrix(self.ft.kx_vector, self.ft.ky_vector, height, layer_thickness)
        """Inverse field matrix that connects the measured field to sources in 2d case"""
        self._units_matrix_attr = "b_to_j_matrix"
        self._units_prefactor_sign = -1

        # Take into the account provided units, if any, otherwise use default `DEFAULT_UNITS`
        self.units = DEFAULT_UNITS.copy()
        if units is not None:
            # .set_units() needs to be called after .j_to_b_matrix is defined. I know it's not optimal
            # but that needs to do for now. I'd prefer for units setting to be order-independent. One
            # option is to have an attribue .prefactor that is multiplied by the j_to_b_matrix, but then  
            self.set_units(units)
            
        self.filters = filters
            
        pass

    def __call__(self, B):
        """Propagates planar 2d current density J of shape (batch_size, 2, width, height) to the magnetic field
        this current density creates at distance `self.height` from the plane where this current is present.
        
        Note that the current density is assumed to be volume density, in units A / mm^2, altough constant across z-layers. 
        Planar current density is obtained by integrating the volume current density along the z-axis, which is just a multiplication
        by the layer thickness in case of the constant in z current density. `CurrentPropagator2d` takes *volume current density* as input.
        
        ..math::
        
            surface current density = volume current density * layer thickness
            
        """
        return self.J_from_B(B)

    def get_j_from_b(self, b):
        """Calculate the Fourier image of the magnetic field b(k_x, k_y, z) from the Fourier image current field j(k_x, k_y, z)."""
        if self.filters is not None:
            for filt in self.filters:
                b = filt(b)
                
        j = torch.einsum("...ijkl,...jkl->...ikl", self.b_to_j_matrix, b)
        return j
    
    def J_from_B(self, B):
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B)

        b = self.ft.forward(B, dim=(-2, -1))
        j = self.get_j_from_b(b)
        J = self.ft.backward(j, dim=(-2, -1))
        if not self.ft.real_signal:
            J = J.real
        return J
    
class MagneticDipolePropagator(Propagator):
    """Propagator from magnetic dipoles at r_source to magnetic field at r_sensor.
    
    Two backends: 'torch' for autograd/optimization, 'numba' for fast inference.
    The implementation is bound at construction time to avoid branches in __call__.
    """
    
    MAX_FFM_SIZE_IN_MB = 100
    
    def __init__(self, r_source, r_sensor, backend='torch', method='matrix', dtype=torch.float32, device='cpu'):
        """Initialize the propagator. Builds data structures and binds forward to the chosen implementation.
        
        Args:
            r_source: (n_source, 3) array of dipole positions
            r_sensor: (n_sensor, 3) array of sensor positions
            backend: 'torch' or 'numba'. Torch for autograd, numba for speed.
            method: 'matrix' or 'fourier'. If 'matrix', precompute FFM. Fourier not implemented here.
            dtype: torch dtype for tensors (torch.float32 or torch.float64)
            device: 'cpu' or 'cuda' (only for torch backend)
        """
        if len(r_source.shape) != 2:
            raise RuntimeError(f"r_source must be 2D, got shape {r_source.shape}")
        if len(r_sensor.shape) != 2:
            raise RuntimeError(f"r_sensor must be 2D, got shape {r_sensor.shape}")
        
        if r_source.shape[1] > r_source.shape[0]:
            warnings.warn(f"r_source expected (N, 3), got {r_source.shape}. Transposing.")
            r_source = r_source.T
        if r_sensor.shape[1] > r_sensor.shape[0]:
            warnings.warn(f"r_sensor expected (N, 3), got {r_sensor.shape}. Transposing.")
            r_sensor = r_sensor.T
        
        expected_size_in_MB = MagneticDipolePropagator.get_expected_ffm_size(r_source, r_sensor)
        if expected_size_in_MB > self.MAX_FFM_SIZE_IN_MB:
            raise RuntimeError(
                "Expected size of the forward-field matrix is {:.2f} MB, which is larger than {:.2f} MB. "
                "This is not feasible.".format(expected_size_in_MB, self.MAX_FFM_SIZE_IN_MB)
            )
        
        self.dtype = dtype
        self.device = device
        self.r_source = self._as_tensor(r_source)
        self.r_sensor = self._as_tensor(r_sensor)
        self.n_source = self.r_source.shape[0]
        self.n_sensor = self.r_sensor.shape[0]
        
        if backend == 'torch':
            if method == 'matrix':
                self.ffm = self.get_ffm(self.r_source, self.r_sensor).to(dtype=dtype, device=device)
                self.forward = self._forward_torch_matrix
            elif method == 'fourier':
                raise NotImplementedError("Fourier method not yet implemented")
            else:
                raise ValueError(f"Unknown method: {method}")
        elif backend == 'numba':
            r_source_np = self.r_source.cpu().numpy().astype(np.float32)
            r_sensor_np = self.r_sensor.cpu().numpy().astype(np.float32)
            self._compiled_func = self._get_compiled_function(r_source_np, r_sensor_np)
            self.forward = self._forward_numba
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _as_tensor(self, x):
        """Convert to torch tensor with instance dtype/device."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device)
        return torch.tensor(x, dtype=self.dtype, device=self.device)
    
    def _forward_torch_matrix(self, m):
        """FFM-based forward. m: (n_source, 3) -> B: (n_sensor, 3)."""
        return torch.einsum('ijkl,jl->ik', self.ffm, m)
    
    def _forward_numba(self, m):
        """Numba-compiled forward. Accepts torch or numpy, returns same type."""
        was_torch = isinstance(m, torch.Tensor)
        m_np = m.detach().cpu().numpy() if was_torch else m
        B_np = self._compiled_func(m_np.astype(np.float32))
        return torch.from_numpy(B_np).to(self.dtype) if was_torch else B_np
    
    def __call__(self, m):
        """Compute B field from dipole moments m. Shape: (n_source, 3) -> (n_sensor, 3)."""
        return self.forward(m)
    
    @staticmethod
    def get_expected_ffm_size(source=None, sensor=None, type="float32", out_units="MB", as_float=False, print_result=False):
        """Get the expected size of the forward-field matrix in MB."""
        if type == "float32":
            element_size = 32
        elif type == "float64":
            element_size = 64
        elif type == "float16":
            element_size = 16
        else:
            element_size = 32
        
        if isinstance(source, (list, np.ndarray)):
            source = torch.tensor(source, dtype=torch.float32)
            M = source.shape[0]
        elif isinstance(source, torch.Tensor):
            M = source.shape[0]
        elif isinstance(source, int):
            M = source
        else:
            raise AttributeError(f"Unexpected type for source: {type(source)}.")
        
        if isinstance(sensor, (list, np.ndarray)):
            sensor = torch.tensor(sensor, dtype=torch.float32)
            N = sensor.shape[0]
        elif isinstance(sensor, torch.Tensor):
            N = sensor.shape[0]
        elif isinstance(sensor, int):
            N = sensor
        else:
            raise AttributeError(f"Unexpected type for sensor: {type(sensor)}.")
        
        if out_units == "KB":
            r = 3 * M * 3 * N * element_size / 8 / 1024
        elif out_units == "MB":
            r = 3 * M * 3 * N * element_size / 8 / 1024 / 1024
        elif out_units == "GB":
            r = 3 * M * 3 * N * element_size / 8 / 1024 / 1024 / 1024
        else:
            raise ValueError("Invalid output units, must be one of 'KB', 'MB', 'GB'")
        
        if as_float:
            r = r.float()
            return r
        else:
            if print_result:
                print(f"{r:.2f} {out_units}")
            else:
                return r
    
    @staticmethod
    def get_ffm(r_source, r_sensor):
        """Get forward-field matrix (FFM) for a magnetic dipole propagator.
        
        Computes the 3x3 matrix G that relates dipole moment m to field B:
        B_i(r_sensor) = G_ij(r_sensor, r_source) * m_j(r_source)
        
        For a magnetic dipole at r_source with moment m, the field at r_sensor is:
        B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / |r|³
        
        where r = r_sensor - r_source, r̂ = r/|r|
        
        This gives: G_ij = (μ₀/4π) * [3*r̂_i*r̂_j - δ_ij] / |r|³
        
        Args:
            r_source: (n_source, 3) tensor of dipole positions
            r_sensor: (n_sensor, 3) tensor of sensor positions
            
        Returns:
            ffm: (n_sensor, n_source, 3, 3) tensor where ffm[i,j] is the 3x3 
                 matrix relating dipole j to field at sensor i
        """
        if isinstance(r_source, list):
            r_source = torch.tensor(r_source, dtype=torch.float32)
        
        if isinstance(r_sensor, list):
            r_sensor = torch.tensor(r_sensor, dtype=torch.float32)
        
        n_sensor = r_sensor.shape[0]
        n_source = r_source.shape[0]
        
        # Compute displacement vectors: r = r_sensor - r_source
        # Shape: (n_sensor, n_source, 3)
        r = r_sensor[:, None, :] - r_source[None, :, :]
        
        # Compute distances
        # Shape: (n_sensor, n_source)
        r_norm = torch.norm(r, dim=-1, keepdim=True)
        
        # Handle cases where r = 0 (dipole at sensor location)
        # Check where r_norm is close to 0:
        if torch.any(r_norm < 1e-30):
            _t, _ = torch.nonzero(r_norm < 1e-30, as_tuple=True)
            num_small_els = _t.shape[0]
            raise ValueError("r_norm < 1e-30 in {} elements.".format(num_small_els))
      
        
        # Compute unit vectors r̂
        # Shape: (n_sensor, n_source, 3)
        r_hat = r / r_norm  # add small epsilon to avoid division by zero
        
        # Compute the 3x3 matrix for each source-sensor pair
        # G_ij = (μ₀/4π) * [3*r̂_i*r̂_j - δ_ij] / |r|³
        # Shape: (n_sensor, n_source, 3, 3)
        
        # Outer product: r̂_i * r̂_j
        r_hat_outer = r_hat[:, :, :, None] * r_hat[:, :, None, :]
        
        # Identity matrix
        eye = torch.eye(3, dtype=r.dtype, device=r.device)
        
        # Combine terms
        G = 3.0 * r_hat_outer - eye[None, None, :, :]
        
        # Scale by (μ₀/4π) / |r|³
        prefactor = MU0 / (4 * torch.pi) / (r_norm[:, :, :, None] ** 3)
        ffm = prefactor * G
          
        return ffm
    
    def plot_ffm_value_distribution(self, ffm=None, source_location=None, sensor_location=None, 
                                       method="histogram", show=False, max_points=50000):
        """Plot the value distribution of the forward-field matrix.
        
        Args:
            ffm: Forward-field matrix. If None, uses self.ffm.
            source_location: Which source indices to include (None = all).
            sensor_location: Which sensor indices to include (None = all).
            method: "histogram" for bar chart with symlog axes, "jitter" for strip plot with density-based alpha.
            show: If True, call plt.show(). Otherwise return the figure.
            max_points: For jitter mode, subsample to this many points for performance.
        """
        if ffm is None:
            ffm = self.ffm
        
        if source_location is None:
            source_location = np.arange(ffm.shape[1])
        elif isinstance(source_location, int):
            source_location = np.array([source_location])
        elif isinstance(source_location, list):
            source_location = np.array(source_location)
        elif isinstance(source_location, torch.Tensor):
            source_location = source_location.cpu().numpy()
        elif isinstance(source_location, float):
            source_location = np.array([int(source_location * ffm.shape[1])])
        else:
            raise ValueError(f"Unexpected type for source_location: {type(source_location)}.")
        
        if sensor_location is None:
            sensor_location = np.arange(ffm.shape[0])
        elif isinstance(sensor_location, int):
            sensor_location = np.array([sensor_location])
        elif isinstance(sensor_location, list):
            sensor_location = np.array(sensor_location)
        elif isinstance(sensor_location, torch.Tensor):
            sensor_location = sensor_location.cpu().numpy()
        elif isinstance(sensor_location, float):
            sensor_location = np.array([int(sensor_location * ffm.shape[0])])
        else:
            raise ValueError(f"Unexpected type for sensor_location: {type(sensor_location)}.")
        
        ffm_values = ffm[sensor_location, :, :, :][:, source_location, :, :]
        if isinstance(ffm_values, torch.Tensor):
            ffm_values = ffm_values.detach().cpu().numpy()
        values = ffm_values.flatten()
        values = values[values != 0]
        
        if values.size == 0:
            fig, ax = plt.subplots()
            ax.set_title("FFM value distribution (no nonzero values)")
            if show:
                plt.show()
            return fig
        
        max_abs = np.abs(values).max()
        min_abs = np.abs(values[values != 0]).min()
        linthresh = min_abs
        
        if method == "histogram":
            return self._plot_ffm_histogram(values, max_abs, min_abs, linthresh, show)
        elif method == "jitter":
            return self._plot_ffm_jitter(values, max_abs, linthresh, max_points, show)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'histogram' or 'jitter'.")
    
    def _plot_ffm_histogram(self, values, max_abs, min_abs, linthresh, show):
        """Histogram mode: symlog x and y axes with bar chart."""
        n_decades = int(np.ceil(np.log10(max_abs))) - int(np.floor(np.log10(min_abs))) + 1
        n_bins_per_decade = 5
        
        pos_edges = np.logspace(np.log10(linthresh), np.log10(max_abs), n_decades * n_bins_per_decade + 1)
        neg_edges = -pos_edges[::-1]
        bin_edges = np.concatenate([neg_edges, pos_edges])
        
        counts, edges = np.histogram(values, bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(centers, counts, width=np.diff(edges), align="center", edgecolor="none", alpha=0.7, color="steelblue")
        
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("FFM value (symlog)")
        ax.set_ylabel("Count (symlog)")
        ax.set_title("FFM value distribution (histogram)")
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        if show:
            plt.show()
        return fig
    
    def _plot_ffm_jitter(self, values, max_abs, linthresh, max_points, show):
        """Jitter mode: strip plot with density-based opacity. X is value (symlog), Y is random jitter."""
        # Subsample if too many points
        if values.size > max_points:
            idx = np.random.choice(values.size, max_points, replace=False)
            values = values[idx]
        
        # Estimate density in symlog-transformed space for alpha calculation
        def symlog_transform(x, linthresh):
            sign = np.sign(x)
            abs_x = np.abs(x)
            return sign * np.where(abs_x <= linthresh, abs_x / linthresh, 1 + np.log10(abs_x / linthresh))
        
        transformed = symlog_transform(values, linthresh)
        
        # Bin-based density estimate in transformed space
        n_bins = 100
        counts, bin_edges = np.histogram(transformed, bins=n_bins)
        bin_idx = np.digitize(transformed, bin_edges[:-1]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        densities = counts[bin_idx]
        
        # Normalize density to alpha: high density = low alpha, low density = high alpha
        max_density = densities.max()
        alpha = 0.02 + 0.5 * (1 - densities / max_density)  # range [0.02, 0.52]
        
        # Random y jitter
        y_jitter = np.random.uniform(-1, 1, size=values.size)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Scatter with per-point alpha via RGBA colors
        colors = np.zeros((values.size, 4))
        colors[:, 0] = 0.2  # R
        colors[:, 1] = 0.4  # G
        colors[:, 2] = 0.8  # B
        colors[:, 3] = alpha
        
        ax.scatter(values, y_jitter, c=colors, s=1, rasterized=True)
        
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel("FFM value (symlog)")
        ax.set_ylabel("Jitter (random)")
        ax.set_title(f"FFM value distribution (jitter, n={values.size})")
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.set_yticks([])
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, axis="x", which="both", ls="--", alpha=0.3)
        
        if show:
            plt.show()
        else:
            return fig
        
    @staticmethod
    def _get_compiled_function(r_source, r_sensor):
        """Create and pre-compile optimized numba function for this geometry.
        
        Args:
            r_source: (n_source, 3) numpy array
            r_sensor: (n_sensor, 3) numpy array
            
        Returns:
            Compiled function that computes B from m
        """
        # Trigger compilation with dummy data
        n_source = r_source.shape[0]
        dummy_m = np.zeros((n_source, 3), dtype=np.float32)
        _ = _compute_dipole_field_optimized(r_source, r_sensor, dummy_m)
        
        # Return the compiled function bound to these specific positions
        def compute_field(m):
            return _compute_dipole_field_optimized(r_source, r_sensor, m)
        
        return compute_field

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _compute_dipole_field_optimized(r_source, r_sensor, m):
    """Optimized numba function to compute magnetic dipole field.
    
    Computes the field directly from positions and dipole moments,
    avoiding storage of large FFM matrices. Uses parallel loops for speed.
    
    For a magnetic dipole at r_source with moment m, field at r_sensor is:
    B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / |r|³
    
    Args:
        r_source: (n_source, 3) array of dipole positions
        r_sensor: (n_sensor, 3) array of sensor positions  
        m: (n_source, 3) array of dipole moments
        
    Returns:
        B: (n_sensor, 3) array of magnetic field
    """
    n_sensor = r_sensor.shape[0]
    n_source = r_source.shape[0]
    B = np.zeros((n_sensor, 3), dtype=np.float32)
    
    mu0_4pi = 1e-7  # μ₀/(4π) in SI units
    
    # Parallel loop over sensors
    for i in range(n_sensor):
        # Accumulate contributions from all source dipoles
        for j in range(n_source):
            # Displacement vector r = r_sensor - r_source
            rx = r_sensor[i, 0] - r_source[j, 0]
            ry = r_sensor[i, 1] - r_source[j, 1]
            rz = r_sensor[i, 2] - r_source[j, 2]
            
            # Distance |r|
            r_norm = np.sqrt(rx*rx + ry*ry + rz*rz)
            
            # Skip if too close (avoid singularity)
            if r_norm < 1e-10:
                continue
            
            # Unit vector r̂
            r_norm_inv = 1.0 / r_norm
            rx_hat = rx * r_norm_inv
            ry_hat = ry * r_norm_inv
            rz_hat = rz * r_norm_inv
            
            # Dot product m·r̂
            m_dot_rhat = m[j, 0]*rx_hat + m[j, 1]*ry_hat + m[j, 2]*rz_hat
            
            # Prefactor: (μ₀/4π) / |r|³
            prefactor = mu0_4pi / (r_norm * r_norm * r_norm)
            
            # B = (μ₀/4π) * [3(m·r̂)r̂ - m] / |r|³
            B[i, 0] += prefactor * (3.0 * m_dot_rhat * rx_hat - m[j, 0])
            B[i, 1] += prefactor * (3.0 * m_dot_rhat * ry_hat - m[j, 1])
            B[i, 2] += prefactor * (3.0 * m_dot_rhat * rz_hat - m[j, 2])
    
    return B
    
    
class CurrentDipolePropagator(Propagator):
        
    def __init__(self, r_source, r_sensor):
        self.ffm = self.get_ffm(r_source=r_source, r_sensor=r_sensor)
        pass
    
    @staticmethod
    def get_ffm(r_source, r_sensor, as_matrix=False):
        """Get forward-field matrix (FFM) for a current dipole propagator. FFM is a matrix
        that connects 3 components of a current dipole at location r_i to the 3 magnetic field 
        components at location r_j"""
        if isinstance(r_source, list):
            r_source = torch.tensor(r_source, dtype=torch.float32)
        
        if isinstance(r_sensor, list):
            r_sensor = torch.tensor(r_sensor, dtype=torch.float32)
        
        n_sensor = r_sensor.shape[0]
        n_source = r_source.shape[0]
        
        tau = torch.empty((n_sensor, n_source, 3))
        tau[:, :] = r_sensor[:, None, :] - r_source[None, :, :]
        tau = MU0 / (4 * torch.pi) * tau / (torch.norm(tau, dim=-1, keepdim=True) ** 3)
        # inf is possible when |τ| = 0, it should return an infinite field as well
        
        """
        cross_product_matrix, indexed `ijr` with shape (3, 3, 3) is such a matrix (tensor), that, 
        when contracted with any vector [v_r] gives a transformation: • × v, where • is another 
        arbitrary vector
        """
        cross_product_matrix = torch.zeros((3, 3, 3))
        cross_product_matrix[:, :, 0] = torch.tensor(
            [[ 0,  0,  0],
            [ 0,  0,  1],
            [ 0, -1,  0]])

        cross_product_matrix[:, :, 1] = torch.tensor(
            [[ 0,  0, -1],
            [ 0,  0,  0],
            [ 1,  0,  0]])

        cross_product_matrix[:, :, 2] = torch.tensor(
            [[ 0,  1,  0],
            [-1,  0,  0],
            [ 0,  0,  0]])
        
        ffm = torch.einsum('smr,ijr->smij', tau, cross_product_matrix)
        if as_matrix:
            ffm = CurrentDipolePropagator.reshape_ffm_to_matrix(ffm)
        return ffm
    
    @staticmethod
    def reshape_ffm_to_matrix(ffm):
        """Reshape forward-field matrix (FFM) to a matrix of shape (3 * n_sensor, 3 * n_source)"""
        n_sensors, n_sources, n_sensor_components, n_source_components = ffm.shape
        
        if n_sensor_components != 3 or n_source_components != 3:
            raise ValueError("FFM must be of shape (n_sensors, n_sources, 3, 3), but got {}.".format(ffm.shape))
        
        ffm_matrix = torch.empty((n_sensors * 3, n_sources * 3))
        
        # Manually assign block values to the big matrix, by selecting corresponding components from the ffm tensor
        ffm_matrix[0::3, 0::3] = ffm[:, :, 0, 0]
        ffm_matrix[0::3, 1::3] = ffm[:, :, 0, 1]
        ffm_matrix[0::3, 2::3] = ffm[:, :, 0, 2]
        
        ffm_matrix[1::3, 0::3] = ffm[:, :, 1, 0]
        ffm_matrix[1::3, 1::3] = ffm[:, :, 1, 1]
        ffm_matrix[1::3, 2::3] = ffm[:, :, 1, 2]
        
        ffm_matrix[2::3, 0::3] = ffm[:, :, 2, 0]
        ffm_matrix[2::3, 1::3] = ffm[:, :, 2, 1]
        ffm_matrix[2::3, 2::3] = ffm[:, :, 2, 2]
        
        return ffm_matrix
    
    @staticmethod
    def reshape_to_vector(vec):
        """Reshape magnetic field matrix to a vector of shape (3 * n_sensor)"""
        return vec.view(-1)
    
    @staticmethod
    def reshape_from_vector(vec):
        """Given a vector of shape (3 * N), reshape it to a matrix of shape (N, 3). Assumes that 
        the vector is ordered as [v_x_1, v_y_1, v_z_1, v_x_2, v_y_2, v_z_2, ...], i.e. the components varies first"""
        return vec.view(-1, 3)
    
    def get_B_from_J(self, J):
        """Shape of J is (n_pts, 3), J can be thought of as current dipoles located at 
        n_pts and having 3 components that define the direction and amplitude of the current."""
        return torch.einsum('smij,mj->si', self.ffm, J)
    
    @staticmethod
    def get_B_at_pts_from_J_at_pts(J, r_source, r_sensor):
        if isinstance(J, np.ndarray):
            J = torch.from_numpy(J)
        elif isinstance(J, list):
            J = torch.tensor([J], dtype=torch.float32)
            
        if isinstance(r_source, np.ndarray):
            r_source = torch.from_numpy(r_source)
        elif isinstance(r_source, list):
            r_source = torch.tensor([r_source], dtype=torch.float32)
            
        if isinstance(r_sensor, np.ndarray):    
            r_sensor = torch.from_numpy(r_sensor)
        elif isinstance(r_sensor, list):
            r_sensor = torch.tensor([r_sensor], dtype=torch.float32)
        
        if J.shape == (3,) and r_source.shape == (3,):
            J = J[None, :]
            r_source = r_source[None, :]
            
        ffm = CurrentDipolePropagator.get_ffm(r_source, r_sensor)
        return torch.einsum('smij,mj->si', ffm, J)
    
    def get_J_from_B(self, B, method="penrose"):
        """Find the current dipole components J from the magnetic field components B using
        the pseudo-inverse (Moore-Penrose inverse) of the forward-field matrix (FFM) if method is `penrose`,
        otherwise use more stable `lstsq` method by torch."""
        ffm = self.reshape_ffm_to_matrix(self.ffm)
        B = self.reshape_to_vector(B)
        if method == "penrose":
            iffm = torch.pinverse(ffm)
            s = torch.einsum('ji,i->j', iffm, B)
        elif method == "lstsq":
            s = torch.linalg.lstsq(ffm, B).solution
        return self.reshape_from_vector(s)
        
    

class AxisProjectionPropagator(Propagator):

    def __init__(self, theta, phi, keepdims=False):
        self.n = SphericalUnitVectorKernel.define_unit_vector(theta, phi)
        self.keepdims = keepdims

    def project(self, x, keepdims=None):
        res = torch.einsum('...cij,c->...ij', x, self.n.type(x.type()))
        keepdims = keepdims if keepdims else self.keepdims
        if keepdims:
            res = res.unsqueeze(-3)
        return res

    def __call__(self, x, keepdims=None):
        return self.project(x, keepdims=keepdims)
    
    def plot_axis_projection(self, ax=None, arc_radius=0.4, elev=None, azim=None, **kwargs):
        """Visualize the projection axis n relative to the xyz coordinate frame.
        
        Camera is auto-positioned so that n lies in the screen plane (maximizes its projected length).
        The azimuth is set perpendicular to n's xy projection, elevation matches n's polar angle.
        Labels are placed away from the camera to avoid overlap with arrows.
        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
        
        nx, ny, nz = float(self.n[0]), float(self.n[1]), float(self.n[2])
        theta_rad = np.arccos(np.clip(nz, -1, 1))
        phi_rad = np.arctan2(ny, nx)
        theta_deg = np.degrees(theta_rad)
        phi_deg = np.degrees(phi_rad)
        
        # Camera angle: look perpendicular to the plane containing z and n,
        # so n appears fully in the screen plane with maximum visual length.
        # azim = phi + 90 makes the camera look from the side of n's xy projection.
        # elev slightly above to see depth.
        if azim is None:
            azim = phi_deg + 90
        if elev is None:
            elev = max(15, min(35, 90 - theta_deg * 0.5))
        
        # Set view early so we can compute screen-space directions for label placement
        ax.view_init(elev=elev, azim=azim)
        
        # Camera direction in 3D (unit vector pointing from scene toward camera).
        # matplotlib uses azim measured from -y toward +x, elev from xy plane toward +z.
        azim_r = np.radians(azim)
        elev_r = np.radians(elev)
        cam = np.array([
            np.cos(elev_r) * np.cos(azim_r),
            np.cos(elev_r) * np.sin(azim_r),
            np.sin(elev_r)
        ])
        
        def label_offset(tip, scale=0.15):
            """Push label away from camera so it doesn't sit on top of the arrow.
            Offset = outward from origin + slightly toward camera for depth clarity."""
            outward = np.array(tip)
            norm = np.linalg.norm(outward)
            if norm > 1e-10:
                outward = outward / norm
            return np.array(tip) + outward * scale
        
        # xyz basis arrows
        arrow_kw = dict(arrow_length_ratio=0.08, linewidth=1.5)
        ax.quiver(0, 0, 0, 1, 0, 0, color='tab:red', alpha=0.6, **arrow_kw)
        ax.quiver(0, 0, 0, 0, 1, 0, color='tab:green', alpha=0.6, **arrow_kw)
        ax.quiver(0, 0, 0, 0, 0, 1, color='tab:blue', alpha=0.6, **arrow_kw)
        
        lx = label_offset([1, 0, 0])
        ly = label_offset([0, 1, 0])
        lz = label_offset([0, 0, 1])
        ax.text(*lx, '$x$', color='tab:red', fontsize=13, ha='center', va='center')
        ax.text(*ly, '$y$', color='tab:green', fontsize=13, ha='center', va='center')
        ax.text(*lz, '$z$', color='tab:blue', fontsize=13, ha='center', va='center')
        
        # Projection axis n
        ax.quiver(0, 0, 0, nx, ny, nz, color='black', linewidth=2.8, arrow_length_ratio=0.1)
        ln = label_offset([nx, ny, nz], 0.18)
        ax.text(*ln, r'$\hat{n}$', color='black', fontsize=14, fontweight='bold', ha='center', va='center')
        
        # Dashed drop lines
        dash_kw = dict(color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.plot([nx, nx], [ny, ny], [0, nz], **dash_kw)
        ax.plot([0, nx], [0, ny], [0, 0], **dash_kw)
        
        # Theta arc: z-axis toward n, in the plane spanned by z_hat and n's xy projection
        r = arc_radius
        n_xy = np.sqrt(nx**2 + ny**2)
        if n_xy > 1e-10:
            u_xy = np.array([nx, ny, 0]) / n_xy
            t = np.linspace(0, theta_rad, 40)
            arc_theta = np.column_stack([r * np.sin(t) * u_xy[0],
                                         r * np.sin(t) * u_xy[1],
                                         r * np.cos(t)])
            ax.plot(arc_theta[:, 0], arc_theta[:, 1], arc_theta[:, 2],
                    color='tab:blue', linewidth=1.5, alpha=0.8)
            # Place theta label at the outer edge of the arc midpoint, pushed away from camera
            mid = len(t) // 2
            mid_pt = arc_theta[mid]
            mid_outward = mid_pt / (np.linalg.norm(mid_pt) + 1e-10)
            lbl = mid_pt + mid_outward * r * 0.5
            ax.text(*lbl, f'$\\theta={theta_deg:.1f}°$', color='tab:blue', fontsize=10, ha='center', va='center')
        
        # Phi arc: x-axis toward n's xy projection, in the xy plane
        if n_xy > 1e-10:
            p = np.linspace(0, phi_rad, 40)
            rp = r * 0.7
            arc_phi = np.column_stack([rp * np.cos(p), rp * np.sin(p), np.zeros_like(p)])
            ax.plot(arc_phi[:, 0], arc_phi[:, 1], arc_phi[:, 2],
                    color='tab:red', linewidth=1.5, alpha=0.8)
            mid = len(p) // 2
            mid_pt = arc_phi[mid]
            mid_outward = mid_pt / (np.linalg.norm(mid_pt) + 1e-10)
            lbl = mid_pt + mid_outward * rp * 0.7
            ax.text(*lbl, f'$\\phi={phi_deg:.1f}°$', color='tab:red', fontsize=10, ha='center', va='center')
        
        # Remove all axis chrome
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        
        ax.set_aspect('equal')
        ax.set_title(f'$\\hat{{n}}=({nx:.3f},\\, {ny:.3f},\\, {nz:.3f})$', fontsize=11, pad=0)
        
        return ax

# TODO: Implement MagneticFieldComponentsPropagator using the kernel