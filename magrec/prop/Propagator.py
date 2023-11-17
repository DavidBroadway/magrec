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
from magrec.prop.constants import DEFAULT_UNITS, get_exponent_from_unit
from magrec.prop.Kernel import (
    UniformLayerFactor2d, 
    MagnetizationFourierKernel2d, 
    CurrentFourierKernel2d, 
    CurrentLayerFourierKernel2d, 
    InverseCurrentLayerFourierKernel2d,
    SphericalUnitVectorKernel
    )


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

        # Take into the account provided units, if any, otherwise use default `DEFAULT_UNITS`
        self.units = DEFAULT_UNITS.copy()
        if units is not None:
            # .set_units() needs to be called after .j_to_b_matrix is defined. I know it's not optimal
            # but that needs to do for now. I'd prefer for units setting to be order-independent. One
            # option is to have an attribue .prefactor that is multiplied by the j_to_b_matrix, but then  
            self.set_units(units)
            
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
    
    def set_units(self, *units):
        """Set the units of the propagator if they are different from the default units.
        
        `units` can be a dict with some of the keys `current`, `length` and `magnetic_field` that 
        specify the units of the current, length, and magnetic field, respectively, to be updated. 
        Example:
        
        units = {
            "current": "A",
            "length": "mm",
            "magnetic_field": "mT"
        } <--- default units
        
        units = {
            "current": "mA",
            "length": "um",  # micrometers
            "magnetic_field": "uT"  # microtesla
        } <--- modified units that warrant changing `self.j_to_b_matrix` 
        
        Also `set_units` accept up to three positional arguments that specify the units of the current, 
        length, and magnetic field, respectively, as string. Order of the arguments is not important, 
        and will be figured out from the major units of the provided strings "T" for magnetic field, 
        "A" for current, and "m" for length.
        
        Values assumed to be affected by the units are the following (from magrec.prop.constants):
        
            MU0: float = 1.25663706212  # [mT * mm / A]
            
        Other values with units as assumed to be already in the proper units, e.g. 
        
            `height`, `layer_thickness`, `dx`, `dy`         units["length"]
            `J`                                             units["current"] / units["length"]^2
            `j` (Fourier image of J)                        units["current"]
            
        To get the proper units for the magnetic field, we modify MU0 to account for the change in units.
        Since MU0 is already contained as a multiplicative factor in `self.j_to_b_matrix`, we modify it by
        the factor we calculate from the change of units from the default to the provided units.
        """
        # copy the default units and update them with the provided units
        updated_units = self.units.copy()
        
        if len(units) == 1:
            if isinstance(units[0], dict):
                units = units[0]
                updated_units.update(units)
            else:
                raise ValueError("If only one argument is provided, it must be a dict with the units to be updated.")
        elif len(units) > 1:
            for unit in units:
                if not isinstance(unit, str):
                    raise ValueError("If more than one argument is provided, they must be strings, \
                        but got type `{}` instead for {}.".format(type(unit), unit))
                if unit[-1] == "m":
                    updated_units["length"] = unit
                elif unit[-1] == "A":
                    updated_units["current"] = unit
                elif unit[-1] == "T":
                    updated_units["magnetic_field"] = unit
                else:
                    raise ValueError("Units specification have one of 'm', 'A', or 'T' endings for length, \
                                     current, and magnetic field, respectively, got `{}`.".format(unit))
        
        # Prepare dicts with exponents of the units, to calculate the multiplicative prefactor
        updated_exponents = {}
        default_exponents = {}
        # Iterate throught the keys of the units dict and get the exponents of the units, 
        # and obtain the exponents of the  already set units
        for key in updated_units.keys():
            default_exponents[key] = get_exponent_from_unit(self.units[key])
            updated_exponents[key] = get_exponent_from_unit(updated_units[key])
        # Prefactor exponent is the order by which the multiplicative factor differs in set units from the updated units
        prefactor_exponent = \
            (default_exponents["magnetic_field"] - updated_exponents["magnetic_field"]) + \
            (default_exponents["length"] - updated_exponents["length"]) - \
            (default_exponents["current"] - updated_exponents["current"])
        # The reason to update the prefactor of this matrix, is that this is the only unit conversion point, which is not
        # supplied by the user, and it uses the constant which connects a new quantity (magnetic field) to those supplied 
        # by the user (current and length). 
        self.j_to_b_matrix = self.j_to_b_matrix * (10 ** (prefactor_exponent))
        self.units = updated_units
        return self

    def B_from_J(self, J):
        if isinstance(J, np.ndarray):
            J = torch.from_numpy(J)

        j = self.ft.forward(J, dim=(-2, -1))
        b = self.get_b_from_j(j)
        B = self.ft.backward(b, dim=(-2, -1))
        if not self.ft.real_signal:
            B = B.real
        return B
    
    
class InverseCurrentPropagator2d(object):
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
    
    def set_units(self, *units):
        """Set the units of the propagator if they are different from the default units.
        
        `units` can be a dict with some of the keys `current`, `length` and `magnetic_field` that 
        specify the units of the current, length, and magnetic field, respectively, to be updated. 
        Example:
        
        units = {
            "current": "A",
            "length": "mm",
            "magnetic_field": "mT"
        } <--- default units
        
        units = {
            "current": "mA",
            "length": "um",  # micrometers
            "magnetic_field": "uT"  # microtesla
        } <--- modified units that warrant changing `self.j_to_b_matrix` 
        
        Also `set_units` accept up to three positional arguments that specify the units of the current, 
        length, and magnetic field, respectively, as string. Order of the arguments is not important, 
        and will be figured out from the major units of the provided strings "T" for magnetic field, 
        "A" for current, and "m" for length.
        
        Values assumed to be affected by the units are the following (from magrec.prop.constants):
        
            MU0: float = 1.25663706212  # [mT * mm / A]
            
        Other values with units as assumed to be already in the proper units, e.g. 
        
            `height`, `layer_thickness`, `dx`, `dy`         units["length"]
            `J`                                             units["current"] / units["length"]^2
            `j` (Fourier image of J)                        units["current"]
            
        To get the proper units for the magnetic field, we modify MU0 to account for the change in units.
        Since MU0 is already contained as a multiplicative factor in `self.j_to_b_matrix`, we modify it by
        the factor we calculate from the change of units from the default to the provided units.
        """
        # copy the default units and update them with the provided units
        updated_units = self.units.copy()
        
        if len(units) == 1:
            if isinstance(units[0], dict):
                units = units[0]
                updated_units.update(units)
            else:
                raise ValueError("If only one argument is provided, it must be a dict with the units to be updated.")
        elif len(units) > 1:
            for unit in units:
                if not isinstance(unit, str):
                    raise ValueError("If more than one argument is provided, they must be strings, \
                        but got type `{}` instead for {}.".format(type(unit), unit))
                if unit[-1] == "m":
                    updated_units["length"] = unit
                elif unit[-1] == "A":
                    updated_units["current"] = unit
                elif unit[-1] == "T":
                    updated_units["magnetic_field"] = unit
                else:
                    raise ValueError("Units specification have one of 'm', 'A', or 'T' endings for length, \
                                     current, and magnetic field, respectively, got `{}`.".format(unit))
        
        # Prepare dicts with exponents of the units, to calculate the multiplicative prefactor
        updated_exponents = {}
        default_exponents = {}
        # Iterate throught the keys of the units dict and get the exponents of the units, 
        # and obtain the exponents of the  already set units
        for key in updated_units.keys():
            default_exponents[key] = get_exponent_from_unit(self.units[key])
            updated_exponents[key] = get_exponent_from_unit(updated_units[key])
        # Prefactor exponent is the order by which the multiplicative factor differs in set units from the updated units
        prefactor_exponent = \
          - (default_exponents["magnetic_field"] - updated_exponents["magnetic_field"]) - \
            (default_exponents["length"] - updated_exponents["length"]) + \
            (default_exponents["current"] - updated_exponents["current"])
        # The reason to update the prefactor of this matrix, is that this is the only unit conversion point, which is not
        # supplied by the user, and it uses the constant which connects a new quantity (magnetic field) to those supplied 
        # by the user (current and length). 
        self.b_to_j_matrix = self.b_to_j_matrix * (10 ** (prefactor_exponent))
        self.units = updated_units
        return self

    def J_from_B(self, B):
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B)

        b = self.ft.forward(B, dim=(-2, -1))
        j = self.get_j_from_b(b)
        J = self.ft.backward(j, dim=(-2, -1))
        if not self.ft.real_signal:
            J = J.real
        return J
    


class AxisProjectionPropagator(object):

    def __init__(self, theta, phi, keepdims=False):
        self.n = SphericalUnitVectorKernel.define_unit_vector(theta, phi)
        self.keepdims = keepdims

    def project(self, x):
        res = torch.einsum('...cij,c->...ij', x, self.n.type(x.type()))
        if self.keepdims:
            res = res.unsqueeze(-3)
        return res

    def __call__(self, x):
        return self.project(x)