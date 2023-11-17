import torch
import numpy as np
from abc import ABC, abstractmethod

from magrec.nn.modules import GaussianFourierFeaturesTransform, ZeroDivTransform
from magrec.prop.Propagator import CurrentPropagator2d
from magrec.nn.utils import load_model_from_ckpt

import pytorch_lightning as L

class Fourier2dMagneticField(torch.nn.Module):
    """2d Fourier space representation of the magnetic field at a fixed measurement plane 
    distance z from a planar source distribution.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 6),  # 6 components for real and imaginary parts of the field Fourier components
        )
        
    def forward(self, x):
        """Returns a vector of the three components of the  magnetic field vector [b_x, b_y, b_z](k_x, k_y | z) 
        as a function of 2d Fourier components k_x, k_y, and z. Each of the three vector components is a Fourier 
        component, thus it is complex. 
        
        Since Fourier components are complex, there's a connection between the real and imaginary parts of the k_y 
        and -k_y components: c(k_y) = c*(-k_y). I tried combining different ouputs of the network to enforce this, such as
        
        ```python
        f_kx_ky = self.fcn(x)
        f_kx_neg_ky = self.fcn(torch.stack([x[:, 0], -x[:, 1]], dim=1))
    
        x = torch.stack([f_kx_neg_ky[:, 3:], -f_kx_ky[:, :3]], dim=2)
        x = torch.view_as_complex(x)  # shape (n_batches, 3), dtype complex64
        ```
        
        But it doesn't seem to work. Below is another a wrong approach, also leading to non-equal values 
        of the real and imaginary parts of the field, as -k_y has just another value, with no connection to
        the value of k_y:
        
        ```python
        f_kx_neg_ky = self.fcn(torch.stack([x[:, 0], -x[:, 1]], dim=1))
        x = torch.stack([f_kx_neg_ky[:, 3:], -f_kx_neg_ky[:, :3]], dim=2)
        x = torch.view_as_complex(x)  # shape (n_batches, 3), dtype complex64
        ```
        
        The best way to proceed would be to consider the values of the network for
        k_y >= 0, and use Fourier transform with explicit real signal set. 
        
        BENEFITS: 
        + Network is run on 1/2 of values.
        + Easy.
        DRAWBACKS:
        - Requires passing only k_y >= 0 to the network.
        """
        # x = (k_x, k_y)
        # Utilize the symmetry of the Fourier transform to reduce the number of parameters by half.
        # The symmetry is that the Fourier transform is symmetric with respect to k_y -> -k_y
        
        x = self.fcn(x)
        # Split the output into real and imaginary parts and combine them into a complex number.
        # The output is a complex number for each of the three components of the field.
        x = torch.stack([x[:, 3:], x[:, :3]], dim=2)
        x = torch.view_as_complex(x)  # shape (n_batches, 3), dtype complex64
        return x
        

class FourierFeatures2dCurrent(torch.nn.Module):
    """2d current distribution function encoded with Fourier features neural network.
    
    By contstruction the current is divergence-free.
    """
    def __init__(self, ff_stds=[(1, 10), (2, 10), (0.5, 10)]) -> None:
        """
        Args:
            ff_stds: a list of standard deviations and the number of frequencies to sample for each std.
                       Should be of the form [(std1, n1), (std2, n2), ...]
        """
        super().__init__()
        self.ffs = torch.nn.ModuleList([GaussianFourierFeaturesTransform(2, n, std) for std, n in ff_stds])
        self.ff_features_n = sum([n * 2 for _, n in ff_stds])
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(self.ff_features_n, self.ff_features_n),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.ff_features_n, 1),
        )
        self.divless = ZeroDivTransform()
        self.requires_grad = True
        
    def forward(self, x):
        # self.divless requires gradients to compute divergence-free field
        # so we enable gradients for x  
        y = torch.cat([ff(x) for ff in self.ffs], dim=1)
        y = self.fcn(y)
        y = self.divless(y, x)
        return y
    
    @classmethod
    def load_model_from_ckpt(cls, version_n=None, ckpt_name_regexp='last.ckpt', folder_name='', ckpt=None, ckpt_path=None):
        """Loads the model from a checkpoint, handling legacy naming of arguments and possible quirks of the model. 
        Uses `load_model_from_ckpt` generic function."""
        return load_model_from_ckpt(cls, version_n=version_n, ckpt_name_regexp=ckpt_name_regexp, 
                                    folder_name=folder_name, type="ff_std_cond",
                                    ckpt=ckpt, ckpt_path=ckpt_path)
        
    

class FourierFeaturesPINN(L.LightningModule):
    
    def __init__(self, ff_sigmas, learning_rate=1e-1) -> None:
        super().__init__()
        self.ff_sigmas = ff_sigmas
        self.soln = FourierFeatures2dCurrent(ff_stds=ff_sigmas)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        J = self.soln(x)
        return J
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model. 
        
        In PINN, expect batch to contain elements of the form (X, cond) where X is a set of 
        points and cond is a Condition object. Condition objects are then used to evaluate the loss. 
        
        At each step, we need to evaluate values of self.soln(X) at a set of points X. 
        Some points are used to connect with the physical model, and some enforce specific boundary conditions
        on `self.soln`. 
        
        """
        # The conditions are the following:
        # 1. Having a grid X_grid, the magnetic field B_hat = self.prop(self.soln(X_grid)) should be equal to the self.cond(X_grid)
        # 2. The current density should be zero in specific regions given by cond2.in_region(x)
        # 3. The derivative should be minimal in specific regions given by cond3.in_region(x) (where the current is constant, we suspect)
        loss = 0 
    
        for x, cond in batch:
            # This becomes a convoluted logics of where we actually enable tracking
            # of gradients for the input x. Should it be a responsibility of dataloader
            # that returns the tensor x? Or should it be modules that operate on x that
            # enable tracking the gradients? 
            x.requires_grad_(True)
            y_hat = self.soln(x)
            cond_loss = cond.loss(x, y_hat)
            self.log(f'{cond.name}_loss', cond_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
            loss += cond_loss
        
        self.log('total_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # For validation, we show plots and corresponding errors from targets, if any. 
        for x, cond in batch:
            
            with torch.enable_grad():
                x = x.requires_grad_(True)
                y_hat = self.soln(x)
                try:
                    cond.validate(x, y_hat, self.logger.experiment, step=self.global_step)
                except AttributeError:
                    self.logger.experiment.add_text(tag='error', text_string=f'No validation for {cond.name}')
                    pass
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return torch.optim.LBFGS(self.parameters(), lr=1, max_iter=20, max_eval=30, history_size=10, line_search_fn='strong_wolfe')
                

class GenericModel(ABC):
    # Super class that other models can be based off.

    def __init__(self, data):
        """
        Args:
            data:   The 2D array that will be used in the training.
                    This is passed to automatically define the size of the network.

        """
        # Define the default values fro the architexure for this model.
        self.arch = dict()
        self.arch["n_channels_in"] = 1
        self.arch["n_channels_out"] = 1

        self.arch["kernel"] = 5
        self.arch["stride"] = 2
        self.arch["padding"] = 2

        # Pad the data to define the size of the neural network.
        padded_data = self.pad_2d_array(data)
        self.arch["size"] = np.shape(padded_data)[0]

    @abstractmethod
    def model(self, nn_output):
        """
        Args:
            nn_output: The output of the neural network

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        raise NotImplementedError("model must be overridden in a child class.")

    def pad_2d_array(self, arr: np.ndarray) -> np.ndarray:
        rows, cols = arr.shape
        new_rows = self.next_power_of_two(rows)
        new_cols = self.next_power_of_two(cols)
        padded_arr = np.pad(arr, [(0, new_rows - rows), (0, new_cols - cols)], mode='constant')
        return padded_arr

    def next_power_of_two(self, x: int) -> int:
        return 1 if x == 0 else 2**(x - 1).bit_length()



class UniformDirectionMagnetisation(GenericModel):
    def __init__(self, data, dx, dy, height, layer_thickness):
        super().__init__(data)

        # Define the propagator so that this isn't performed during a loop.
        self.define_propagtor(data, dx, dy, height, layer_thickness)

    def define_propagtor(self, data, dx, dy, height, layer_thickness):
        from magrec.prop.Propagator import MagnetizationPropagator2d as Propagator
        self.propagator = Propagator(data.shape, dx, dy, height, layer_thickness)



    def model(self, nn_output):
        """
        Args:
            nn_output:  The output of the neural network which is a be a 2D array of
                        magnetisation along a single direction

        Returns:
            magnetic_field: The magnetic field produced from the output of the network
        """
        b = self.propagator.get_B(nn_output)
        return b


