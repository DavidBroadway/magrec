import torch
import numpy as np
from abc import ABC, abstractmethod

from magrec.nn.modules import GaussianFourierFeaturesTransform, ZeroDivTransform
from magrec.prop.Propagator import CurrentPropagator2d

import pytorch_lightning as L

class FourierFeatures2dCurrent(torch.nn.Module):
    """2d current distribution function encoded with Fourier features neural network.
    
    By contstruction the current is divergence-free.
    """
    def __init__(self, ff_sigmas=[(1, 10), (2, 10), (0.5, 10)]) -> None:
        """
        Args:
            ff_sigmas: a list of standard deviations and the number of frequencies to sample for each sigma.
                       Should be of the form [(sigma1, n1), (sigma2, n2), ...]
        """
        super().__init__()
        self.ffs = torch.nn.ModuleList([GaussianFourierFeaturesTransform(2, n, sigma) for sigma, n in ff_sigmas])
        self.ff_features_n = sum([n * 2 for _, n in ff_sigmas])
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
    

class FourierFeaturesPINN(L.LightningModule):
    
    def __init__(self, ff_sigmas, learning_rate=1e-1) -> None:
        super().__init__()
        self.ff_sigmas = ff_sigmas
        self.soln = FourierFeatures2dCurrent(ff_sigmas=ff_sigmas)
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


