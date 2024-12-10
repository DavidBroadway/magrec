from typing import Union
from abc import ABC, abstractmethod
import sys

import torch
import numpy as np

from magrec.nn.modules import GaussianFourierFeaturesTransform, DivergenceFreeTransform2d
from magrec.prop.Propagator import CurrentPropagator2d
from magrec.nn.utils import load_model_from_ckpt, plot_ffs_params

from magrec.misc.sampler import GridSampler

import magpylib
import matplotlib.pyplot as plt
import math

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
        self.divless = DivergenceFreeTransform2d()
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
        
        
class FourierFeaturesNd(torch.nn.Module):
    """N-d Fourier features encoded neural network.
    """
    def __init__(self, n_inputs, n_outputs, ff_stds=[(1, 10), (2, 10), (0.5, 10)]) -> None:
        """
        Args:
            ff_stds: a list of standard deviations and the number of frequencies to sample for each std.
                       Should be of the form [(std1, n1), (std2, n2), ...]
        """
        super().__init__()
        self.ffs = torch.nn.ModuleList([GaussianFourierFeaturesTransform(n_inputs, n, std) for std, n in ff_stds])
        self.ff_features_n = sum([n * 2 for _, n in ff_stds])
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(self.ff_features_n, self.ff_features_n),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.ff_features_n, n_outputs),
        )
        
    def forward(self, x):
        y = torch.cat([ff(x) for ff in self.ffs], dim=-1)  # -1 intended to allow for batched and single-pt calculations
        y = self.fcn(y)
        return y
    
    def set_ffs(self, fourier_features: Union[torch.nn.Module, list, tuple], verbose=False):
        """Sets the Fourier features to the given list of Fourier features modules. Can be used
        to substitute e.g. GaussianFourierFeaturesTransform with UniformFourierFeaturesTransform, or 
        a custom Fourier features module."""
        
        if not isinstance(fourier_features, (list, tuple)):
            fourier_features = [fourier_features]
        
        if verbose:
            # Prepare a plot of the Fourier features parameters if verbose
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            plot_ffs_params(self, axs[0])
        
        # Save current state
        prev_ffs = self.ffs
        prev_ff_features_n = self.ff_features_n
        prev_fcn = self.fcn

        # Update with new Fourier features
        self.ffs = torch.nn.ModuleList(list(fourier_features))
        self.ff_features_n = sum([ff.B.shape[1] * 2 for ff in fourier_features])
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(self.ff_features_n, self.ff_features_n),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.ff_features_n, self.fcn[-1].out_features),
        )

        print("The model has been initialized with new params.")
        
        if verbose:
            plot_ffs_params(self, axs[1])
            plt.show()
            
        print("If reinitializing the trained model is not intended, press undo.")

        if 'ipykernel' in sys.modules:
            from IPython.display import display
            import ipywidgets as widgets

            undo = widgets.Button(description="Undo")
            output = widgets.Output()

            def on_undo_clicked(_):
                with output:
                    output.clear_output()
                    output.append_stdout("Undoing changes...\n")
                    self.ffs = prev_ffs
                    self.ff_features_n = prev_ff_features_n
                    self.fcn = prev_fcn
                    output.append_stdout("Changes undone.")

            undo.on_click(on_undo_clicked)
            display(undo, output)
        else:
            print("Confirm undo (y/n): ")
            if input().lower() == 'y':
                self.ffs = prev_ffs
                self.ff_features_n = prev_ff_features_n
                self.fcn = prev_fcn
                print("Changes undone.")
    
# class FourierFeaturesNd(torch.nn.Module):
#     """N-d Fourier features encoded neural network.
#     """
#     def __init__(self, n_inputs, n_outputs, ffs_module_class, **kwargs) -> None:
#         """
#         Args:
#             ff_stds: a list of standard deviations and the number of frequencies to sample for each std.
#                        Should be of the form [(std1, n1), (std2, n2), ...]
#         """
#         if ff_stds in kwargs:
#             self.ffs = torch.nn.ModuleList([GaussianFourierFeaturesTransform(n_inputs, n, std) for std, n in ff_stds])
#         else:
#             self.ffs = torch.nn.ModuleList([ffs_module_class(n_inputs, n, std) for std, n in ff_stds])
#         super().__init__()
#         self.ff_features_n = sum([n * 2 for _, n in ff_stds])
#         self.fcn = torch.nn.Sequential(
#             torch.nn.Linear(self.ff_features_n, self.ff_features_n),
#             torch.nn.Sigmoid(),
#             torch.nn.Linear(self.ff_features_n, n_outputs),
#         )
        
#     def forward(self, x):
#         y = torch.cat([ff(x) for ff in self.ffs], dim=-1)  # -1 intended to allow for batched and single-pt calculations
#         y = self.fcn(y)
#         return y

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


class WireNet(L.LightningModule):
    """A mapping s → (x, y) for recreating a path of a wire. Optional parameter is t`he current which
    can be either learned from the reconstruction, or given as a constraint."""
    
    class WirePath(torch.nn.Module):
            def __init__(self, I=0.0, learnable_I=False):
                super().__init__()
                self.I = I
                if learnable_I:
                    self.I = torch.nn.Parameter(torch.tensor(I))
                ff_stds = [(1, 5), (2, 5), (0.5, 5)]
                self.ffs = torch.nn.ModuleList([GaussianFourierFeaturesTransform(1, n, std) for std, n in (ff_stds)])
                self.ff_features_n = sum([n * 2 for _, n in ff_stds])
                self.fcn = torch.nn.Sequential(
                    torch.nn.Linear(self.ff_features_n, self.ff_features_n),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(self.ff_features_n, 2),
                )
                
            def forward(self, s):
                r = torch.cat([ff(s) for ff in self.ffs], dim=1)
                r = self.fcn(r)
                return r
        
                
    def __init__(self, I=0.0, learnable_I=False):
        super().__init__()
        self.net = self.WirePath(I, learnable_I)
                    
    def forward(self, s):
        r = self.net(s)
        return r
    
    def get_B(self, pos):
        """Returns the magnetic field at the points r generated by the wire."""
        # Sample the wire with s
        n_pts = 100
        r = self.net(torch.linspace(0, 1, n_pts).unsqueeze(1))
        # Add z component to the wire with convention that it is at the plane z = 0
        r = torch.cat([r, torch.zeros((n_pts, 1))], dim=1)
        r = r.detach().numpy()  
        B = magpylib.current.Polyline(position=(0, 0, 0), vertices=r, current=self.net.I).getB(pos)
        return B
    
    def fit_to_field(self, field, n_points=None, max_epochs=1000):
        """Fits the wire to a given magnetic field using the given optimizer setting.
        
        field (torch.Tensor or numpy.array): 2d field distribution to fit to. The field should be a 2d array
            with shape (3, n, m) where n is the number of x points and m is the number of y points, and 3 
            corresponds to the number of components of the magnetic field. 
        
        """
        nx_points, ny_points = field.shape[-2:]
        grid_pts = GridSampler.sample_grid(nx_points, ny_points, origin=[0, 0], diagonal=[1, 1])
        train_dataset = torch.utils.data.TensorDataset(grid_pts, field)
        # Use provided number of points or determine the batch size based on the number of points using a sane heuristic
        batch_size = max(32, 2 ** int(math.log2(nx_points * ny_points) - 4)) if n_points is None else n_points
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size,
                                                   num_workers=4)
        
        val_dataset = torch.utils.data.TensorDataset(grid_pts, field)
        # For validation, pass all points at once
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=nx_points * ny_points)
        
        self.training_step = self._field_training_step
        
        self.trainer = L.Trainer(max_epochs=max_epochs)
        self.trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
    def _field_training_step(self, batch, batch_idx):
        pos, B = batch
        B_pred = torch.tensor(self.get_B(pos))
        loss = torch.nn.functional.mse_loss(B_pred, B)
        return loss

    def fit_to_path(self, path, n_points, max_epochs=1000):
        """Fits the wire to a given path using the given optimizer setting.
        
        path: callable
            A function that takes a torch.Tensor s and returns a torch.Tensor r that gives a n-dimensional
            point of the path corresponding to the parametrization value s.
        """
        # First, get observed values for the path and prepare a dataset with them
        uniform_s = torch.distributions.uniform.Uniform(0, 1).sample((n_points * 10,)).unsqueeze(1)
        uniform_r = path(uniform_s)
        train_dataset = torch.utils.data.TensorDataset(uniform_s, uniform_r)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        
        # Record the path that is learned
        self.path = path
        
        ss = torch.linspace(0, 1, n_points).unsqueeze(1)
        rr = path(ss)
        val_dataset = torch.utils.data.TensorDataset(ss, rr)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50)
        
        self.training_step = self._path_training_step
        
        self.trainer = L.Trainer(max_epochs=max_epochs)
        self.trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
    def _path_training_step(self, batch, batch_idx):
        s, r = batch
        r_pred = self(s)
        loss = torch.nn.functional.mse_loss(r_pred, r)
        return loss
        
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("This method should be overriden.")

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def plot_wire(self, s_range=[0, 1]):
        """Create a plot illustrating the obtained solution from the net."""
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        ss = torch.linspace(*s_range, 100).unsqueeze(1)
        self.net.eval()
        rr = self(ss).detach()
        ax.plot(*rr.T)
        ax.plot(*self.path(ss).T, 'r--')
        