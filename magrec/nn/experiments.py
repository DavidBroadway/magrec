# Data Setup / Imports
from magrec import __logpath__, __datapath__

from copy import deepcopy
import inspect  # to print source code of a function


import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

import tqdm

import pytorch_lightning as L

from scipy.ndimage import gaussian_filter, median_filter

from magrec.nn.modules import GaussianFourierFeaturesTransform, UniformFourierFeaturesTransform, RegularFourierFeaturesTransform
from magrec.nn.models import FourierFeaturesNd, WireNet
from magrec.nn.utils import batched_curl, batched_div, batched_grad, save_model_for_experiment, load_model_for_experiment

from magrec.misc.data import DataBlock, MagneticFieldUnstructuredGrid, MagneticFieldImageData

from magrec.misc.plot import plot_n_components, plot_vector_field_2d, plot_check_aligned_data
from magrec.prop.constants import twopi
from magrec.prop.Propagator import CurrentPropagator2d

from magrec.misc.sampler import GridSampler, NDGridPoints
from magrec.nn.utils import rotate_vector_field_2d, get_ckpt_path_by_regexp, load_model_from_ckpt, plot_ffs_params, \
    reshape_rect


# Base configuration for all experiments
class ExperimentConfig:
    def __init__(self, **kwargs):
        # Core training params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = 1000
        self.batch_size = 32
        self.learning_rate = 1e-3
        
        # Physical params
        self.height = None  # μm 
        self.dx = None      # μm
        self.dy = None      # μm
        
        # Data params
        self.data_path = None
        
        self.log_interval = 10
        self.model = None
        self.model_class = None
        self.model_params = {}
        self.data_loader = None
        
        # Set additional parameters as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def validate(self):
        if None in [self.height, self.dx, self.dy, self.data_path]:
            raise ValueError("Required parameters not set")
        

# Jerschow-specific configuration
class JerschowExperimentConfig(ExperimentConfig):
    def __init__(self, data_filename='Sine_wire.txt', 
                 background_filename='Sine_wire_blank.txt',
                 model_class=FourierFeaturesNd,     
                 model_params={'input_dim': 3, 'output_dim': 3},
                 **kwargs):
        super().__init__(**kwargs)
        self.data_filename = data_filename
        self.background_filename = background_filename
        self.model_class = model_class
        self.model_params = model_params
        self.units = {
            "length": "mm",
            "magnetic_field": "nT",
            "current": "mA"
        }
        self.data = None

    def load_data(self, data_filename=None, background_filename=None):
        if data_filename is not None:
            self.data_filename = data_filename
        if background_filename is not None: 
            self.background_filename = background_filename
            
        field_data = MagneticFieldUnstructuredGrid().from_file(__datapath__ / 'Jerschow' / self.data_filename)
        # Convert unstructured grid to ImageData() regular grid
        field_grid = field_data.resample_to_regular_grid()
        
        background_data = MagneticFieldUnstructuredGrid().from_file(__datapath__ / 'Jerschow' / self.background_filename)
        # Create a regular grid with ImageData where to put the background data
        background_grid = field_grid.interpolate(background_data, pass_cell_data=False, pass_point_data=False)
        
        # Subtract background from the measurements
        field_grid.point_data["B"] = field_grid.point_data["B"] - background_grid.point_data["B"]
        
        self.height = 12.0  # mm
        field_grid.translate([0, 0, self.height])  # move up to the specific height
        
        # Set data_pts and data_vals as attributes
        self.data_pts = torch.tensor(field_grid.points, dtype=torch.float)
        self.data_vals = torch.tensor(field_grid["B"], dtype=torch.float)
        
        self.nx_points = field_grid.dimensions[0]
        self.ny_points = field_grid.dimensions[1]
        # CHECK IF HEIGHT IS APPROPRIATE for sampling
        self.data = field_grid
        self.field = field_grid.get_as_grid("B")
        self.background = background_grid.get_as_grid("B")
        
        self.dx = field_grid.spacing[0]
        self.dy = field_grid.spacing[1]
        
        return field_grid

    def to_dict(self):
        return {
            "data_filename": self.data_filename,
            "background_filename": self.background_filename,
            "model_class": self.model_class.__name__,  # Serialize the class name
            "model_params": self.model_params,
            "units": self.units,
        }

    def validate(self):
        super().validate()
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data().")
        
        
class NbWireConfig(JerschowExperimentConfig):
    def __init__(self, 
                 model_class=FourierFeaturesNd,     
                 model_params={'n_inputs': 3, 'n_outputs': 3},
                 **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.model_params = model_params
        self.units = {
            "length": "mm",
            "magnetic_field": "nT",
            "current": "mA"
        }
        self.data = None

    def load_data(self):
        # transposition is needed here to make the datapoints align with the way
        # pyvista is creating the grid in ImageData()
        Bx = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "Bx.txt").T
        By = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "By.txt").T
        Bz = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "Bz.txt").T
        B = torch.tensor(np.array([Bx, By, Bz]), dtype=torch.float32)
        
        _, nx, ny = B.shape
        self.dx = 0.408  # um
        self.dy = 0.408  # um
         
        field_grid = MagneticFieldImageData()
        field_grid.dimensions = [nx, ny, 1]
        field_grid.spacing = [self.dx, self.dy, 1]
        field_grid.origin = [0, 0, 0]

        field_grid["B"] = B.reshape(3, -1).T
        
        self.height = 0.015  # um
        field_grid.translate([0, 0, self.height])  # move up to the specific height
        
        # Set data_pts and data_vals as attributes
        self.data_pts = torch.tensor(field_grid.points, dtype=torch.float)
        self.data_vals = torch.tensor(field_grid["B"], dtype=torch.float)
        
        self.nx_points = field_grid.dimensions[0]
        self.ny_points = field_grid.dimensions[1]
        self.data = field_grid
        self.field = field_grid.get_as_grid("B")
        
        self.dx = field_grid.spacing[0]
        self.dy = field_grid.spacing[1]
        
        return B

    def to_dict(self):
        return {
            "data_filename": self.data_filename,
            "background_filename": self.background_filename,
            "model_class": self.model_class.__name__,  # Serialize the class name
            "model_params": self.model_params,
            "units": self.units,
        }

    def validate(self):
        super().validate()
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data().")


# -------------------------------------------------------------#
#                                        EXPERIMENT classes                             #
# -------------------------------------------------------------#

# Base Experiment class
class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.losses = {'train': []}

    def build_model(self):
        # If a model is not provided, build a new one
        if not self.config.model:
            model_class = self.config.model_class
            model_params = self.config.model_params
            model = model_class(**model_params)
        else:
            model = self.config.model
        
        return model.to(self.device)

    def load_data(self):
        self.config.load_data()

    def train(self):
        # Implement the training loop
        pass

    def evaluate(self):
        # Implement evaluation logic if needed
        pass

    def plot_losses(self, show=False):
        # Create a plot with subplots for each loss in `self.losses`
        fig, axs = plt.subplots(len(self.losses), 1, figsize=(6, 12), sharex=True)
        for i, (k, v) in enumerate(self.losses.items()):
            axs[i].plot(v)
            axs[i].set_ylabel(k + ' loss')
        
        axs[i].set_xlabel('Epoch')
        
        if not show:
            plt.close()
            
        return fig
    
    
class PINNExperiment(Experiment):
    def __init__(self, config):
        """PINNExperiment defines curl and divergence calculation of the model."""
        super().__init__(config)

    # So far it is ok to compute jacobian multiple times, because we are not necessarily sure 
    # that the model is evaluated at the same points. 
    def get_curl_vals(self, pts: torch.Tensor):
        pts.requires_grad = True
        # compute curl of y wrto inputs (pts) at pts
        r = batched_curl(self.model)(pts)
        return r

    def get_div_vals(self, pts: torch.Tensor):
        pts.requires_grad = True
        # compute div of y wrto inputs (pts) at pts
        r = batched_div(self.model)(pts)
        return r

# Jerschow Experiment class
class JerschowExperiment(PINNExperiment):
    def __init__(self, config: JerschowExperimentConfig):
        super().__init__(config)
        self.units = config.units
        
    def load_model(self, ckpt_path=None):
        self.model = load_model_from_ckpt(self.model, ckpt_path)
        self.model.eval()
        
    def load_data(self):
        super().load_data()

        data_pts = self.config.data_pts
        data_vals = self.config.data_vals
        
        # Store the data as instance attributes
        self.data_pts = data_pts
        self.data_vals = data_vals

    def plot_data(self):
        """Plot the loaded data."""
        # Plot field, background, and data
        fig = plot_n_components(
            [self.config.field, self.config.background, self.config.data.get_as_grid("B")], 
            labels=[r"$B_x$", r"$B_y$", r"$B_z$"], 
            cmap="RdBu_r", 
            units=r"{}".format(self.config.units["magnetic_field"]))    
        
        return fig
    
    def plot_results(self):
        """Plot the results of the experiment."""
        self.model.eval()
        res_vals = self.model(self.data_pts).detach().cpu().numpy()
        self.config.data["B_fit"] = res_vals  # Store the fit results in the data object
        # This ensures that the order of data values and space points matches 
        res_vals = self.config.data.get_as_grid("B_fit")
        data_vals = self.config.data.get_as_grid("B")
        fig = plot_n_components(
            [data_vals, res_vals], 
            title=["Input data", "Fit results"],
            labels=[r"$B_x$", r"$B_y$", r"$B_z$"], 
            cmap="RdBu_r", 
            units=r"{}".format(self.config.units["magnetic_field"]))
        
        return fig

    def train(self):
        # Implement the training loop specific to Jerschow Experiment
        pass


class JerschowExperimentLearnB(JerschowExperiment):
    def __init__(self, config):
        super().__init__(config)
        # define extra loss terms to track
        self.losses = {
            "total": [],
            "data": [],
            "curl": [],
            "div": [],
        }
        self.is_trained = False
        
    def load_data(self):
        # Define extra data points for curl and div calculations used in this experiment
        super().load_data()
        
    def get_data_sample(self, n):
        """Returns a tensor of shape (n, l) with n batched samples from an
        l-d space where data is available."""
        list_of_indices = torch.randint(low=0, high=len(self.data_pts), size=(n,))
        return self.data_pts[list_of_indices], self.data_vals[list_of_indices]
        
    def get_curl_sample(self, n):
        """Returns a tensor of shape (n, l) with n batched samples from an 
        l-d space where to calculate curl."""
        n = int(n ** (1/3))
        pts = NDGridPoints.get_grid_pts(n, n, n, origin=(0, 0, 0), diagonal=(1, 1, 2))
        vals = torch.zeros(pts.shape[0], 3)  # curl is to be zero
        return pts, vals

    def get_div_sample(self, n):
        n = int(n ** (1/3))
        pts = NDGridPoints.get_grid_pts(n, n, n, origin=(0, 0, 0), diagonal=(1, 1, 2))
        vals = torch.zeros(pts.shape[0])  # div is to be zero, it's values are a list of points
        return pts, vals
    
    def train(self, n_iters=1000, **kwargs):
        data_batch_size = self.config.data_batch_size
        batch_size = self.config.batch_size
        eps_data = self.config.eps_data
        eps_curl = self.config.eps_curl
        eps_div = self.config.eps_div
    
        if "lr" in kwargs:
            self.optimizer.lr = kwargs["lr"]
        
        try:
            for i in tqdm.trange(n_iters):
                # Implement the training loop specific to Jerschow Experiment
                # sample data points 
                train_pts, data_targets = self.get_data_sample(n=data_batch_size)
                curl_pts, curl_targets = self.get_curl_sample(n=batch_size)
                div_pts, div_targets = self.get_div_sample(n=batch_size)
                
                data_est = self.model(train_pts)
                data_loss = self.loss_fn(data_est, data_targets)
                
                curl_vals = self.get_curl_vals(curl_pts)
                curl_loss = self.loss_fn(curl_vals, curl_targets)
                
                div_vals = self.get_div_vals(div_pts)
                div_loss = self.loss_fn(div_vals, div_targets)
                
                loss = eps_data * data_loss + eps_curl * curl_loss + eps_div * div_loss
                
                self.losses["curl"].append(curl_loss.item())
                self.losses["div"].append(div_loss.item())
                self.losses["data"].append(data_loss.item())
                self.losses["total"].append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                # zero the gradients
                self.optimizer.zero_grad()
        except KeyboardInterrupt:
            print("Training interrupted.")
        
        self.is_trained = True
                
        return self
    
    
class JerschowExperimentLearnBRandomPts(JerschowExperimentLearnB):

    def get_curl_sample(self, n):
        dx = self.config.dx
        dy = self.config.dy
        height = self.config.height
        nx_points = self.config.nx_points
        ny_points = self.config.ny_points
        
        volume_pts = NDGridPoints.get_random_pts(n,
            origin=(0, 0, 0), 
            diagonal=(dx * nx_points, dy * ny_points, 2 * height))
        
        return volume_pts, torch.zeros(n, 3)

    def get_div_sample(self, n):
        dx = self.config.dx
        dy = self.config.dy
        height = self.config.height
        nx_points = self.config.nx_points
        ny_points = self.config.ny_points
        
        volume_pts = NDGridPoints.get_random_pts(n,
            origin=(0, 0, 0), 
            diagonal=(dx * nx_points, dy * ny_points, 2 * height))
        
        return volume_pts, torch.zeros(n)
    
    
class JerschowExperimentLearnDecayB(JerschowExperimentLearnBRandomPts):
    def __init__(self, config):
        super().__init__(config)
        # define extra loss terms to track
        self.losses = {
            "total": [],
            "data": [],
            "curl": [],
            "div": [],
        }
        self.is_trained = False
        self.load_data()
        
        regions = DataBlock()
        regions["original"] = self.config.data
        regions["original"]["B_norm"] = np.linalg.norm(regions["original"]["B"], axis=1)
        regions["all_expanded"] = self.config.data.expand_bounds_2d([(1, 1), (1, 1)])
        regions["y_expanded"] = regions["original"].expand_bounds_2d([(0, 0), (1, 1)])
        regions["sides_expanded"] = (regions["all_expanded"] - regions["y_expanded"])
        regions["sides_expanded"].extend_data(source=regions["original"], source_name="B", target_name="B_decay")
        self.outer_region_pts = torch.tensor(regions["sides_expanded"].points, dtype=torch.float)
        self.outer_region_vals = torch.tensor(regions["sides_expanded"]["B_decay"], dtype=torch.float)
                
    def get_data_sample(self, n):
        """Returns a tensor of shape (n, l) with n batched samples from an
        l-d space where data is available."""
        list_of_indices = torch.randint(low=0, high=len(self.data_pts), size=(n,))
        return self.data_pts[list_of_indices], self.data_vals[list_of_indices]

    def get_decay_sample(self, n):
        """Return n random points in the region where the solution should decay."""
        list_of_indices = torch.randint(low=0, high=len(self.outer_region_pts), size=(n,))
        # `pts` are points in the region where the decay is probed
        pts = self.outer_region_pts[list_of_indices]        
        # `vals` are target values of the B-field at those points 
        # (it is OK if the values of the field are less there, 
        # but it is very bad if they are larger)
        vals = self.outer_region_vals[list_of_indices]
        return pts, vals
    
    def train(self, n_iters=1000, **kwargs):
        data_batch_size = self.config.data_batch_size
        batch_size = self.config.batch_size
        try:
            decay_batch_size = self.config.decay_batch_size
        except AttributeError:  # compatibility with the previous version, where decay_batch_size was not defined
            decay_batch_size = self.config.batch_size
        eps_data = self.config.eps_data
        eps_curl = self.config.eps_curl
        eps_div = self.config.eps_div
        eps_decay = self.config.eps_decay
    
        if "lr" in kwargs:
            self.optimizer.lr = kwargs["lr"]
        
        try:
            for i in tqdm.trange(n_iters):
                # Implement the training loop specific to Jerschow Experiment
                # sample data points 
                train_pts, data_targets = self.get_data_sample(n=data_batch_size)
                curl_pts, curl_targets = self.get_curl_sample(n=batch_size)
                div_pts, div_targets = self.get_div_sample(n=batch_size)
                bd_decay_pts, bd_decay_targets = self.get_decay_sample(n=decay_batch_size)
                
                data_est = self.model(train_pts)
                data_loss = self.loss_fn(data_est, data_targets)
                
                curl_vals = self.get_curl_vals(curl_pts)
                curl_loss = self.loss_fn(curl_vals, curl_targets)
                
                div_vals = self.get_div_vals(div_pts)
                div_loss = self.loss_fn(div_vals, div_targets)
                
                bd_decay_vals = self.model(bd_decay_pts)
                bd_decay_loss = self.loss_fn(bd_decay_vals, bd_decay_targets)
                
                loss = eps_data * data_loss + eps_curl * curl_loss + eps_div * div_loss + eps_decay * bd_decay_loss
                
                self.losses["curl"].append(curl_loss.item())
                self.losses["div"].append(div_loss.item())
                self.losses["data"].append(data_loss.item())
                self.losses["bd_decay"].append(bd_decay_loss.item())
                self.losses["total"].append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                # zero the gradients
                self.optimizer.zero_grad()
        except KeyboardInterrupt:
            print("Training interrupted.")
        
        self.is_trained = True
        return self

    
class ExperimentDenoise(JerschowExperimentLearnBRandomPts):  # names become like in Java now
    def __init__(self, config, noise_level=0.1):
        super().__init__(config)
        self.noise_level = noise_level
        self.is_trained = False

    def load_data(self):
        super().load_data()
        self.config.data["B_noisy"] = self.data_vals + self.noise_level * torch.randn_like(self.data_vals)
        self.data_vals = torch.tensor(self.config.data["B_noisy"], dtype=torch.float)
        
    def plot_data(self):
        """Plot the loaded data."""
        # Plot field, background, and data
        fig = plot_n_components(
            [self.config.data.get_as_grid("B"), self.config.data.get_as_grid("B_noisy")], 
            title=["Input data", "Noisy data"],
            labels=[r"$B_x$", r"$B_y$", r"$B_z$"], 
            cmap="RdBu_r", 
            units=r"{}".format(self.config.units["magnetic_field"]))    
        
        return fig
        
    def plot_results(self, plot_original=False, plot_comparison=False, gaussian_sigma=1.5, median_size=5):
        """Plot the results of the experiment."""
        self.model.eval()
        res_vals = self.model(self.data_pts).detach().cpu().numpy()
        self.config.data["B_fit"] = res_vals  # Store the fit results in the data object
        res_vals = self.config.data.get_as_grid("B_fit")
        data_vals = self.config.data.get_as_grid("B")
        noisy_data = self.config.data.get_as_grid("B_noisy")
        
        data_list_to_plot = [noisy_data, res_vals]
        title_list = ["Noisy data", "Fit results"]
        
        if plot_comparison:
            # Apply Gaussian filter
            gaussian_smoothed_data = gaussian_filter(noisy_data, sigma=gaussian_sigma, axes=(1,2))
            data_list_to_plot.insert(1, gaussian_smoothed_data)
            title_list.insert(1, "Gaussian smoothed")
            
            # Apply median filter
            median_smoothed_data = median_filter(noisy_data, size=median_size, axes=(1,2))
            data_list_to_plot.insert(2, median_smoothed_data)
            title_list.insert(2, "Median smoothed")
            
        if plot_original:
            data_list_to_plot.insert(0, data_vals)
            title_list.insert(0, "Original data")    
        
        fig = plot_n_components(
                data=data_list_to_plot, 
                title=title_list,
                labels=[r"$B_x$", r"$B_y$", r"$B_z$"], 
                cmap="RdBu_r", 
                units=r"{}".format(self.config.units["magnetic_field"]))
        
        return fig
        
class JerschowExperimentDenoise(ExperimentDenoise):  # mostly dummy alias
    
    def plot_data(self):
        """Plot the loaded data."""
        # Plot field, background, and data
        fig = plot_n_components(
            [self.config.field, self.config.background, self.config.data.get_as_grid("B")], 
            labels=[r"$B_x$", r"$B_y$", r"$B_z$"], 
            cmap="RdBu_r", 
            units=r"{}".format(self.config.units["magnetic_field"]))    
        
        return fig
    
    
def check_curl_div_at_rnd_pts(experiment):
    """
    Evaluate and print the mean squared error loss for curl and divergence values 
    at random points within the experiment's grid.

    Args:
        experiment: An object containing the experiment configuration and methods 
                    to compute curl and divergence values.
    """
    dx = experiment.config.dx
    dy = experiment.config.dy
    height = experiment.config.height
    nx_points = experiment.config.nx_points
    ny_points = experiment.config.ny_points
    
    rnd_pts = NDGridPoints.get_random_pts(1000, origin=(0, 0, 0), 
        diagonal=(dx * nx_points, dy * ny_points, 2 * height))
    
    curl_vals = experiment.get_curl_vals(rnd_pts)
    div_vals = experiment.get_div_vals(rnd_pts)
    
    print(
        'Curl loss: ',
        torch.nn.functional.mse_loss(
        curl_vals, torch.zeros_like(curl_vals)).item(), 
        ' vs ', 
        experiment.losses["curl"][-1])
    print(
        'Div loss: ',
        torch.nn.functional.mse_loss(div_vals, torch.zeros_like(div_vals)).item(),
        ' vs ',
        experiment.losses["div"][-1])