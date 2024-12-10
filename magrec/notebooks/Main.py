#!/usr/bin/env python
# coding: utf-8

# # Imports
# Import libraries and any data or parameters needed for the project.

# In[239]:


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

from magrec.nn.modules import GaussianFourierFeaturesTransform
from magrec.nn.models import FourierFeaturesNd, WireNet
from magrec.nn.utils import batched_curl, batched_div, batched_grad

from magrec.misc.data import MagneticFieldUnstructuredGrid

from magrec.misc.plot import plot_n_components, plot_vector_field_2d, plot_check_aligned_data
from magrec.prop.constants import twopi
from magrec.prop.Propagator import CurrentPropagator2d

from magrec.misc.sampler import GridSampler, NDGridPoints
from magrec.nn.utils import rotate_vector_field_2d, get_ckpt_path_by_regexp, load_model_from_ckpt, plot_ffs_params, \
    reshape_rect



# # Common functions and data loading

# In[ ]:


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
        self.data = field_grid
        
        self.dx = field_grid.spacing[0]
        self.dy = field_grid.spacing[1]
        
        return field_grid


    def validate(self):
        super().validate()
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data().")



# In[ ]:


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
        from magrec.misc.plot import plot_n_components

        # Plot field, background, and data
        fig = plot_n_components(
            [self.config.field, self.config.background, self.config.data], 
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


# # Jerschow — Reconstruct current
# 
#  

# In[160]:


model = load_model_from_ckpt(version_n=26, ckpt_name_regexp='last', folder_name='jerschow', type='ff_std_cond')

units = {
    "length": "mm",
    "magnetic_field": "nT",
    "current": "mA"
}

nx = 300
ny = 300
dx = 2.0
dy = 2.0
height = 12.0
layer_thickness = 0.4

rect = {
    "origin": [0, 0],
    "diagonal": [3, 3],
    "nx": nx,
    "ny": ny,
    "dx": dx,
    "dy": dy,
}

rect = reshape_rect(rect, origin=(-4, -4), diagonal=(4, 4))

prop = CurrentPropagator2d(
    source_shape=(rect["nx"], rect["ny"]),
    dx=rect["dx"],
    dy=rect["dy"],
    height=height,
    layer_thickness=layer_thickness,
    real_signal=False,
    units=units
)

pts = GridSampler.sample_grid(rect["nx"], rect["ny"], origin=rect["origin"], diagonal=rect["diagonal"])
pts.requires_grad_(True)
y_hat = model(pts).detach()
y_hat_grid = GridSampler.pts_to_grid(y_hat, rect["nx"], rect["ny"])

flow_fig = plot_vector_field_2d(y_hat_grid, cmap="plasma", units=r"{}/{}$^2$".format(units["current"], units["length"]))

values_hat = prop(y_hat_grid).real
mag_fig = plot_n_components(
    values_hat,
    labels=[r"B_x", r"B_y", r"B_z", r"B_{NV}"],
    cmap="bwr",
    units=units["magnetic_field"]
)


# In[68]:


flow_fig


# In[69]:


mag_fig


# # Jerschow — Learn magnetic field

# Goal of this section is to make a neural network $u_{NN}(x, y) = \vec{B}(x, y)$ that learns parameters given a set of measurement $N$ values $\{B^i_{NV}\}_{i = 1, \dots, N}$ obtained for a single component of the magnetic field $B_{NV}$ at different points ${(x_i, y_i)}_{i = 1, \dots, N}$
# 
# For this, we take any architecture of the neural network. 
# 
# We implement sampling of coordinates on a grid in a rectangle. There are two types of points: those with measurement values and those without. Those with measurement values should have corresponding values of the field. Those without can still be sampled to enforce additional constraints, besides the experimental data constraints. Such constraints, in the case of the magnetic field, are $\nabla\cdot\vec{B} = 0$ and $\nabla \times \vec{B} = 0$. 

# ## Neural network architecture
# 
# We have two neural networks to try. 
# 
# 1. One learnes a mapping $(x, y, z) \mapsto \vec{B}(x, y, z)$ and receives the constraint $\nabla\cdot\vec{B} = 0$ together with $\nabla\times\vec{B} = 0$ as an additional loss term. 
# 
# 2. The other learns a mapping $(x, y) \mapsto \vec{B}(x, y)$ and outputs a 2D vector field already free from divergence by construction.
# 
# Beyond that, there are also other approaches:
# 
# 3. Use a neural network to learn a mapping $(x, y) \mapsto \vec{J}(x, y)$, again with the constraint $\nabla\cdot\vec{J} = 0$ either enforced by construction or as an additional loss term. Then, integrate the current to obtain the magnetic field.
# 
# 4. We can also do all of the above directly in the Fourier space.
# 
# Below let's implement the first approach.

# ## Data to learn
# 
# What data should we try to learn? Let's start with a squiggly wire from Jerschow's dataset.

# In[170]:


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
            
        return self


# ## GaussianFourierFeatures model

# In[162]:


# 1. Instantiate Configuration
config = JerschowExperimentConfig('Sine_wire.txt',
                                  model_params=dict(n_inputs=3, n_outputs=3, 
                                                    ff_stds=[(0.1, 10), (20, 5), (5, 5)]),
                                  data_batch_size=1000,
                                  batch_size=1000,
                                  nz_points=100,
                                  eps_curl=1,
                                  eps_data=1,
                                  eps_div=1,)


# I have a suspicion that loading and sampling datapoints is not made correctly. As you see, $B_x$, $B_y$, $B_z$ maps have some step artifacts, and they are supposed to be outputs of the differentiable neural network, and therefore smooth. So I assume that something isn't being fed right, and neural network tries very hard to do that at the expense… but what does it not load right? I have a faint hint that the error happens here:
# 
# ```python
# def get_data_sample(self, n):
#         """Returns a tensor of shape (n, l) with n batched samples from an
#         l-d space where data is available."""
#         list_of_indices = torch.randint(low=0, high=len(self.data_pts), size=(n,))
#         return self.data_pts[list_of_indices], self.data_vals[list_of_indices]
# ```    
# 
# Because `data_pts` and `data_vals` aren't aligned together. That is, `data_pts` values do not correspond to `data_vals` at those pts.  
# 
# We check that by plotting the data points and the values at those points.

# Now let's plot `experiment.data_pts` and `experiment.data_vals` to see if they are aligned.

# Aha, so that's the problem. The data points are not aligned with the data values. We need to fix that.

# Looking at how the FourierFeatures choose the frequencies, I see that it is not optimal. Setting std still allows for large k-vectors to be chosen. A better approach is to choose some frequencies manually. 

# In[163]:


fcnn_config = JerschowExperimentConfig('Sine_wire.txt',
                                  model_params=dict(n_inputs=3, n_outputs=3, 
                                                    ff_stds=[(0.1, 10), (20, 5), (5, 5)]),
                                  data_batch_size=1000,
                                  batch_size=1000,
                                  lr=1e-3,
                                  nz_points=100,
                                  eps_curl=1,
                                  eps_data=1,
                                  eps_div=1,)

fcnn_config.model = torch.nn.Sequential(
    torch.nn.Linear(3, 10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10, 10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10, 3),
)


# In[142]:


fcnn_experiment = JerschowExperimentLearnB(fcnn_config)
fcnn_experiment.load_data()


# In[ ]:


fcnn_experiment.train(n_iters=100000, lr=1e-1)


# In[116]:


fcnn_experiment.plot_losses()


# Define a proper version of sampling pts for curl and div losses. Before it used to be a grid, that is, specific points. Now it should be a random sample of points.

# In[164]:


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


# In[165]:


rnd_pts_experiment_fcnn = JerschowExperimentLearnBRandomPts(fcnn_config)


# In[219]:


rnd_pts_experiment_fcnn.optimizer = torch.optim.Adam(rnd_pts_experiment_fcnn.model.parameters(), lr=1e-3)


# In[220]:


rnd_pts_experiment_fcnn.train(n_iters=500000)


# In[223]:


rnd_pts_experiment_fcnn.plot_losses()


# In[224]:


rnd_pts_experiment_fcnn.plot_results()


# In[225]:


# Save model
torch.save(rnd_pts_experiment_fcnn.model, __datapath__ / 'fcnn_model.pt')


# In[186]:


rnd_pts_experiment_fcnn.optimizer.lr = 0.01


# In[192]:


rnd_pts_experiment_fcnn.train(n_iters=400000)


# Let's manually check at random points whether the curl and div are zero.

# In[ ]:





# In[226]:


def check_curl_div_at_rnd_pts(experiment):
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
    
check_curl_div_at_rnd_pts(rnd_pts_experiment_fcnn)


# In[227]:


config = JerschowExperimentConfig('Sine_wire.txt',
                                  model_params=dict(n_inputs=3, n_outputs=3, 
                                                    ff_stds=[(0.1, 10), (2, 5), (5, 5)]),
                                  data_batch_size=1000,
                                  batch_size=1000,
                                  lr=1e-2,
                                  nz_points=100,
                                  eps_curl=1,
                                  eps_data=1,
                                  eps_div=1,)

experiment = JerschowExperimentLearnB(config)
experiment.load_data()


# In[229]:


experiment.train(n_iters=100000)


# In[230]:


experiment.plot_results()


# # Jerschow — Learn magnetic potential

# Goal of this section is to make a neural network $u_{NN}(x, y, z) = {\Phi}(x, y, z)$ that learns parameters given a set of measurement $N$ values $\{B^i_{NV}\}_{i = 1, \dots, N}$ obtained for a single component of the magnetic field $B_{NV}$ at different points ${(x_i, y_i)}_{i = 1, \dots, N}$ such that $\vec{B} = \nabla \Phi$.

# In[236]:


class JerschowExperimentLearnPot(JerschowExperiment):
    def __init__(self, config):
        super().__init__(config)
        # define extra loss terms to track                                                         
        self.losses = {
            "total": [],
            "data": [],
            "curl": [],
            "div": [],
        }
        
    def load_data(self):
        # Define extra data points for curl and div calculations used in this experiment
        super().load_data()
        
    def get_data_sample(self, n):
        """Returns a tensor of shape (n, l) with n batched samples from an
        l-d space where data is available."""
        list_of_indices = torch.randint(low=0, high=len(self.data_pts), size=(n,))
        return self.data_pts[list_of_indices], self.data_vals[list_of_indices]
    
    def get_B(self, pts: torch.Tensor):
        """The magnetic field B is obtained by applying the gradient to the output of the model."""
        pts.requires_grad = True
        # compute grad of y wrto inputs (pts) at pts
        return batched_grad(self.model)(pts)
    
    
    def train(self, n_iters=1000, **kwargs):
        data_batch_size = self.config.data_batch_size
    
        if "lr" in kwargs:
            self.optimizer.lr = kwargs["lr"]
        
        for i in tqdm.trange(n_iters):
            # Implement the training loop specific to Jerschow Experiment
            # sample data points 
            train_pts, data_targets = self.get_data_sample(n=data_batch_size)
            
            data_est = self.get_B(train_pts)
            data_loss = self.loss_fn(data_est, data_targets)
            
            loss = data_loss
            self.losses["data"].append(data_loss.item())
            
            loss.backward()
            self.optimizer.step()
            # zero the gradients
            self.optimizer.zero_grad()
            
        return self


# In[237]:


pot_config = JerschowExperimentConfig('Sine_wire.txt',
                                  data_batch_size=1000,
                                  batch_size=1000,
                                  lr=1e-3,)

pot_config.model = torch.nn.Sequential(
    torch.nn.Linear(3, 10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10, 10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10, 1),
)


# In[ ]:


pot_experiment = JerschowExperimentLearnPot(pot_config)
pot_experiment.load_data()
pot_experiment.train(n_iters=100000)


# # Wire — Reconstruct path with the neural network

# In[ ]:


def test_path(s):
    r = torch.cat([s, torch.sin(30*s)], dim=1)
    r = rotate_vector_field_2d(r, 45 / 360 * 2 * torch.pi)
    return r

net = WireNet()


# In[9]:


net.fit_to_path(test_path, n_points=200, max_epochs=40)


# In[10]:


net.plot_wire()


# In[11]:


net.net.I = 1
plane_pos = GridSampler.sample_grid(100, 100, origin=[-1, -1], diagonal=[2, 2], z=0.1)
plane_pos = plane_pos.numpy()

B = net.get_B(pos=plane_pos)
B = GridSampler.pts_to_grid(B, 100, 100)

plot_n_components(B)


# # To-do

# 1. Set `.data_pts` and `.data_vals` to the values directly from the file. 
# 2. Try fitting the magnetic field on the curve defined by the neural network.
# 3. Try implementing expansion CNN: take a magnetic field image, form a training set of images obtained by (a) different rectangular or other sized cutouts of the original image, (b) same but also rotations and scales. Then feed it to CNN that receives cutouts as inputs and adds pixels to the sides. 
# 
# ## Done
# 
# + Make sure refactored section works and produces the same results as the old one. 

# # Questions
# 
# 

# Where to put `ExperimentConfig` class and `JerschowExperiment`? 
# 
# Where to put plotting function to visualize loaded data, which is load function specific?
