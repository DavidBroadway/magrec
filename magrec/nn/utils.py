import copy
import os
import re
import time
import tracemalloc
import psutil
import json

import torch
from pytorch_lightning import Callback

import matplotlib.pyplot as plt
import numpy as np

from magrec import __logpath__


def get_rotation_matrix_2d(theta):
    """
    Returns a 2D rotation matrix for an angle theta in radians.
    """
    # Make sure theta is a tensor, usually a singleton.
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta)
        
    m = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    return m


def rotate_vector_field_2d(v, theta):
    """
    Rotates a vector field v by an angle theta in radians.
    
    v: torch.Tensor of shape (..., 2) where ... is a number of samples (batch size) and 2 is the number of components
    """
    m = get_rotation_matrix_2d(theta)
    v_rot = torch.einsum("...ij,...j->...i", m, v)
    return v_rot


class ROI(object):
    """
    Class to handle defining region of interest (ROI) in images. To be used with padded images, for example.
    """
    pass


class MagneticFieldKernel(object):
    """
    Kernel that computes three components of the magnetic field from only one component of the magnetic field in the
    source free region. There, the components of the magnetic field are connected to each other through a scalar potential
    and can be computed as:
    """
    def __init__(self):
        pass


def isdifferentiable(model, input):
    """A quick utility function to check if model properly preserves gradiends
    wrt its parameters. That is, if all operations are differentiable with torch."""
    output = model(input)
    target = torch.zeros_like(output)
    error = torch.nn.functional.mse_loss(output, target)
    error.backward()
    
def isoptimizable(model: torch.nn.Module, input: torch.Tensor):
    """Performs a single optimization step of model parameters on the input
    assuming a dummy `target = torch.zeros_like(ouput)` and a simplest optimizer case."""
    try:
        optim = torch.optim.Adam(params=model.parameters())
        output = model(input)
        target = torch.zeros_like(output)
        error = torch.nn.functional.mse_loss(output, target)
        error.backward()
        optim.step()
        optim.zero_grad()
        return True
    except Exception as e:
        print("Encountered error:\n{}".format(e))
        return False
    
    
def save_model_for_experiment(experiment, path_to_model_dict):
    if experiment.is_trained:
        if os.path.exists(path_to_model_dict):
            raise ValueError("Model state_dict already exists. Please remove it to save the new one.")
        else:
            torch.save(experiment.model.state_dict(), path_to_model_dict)
    else:
        raise ValueError("Model is not trained. Saving its state_dict would ruin the safe.")
    
        
def load_model_for_experiment(experiment, path_to_model_dict):
    experiment = copy.deepcopy(experiment)
    model_state_dict = torch.load(path_to_model_dict)
    new_model = experiment.config.model_class(**experiment.config.model_params)
    new_model.load_state_dict(model_state_dict)
    experiment.model = new_model
    return experiment
    

def get_ckpt_path_by_regexp(version_n, ckpt_name_regexp, folder_name='jerschow'):
    version_name = f'version_{version_n}'
    # match the checkpoint name using regexp
    ckpt_names = [x for x in os.listdir(__logpath__ / folder_name / 'lightning_logs' / version_name / 'checkpoints') if re.match(ckpt_name_regexp, x)]
    if len(ckpt_names) == 0:
        raise ValueError('No checkpoint found')
    elif len(ckpt_names) > 1:
        raise ValueError('Multiple checkpoints found: {}\nProvide a more specific regexp.'.format(ckpt_names))
    else:
        ckpt_name = ckpt_names[0]
    ckpt_path = __logpath__ / folder_name / 'lightning_logs' / version_name / 'checkpoints' / ckpt_name
    return ckpt_path


def load_model_from_ckpt(cls=None, version_n=None, ckpt_name_regexp='last.ckpt', 
                         folder_name='', type='', 
                         ckpt=None, ckpt_path=None):
    """Loads a model from a checkpoint."""
    if ckpt_path is None and ckpt is None:
        ckpt_path = get_ckpt_path_by_regexp(version_n=version_n, ckpt_name_regexp=ckpt_name_regexp, folder_name=folder_name)
        ckpt = torch.load(ckpt_path)
    elif ckpt_path is not None and ckpt is None:
        ckpt = torch.load(ckpt_path)
    elif ckpt_path is None and ckpt is not None:
        pass
    elif ckpt_path is not None and ckpt is not None:
        # In case both are provided, check if behaviour is expected. Perhaps this is a repitios data, but it's compatible.
        ckpt_tmp = torch.load(ckpt_path)
        if ckpt_tmp != ckpt:
            raise ValueError('Both ckpt and ckpt_path are provided and they are not the same. Provide only one.')
    else:
        RuntimeError('Well, that\'s unexpected. \
            I thought the previous if-else block should have covered all cases, \
                but it did not. Check the code.')
    
    # SELECT LOADING SCHEME based on TYPE of the model. 
    # Known cases: 'ff_std_cond' - FourierFeatures2dCurrent with Gaussian Fourier features, setup 
    #                              with torchphysics architecture in mind, where FourierFeatures2dCurrent
    #                              has multiple train_conditions modules. 
    
    # I've found no better way to maintain different loading schemes for different models
    # It can be also a Class method of the model, but then it needs to repeat for all similar models
    if cls is None and type == 'ff_std_cond':
        # Type is enough to determine the class of the model to load, 
        # otherwise this function should be called as a class method of the model.
        from magrec.nn.models import FourierFeatures2dCurrent
        cls = FourierFeatures2dCurrent
    
    if type == 'ff_std_cond':    
        # Since model checkpoint is a confusing dict, we need to extract from the repeated module parameters
        stdc = {}
        # We also need to figure out now how many ffs we have and what were their size, i.e. what ff_stds to use
        # Reason: FourierFeatures2dCurrent() call initializes GaussianFourierFeaturesTransform with default
        # ff_stds, which may not be the same as in the checkpoint. .load_state_dict() then fails because of 
        # size mismatch. 
        ff_stds = []
        ff_std = 0  # variable to hold detected std from the checkpoint dict
        for k, v in ckpt['state_dict'].items():
            if '0.module' in k:
                stdc.update({k.replace('train_conditions.0.module.', ''):  v})
                if k[-1:] == "B":  
                    # Infer ff_stds from here, append the shape to a list. First number is
                    # irrelevant, second gives the shape to initiate the FourierFeatures2dCurrent() with
                    ff_stds += [[ff_std, v.shape[1]]]
                elif k.split('.')[-1] == "sigma":
                    # rename the key to match the model's state_dict, 
                    # replace the last element of ff_stds with the std that we found here
                    ff_std = v.item()  # implicitly assumes that "ff_stds" spec and B tensor comes after sigma/std
                    # sigma → std
                    stdc.update({k.replace('sigma', 'std'):  ff_std})
                    
        
        # Initialize a fresh model and load state dict 
        model = cls(tuple(ff_stds))
        model.load_state_dict(stdc, strict=False, assign=True)
        model.step = ckpt["global_step"]
        
    return model


class MemoryProfilingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        tracemalloc.start()

    def on_train_end(self, trainer, pl_module):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for index, stat in enumerate(top_stats[:10], 1):
            print(f"#{index}: {stat}")

        tracemalloc.stop()
        

class ProcessMemoryMonitorCallback(Callback):
    
    def on_train_end(self, trainer, pl_module):
        main_process_id = os.getpid()
        print(f"Memory consumption by child processes of main process (PID={main_process_id}) after training:")
        
        for proc in psutil.process_iter():
            try:
                # Fetch process details
                p_info = proc.as_dict(attrs=['pid', 'ppid', 'name', 'memory_percent'])
                
                # Check if this process is a child of the main process
                if p_info['ppid'] == main_process_id:
                    # Extract the memory percentage information
                    mem_percent = p_info['memory_percent']
                    # Print
                    print(f"PID={p_info['pid']}, Name={p_info['name']}, Memory Percent={mem_percent}%")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            
            
# Write a convenience function to reshape `rect` objects (dicts).
# It should be able to reshape the following:
# - change origin or diagonal, which will change either nx, ny or dx, dy

def reshape_rect(rect, origin=None, diagonal=None):
    """Reshape a rectangle.
    
    Parameters
    ----------
    rect : dict
        Rectangle to reshape.
    origin : tuple, optional
        New origin.
    diagonal : tuple, optional
        New diagonal.
    nx : int, optional
        New number of points in x direction.
    ny : int, optional
        New number of points in y direction.
    dx : float, optional
        New step size in x direction.
    dy : float, optional
        New step size in y direction.
    
    Returns
    -------
    rect : dict
        Reshaped rectangle.
    """
    # Prepare a copy of previous rectangle to save the history
    new_rect = rect.copy()
    new_rect.update({"prev_rect": rect})
    
    if origin is not None:
        prev_origin = rect["origin"]
        new_rect["origin"] = origin    
    if diagonal is not None:
        prev_diagonal = rect["diagonal"]
        new_rect["diagonal"] = diagonal
    
    # Update the rectangle
    if not (prev_origin[0] == origin[0] and prev_diagonal[0] == diagonal[0]):
            scale_dx = (diagonal[0] - origin[0]) / (prev_diagonal[0] - prev_origin[0])
            new_rect["dx"] = rect["dx"] * scale_dx
    if not (prev_origin[1] == origin[1] and prev_diagonal[1] == diagonal[1]):
            scale_dy = (diagonal[1] - origin[1]) / (prev_diagonal[1] - prev_origin[1])
            new_rect["dy"] = rect["dy"] * scale_dy
    
    return new_rect



def lift_to_batched(f):
    def batched_f(x):
        return torch.vmap(f, in_dims=0, out_dims=0)(x)
    return batched_f

def batched_jacrev(f):
    """Compute Jacobian matrix [∂y_i/∂x_j]_{i, j} of a function f: R^m -> R^n
    for `b` batched vectors x in R^m. The output is a tensor of shape (b, n, m) 
    where b is the batch size."""
    def f_jac(x):
        return torch.vmap(torch.func.jacrev(f), in_dims=0, out_dims=0)(x)
    return f_jac

def curl(f):
    """
    Returns a function to compute the curl of a vector field f: ℝ³ → ℝ³ at given points x.

    Args:
        f (callable): Vector field function.

    Returns:
        callable: Function that computes the curl at given points x.
    """
    def curl_fn(x):
        y = f(x)
        jac = torch.func.jacrev(f)(x)
        v_1 = jac[2, 1] - jac[1, 2]
        v_2 = jac[0, 2] - jac[2, 0]
        v_3 = jac[1, 0] - jac[0, 1]
        return torch.stack((v_1, v_2, v_3), dim=0)
    return curl_fn

def div(f):
    """
    Returns a function to compute the divergence of a vector field f: ℝⁿ → ℝⁿ at given points x.

    Args:
        f (callable): Vector field function.

    Returns:
        callable: Function that computes the divergence at given points x.
    """
    def div_fn(x):
        y = f(x)
        jac = torch.func.jacrev(f)(x)
        return torch.trace(jac)
    return div_fn

def grad(f):
    """
    Returns a function to compute the gradient of a scalar field f: ℝ³ → ℝ at given points x.

    Args:
        f (callable): Scalar field function.

    Returns:
        callable: Function that computes the gradient at given points x.
    """
    def grad_fn(x):
        y = f(x)
        jac = torch.func.jacrev(f)(x)
        return jac
    return grad_fn


def batched_curl(f):
    """
    Returns a function to compute the curl of a vector field f: ℝ³ → ℝ³ for batched inputs.

    Args:
        f (callable): Vector field function.

    Returns:
        callable: Batched function to compute the curl.
    """
    curl_fn = curl(f)
    return torch.vmap(curl_fn, in_dims=0, out_dims=0)

def batched_div(f):
    """
    Returns a function to compute the divergence of a vector field f: ℝⁿ → ℝⁿ for batched inputs.

    Args:
        f (callable): Vector field function.

    Returns:
        callable: Batched function to compute the divergence.
    """
    div_fn = div(f)
    return torch.vmap(div_fn, in_dims=0, out_dims=0)    

def batched_grad(f):
    """
    Returns a function to compute the gradient of a scalar field f: ℝ³ → ℝ at given points x.

    Args:
        f (callable): Scalar field function.

    Returns:
        callable: Function that computes the gradient at given points x.
    """
    grad_fn = grad(f)
    return torch.vmap(grad_fn, in_dims=0, out_dims=0)