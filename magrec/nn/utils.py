import os
import re
import tracemalloc
import psutil

import torch
from pytorch_lightning import Callback

from magrec import __logpath__

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
    # Known cases: 'ff_std_cond' - FourierFeatures2dCurrent with standard Fourier features, setup with torchphysics architecture in mind,
    #                              where FourierFeatures2dCurrent has multiple train_conditions modules. 
    
    # I've found no better way to maintain different loading schemes for different models
    # It can be also a Class method of the model, but then it needs to repeat for all similar models
    if cls is None and type == 'ff_std_cond':
        # Type is enough to determine the class of the model to load, 
        # otherwise this function should be called as a class method of the model.
        from magrec.nn.models import FourierFeatures2dCurrent
        cls = FourierFeatures2dCurrent
    
    if type == 'ff_std_cond':    
        # Since model checkpoint is a confused dict, we need to extract from the repeated module parameters
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