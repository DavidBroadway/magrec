from typing import Union

import torch
import torch.nn as nn
import torch.utils.data

from magrec.prop.Propagator import CurrentFourierPropagator3d as Propagator
from magrec.nn.arch import GeneratorCNN, GeneratorMultipleCNN

import numpy as np


def create_zero_padder(required_shape: tuple[int, int], W: int, H: int):
    # calculate the required padding to get to the expanded shape
    pad_width = (required_shape[0] - W*3) 
    pad_height = (required_shape[1] - H*3)

    # if the padding is odd, we need to add one more pixel to the right and bottom
    if pad_width % 2 == 1:
        pad_left = pad_width // 2 + 1
        pad_right = pad_width // 2
    else:
        pad_left = pad_width // 2
        pad_right = pad_width // 2

    # similar to height
    if pad_height % 2 == 1:
        pad_bottom = pad_height // 2 + 1
        pad_top = pad_height // 2
    else:
        pad_bottom = pad_height // 2
        pad_top = pad_height // 2

    # calculate the ROI to crop the result to
    roi_left = pad_left + W
    roi_bottom = pad_bottom + H

    pad_zero = torch.nn.ZeroPad2d((pad_bottom, pad_top, pad_left, pad_right))
    return pad_zero, (roi_left, roi_bottom)


def train(net: GeneratorCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
          loss_function='L1', n_epochs=25, batch_size=100, print_every_n=1):
    """
    Run gradient descent on the network to optimize the weights and biases.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape))
    optimizer       : torch.optim.Optimizer object
    loss_function   : str, L1, L2 or CE (for Cross Entropy Loss) (default: L1)
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)

    Returns
    -------
    The final loss and accuracy
    """
    if loss_function == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_function == 'L2':
        loss_fn = nn.MSELoss()
    elif loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        ValueError(f'Loss function must be L1, L2, or CE. Got `{loss_function}` instead')

    # Create a train_loader to load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    track_loss = []

    net.train()

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch in train_loader:
            # Get the batch data and labels
            inputs, labels = batch

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            outputs = net(inputs)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append(loss.item())

        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1]: .2e}')

    # Return the loss and accuracy
    return track_loss


def load_tensors_data_labels_from_npz(path: str) -> (torch.Tensor, torch.Tensor):
    """
    Load sets of data and labels from a .npz file.

    Args:
        path: str
            Path to the .npz file

    Returns:
        (torch.Tensor with data, torch.Tensor with labels) tuple
        Shape of each torch.Tensor: (n_samples, data.shape) and (n_samples, labels.shape) respectively
    """
    with np.load(path) as file:
        data = file['data']
        labels = file['labels']

    # By default, numpy saves in double (float64) precision, but we need to convert to float32
    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels).float()

    # Return the set of data and labels
    return data_tensor, labels_tensor


def load_tensors_from_npz(*path: str) -> dict[str: torch.Tensor]:
    """
    Load tensors from a .npz file as a dict with keys.

    Parameters
    ----------
    path: str                   Path of the .npz file

    Returns
    -------
    dict[str: torch.Tensor]     Dict with keys which are names of the arrays as they were saved,
                                and tensors as torch.Tensors objects converted from numpy arrays into
                                float32 precision.

    """
    tensors = {}
    for p in path:
        with np.load(p) as file:
            for key, value in file.items():
                tensors[key] = torch.from_numpy(value).float()
    return tensors


def pinn_trainer(
        net: GeneratorMultipleCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
        n_epochs=25, batch_size=100, print_every_n=1, device='cpu'
):
    """
    Run physics-informed training on network to optimize the weights and biases.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape))
    optimizer       : torch.optim.Optimizer object
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)
    device (str)    : name of the device to use for training, e.g. 'cpu' or 'cuda'

    Returns
    -------
    The final loss and accuracy
    """
    # Create a train_loader to load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # https://doi.org/10.1007/978-3-031-11349-9_30
    # Based on Khare et al. (2022). Analysis of Loss Functions for Image Reconstruction Using Convolutional Autoencoder,
    # MSE (L2) is the most perfomant function for conv autoencoders, so we will use that.
    # Indeed, I see the error steadily decreasing, especially with reduction='sum'.
    loss_fn = nn.MSELoss(reduction='sum')

    track_loss = []

    propagator = Propagator(shape=(16, 16, 16), width=10.0, depth=10.0, height=10.0, D=0.5, padding=-1).to(device)

    net.train()

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch in train_loader:
            # Get the batch data and labels
            inputs, labels = batch

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            outputs = net(inputs)

            b = propagator.get_B(outputs)

            # Compute the loss
            loss1 = loss_fn(b, inputs) * 100
            loss2 = loss_fn(outputs, labels)

            loss = loss1 + loss2

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append((loss1.item(), loss2.item()))

        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1]}')

    # Return the loss and accuracy
    return track_loss


def pinn_trainer_adj1(
        net: GeneratorMultipleCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
        n_epochs=25, batch_size=100, print_every_n=1, device='cpu'
):
    """
    Run physics-informed training on network to optimize the weights and biases that tries to adjust
    the physical backward loop error according to the amplitude of the features.

    Previously the issue, I think, was that the error was smaller for those magnetic field maps,
    where the variation and the amplitude of the magnetic field were small.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape))
    optimizer       : torch.optim.Optimizer object
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)
    device (str)    : name of the device to use for training, e.g. 'cpu' or 'cuda'

    Returns
    -------
    The final loss and accuracy
    """
    # Create a train_loader to load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # https://doi.org/10.1007/978-3-031-11349-9_30
    # Based on Khare et al. (2022). Analysis of Loss Functions for Image Reconstruction Using Convolutional Autoencoder,
    # MSE (L2) is the most perfomant function for conv autoencoders, so we will use that.
    mse_loss = torch.nn.MSELoss()

    gaussian_loss = torch.nn.GaussianNLLLoss()

    def my_rms_loss(estimate, target):
        dims = (0, 1, 2, 3, 4)[2:]
        dims = (1,) + (-3, -2, -1)[-(estimate.dim() - 2):]
        target_rms = \
            torch.sqrt(
                torch.mean(torch.pow(target, 2), dim=dims, keepdim=True)
            ).expand_as(estimate)
        norm_estimate = estimate / target_rms
        norm_target = target / target_rms
        return mse_loss(norm_estimate, norm_target)

    def my_gauss_loss(estimate, target, scale=1e-2):
        dims = (-3, -2, -1)[-(estimate.dim() - 2):]
        var = target.std(dim=dims, keepdim=True).expand_as(estimate)
        return gaussian_loss(estimate, target, var * scale)

    track_loss = []

    propagator = Propagator(shape=(16, 16, 16), width=10.0, depth=10.0, height=10.0, D=0.5, padding=-1).to(device)

    net.train()

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch in train_loader:
            # Get the batch data and labels
            features, targets = batch

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            estimates = net(features)

            B = propagator.get_B(estimates)

            # Compute the loss
            loss1 = my_rms_loss(B, features)
            loss2 = my_rms_loss(estimates, targets)

            loss = loss1 + loss2

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append((loss1.item(), loss2.item()))

        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1]}')

    # Return the loss and accuracy
    return track_loss


def pinn_trainer_adj2(
        net: GeneratorMultipleCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
        n_epochs=25, batch_size=100, print_every_n=1, device='cpu', features_std_mean: tuple = (1.0, 0), target_std_mean: tuple = (1.0, 0),
        alpha=100000
):
    """
    Run physics-informed training on network to optimize the weights and biases that tries to adjust
    the physical backward loop error according to the amplitude of the features.

    Learns to reconstruct normalized estimates from normalized targets. Learns to also have normalized estimates
    such that if they are denormalized to original distributions, the magnetic field they produce matches with the
    denormalized features.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape)).
                      Features and targets are assumed to have zero mean and unit variance (each separately).
    optimizer       : torch.optim.Optimizer object
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)
    device (str)    : name of the device to use for training, e.g. 'cpu' or 'cuda'
    alpha           : hyperparameter weight of the loss physical loss term (default: 100000)

    Returns
    -------
    The final loss and accuracy
    """
    # Create a train_loader to load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # https://doi.org/10.1007/978-3-031-11349-9_30
    # Based on Khare et al. (2022). Analysis of Loss Functions for Image Reconstruction Using Convolutional Autoencoder,
    # MSE (L2) is the most perfomant function for conv autoencoders, so we will use that.
    mse_loss = torch.nn.MSELoss(reduction='sum')

    track_loss = []

    propagator = Propagator(shape=(16, 16, 16), width=10.0, depth=10.0, height=10.0, D=0.5, padding=-1).to(device)

    net.train()

    f_std, f_mean = features_std_mean
    t_std, t_mean = target_std_mean

    def physical_loss(estimated_normalized_current, features):
        denormalized_features = (features * f_std) + f_mean
        J = (estimated_normalized_current * t_std) + t_mean
        B = propagator.get_B(J)
        return mse_loss(B, denormalized_features)

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch in train_loader:
            # Get the batch data and labels
            features, targets = batch

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            estimates = net(features)

            B = propagator.get_B(estimates)

            # Compute the loss
            loss1 = physical_loss(estimates, features) * alpha
            loss2 = mse_loss(estimates, targets)

            loss = loss1 + loss2

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append((loss1.item(), loss2.item()))

        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1]}')

    # Return the loss and accuracy
    return track_loss


def pinn_trainer_two_wires_adj2(
        net: GeneratorMultipleCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
        n_epochs=25, batch_size=100, print_every_n=1, device='cpu', features_std_mean: tuple = (1.0, 0), target_std_mean: tuple = (1.0, 0),
        alpha=100000
):
    """
    For two wires, run physics-informed training on network to optimize the weights and biases that tries to adjust
    the physical backward loop error according to the amplitude of the features.

    Learns to reconstruct normalized estimates from normalized targets. Learns to also have normalized estimates
    such that if they are denormalized to original distributions, the magnetic field they produce matches with the
    denormalized features.

    Two wires are simulated by taking two random batches from the train set and adding them up. Note that STD and MEAN of
    the added batches are not the same as the STD and MEAN of the original batches. We ignore this in this function.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape)).
                      Features and targets are assumed to have zero mean and unit variance (each separately).
    optimizer       : torch.optim.Optimizer object
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)
    device (str)    : name of the device to use for training, e.g. 'cpu' or 'cuda'
    alpha           : hyperparameter weight of the loss physical loss term (default: 100000)

    Returns
    -------
    The final loss and accuracy
    """
    N_WIRES = 2

    # Create a train_loader to load the training data in batches
    train_loader1 = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # https://doi.org/10.1007/978-3-031-11349-9_30
    # Based on Khare et al. (2022). Analysis of Loss Functions for Image Reconstruction Using Convolutional Autoencoder,
    # MSE (L2) is the most perfomant function for conv autoencoders, so we will use that.
    mse_loss = torch.nn.MSELoss(reduction='sum')

    track_loss = []

    propagator = Propagator(shape=(16, 16, 16), width=10.0, depth=10.0, height=10.0, D=0.5, padding=-1).to(device)

    net.train()

    f_std, f_mean = features_std_mean
    t_std, t_mean = target_std_mean

    def physical_loss(estimated_normalized_current, features):
        denormalized_features = (features * f_std) + N_WIRES * f_mean
        J = (estimated_normalized_current * t_std) + N_WIRES * t_mean
        B = propagator.get_B(J)
        return mse_loss(B, denormalized_features)

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch1, batch2 in zip(train_loader1, train_loader2):
            # Get the batch data and labels
            features1, targets1 = batch1
            features2, targets2 = batch2

            # features are the magnetic field produced by the sum of the current distributions
            features = features1 + features2
            # source of the magnetic field is a sum of two random current distribution batches
            targets = targets1 + targets2

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            estimates = net(features)

            # Compute the loss
            loss1 = physical_loss(estimates, features) * alpha
            loss2 = mse_loss(estimates, targets)

            loss = loss1 + loss2

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append((loss1.item(), loss2.item()))

        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1]}')

    # Return the loss and accuracy
    return track_loss


def get_3d_from_long_2d(long_2d: torch.Tensor, nx_pixels=16, ny_pixels=16, nz_pixels=16) -> torch.Tensor:
    """
    Convert a 2D tensor of shape (n_samples, n_components, x_pixels * z_pixels, y_pixels) to a 3D tensor of shape (
    n_samples, components, x_pixels, y_pixels, z_pixels)
    """
    n_samples = long_2d.shape[0]
    n_components = long_2d.shape[1]
    return long_2d.transpose(-1, -2).view(n_samples, n_components, nx_pixels, ny_pixels, nz_pixels).permute(0, 1, -1, -3, -2)


def pinn_trainer_multi_wires_adj2(
        net: GeneratorMultipleCNN, train_set: torch.utils.data.TensorDataset, optimizer: torch.optim.Optimizer,
        n_epochs=25, batch_size=100, print_every_n=1, device='cpu', features_std_mean: tuple = (1.0, 0), target_std_mean: tuple = (1.0, 0),
        alpha=100000
):
    """
    For two wires, run physics-informed training on network to optimize the weights and biases that tries to adjust
    the physical backward loop error according to the amplitude of the features.

    Learns to reconstruct normalized estimates from normalized targets. Learns to also have normalized estimates
    such that if they are denormalized to original distributions, the magnetic field they produce matches with the
    denormalized features.

    Two wires are simulated by taking two random batches from the train set and adding them up. Note that STD and MEAN of
    the added batches are not the same as the STD and MEAN of the original batches. We ignore this in this function.

    Parameters
    ----------
    net             : torch.nn.Module object containing the network
    train_set       : dataset to train on, that returns batches of ((n_samples, data.shape), (n_samples, labels.shape)).
                      Features and targets are assumed to have zero mean and unit variance (each separately).
    optimizer       : torch.optim.Optimizer object
    n_epochs        : number of epochs to cover, in each epoch, the entire dataset is seen by the net (default: 25)
    batch_size      : number of samples in each batch (default: 100)
    device (str)    : name of the device to use for training, e.g. 'cpu' or 'cuda'
    alpha           : hyperparameter weight of the loss physical loss term (default: 100000)

    Returns
    -------
    The final loss and accuracy
    """
    N_WIRES = 2

    # Create a train_loader to load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # https://doi.org/10.1007/978-3-031-11349-9_30
    # Based on Khare et al. (2022). Analysis of Loss Functions for Image Reconstruction Using Convolutional Autoencoder,
    # MSE (L2) is the most perfomant function for conv autoencoders, so we will use that.
    mse_loss = torch.nn.MSELoss(reduction='sum')

    track_loss = []

    propagator = Propagator(shape=(16, 16, 16), width=10.0, depth=10.0, height=10.0, D=0.5, padding=-1).to(device)

    net.train()

    f_std, f_mean = features_std_mean
    t_std, t_mean = target_std_mean

    def physical_loss(estimated_normalized_current, features):
        denormalized_features = (features * f_std) + f_mean
        J = (estimated_normalized_current * t_std) + t_mean
        B = propagator.get_B(J)
        return mse_loss(B, denormalized_features)

    # Iterate for each epoch
    for epoch_n in range(n_epochs):
        # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
        for batch in train_loader:
            # Get the batch data and labels
            features, targets = batch['B'], batch['J']

            # zero the parameter gradients
            # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
            optimizer.zero_grad()

            # Forward pass
            # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
            # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
            # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
            estimates = net(features)

            # Compute the loss
            loss1 = physical_loss(estimates, features) * alpha
            loss2 = mse_loss(estimates, targets)

            loss = loss1 + loss2

            # Backpropagate the loss
            loss.backward()

            # Update the weights and biases
            optimizer.step()

            # Keep track of loss at each iteration
            track_loss.append((loss1.item(), loss2.item()))

        if epoch_n == 0:
            print(f'epoch {"":5s} | {"physical loss":>40s} | {"mse loss":9s}')
        if epoch_n % print_every_n == 0 or epoch_n == 0:
            print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {track_loss[-1][0]: >#15.3e} | {track_loss[-1][1]:#1.3e}')

    # Return the loss and accuracy
    return track_loss


class MultiWireDataset(torch.utils.data.Dataset):
    def __init__(self, B, J, indices, transform=None):
        """
        Args:
            B (tensor): Tensor with magnetic field distributions
            J (tensor): Tensor with current distributions
            indices: List of tuples with indices of wires which correspond to the samples in B and J or list of indices list(int)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.B = B
        self.J = J
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idxs):
        if torch.is_tensor(idxs):
            idx: torch.Tensor
            idxs = idxs.tolist()

        match idxs:
            case (*tupl,):
                sample = {'B': torch.sum(self.B[tupl], dim=0),
                          'J': torch.sum(self.J[tupl], dim=0),
                          'wire_tuple': torch.tensor(tupl),}
            case int(_):
                # passed index from the list of indices
                sample = self.__getitem__(self.indices[idxs])
                sample['tuple_index'] = idxs
            case _:
                raise ValueError(f'index must be an integer or tuple specifying wires, got {idxs}')

        if self.transform:
            sample = self.transform(sample)

        # return the sum of magnetic field and the corresponding sum of the current distributions
        return sample
