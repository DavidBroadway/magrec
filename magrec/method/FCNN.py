# Example of a fitting method for a model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from magrec.image_processing.Filtering import DataFiltering
from magrec.transformation.Fourier import FourierTransform2d
from magrec.image_processing.Padding import Padder

class FCNN(object):

    def __init__(self, model, dataset, learning_rate = 0.001, fourier_model = False):
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.fourier_model = fourier_model
        self.ft = FourierTransform2d(grid_shape=dataset.target.size(), dx=dataset.dx, dy=dataset.dy, real_signal=True)
        self.Padder = Padder()

    def prepare_fit(self, 
                    n_channels_in=1, 
                    n_channels_out=1, 
                    loss_weight = None
                    ):
        # Prepare the method for fitting.
       
        # Check the size of the data and pad it if necessary.

        # check model requirements
        self.model.requirements()
        if "num_sources" in self.model.require:
            n_channels_out = self.model.require["num_sources"]
            print("Number of sources: {}".format(n_channels_out))
        if "num_targets" in self.model.require:
            n_channels_in = self.model.require["num_targets"]
            print("Number of targets: {}".format(n_channels_in))


        # check if the dataset meets the requirements of the model
        self.model.prepareTargetData()    
        training_target = self.model.training_target


        # define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the network.
        self.Net = Net(training_target, 
                       n_channels_in=n_channels_in, 
                       n_channels_out=n_channels_out).to(self.device)

        # Define the data for loading into the network.
        self.img_comp = torch.Tensor(training_target)
        self.img_input = torch.Tensor(torch.flatten(training_target))
        self.mask =np.where(self.img_input.numpy()  == 0,0,1) 

        if n_channels_in > 1:
            self.img_comp = torch.FloatTensor(self.img_comp[np.newaxis])
        else:
            self.img_comp = torch.FloatTensor(self.img_comp[np.newaxis, np.newaxis])
        # self.img_comp = torch.FloatTensor(self.img_comp[np.newaxis, np.newaxis])

        self.img_input = torch.FloatTensor(self.img_input[np.newaxis, np.newaxis])
        self.mask_t = torch.FloatTensor(self.mask[np.newaxis,np.newaxis])

        # Define the wieght matrix for the loss function, if it is in Fourier space transform it. 
        if loss_weight is not None:
            if self.fourier_model:
                self.loss_weight = self.ft.forward(loss_weight, dim=(-2, -1) )
            else:
                self.loss_weight = loss_weight
        else:
             self.loss_weight = loss_weight

        self.train_data_cnn = TensorDataset(self.img_input, self.mask_t)
        self.train_loader = DataLoader(self.train_data_cnn)

        # Define the optimizer
        self.optimizer = optim.Adam(self.Net.parameters(), self.learning_rate, eps=1e-08, weight_decay=0, amsgrad=False)


    def fit(self, n_epochs=25, print_every_n=10, weight = None, add_filter = False):    
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

        # Create a train_loader to load the training data in batches
        self.track_loss = []

        # Set the network to training mode
        self.Net.train()


        # # Define the wieght matrix for the output, if it is in Fourier space transform it. 
        weight = self.Padder.pad_zeros2d(weight)
        # if weight is not None:
        #     if self.fourier_model:

        #         weight = self.Padder.pad_zeros2d(weight)
        #         weight = self.ft.forward(weight, dim=(-2, -1))
        #         # weight = torch.cat((weight, weight), 0)
        #         print('Converting weight matrix to Fourier space')


        # Iterate for each epoch
        for epoch_n in range(n_epochs):
            # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
            # for batch_idx, data in enumerate(self.train_loader):
            for batch_idx, (data,mask_t) in enumerate(self.train_loader):
                # Get the batch data and labels
                # inputs = batch
                # print(inputs[0].size() )
                data, mask_t= (data).to(self.device), (mask_t).to(self.device)
                data=data*1

                # zero the parameter gradients
                # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
                self.optimizer.zero_grad()

                # Forward pass
                # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
                # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
                # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
                outputs = self.Net(data)

                # Add restrictions to the output of the NN 
                if self.fourier_model:
                    
                    # add a hanning filter for the Fourier method
                    nn_shape = outputs.shape
                    real_component =  outputs[..., 0:int(0.5*nn_shape[-1])]
                    imag_component =  outputs[..., int(0.5*nn_shape[-1]):nn_shape[-1]]
                    if add_filter:
                        filter_size = 1*self.dataset.height

                        Filtering = DataFiltering(real_component, self.dataset.dx, self.dataset.dy)
                        real_component = Filtering.apply_hanning_filter(filter_size, data=real_component, in_fourier_space=True)
                        real_component = Filtering.apply_short_wavelength_filter(filter_size, data=real_component, in_fourier_space=True,  print_action=False) 
                        
                        imag_component = Filtering.apply_hanning_filter(filter_size, data=imag_component, in_fourier_space=True)
                        imag_component = Filtering.apply_short_wavelength_filter(filter_size, data=imag_component, in_fourier_space=True,  print_action=False) 
                        
                    if weight is not None:
                        
                        Real_space_component =  self.ft.backward(real_component, dim=(-2, -1))
                        Real_space_component = Real_space_component * weight
                        real_component = self.ft.forward(Real_space_component, dim=(-2, -1))

                        Real_space_I_component =  self.ft.backward(imag_component, dim=(-2, -1))
                        Real_space_I_component = Real_space_I_component * weight
                        imag_component = self.ft.forward(Real_space_I_component, dim=(-2, -1))
                        # real_component = torch.einsum("...kl,kl->...kl", real_component, weight)
                        # imag_component = torch.einsum("...kl,kl->...kl", imag_component, weight)

                    outputs[..., 0:int(0.5*nn_shape[-1])] = real_component
                    outputs[..., int(0.5*nn_shape[-1]):nn_shape[-1]] = imag_component
                else:
                    if weight is not None:
                        outputs = torch.einsum("...kl,kl->...kl", outputs, weight)

                # Convert to magnetic field
                b = self.model.transform(outputs)

                # Compute the loss
                loss = self.model.calculate_loss(b, self.img_comp, loss_weight = self.loss_weight, nn_output=outputs)

                # Backpropagate the loss
                loss.backward()

                # Update the weights and biases
                self.optimizer.step()

                # Keep track of loss at each iteration
                self.track_loss.append(loss.item())

                if epoch_n % print_every_n == 0 or epoch_n == 0:
                    print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {self.track_loss[-1]: .2e}')

        self.final_output = outputs.detach()
        self.final_b = b.detach()

        # Return the loss and accuracy
        return 


    def extract_results(self, remove_padding = True):
        # Extract the results from the model and return them.
        self.results = self.model.extract_results(self.final_output, self.final_b, remove_padding = remove_padding)

    def plot_results(self, remove_padding = True):
        # Plot the results from the model.

        # check is results have been unpacked
        if not hasattr(self, "results"):
            self.results = self.model.extract_results(self.final_output, self.final_b, remove_padding = remove_padding)

        self.model.plot_results(self.results)

    def plot_loss(self):
        # Plot the evolution of the loss function at the end of the train
        plt.figure()
        plt.plot(self.track_loss, label='Loss function')
        plt.xlim([0, len(self.track_loss)])
        plt.ylabel('Average difference B (mT)')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')


# Subclass for the architecture of the NN
class Net(nn.Module):
    # class to create a fully connected neural network for magnetisation reconstruction
    def __init__(self, dataset, n_channels_in=1, n_channels_out=1):
        super(Net, self).__init__()

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        self.output_size = dataset.shape
        self.input_size = len(torch.flatten(dataset))

        # Larger network
        # self.enc1 = nn.Linear(in_features=self.input_size, out_features=1024)
        # self.enc2 = nn.Linear(in_features=1024, out_features=512)
        # self.enc3 = nn.Linear(in_features=512, out_features=256)
        # self.enc4 = nn.Linear(in_features=256, out_features=128)
        # self.enc5 = nn.Linear(in_features=128, out_features=64)
        # self.enc6 = nn.Linear(in_features=64, out_features=1)

        # self.dec1 = nn.Linear(in_features=64, out_features=128)
        # self.dec2 = nn.Linear(in_features=128, out_features=256)
        # self.dec3 = nn.Linear(in_features=256, out_features=512)
        # self.dec4 = nn.Linear(in_features=512, out_features=1024)
        # self.dec5 = nn.Linear(in_features=1024, out_features= self.n_channels_out * self.input_size)

        # Middle network
        # self.enc1 = nn.Linear(in_features=self.input_size, out_features=256)
        # self.enc2 = nn.Linear(in_features=256, out_features=128)
        # self.enc3 = nn.Linear(in_features=128, out_features=64)
        # self.enc4 = nn.Linear(in_features=64, out_features=32)
        # self.enc5 = nn.Linear(in_features=32, out_features=16)


        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        # self.dec5 = nn.Linear(in_features=256, out_features= self.n_channels_out * self.input_size)

        # Smaller network
        self.enc1 = nn.Linear(in_features=self.input_size, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=64)
        self.enc3 = nn.Linear(in_features=64, out_features=32)
        self.enc4 = nn.Linear(in_features=32, out_features=16)
        self.enc5 = nn.Linear(in_features=16, out_features=8)

        self.dec1 = nn.Linear(in_features=8, out_features=16)
        self.dec2 = nn.Linear(in_features=16, out_features=32)
        self.dec3 = nn.Linear(in_features=32, out_features=64)
        self.dec4 = nn.Linear(in_features=64, out_features=128)
        self.dec5 = nn.Linear(in_features=128, out_features= self.n_channels_out * self.input_size)

        # tiny network
        # self.m = nn.Linear(in_features=self.input_size, out_features= self.n_channels_out * self.input_size)

    def forward(self,input):

        enc1 = F.relu(self.enc1(input))
        enc2 = F.relu(self.enc2(enc1))
        enc3 = F.relu(self.enc3(enc2))
        enc4 = F.relu(self.enc4(enc3))
        enc5 = F.relu(self.enc5(enc4))

        dec1 = F.relu(self.dec1(enc5))
        dec2 = F.relu(self.dec2(dec1))
        dec3 = F.relu(self.dec3(dec2))
        dec4 = F.relu(self.dec4(dec3))
        out = self.dec5(dec4)

        # out = self.enc1(input)


        return torch.reshape(out, (1, self.n_channels_out, self.output_size[-2], self.output_size[-1]))
