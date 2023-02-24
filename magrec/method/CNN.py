# Example of a fitting method for a model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class CNN(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset


    def prepare_fit(self, n_channels_in=1, n_channels_out=1, size=2, kernel=5, stride=2, padding=2):
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

        # define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the network.
        self.Net = Net(n_channels_in=n_channels_in, 
            n_channels_out=n_channels_out, 
            ImageSize = self.dataset.target.size()[-1], 
            kernel=kernel, 
            stride=stride, 
            padding=padding).to(self.device)

        # Define the data for loading into the network.
        self.mask = np.where(self.dataset.target.numpy() == 0,0,1)  

        # Normalise the data for the network.
        # nomalized_data = (self.data.target - self.data.target.mean()) / torch.sqrt(self.data.target.var())

        # Need to add two dimensions to the data to make it a batch
        # if n_channels_in > 1:
        #     self.img_input = torch.FloatTensor(self.data.target[np.newaxis])
        #     self.mask_t = torch.FloatTensor(self.mask[np.newaxis])
        # else:
        self.img_input = torch.FloatTensor(self.dataset.target[np.newaxis, np.newaxis])
        self.mask_t = torch.FloatTensor(self.mask[np.newaxis,np.newaxis])
        

        self.train_data_cnn = TensorDataset(self.img_input, self.mask_t)
        self.train_loader = DataLoader(self.train_data_cnn)

        # Define the optimizer
        self.optimizer = optim.Adam(self.Net.parameters())


    def fit(self, n_epochs=25, print_every_n=10, weight = None):    
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

        # Iterate for each epoch
        for epoch_n in range(n_epochs):
            # Note that the training data is shuffled every epoch by the DataLoader

        # Iterate for each batch
            for batch_idx, (data,mask_t) in enumerate(self.train_loader):
                # Get the batch data and labels
                # inputs = batch
                # print(inputs[0].size() )
                data,mask_t= (data).to(self.device), mask_t.to(self.device)
                data=data*1

                # zero the parameter gradients
                # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
                self.optimizer.zero_grad()

                # Forward pass
                # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
                # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
                # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
                outputs = self.Net(data)

                if weight is not None:
                    outputs[0,0,:,:] = outputs[0,0,:,:]*weight

                # Convert to magnetic field
                b = self.model.transform(outputs)

                # Compute the loss
                loss = self.model.calculate_loss(b, self.img_input)

                # Backpropagate the loss
                loss.backward()

                # Update the weights and biases
                self.optimizer.step()

                # Keep track of loss at each iteration
                self.track_loss.append(loss.item())

                if epoch_n % print_every_n == 0 or epoch_n == 0:
                    print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {self.track_loss[-1]: .2e}')

        self.final_output = outputs[0,0,:,:].detach().numpy() 
        self.final_b = b[0,0,:,:].detach().numpy() 

        # Return the loss and accuracy
        return 


    def extract_results(self):
        # Extract the results from the model and return them.
        self.results = self.model.unpack_results(self.final_output)

    def plot_results(self):
        # Plot the results from the model.
        self.model.plot_results()

    def plot_loss(self):
        # Plot the evolution of the loss function at the end of the train
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        plt.plot(self.track_loss, label='Loss function')
        plt.xlim([0, len(self.track_loss)])
        plt.ylabel('Average difference, $\Delta B (T)$')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')

    def save_results(self):
        # Save the results from the model.
        results = self.extract_results()
        self.dataset.save_results(results)


# Subclass for the architecture of the NN
class Net(nn.Module):
    """
    Architecture for 2d → 2d image reconstruction, which learns to reconstruct 2d image from another 2d image.
    """
    def __init__(self, Size=1, ImageSize=256, kernel = 5, stride = 2, padding = 2, n_channels_in = 1, n_channels_out=1):

        """
        Create the net that takes an image of currents of size (3, W, H) and creates another image (3, W, H).
        3 corresponds to the number of channels in the input image. W and H must be multiples of 2^4 = 16, because the
        net has 4 convolution layers if stride = 2.

        Args:
            n_channels_in:  number of channels in input image (number of components)
            size:           kinda channel inflation parameter, inner convolution layers give size * 8 or size * 16 output parameters
            kernel:         kernel size
            stride:         step in which to do the convolution
            padding:        whether to pad input image for convolution and by how much

        Returns:
            GeneratorCNN:   the net
        """
        super().__init__()

        M=Size
    
        if ImageSize == 512:
            ConvolutionSize = 32
        elif ImageSize == 256:
            ConvolutionSize = 16
        else: # size is 128
            ConvolutionSize = 8
        # first index is the number of channels
        self.convi = nn.Conv2d(n_channels_in, 8*M, kernel, 1, padding)
        self.conv_r0 = nn.Conv2d(1, 8*M, kernel, 1, padding)
        self.conv1 = nn.Conv2d(8*M, 8*M, kernel, stride, padding)
        self.bn1  = nn.BatchNorm2d(8*M)
        self.conv2 = nn.Conv2d(8*M, 16*M, kernel, stride, padding)
        self.bn2  = nn.BatchNorm2d(16*M)
        self.conv3 = nn.Conv2d(16*M, 32*M, kernel, stride, padding)
        self.bn3  = nn.BatchNorm2d(32*M)
        self.conv4 = nn.Conv2d(32*M, 64*M, kernel, stride, padding)
        self.bn4  = nn.BatchNorm2d(64*M)

        self.conv5 = nn.Conv2d(64*M, 128*M, 5, 1, 2)
        self.bn5  = nn.BatchNorm2d(128*M)

        self.trans1 = nn.ConvTranspose2d(128*M, 64*M, kernel, stride, padding,1)
        self.trans2 = nn.ConvTranspose2d(64*M+32*M, 32*M, kernel, stride, padding,1)
        self.trans3 = nn.ConvTranspose2d(32*M+16*M, 16*M, kernel, stride, padding,1)
        self.trans4 = nn.ConvTranspose2d(16*M+8*M, 8*M, kernel, stride, padding,1)
        self.conv6 = nn.Conv2d(8*M, n_channels_out, kernel, 1, padding)
        self.conv7 = nn.Conv2d(n_channels_out, n_channels_out, kernel, 1, padding)
        # self.conv6 = nn.Conv2d(8*M, 2, 5, 1, 2)
        # self.conv7 = nn.Conv2d(2, 1, kernel, 1, padding)
        # add the 2 at the first index for 2 outputs. 

        self.fc11 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc12 = nn.Linear(120, 84)
        self.fc13 = nn.Linear(84, 1)

        self.fc21 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc22 = nn.Linear(120, 84)
        self.fc23 = nn.Linear(84, 1)

        self.fc31 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc32 = nn.Linear(120, 84)
        self.fc33 = nn.Linear(84, 1)

        self.fc41 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc42 = nn.Linear(120, 84)
        self.fc43 = nn.Linear(84, 1)

        self.fc51 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.fc52 = nn.Linear(120, 84)
        self.fc53 = nn.Linear(84, 1)

        self.transfc1 = nn.Linear(64*M * ConvolutionSize*ConvolutionSize, 120)
        self.transfc2 = nn.Linear(120, 256)
        self.transfc3 = nn.Linear(256, 65536)

    def forward(self, input):

        conv0 = self.convi(input)
        conv0 = F.leaky_relu(conv0, 0.2)
        conv1 = F.leaky_relu(self.bn1(self.conv1(conv0)), 0.2)
        conv2 = F.leaky_relu(self.bn2(self.conv2(conv1)), 0.2)
        conv3 = F.leaky_relu(self.bn3(self.conv3(conv2)), 0.2)
        conv4 = F.leaky_relu(self.bn4(self.conv4(conv3)), 0.2)

        conv5 = F.leaky_relu(self.conv5(conv4), 0.2)

        trans1 = F.leaky_relu(self.bn4(self.trans1(conv5)), 0.2)
        trans2 = F.leaky_relu(self.bn3(self.trans2(torch.cat([conv3, trans1], dim=1))), 0.2)
        trans3 = F.leaky_relu(self.bn2(self.trans3(torch.cat([conv2, trans2], dim=1))), 0.2)
        trans4 = F.leaky_relu(self.bn1(self.trans4(torch.cat([conv1, trans3], dim=1))), 0.2)

        conv6 = self.conv6(trans4)
        conv7 = self.conv7(conv6)

        return conv7

