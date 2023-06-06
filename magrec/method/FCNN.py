# Example of a fitting method for a model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import conv2d, conv3d
from torch.distributions import Normal, Cauchy

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class FCNN(object):

    def __init__(self, 
                 model: object, 
                 learning_rate: float = 0.001):
        """
        Args:
            model: The model to be fitted.
            learning_rate: The learning rate for the optimizer.
        """

        # Defining all of the parameters.
        self.model = model
        self.learning_rate = learning_rate
        torch.autograd.set_detect_anomaly(True)

        self.prepare_fit()

    def prepare_fit(self, 
                    n_channels_in=1, 
                    n_channels_out=1, 
                    ):

        # check model requirements
        self.model.requirements()
        if "num_sources" in self.model.require:
            n_channels_out = self.model.require["num_sources"]
            print("Number of sources: {}".format(n_channels_out))
        if "num_targets" in self.model.require:
            n_channels_in = self.model.require["num_targets"]
            print("Number of targets: {}".format(n_channels_in))
        
        if "source_angles" in self.model.require:
            self.source_angles = self.model.require["source_angles"]
            print("Including source angles in the neural network: {}".format(self.source_angles))
        else:
            self.source_angles = False

        # check if the dataset meets the requirements of the model
        self.model.prepareTargetData()    
        training_target = self.model.training_target

        # define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the network.
        self.Net = Net(training_target, 
                       n_channels_in=n_channels_in, 
                       n_channels_out=n_channels_out,
                       source_angles=self.source_angles).to(self.device)

        # Define the data for loading into the network.
        self.img_comp = torch.Tensor(training_target)
        self.img_input = torch.Tensor(torch.flatten(training_target))
        self.mask =np.where(self.img_input.numpy()  == 0,0,1) 

        # Expand the size of the tensors to include the batch size and channel size
        self.img_comp = torch.FloatTensor(self.img_comp[np.newaxis, np.newaxis])
        self.img_input = torch.FloatTensor(self.img_input[np.newaxis, np.newaxis])
        self.mask_t = torch.FloatTensor(self.mask[np.newaxis,np.newaxis])

        self.train_data_cnn = TensorDataset(self.img_input, self.mask_t)
        self.train_loader = DataLoader(self.train_data_cnn)

        # Define the optimizer
        self.optimizer = optim.Adam(self.Net.parameters(), self.learning_rate, eps=1e-08, weight_decay=0, amsgrad=False)


    def fit(self, n_epochs=25, print_every_n=25):    
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
            # for batch_idx, data in enumerate(self.train_loader):
            for batch_idx, (data,mask_t) in enumerate(self.train_loader):
                # Get the batch data and labels
                data, mask_t= (data).to(self.device), (mask_t).to(self.device)
                data=data*1

                # zero the parameter gradients
                # As described here: https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
                self.optimizer.zero_grad()

                # Forward pass
                # Calling .forward(inputs) skips any PyTorch hooks before the call, and should avoided:
                # Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
                # See also: https://stackoverflow.com/questions/55338756/why-there-are-different-output-between-model-forwardinput-and-modelinput
                if self.source_angles:
                    outputs, self.source_theta, self.source_phi = self.Net(data,
                                                theta_guess = self.model.m_theta, 
                                                phi_guess =self.model.m_phi)

                    # Convert to magnetic field
                    b, outputs = self.model.transform(outputs, self.source_theta, self.source_phi)

                else:
                    outputs = self.Net(data)
                    # Convert to magnetic field
                    b, outputs = self.model.transform(outputs)


                # Compute the loss
                loss = self.model.calculate_loss(b, self.img_comp, nn_output=outputs)

                # Backpropagate the loss
                loss.backward()

                # Update the weights and biases
                self.optimizer.step()

                # Keep track of loss at each iteration
                self.track_loss.append(loss.item())

                if epoch_n % print_every_n == 0 or epoch_n == 0:
                    print(f'epoch {epoch_n + 1:5d} | loss on last mini-batch: {self.track_loss[-1]: .2e}')

        self.final_output = outputs.detach()
        self.final_Jxy = outputs.detach()
        # self.final_Jxy = self.Jxy.detach()
        self.final_b = b.detach()

        if self.source_angles:
            self.final_theta = self.source_theta.detach()[0]
            self.final_phi = self.source_phi.detach()[0]
        if not self.model.fit_m_theta:
            self.final_theta = torch.tensor(self.model.m_theta)
        if not self.model.fit_m_phi:
            self.final_phi = torch.tensor(self.model.m_phi)
        # Return the loss and accuracy
        return 


    def extract_results(self, remove_padding = True, additional_roi=None):
        # print the final angles that were used
        print("Final reconstruction from the network used the following angles:")
        print(f"theta: {self.final_theta:.2f}")
        print(f"phi: {self.final_phi+90:.2f}")

        # Extract the results from the model and return them.
        self.results = self.model.extract_results(self.final_output, 
                                                  self.final_b, 
                                                  remove_padding = remove_padding)

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
    def __init__(self, target, 
                 n_channels_in=1, 
                 n_channels_out=1,
                 source_angles = False):
        super(Net, self).__init__()

        self.source_angles = source_angles

        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out

        self.output_size = target.shape
        self.input_size = len(torch.flatten(target))

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

        self.theta = nn.Linear(in_features=2, out_features=1)  # output layer for theta
        self.phi = nn.Linear(in_features=2, out_features=1)  # output layer for phi

        
        # self.fc11 = nn.Linear(in_features=128,  out_features=120)
        # self.fc12 = nn.Linear(120, 84)
        # self.fc13 = nn.Linear(84, 1)

        # self.fc21 = nn.Linear(in_features=128,  out_features=120)
        # self.fc22 = nn.Linear(120, 84)
        # self.fc23 = nn.Linear(84, 1)


    def forward(self, input, theta_guess = None, phi_guess = None):

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

        final_output =  torch.reshape(out, (1, self.n_channels_out, self.output_size[-2], self.output_size[-1]))

        if self.source_angles:
            # theta = torch.flatten(enc1, 1)
            # theta= F.leaky_relu(self.fc11(theta),0)
            # theta= F.leaky_relu(self.fc12(theta),0.)
            # theta= 180*F.sigmoid(self.fc13(theta))

            # phi = torch.flatten(enc1, 1)
            # phi = F.leaky_relu(self.fc21(phi),0)
            # phi = F.leaky_relu(self.fc22(phi),0.)
            # phi = 360*F.sigmoid(self.fc23(phi))

            theta_guess = torch.tensor([theta_guess, theta_guess], dtype=torch.float32)
            phi_guess = torch.tensor([phi_guess, phi_guess], dtype=torch.float32)
            
            theta = 180*F.sigmoid(self.theta(theta_guess/180))  # output of the theta layer
            phi = 360*F.sigmoid(self.phi(phi_guess/360))  # output of the phi layer
            return final_output, theta, phi
        else:
            return final_output
    
    

    
