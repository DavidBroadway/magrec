# Example of a fitting method for a model.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericMethod(object):

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def prepare_fit(self):
        # Prepare the method for fitting.
        # In the case of a NN this is making the network.

        # Check the size of the and pad it if necessary.

        # For example for NN:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Network = self.arch(device)
        self.Network.generate_network()


    def fit(self):    
        # fit the data with the model.
        # in the case of a NN this is training
        
        # For example:
        for Epoch in tqdm(range(Epochs)):
            for batch in range(n_batches):
                # Get the data for the batch.
                batch_data = self.data[batch*batch_size:(batch+1)*batch_size]

                # Get the output of the network.
                nn_output = self.model(batch_data)

                # convert the output to a magnetic field.
                magnetic_field = self.model.transform(nn_output)

                # Get the loss.
                loss = self.model.loss_function(magnetic_field, batch_data)

                # Backpropogate the loss.
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()


    def extract_results(self):
        # Extract the results from the model and return them.
        return self.model.unpack_results()

    def plot_results(self):
        # Plot the results from the model.
        self.model.plot_results()

    def save_results(self):
        # Save the results from the model.
        results = self.extract_results()
        self.data.save_results(results)


    # Subclass for the architecture of the NN
    class arch(self, nn.Module):
            # defin the architecture of the model.
            # CNN, MLP, etc. 
                def __init__(self, Size=2, ImageSize=256, kernel = 5, stride = 2, padding = 2, channels_in = 1, channels_out=1):
                    super(generator_CNN, self).__init__()

                    M=Size
                    #etc
