# Example of a fitting method for a model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

    def plot_loss(self):
        # Plot the evolution of the loss function at the end of the train
        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        plt.plot(self.track_loss, label='Loss function')
        plt.xlim([0, len(self.track_loss)])
        plt.ylabel('Average difference, B (mT)')
        plt.title('Error function evolution')
        plt.xlabel('Epochs')
