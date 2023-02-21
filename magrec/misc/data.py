# Class and fucntions for handling the data and the parameters of the data.

import torch
import numpy as np
import matplotlib.pyplot as plt
from magrec.prop.Filtering import DataFiltering
from magrec.prop.Padding import Padder


class Data(object):
    # class for the data that containes the data, the parameters, and plotting functions.

    def __init__(self):
        """ Initialise the data class. """
        self.actions = []
        self.Padder = Padder()

    def load_data(self, image):
        """ Load the data. """

        # Convert the data to a torch tensor.
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)
        elif type(image) == list:
            image = torch.from_numpy(np.array(image))

        self.target = image
        self.data_modifications = dict()
        self.data_modifications["0"] = image

    def define_pixel_size(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    def define_height(self, height):
        self.height = height

    def define_sensor_angles(self, theta, phi):
        self.sensor_theta = theta
        self.sensor_phi = phi

    def define_layer_thickness(self, layer_thickness):
        self.layer_thickness = layer_thickness


    # ------------------------- Data Modification Functions ------------------------- #
    
    def add_hanning_filter(self, wavelength):
        Filtering = DataFiltering(self.target, self.dx, self.dy)
        self.target = Filtering.apply_hanning_filter(wavelength, data=self.target, plot_results=False)
        self.track_actions('hanning_filter')

    def remove_DC_background(self):
        Filtering = DataFiltering(self.target, self.dx, self.dy)
        self.target = Filtering.remove_DC_background(data=self.target)
        self.track_actions('remove_DC_background')

    def add_short_wavelength_filter(self, wavelength):
        Filtering = DataFiltering(self.target, self.dx, self.dy)
        self.target = Filtering.apply_short_wavelength_filter(wavelength, data=self.target, plot_results=False)
        self.track_actions('short_wavelength_filter')
    
    def add_long_wavelength_filter(self, wavelength):
        Filtering = DataFiltering(self.target, self.dx, self.dy)
        self.target = Filtering.apply_long_wavelength_filter(wavelength, data=self.target, plot_results=False)
        self.track_actions('long_wavelength_filter')


    # ------------------------- Data Padding Functions ------------------------- #

    def pad_data(self, padding):
        """ Pad the data with zeros. """
        self.target = self.Padder.pad_data(padding)
        self.track_actions('pad_data')
    
    def crop_data(self, padding):
        """ Crop the data by removing the padding. """
        self.target = self.Padder.crop_data(padding)
        self.track_actions('crop_data')

    def pad_data_to_power_of_two(self):
        """ Pad the data to a power of two. """
        self.target = self.Padder.pad_to_next_power_of_2(self.target)
        self.track_actions('pad_data_to_power_of_two')

    def crop_data_to_power_of_two(self):
        """ Crop the data to a power of two. """
        self.target = self.Padder.crop_data_to_power_of_two()
        self.track_actions('crop_data_to_power_of_two')


    # ------------------------- Data Modification Tracking Functions ------------------------- #
    def track_actions(self, action):
        """ Track the actions performed on the data. """
        self.actions.append(action)
        action_num = len(self.actions)
        self.data_modifications[str(action_num)] = self.target


    # ------------------------- Data Plotting Functions ------------------------- #


    def plot_target(self):
        """ Plot the target image """
        # REPLACE THIS WITH THE ALREADY EXISTING PLOT FUNCTION IN MISC

        range = torch.max(torch.abs(self.target*1e3))
        size = self.target.size()
        x = torch.linspace(0, size[0]*self.dx, size[0])
        y = torch.linspace(0, size[1]*self.dy, size[1])
        extent = [0, size[0]*self.dx, 0, size[1]*self.dy]
        plt.figure()
        plt.imshow(self.target*1e3, extent=extent, cmap="bwr", vmin=-range, vmax=range)
        cb = plt.colorbar()
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        cb.set_label("Magnetic Field (mT)")
        plt.show()
