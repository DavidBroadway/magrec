# Class and fucntions for handling the data and the parameters of the data.

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.Filtering import DataFiltering
from magrec.prop.Padding import Padder


class Data(object):
    # class for the data that containes the data, the parameters, and plotting functions.

    def __init__(self):
        """ Initialise the data class. """
        self.actions = pd.DataFrame(
            columns=[
                "action type", 
                "reverseable",
                "reverse action", 
                "description", 
                "parameters"])
        self.actions["reverseable"] = self.actions["reverseable"].astype(bool)

    def load_data(self, image, dx, dy, height, theta, phi, layer_thickness):
        """ Load the data. """

        # Convert the data to a torch tensor.
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)
        elif type(image) == list:
            image = torch.from_numpy(np.array(image))
        
        # Define the parameters of the data.
        self.define_pixel_size(dx, dy)
        self.define_height(height)
        self.define_sensor_angles(theta, phi)
        self.define_layer_thickness(layer_thickness)

        # Define the data as the target of the reconstruction.
        self.target = image

        # Define the data objects for tracking the actions on the data.
        self.data_modifications = tuple()
        self.reverse_parameters = tuple()
        action = pd.DataFrame({
            'action type': "load_data",
            'reverseable': False,
            'reverse action': None,
            'description': "loaded the data",
            'parameters': None
        }, index=[0])
        self.track_actions(action)

        # load classes
        self.load_filter_Class()
        self.Padder = Padder()
        self.Ft = FourierTransform2d(self.target.shape, dx=self.dx, dy=self.dy)


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
    def load_filter_Class(self):
        """ Load the data filtering class. """
        self.Filtering = DataFiltering(self.target, self.dx, self.dy)

    def add_hanning_filter(self, wavelength):
        """ Add a hanning filter to the data. """
        self.target = self.Filtering.apply_hanning_filter(wavelength, data=self.target, plot_results=False)
        # Track the action on the dataset
        self.filter_action(
            'hanning_filter', 
            f"Applied a low frequency filter, removing all components larger than {wavelength} um", 
            f"wavelength = {wavelength}")


    def remove_DC_background(self):
        """ Remove the DC background from the data. """
        self.target = self.Filtering.remove_DC_background(data=self.target)
        # Track the action on the dataset
        self.filter_action(
            'remove_DC_background',
            "Removed the DC background from the data",
            None)


    def add_short_wavelength_filter(self, wavelength):
        """ Add a short wavelength filter to the data. """
        self.target = self.Filtering.apply_short_wavelength_filter(wavelength, data=self.target, plot_results=False)
        # Track the action on the dataset
        self.filter_action(
            'short_wavelength_filter',
            f"Applied a high frequency filter, removing all components smaller than {wavelength} um",
            f"wavelength = {wavelength}")

    def add_long_wavelength_filter(self, wavelength):
        """ Add a long wavelength filter to the data. """
        self.target = self.Filtering.apply_long_wavelength_filter(wavelength, data=self.target, plot_results=False)
        # Track the action on the dataset
        self.filter_action(
            'long_wavelength_filter',
            f"Applied a low frequency filter, removing all components larger than {wavelength} um",
            f"wavelength = {wavelength}")

    def filter_action(self, action, description, parameters):
        """ Defines the dictionary of the filter action and tracks the action. """
        action = pd.DataFrame({
            'action type': action,
            'reverseable': False,
            'reverse action': None,
            'description': description,
            'parameters': parameters
        }, index=[0])
        self.track_actions(action)
        return

    # ------------------------- Data Padding Functions ------------------------- #

    def pad_data(self, padding):
        """ Pad the data with zeros. """
        self.target = self.Padder.pad_data(padding)
        self.track_actions('pad_data')
    
    def crop_data(self, roi):
        """ Crop the data by removing the padding. """
        self.target = self.Padder.crop_data(self.target, roi)

        roi_string = ''.join(str(x) + ',' for x in roi)

        self.crop_action(
            'crop_data',
            "crop the data with the given region of interest",
            "roi = [" + roi_string  + "]"
            )

    def pad_data_to_power_of_two(self):
        """ Pad the data to a power of two. """
        self.target, original_roi = self.Padder.pad_to_next_power_of_2(self.target)
        self.pad_action(
            'pad_data',
            "Padded the data to a square image with dimension that are a power of two",
            None,
            original_roi)

    def pad_action(self, action, description, parameters, reverse_parameters):
        """ Defines the dictionary of the filter action and tracks the action. """
        action = pd.DataFrame({
            'action type': action,
            'reverseable': True,
            'reverse action': 'crop_data',
            'description': description,
            'parameters': parameters
        }, index=[0])
        self.track_actions(action, reverse_parameters=reverse_parameters)
        return

    def crop_action(self, action, description, parameters):
        """ Defines the dictionary of the filter action and tracks the action. """
        action = pd.DataFrame({
            'action type': action,
            'reverseable': False,
            'reverse action': 'crop_data',
            'description': description,
            'parameters': parameters
        }, index=[0])
        self.track_actions(action)
        return


    # ------------------------- Data Transformation Functions ------------------------- #

    def set_transformer(self, transform_class, **kwargs):
        """ Set the transformation class. """
        self.Transformer = transform_class(self, **kwargs)

    def transform_data(self):
        """ Transform the data. """
        self.transformed_target = self.Transformer.transform()


    # ------------------------- Data Modification Tracking Functions ------------------------- #
    def track_actions(self, action, reverse_parameters=None):
        """ Track the actions performed on the data. """
        self.actions = pd.concat([self.actions, action], ignore_index=True)
        self.data_modifications = self.data_modifications + (self.target,)
        self.reverse_parameters = self.reverse_parameters + (reverse_parameters,)




    # ------------------------- Data Plotting Functions ------------------------- #


    def plot_target(self):
        """ Plot the target image """
        # REPLACE THIS WITH THE ALREADY EXISTING PLOT FUNCTION IN MISC

        range = torch.max(torch.abs(self.target*1e3))
        # size = self.target.size()
        # x = torch.linspace(0, size[0]*self.dx, size[0])
        # y = torch.linspace(0, size[1]*self.dy, size[1])
        # extent = [0, size[0]*self.dx, 0, size[1]*self.dy]
        plt.figure()
        plt.imshow(self.target*1e3, cmap="bwr", vmin=-range, vmax=range)
        cb = plt.colorbar()
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        cb.set_label("Magnetic Field (mT)")
        plt.show()
