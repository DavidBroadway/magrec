# Class and fucntions for handling the data and the parameters of the data.

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from magrec.transformation.Fourier import FourierTransform2d
from magrec.image_processing.Filtering import DataFiltering
from magrec.image_processing.Padding import Padder
from magrec.misc.plot import plot_n_components


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
            image = torch.from_numpy(image.astype(np.float32))
        elif type(image) == list:
            image = torch.from_numpy(np.array(image, dtype=np.float32))
        


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

    def numpy_pad_data(self, pad_width: int, mode: str, plot: bool = False):
        """ Pad the data with zeros. """
        self.target, original_roi = self.Padder.numpy_pad_2d(self.target, pad_width, mode, plot)
        self.pad_action(
            'pad_data',
            "Padded the data with numpy pad function",
            None,
            original_roi)
    
    def crop_data(self, roi, image=None):
        """ Crop the data by removing the padding. """
        if image is None:
            image = self.target
            btarget = True
        else:
            btarget = False
        image = self.Padder.crop_data(image, roi)

        roi_string = ''.join(str(x) + ',' for x in roi)

        self.crop_action(
            'crop_data',
            "crop the data with the given region of interest",
            "roi = [" + roi_string  + "]"
            )
        
        if btarget:
            self.target = image
        else:
            return image

    def pad_data_to_power_of_two(self):
        """ Pad the data to a power of two. """
        self.target, original_roi = self.Padder.pad_to_next_power_of_2(self.target)
        self.pad_action(
            'pad_data',
            "Padded the data to a square image with dimension that are a power of two",
            None,
            original_roi)

    def pad_reflective2d(self):
        """ Pad the data with zeros. """
        self.target, original_roi = self.Padder.pad_reflective2d(self.target)
        self.pad_action(
            'pad_data',
            "Padded with reflective boundary conditions",
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
    

    def remove_padding_from_results(self, array):
        # Remove the padding from the results.
        padding = Padder()
        print('Removed the padding that was applied to the data')

        for idx in range(len(self.actions)):
            if self.actions.loc[len(self.actions) - 1-idx].reverseable:
                    roi = self.reverse_parameters[-1-idx]
                    if len(array.shape) > 2: 
                        old_array = array
                        array = torch.zeros((array.shape[0], roi[1] - roi[0], roi[3] - roi[2]))
                        for idx in range(len(array.shape)):
                            array[idx,::] = padding.crop_data(old_array[idx,::], roi)
                    else:
                        array = padding.crop_data(array, roi)
        return array

                        


    # ------------------------- Data Transformation Functions ------------------------- #

    def set_transformer(self, transform_class, **kwargs):
        """ Set the transformation class. """
        self.Transformer = transform_class(self, **kwargs)

    def transform_data(self, data_to_transform=None):
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


        if len(self.target.size()) > 2:
            range = torch.max(torch.abs(self.target*1e3))
            size = self.target.size()
            extent = [0, size[1]*self.dx, 0, size[0]*self.dy]
            plt.figure()
            plt.imshow(self.target*1e3, cmap="bwr", extent = extent, vmin=-range, vmax=range)
            cb = plt.colorbar()
            plt.xlabel("x (um)")
            plt.ylabel("y (um)")
            cb.set_label("Magnetic Field (mT)")
            plt.show()

        else:

            range = torch.max(torch.abs(self.target*1e3))
            size = self.target.size()
            extent = [0, size[1]*self.dx, 0, size[0]*self.dy]
            plt.figure()
            plt.imshow(self.target*1e3, cmap="bwr", extent = extent, vmin=-range, vmax=range)
            cb = plt.colorbar()
            plt.xlabel("x (um)")
            plt.ylabel("y (um)")
            cb.set_label("Magnetic Field (mT)")
            plt.show()
