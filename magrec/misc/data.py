# Class and fucntions for handling the data and the parameters of the data.
# Author: David A. Broadway and Mykalio Flakes

import numpy as np
import pytorch as torch


class Data(object):
    # class for the data that containes the data, the parameters, and plotting functions.

    def __init__(self,):
        """ Initialise the data class. """


    def load_data(self, data):
        self.data = data

    def define_pixel_size(self, dx, dy):
        self.dx = dx
        self.dy = dy
    
    def define_height(self, height):
        self.height = height

    def define_layer_thickness(self, layer_thickness):
        self.layer_thickness = layer_thickness

