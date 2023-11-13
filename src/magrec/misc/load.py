"""
Data loaderers to be used with magrec.
"""

import os
import json

import numpy as np


def load_matlab_simulation(datapath):
    """
    Load a magnetic field simulation from a .json file that contains 
    the source magnetization patterns and the magnetic field from a simulation.

    Args:
        datapath (str): Path to the .json file.

    Returns:
        dict: Dictionary containing the source magnetization patterns and the magnetic field.
    """
    with open(datapath) as f:
        data = json.load(f)

    # Import below follows `magnetisation_reconstruction.2d.Magnetisation.utils.LoadData` functions for key names.
    
    magnetic_propagation = data["MagnetisationSimulation"]["MagneticPropagation"]
    magnetic_field = magnetic_propagation["MagneticField"]
    source_magnetisation = magnetic_propagation["PropStruct"]
    
    datadict = {
        "M": {
            "data": source_magnetisation["Mag"]["Data"],
            "direction": source_magnetisation["Kernal"]["AssumedMagDir"],
        },
        "B": {
            "data": magnetic_field["BNV"],
            "height": source_magnetisation["BNV"]["Height"],  # looks like it is in meters in the simulation files
        },
        "NV": {
            "theta": source_magnetisation["BNV"]["Theta"],
            "phi": source_magnetisation["BNV"]["Phi"],
        },
        "dx": source_magnetisation["PixelSizeX"],
        "dy": source_magnetisation["PixelSizeY"],
        "shape": None  # to be computed below after the conversion of the data to numpy arrays
    }

    # convert all data to numpy arrays
    datadict["M"]["data"] = np.array(datadict["M"]["data"])
    datadict["B"]["data"] = np.array(datadict["B"]["data"])

    # calculate shape of the data and check that it is valid
    datadict["shape"] = datadict["M"]["data"].shape
    if datadict["shape"] != datadict["B"]["data"].shape:
        raise ValueError("Shape of the magnetic field data does not match the shape of the magnetization data.")
            
    return datadict


def load_matlab_data(datapath):
    """
    Load a general data set from a .mat file produced by matlab.
    """
    import scipy.io as sio
    mat = sio.loadmat(datapath)
    return mat