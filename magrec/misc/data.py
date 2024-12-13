# Class and fucntions for handling the data and the parameters of the data.

import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from magrec.misc.plot import plot_n_components

import pyvista as pv


# class Data(object):
#     # class for the data that containes the data, the parameters, and plotting functions.

#     def __init__(self):
#         """ Initialise the data class. """
#         self.actions = pd.DataFrame(
#             columns=[
#                 "action type", 
#                 "reverseable",
#                 "reverse action", 
#                 "description", 
#                 "parameters"])
#         self.actions["reverseable"] = self.actions["reverseable"].astype(bool)

#     def load_data(self, image, dx, dy, height, theta, phi, layer_thickness):
#         """ Load the data. """

#         # Convert the data to a torch tensor.
#         if type(image) == np.ndarray:
#             image = torch.from_numpy(image.astype(np.float32))
#         elif type(image) == list:
#             image = torch.from_numpy(np.array(image, dtype=np.float32))
        


#         # Define the parameters of the data.
#         self.define_pixel_size(dx, dy)
#         self.define_height(height)
#         self.define_sensor_angles(theta, phi)
#         self.define_layer_thickness(layer_thickness)

#         # Define the data as the target of the reconstruction.
#         self.target = image

#         # Define the data objects for tracking the actions on the data.
#         self.data_modifications = tuple()
#         self.reverse_parameters = tuple()
#         action = pd.DataFrame({
#             'action type': "load_data",
#             'reverseable': False,
#             'reverse action': None,
#             'description': "loaded the data",
#             'parameters': None
#         }, index=[0])
#         self.track_actions(action)

#         # load classes
#         self.load_filter_Class()
#         self.Padder = Padder()
#         self.Ft = FourierTransform2d(self.target.shape, dx=self.dx, dy=self.dy)


#     def define_pixel_size(self, dx, dy):
#         self.dx = dx
#         self.dy = dy
    
#     def define_height(self, height):
#         self.height = height

#     def define_sensor_angles(self, theta, phi):
#         self.sensor_theta = theta
#         self.sensor_phi = phi

#     def define_layer_thickness(self, layer_thickness):
#         self.layer_thickness = layer_thickness


#     # ------------------------- Data Modification Functions ------------------------- #
#     def load_filter_Class(self):
#         """ Load the data filtering class. """
#         self.Filtering = DataFiltering(self.target, self.dx, self.dy)

#     def add_hanning_filter(self, wavelength):
#         """ Add a hanning filter to the data. """
#         self.target = self.Filtering.apply_hanning_filter(wavelength, data=self.target, plot_results=False)
#         # Track the action on the dataset
#         self.filter_action(
#             'hanning_filter', 
#             f"Applied a low frequency filter, removing all components larger than {wavelength} um", 
#             f"wavelength = {wavelength}")


#     def remove_DC_background(self):
#         """ Remove the DC background from the data. """
#         self.target = self.Filtering.remove_DC_background(data=self.target)
#         # Track the action on the dataset
#         self.filter_action(
#             'remove_DC_background',
#             "Removed the DC background from the data",
#             None)


#     def add_short_wavelength_filter(self, wavelength):
#         """ Add a short wavelength filter to the data. """
#         self.target = self.Filtering.apply_short_wavelength_filter(wavelength, data=self.target, plot_results=False)
#         # Track the action on the dataset
#         self.filter_action(
#             'short_wavelength_filter',
#             f"Applied a high frequency filter, removing all components smaller than {wavelength} um",
#             f"wavelength = {wavelength}")

#     def add_long_wavelength_filter(self, wavelength):
#         """ Add a long wavelength filter to the data. """
#         self.target = self.Filtering.apply_long_wavelength_filter(wavelength, data=self.target, plot_results=False)
#         # Track the action on the dataset
#         self.filter_action(
#             'long_wavelength_filter',
#             f"Applied a low frequency filter, removing all components larger than {wavelength} um",
#             f"wavelength = {wavelength}")

#     def filter_action(self, action, description, parameters):
#         """ Defines the dictionary of the filter action and tracks the action. """
#         action = pd.DataFrame({
#             'action type': action,
#             'reverseable': False,
#             'reverse action': None,
#             'description': description,
#             'parameters': parameters
#         }, index=[0])
#         self.track_actions(action)
#         return

#     # ------------------------- Data Padding Functions ------------------------- #

#     def pad_data(self, padding):
#         """ Pad the data with zeros. """
#         self.target = self.Padder.pad_data(padding)
#         self.track_actions('pad_data')
    
#     def crop_data(self, roi, image=None):
#         """ Crop the data by removing the padding. """
#         if image is None:
#             image = self.target
#             btarget = True
#         else:
#             btarget = False
#         image = self.Padder.crop_data(image, roi)

#         roi_string = ''.join(str(x) + ',' for x in roi)

#         self.crop_action(
#             'crop_data',
#             "crop the data with the given region of interest",
#             "roi = [" + roi_string  + "]"
#             )
        
#         if btarget:
#             self.target = image
#         else:
#             return image

#     def pad_data_to_power_of_two(self):
#         """ Pad the data to a power of two. """
#         self.target, original_roi = self.Padder.pad_to_next_power_of_2(self.target)
#         self.pad_action(
#             'pad_data',
#             "Padded the data to a square image with dimension that are a power of two",
#             None,
#             original_roi)

#     def pad_reflective2d(self):
#         """ Pad the data with zeros. """
#         self.target, original_roi = self.Padder.pad_reflective2d(self.target)
#         self.pad_action(
#             'pad_data',
#             "Padded with reflective boundary conditions",
#             None,
#             original_roi)
    

#     def pad_action(self, action, description, parameters, reverse_parameters):
#         """ Defines the dictionary of the filter action and tracks the action. """
#         action = pd.DataFrame({
#             'action type': action,
#             'reverseable': True,
#             'reverse action': 'crop_data',
#             'description': description,
#             'parameters': parameters
#         }, index=[0])
#         self.track_actions(action, reverse_parameters=reverse_parameters)
#         return

#     def crop_action(self, action, description, parameters):
#         """ Defines the dictionary of the filter action and tracks the action. """
#         action = pd.DataFrame({
#             'action type': action,
#             'reverseable': False,
#             'reverse action': 'crop_data',
#             'description': description,
#             'parameters': parameters
#         }, index=[0])
#         self.track_actions(action)
#         return


#     # ------------------------- Data Transformation Functions ------------------------- #

#     def set_transformer(self, transform_class, **kwargs):
#         """ Set the transformation class. """
#         self.Transformer = transform_class(self, **kwargs)

#     def transform_data(self):
#         """ Transform the data. """
#         self.transformed_target = self.Transformer.transform()


#     # ------------------------- Data Modification Tracking Functions ------------------------- #
#     def track_actions(self, action, reverse_parameters=None):
#         """ Track the actions performed on the data. """
#         self.actions = pd.concat([self.actions, action], ignore_index=True)
#         self.data_modifications = self.data_modifications + (self.target,)
#         self.reverse_parameters = self.reverse_parameters + (reverse_parameters,)

#     # ------------------------- Data Plotting Functions ------------------------- #


#     def plot_target(self):
#         """ Plot the target image """
#         # REPLACE THIS WITH THE ALREADY EXISTING PLOT FUNCTION IN MISC


#         if len(self.target.size()) > 2:
#             range = torch.max(torch.abs(self.target*1e3))
#             size = self.target.size()
#             extent = [0, size[1]*self.dx, 0, size[0]*self.dy]
#             plt.figure()
#             plt.imshow(self.target*1e3, cmap="bwr", extent = extent, vmin=-range, vmax=range)
#             cb = plt.colorbar()
#             plt.xlabel("x (um)")
#             plt.ylabel("y (um)")
#             cb.set_label("Magnetic Field (mT)")
#             plt.show()

#         else:

#             range = torch.max(torch.abs(self.target*1e3))
#             size = self.target.size()
#             extent = [0, size[1]*self.dx, 0, size[0]*self.dy]
#             plt.figure()
#             plt.imshow(self.target*1e3, cmap="bwr", extent = extent, vmin=-range, vmax=range)
#             cb = plt.colorbar()
#             plt.xlabel("x (um)")
#             plt.ylabel("y (um)")
#             cb.set_label("Magnetic Field (mT)")
#             plt.show()


class MagneticFieldDataMixin:
    """Mixin class to add common magnetic field data functionality.
    
    + Allows to access field_data and point_data keys as attributes and converts
    them to torch.Tensor automatically."""            
    
    def __getattr__(self, name):
        """Dynamically expose field_data and point_data keys as attributes."""
        # Avoid recursion by using `object.__getattribute__`
        try:
            # Check if the attribute is in `field_data`
            field_data = object.__getattribute__(self, 'field_data')
            if name in field_data:
                return torch.tensor(field_data[name])
            
            # Check if the attribute is in `point_data`
            point_data = object.__getattribute__(self, 'point_data')
            if name in point_data:
                return torch.tensor(point_data[name])
            
            return object.__getattribute__(self, name)

        except AttributeError as e:
            raise e
        
    def map(self, func: callable, name):
        """Map a function over all points in the ImageData, assign the 
        result of the computation to a new `point_data` array."""
        self.point_data[name] = func(torch.tensor(self.points, dtype=torch.float32)).detach().numpy()
        return self

            
class DataBlock(pv.MultiBlock):
    """Data structure for handling multiple data sets, i.e. that pertain 
    to different regions in space."""
    
    def append(self, dataset, name=None):
        """Append a dataset to the block, same as in parent, but
        enforce unique names."""
        if name is None:
            name = f"block_{len(self)}"
        elif name in self.keys():
            raise ValueError(f"A block with the name '{name}' already exists.")
        return super().append(dataset, name)

    def __repr__(self):
        return self[0:len(self)].__repr__()
    
    
class MagneticFieldUnstructuredGrid(MagneticFieldDataMixin, pv.PolyData):
    """Class for handling magnetic field data on an unstructured grid.
    Used with experimental data (i.e. measurements) where coordinates of the
    points are given explicitly."""
    
    def __init__(self, *args, **kwargs):
        # Load data from file
        super().__init__(*args, **kwargs)
        
    def from_file(self, path_to_file):
        datadict = self.load_data(path_to_file)
        B = datadict["B"]  # Store B field values
        points = datadict["points"]  # Original measurement points
        
        self.__init__()
        
        self.points = points  # Add points to the grid
        self.point_data["B"] = np.array(B).T  # Add field vectors 
        self.field_data["dx"] = datadict["dx"]
        self.field_data["dy"] = datadict["dy"]
        
        return self

    def load_data(self, path):
        """Load data from file and organize into points and fields."""
        data = np.loadtxt(path, delimiter=",")

        # Extract coordinates and magnetic field components
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        points = np.column_stack((x_coords, y_coords, np.zeros_like(x_coords)))  # 2D points in 3D space

        # Bx, By, Bz values with transformations as needed
        Bx = data[:, 2]
        By = data[:, 4]
        Bz = -data[:, 3]

        # Unique x and y coordinates to compute grid spacing (for later use)
        x_positions = np.unique(x_coords)
        y_positions = np.unique(y_coords)
        dx_avg = np.mean(np.diff(x_positions))
        dy_avg = np.mean(np.diff(y_positions))

        # Return all relevant data as a dictionary
        return {
            "points": points,
            "B": [Bx, By, Bz],
            "dx": dx_avg,
            "dy": dy_avg,
        }

    def resample_to_regular_grid(self):
        """Resample data onto a regular grid."""
        # Define bounds and spacing for a regular grid
        grid = MagneticFieldImageData()
        
        bounds = self.bounds
        nx = int((bounds[1] - bounds[0]) / self.dx) + 1
        ny = int((bounds[3] - bounds[2]) / self.dy) + 1
        
        grid.dimensions = (nx, ny, 1)
        grid.spacing = (self.dx, self.dy, 1)
        grid.origin = (bounds[0], bounds[2], bounds[4])  # Set the origin
        
        # Resample UnstructuredGrid data onto this UniformGrid
        sampled_grid = grid.interpolate(self)
        
        return sampled_grid
    
    def get_random_pts_vals(self, n):
        """Sample random points from the UnstructuredGrid."""
        # Sample random indices
        n = min(n, self.n_points)
        indices = np.random.choice(self.n_points, n, replace=False)
        
        # Extract the points
        pts = self.points[indices]
        vals = self.B[indices]
        return pts, vals
    
    def extend_data(self, source, source_name=None, strategy="closest_constant", target_name=None):
        """Extend the data from `source` to `self` using a specific `strategy`."""
        if strategy == "closest_constant":
            # Find the closest point from the source and extend the data with the same value
            tree = scipy.spatial.KDTree(source.points)
            distances, indices = tree.query(self.points)
            
            if source_name is None and target_name is None:
                # Extend the data by copying the values from the closest points in the source
                for key in source.point_data.keys():
                    self.point_data[key] = source.point_data[key][indices]
            elif source_name is not None and target_name is None:
                # if source_name is given and target_name is not, extend the data
                # and copy the source_name
                self.point_data[source_name] = source.point_data[source_name][indices]
            elif source_name is not None and target_name is not None:
                # if both source_name and target_name are given, use target_name 
                # when extending the data
                self.point_data[target_name] = source.point_data[source_name][indices]
            else:
                raise ValueError("""Invalid `source_name` ({} of type {})\
                    and `target_name` ({} of type {}) combination.""".format(
                    source_name, type(source_name), target_name, type(target_name)
                    ))
                
        return self
    

class MagneticFieldImageData(MagneticFieldDataMixin, pv.ImageData):
    """Class for handling magnetic field data on a regular grid."""
    
    def __sub__(self, other, threshold_distance=1e-2):
        """Subtract points of one grid from another. Sub"""
        if isinstance(other, MagneticFieldImageData):
            # Get points from both datasets
            points1 = np.array(self.points)
            points2 = np.array(other.points)

            # Build a KDTree for fast nearest-neighbor search
            tree = scipy.spatial.KDTree(points2)

            # Query distances to the nearest neighbor in points2 for each point in points1
            distances, _ = tree.query(points1)

            # Create a mask for points in self that are NOT close to any points in other
            mask = (distances > threshold_distance)

            # Extract points from self that are not close to any points in other, 
            # that will be the difference
            filtered_points = points1[mask]

            # Create a new dataset with these points
            # filtered_data = pv.UnstructuredGrid()
            # filtered_data.points = filtered_points
            filtered_data = MagneticFieldUnstructuredGrid(pv.PolyData(filtered_points))

            # If you want to keep point data (e.g., scalars), extract those too
            filtered_point_data = {key: value[mask] for key, value in self.point_data.items()}
            for key, value in filtered_point_data.items():
                filtered_data.point_data[key] = value

            return filtered_data
        else:
            raise TypeError("Unsupported type {} to subtract from {}.".format(
                type(other), type(self)))
    
    def expand_bounds_3d(self, factor=1, name=None):
        """Expand the bounds of the grid in x, y, amd z directions by a factor."""
        if isinstance(factor, (list, tuple)):
            if all((isinstance(f, (list, tuple)) and len(f) == 2) for f in factor) and len(factor == 3):
                # treat this case as a list of factors [(a_factor, b_factor), ..., (e, f)]
                a, b = factor[0]
                c, d = factor[1]
                e, f = factor[2]
            elif all(isinstance(f, (float, int)) for f in factor) and len(factor) == 3:
                # treat factors as symmetric of both sides of a dimension
                # (2, 2, 2) will enlarge the field of view by 2 in each direction
                a = b = factor[0]
                c = d = factor[1]
                e = f = factor[2]
            else:
                TypeError("Unsupported factor type {} in factor specification. If factor is an iterable, \
                          it's elements should be either tuples of length 2, or numbers.")
        elif isinstance(factor, (int, float)):
            a = b = c = d = e = f = factor
            
        dims = self.dimensions
        spacing = self.spacing
        origin = self.origin
        extent = self.extent
        
        expanded_dims = (dims[0] * (1 + a + b), dims[1] * (1 + c + d), dims[2] * (1 + e + f))
        expanded_dims = (int(e_d) for e_d in expanded_dims)  # convert all to integers, how? idk
        # now we calculate how much we need to shift the origin by to get the expansion
        # in each direction by the appropriate factor
        expanded_origin = (
            origin[0] - spacing[0] * dims[0] * a, 
            origin[1] - spacing[1] * dims[1] * c, 
            origin[2] - spacing[2] * dims[2] * e
            )
        
        expanded_grid = MagneticFieldImageData()
        expanded_grid.dimensions = expanded_dims
        expanded_grid.origin = expanded_origin
        
        blocks = DataBlock()
        try:
            blocks.append(self)  # you need to give blocks a name, in particular
        except AttributeError as e:
            blocks.append(self)
        # in particular, each data object should have a name already
        blocks.append(expanded_grid)
        return blocks
    
    def expand_bounds_2d(self, factor=1, name=None) -> DataBlock:
        """Expand the bounds of the grid in x and y directions by a factor."""
        if isinstance(factor, (list, tuple)):
            if all([(isinstance(f, (list, tuple)) and len(f) == 2 for f in factor)]) and len(factor) == 2:
                # treat this case as a list of factors [(a_factor, b_factor), (c, d)]
                a, b = factor[0]
                c, d = factor[1]
            elif all(isinstance(f, (float, int)) for f in factor) and len(factor) == 2:
                # treat factors as symmetric of both sides of a dimension
                # (2, 2) will enlarge the field of view by 2 in each direction
                a = b = factor[0]
                c = d = factor[1]
            else:
                raise TypeError("Unsupported factor type {} in factor specification. If factor is an iterable,\n \
                          it's elements should be either tuples of length 2, or numbers, and it length should\n \
                              be 2 for `expand_bounds_2d`.".format(factor))
        elif isinstance(factor, (int, float)):
            a = b = c = d = factor
        else:
            raise TypeError("Unsupported factor {} in factor specification. If factor is an iterable,\n \
                      it's elements should be either tuples of length 2, or numbers, and its length should\n \
                      be 2 for `expand_bounds_2d`.".format(factor))
            
        dims = self.dimensions
        spacing = self.spacing
        origin = self.origin
        extent = self.extent
        
        expanded_dims = (dims[0] * (1 + a + b), dims[1] * (1 + c + d), dims[2])
        expanded_dims = (int(e_d) for e_d in expanded_dims)  # convert all to integers, how? idk
        # now we calculate how much we need to shift the origin by to get the expansion
        # in each direction by the appropriate factor
        expanded_origin = (
            origin[0] - spacing[0] * dims[0] * a, 
            origin[1] - spacing[1] * dims[1] * c, 
            origin[2]
            )
        
        expanded_grid = MagneticFieldImageData()
        expanded_grid.dimensions = expanded_dims
        expanded_grid.origin = expanded_origin
        expanded_grid.spacing = spacing
        
        return expanded_grid

    
    def get_as_grid(self, point_data_name):
        """Return point data reshaped into a grid matching the mesh structure.
        Only available for ImageData since it has a regular grid structure.
        """
        data = self.__getattr__(point_data_name)
        # We need to cast data to a (nx, ny, nz) tensor. ImageData uses
        # Fortran ordering, meaning that pts and their vals in point_data are
        # stored in the following way: val[k] = val[i, j] and k = i + j*nx, so
        # first index changes the fastest. If we want to reshape to (nx, ny, nz), 
        # we need to view the tensor as (nz, ny, nx) and then permute the axes.
        # Viewing guarantees that the data is contiguous in memory and the layout 
        # is as expected.
        nx, ny, nz = self.dimensions
        shape = (nz, ny, nx, 3) if nz > 1 else (ny, nx, 3)
        if nz > 1:
            return data.reshape(*shape).permute(2, 1, 0, 3)
        elif nz == 1:
            return data.reshape(*shape).permute(2, 1, 0)
        else:
            raise ValueError("Invalid dimensions for the grid.")
    
    def interpolate(self, *args, **kwargs):
        """Call the interpolate method of the superclass but ensure that the
        output is a MagneticFieldImageData object."""
        interpolated = super().interpolate(*args, **kwargs)
        return MagneticFieldImageData(interpolated)
    
    def plot_n_components(self, point_data_name):
        """Use `plot_n_components` function to extract a 2d image of a 
        point data with name `point_data_name`."""
        grid_data = self.get_as_grid(point_data_name=point_data_name)
        return plot_n_components(data=grid_data)
    
    
    # def plot(self, only_points=False, point_data_name=None):
    #     """Plot the point data with name `point_data_name`."""
    #     if only_points:
    #         plotter = pv.Plotter()
    #         # Add points to the plot as spheres
    #         plotter.add_mesh(self.points, point_size=5, render_points_as_spheres=True)
            
    #         plotter.add_bounding_box(color="black")
    #         plotter.show_bounds(
    #             grid="front",           # Display the grid on the front face
    #             location="outer",       # Place the ruler outside the bounds
    #             ticks="both",           # Display both major and minor ticks
    #             color="black",          # Ruler color
    #             font_size=10,           # Font size for labels
    #         )
    #         plotter.view_xy()  # Show the XY plane
    #         return plotter.show()