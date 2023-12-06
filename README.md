#  Magnetic field reconstruction

Magrec is a package for the reconstruction of the source quantity from the measured magnetic field. 
The source quantity can be the magnetisation or the current density in 2 dimensions.
The task is completed by a untrained physics informed neural networks that learn on the fly on each new single image.

## 1.Installation and Requirements
### 1.1. Installation

The software cloned with:
```
git@github.com:DavidBroadway/magrec.git
```
or 
```
https://github.com/DavidBroadway/magrec.git
```

### 1.1. Required libraries
The system requires the following:

	
- [Python](https://www.python.org/downloads/): Python 3 by default (works for python 2, but no future guarantees).
- [Pytorch](https://www.pytorch.org/): The Deep Learning library for back end.
- [matplotlib](http://matplotlib.org/): visualization library
- [matplotlib-scalebar](https://pypi.org/project/matplotlib-scalebar/): Provides a new artist for matplotlib to display a scale bar
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
- [tqdm](http://www.tqdm.org/) : progress bar


#### Install using Conda
Make a new conda environment and install python:
```bash
conda create -n magrec  
conda activate magrec
conda install python=3.11
```

We have some issues with install pytorch via pip so use conda to install this first. 
```bash
conda install pytorch
```

Then navigate to the magrec folder and use pip to install
```bash
pip install -e .
```

#  2. Usage

Simple examples of use can be found in the example notebooks test notebooks (E.g. Test Magnetisation Reconstruction)


## 2.1. Data format

To faciliate passing and manipulation of the image data we use a data class. This takes a series of required arguments. An example of initalising the data class is given below. 

```
from magrec.misc.data import Data
import numpy as np

# Make some fake data
Bsensor = np.random.rand((256,256))

# Define the pixel dimensions. 
dx = 0.1 # (um)
dy = dx # (um)

# Define the properties related to the sensor
sensor_theta = 54 # (deg)
sensor_phi = 45 # (deg)
height = 30e-3 # (um)
layer_thickness = 0 # set this to zero as it can cause issues and doesn't change results significantly

# Initialise the data class
dataset = Data()
# load the data
dataset.load_data(Bsensor, dx, dy, height, sensor_theta, sensor_phi, layer_thickness)
```

## 2.2 Data manipulation

The principle behind the software is that every modification of the data is tracked. We refers to anything that interacts with the data as an action and then the information and order of the actions is tracked. The order of different actions is important for the reconstruction process so this is an important tool for debugging with difficult to reconstruct data. 

Below is an example of applying actions to a dataset in the data class. 

```
# Initialise the data class
dataset = Data()
# load the data
dataset.load_data(Bsensor, dx, dy, height, sensor_theta, sensor_phi, layer_thickness)

# Add spatial filters and perform other actions on the dataset 
dataset.add_hanning_filter(height)
dataset.add_short_wavelength_filter(height)
dataset.remove_DC_background()
dataset.crop_data([0,256,0,256])
dataset.pad_data_to_power_of_two()

# Plot the current dataset at any stage by calling dataset.plot_target()
# The Bsensor data is refered to as the target because it acts as the fitting 
# image in the neural network based reconstructions. 

dataset.plot_target()

# Display all of the actions that have been performed on the data
dataset.actions
```
![actions](images/actions.png)


## 2.3 Transformations
Various transformations can be performed without the add of the neural network. All transformations are contains in magrec.transformation, including those used by the neural network reconstruction. To perform the transformation using a Fourier space method you can use the following code. This is a suitable approach when reconstructing Bsensor -> Bxyz, B -> Mz, and B -> Jxy. 

```
# Example of transforming a magnetic field with an arbitray sensor angle in the cartesian components.

from magrec.transformation.MagneticFields import MagneticFields   
from magrec.misc.plot import plot_n_components

# Initialise the data class
dataset = Data()
# load the data
dataset.load_data(Bsensor, dx, dy, height, sensor_theta, sensor_phi, layer_thickness)

# Set the transformation to be performed on the data
dataset.set_transformer(MagneticFields)
# Perform the transformation
dataset.transform_data()

# Plot the results
plot_n_components(dataset.transformed_target, symmetric=True, labels=[r"$B_x$", r"$B_y$", r"$B_z$"], cmap="bwr")
```

## 2.3 Models 
For reconstruction we define a model that is used to transform from the neural network output back into the target magnetic field. These models go beyond a the transformation itself by containing addation restrictions like mask. 

Example of the model for a uniform magnetisation direction.

```
from magrec.models.UniformMagnetisation import UniformMagnetisation


dataset = Data()
dataset.load_data(Bz_data, 
                        dx = dx, 
                        dy = dy, 
                        height = height, 
                        theta = 0, 
                        phi = 0, 
                        layer_thickness = 0)

# Define the model of the source that will be reconstructed
Model = UniformMagnetisation(dataset, 
                             loss_type = "MSE", 
                             m_theta = 0, 
                             m_phi = 0,
                             scaling_factor = 1e6)
```

Here the model takes a series of important parameters.

dataset: is the data class that contains the target magnetic field

loss_type: is an options to switch from different loss function definitions. The default is type is mean square error "MSE" 

scaling_factor: is a number to multiple the magnetic field by for the fitting. In most cases the magnetic field is small which can lead to poor convergence of the NN. We multiple by this factor to improve the fit convergence. The default of 1e6 is good starting position. If the network doesn't converge this parameter can be optimised. 

source_weight and loss_weight: Masks for calculating the source and loss functions. This is discussed later. 


### 2.4.1 Masks and wieghts
The neural network approach allows for the inclusion of masks or weights. This method has only been tested with hard masks, it is in principle possible to use them a wieghting for the fitting process. These masks are passed to the model.

There are two types of masks. 
**source_weight** 
This is applied to the source image itself. Which means it can be used to restrict the region in which the source quanity can be reconstructed. This is a excellent tool for removing background values when the image is known to have zero source in the background. This is a distinct advantage over the Fourier method which often added a source value into the background which has no obvious solution for removal both raw subtraction and fitting of the background can lead to an offset in the true source value.

**loss_weight**
This is applied to the loss calculation and can be used as a traditional wieghting to minimise the allocation of source values to regions that are more error prone or background. This can be useful when using padding to get better Fourier transforms. In this case you don't want the NN to focus on matching the padding region. Including this mask can help to remove edge artifacts when using padding. 

**Defining a mask**
To help define quick and simple masks there is a vertical and horizontial mask function. This iterates through each line of the image from both edges and finds the first value that is larger than the threshold. The magnetic field tends to have the largest value at the edge of the material so this can be used to roughly detect the edges. 

Example of definng a mask. 
```
import magrec.image_processing.Masks as masks
# Deifne the source weight
threshold = 0.8e-4 # Threshold value in Tesla
# define a vertical based mask
source_weight = masks.mask_vert_dir(dataset.transformed_target[2,::], threshold, plot = True)
```
The mask can be plotted when requested
![mask](images/mask_example.png)


## 2.4.2 Spatial filters
Only used when using a fully connected neural network. This acts to introduce the measurement spatial resolution into the dataset, otherwise the FCNN will converge to an image that is too sharp to be a realisic result. 

Options and defualt values. 
```
spatial_filter: bool = False,
spatial_filter_type: str = "Gaussian",
spatial_filter_kernal_size: int = 3,
spatial_filter_width: float = 0.5
```


## 2.5 Neural Network types 
Different networks are available depending of the reconstruction task(magnetisation or current density). All of these reconstruction methods inhert from the generic_method parent class.

1) CNN : Convolutionnal neural network
init: takes the model and a leanring_rate

preparing the fit has options about the network and doesn't need to be called as the default values will be assumed in the init statement. 

If you want to play with the size of the network these are the parameters. 
```
n_channels_in=1 # The number of magnetic field components
n_channels_out=1 # the number of source componets (e.g. Mz)

# Kernal stride and padding as defined in the pytorch documentation
kernel=5 # Size of the convolutional kernal
stride=2 # step in which to do the convolution
padding=2 # whether to pad input image for convolution and by how much
```

2) FCNN : Fully connected neural network
The fully connected network doesn't have any convolutions but behaves in a similar fashion to the CNN method. But because there is no convolution the FCNN will tend to produce an artifically sharp image. To overcome this you can introduce a blurring function to the model. This will blur the output of the network with the desired function and width to match the spatial resolution of the measurement. 
For example
```
# Define the model of the source that will be reconstructed
Model = UniformMagnetisation(NN_recon_data, 
                             loss_type = "MSE", 
                             m_theta = 0, 
                             m_phi = 0,
                             scaling_factor = 1e6,
                             source_weight = source_weight,
                             loss_weight = None, 
                             spatial_filter = True, 
                             spatial_filter_type = "Lorentzian",
                             spatial_filter_width = [height, height])
``` 
Here a lorentzian spatial filter has been applied to the output (gaussian also available) that has a width of the hieght of the sensor. While this can match the spatial resolution of the measurement it can also result in local bunching in the source values. This can be seen as oscilations in the value that are not physical. As a concequence it can be tricker to get this method to work, however, the benefit is that because the model is more accurate to the physical measurement is can produce a better approximation of the source quantity by remove low frequency oscillations that can be introduced in the convolution and fouier methods. See the comparison of methods. 

### 2.6. Fitting process
To perform the fitting we define the neural network and pass it the model.
```
# Define the fitting method and pass it the model
FittingMethod = CNN(Model, learning_rate = 0.1)
```
We optionally pass the learning rate for the network. This modifies how large of a step the neural network can take between each epoch. See the pytorch documentation for more details, but generally the default value is fine. 


The fitting is performed by calling
```
# Perform the fit using the NN
FittingMethod.fit(n_epochs=200)
```
where n_epochs is the number of epochs that the neural network will go through. 

```
# Plot the exvolution of the loss function
FittingMethod.plot_loss()
```
The error function should evolve until it flattens off. At this stage the NN is likely optimising on noise. 
![error_evol](images/error_evolution.png)

Like most fitting algorithms you may need to run the network several times and play around with the number of epochs and learning rate to get a better fit. 


### 2.7 Comparison of techniques
Here is an example of using different methods to reconstruct the Mz component of a CrI3 flake from the publication 10.1126/science.aav6926. No additional background subtractions are performed to give a straight comparison of the methods. 

![method_comp](images/method_comp.png)



## 3. Admin

#### 3.1. Citation
If you are publishing scientific results, mentioning this package in your methods description is the least you can do as good scientific practice. 

You should cite our paper : 
Dubois, A. E. E., Broadway, D. A., Stark, A., Tschudin, M. A., Healey, A. J., Huber, S. D., Tetienne, J.-P., Greplova, E., & Maletinsky, P. (2022). Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources. Physical Review Applied, 18(6), 064076. <https://doi.org/10.1103/PhysRevApplied.18.064076>

### 3.2. Documentation

Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements.

Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources.

### 3.3 Collaboration

If you wanna collaborate or have questions please contact me at broadwayphysics@gmail.com or my relavant univeristy account currently david.broadway@rmit.edu.au

### 3.4. License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


### 3.5. Contributors

Adrien E. E. Dubois (adr.dubois@gmail.com)
Mykhailo Flaks (mykhailo.flaks@unibas.ch)
David A. Broadway (davidaaron.broadway@unibas.ch)


[1] Broadway, D. A., Lillie, S. E., Scholten, S. C., Rohner, D., Dontschuk, N., Maletinsky, P., Tetienne, J.-P., & Hollenberg, L. C. L. (2020). Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements. Physical Review Applied, 14(2), 024076. <https://doi.org/10.1103/PhysRevApplied.14.024076>

[2] Dubois, A. E. E., Broadway, D. A., Stark, A., Tschudin, M. A., Healey, A. J., Huber, S. D., Tetienne, J.-P., Greplova, E., & Maletinsky, P. (2022). Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources. Physical Review Applied, 18(6), 064076. <https://doi.org/10.1103/PhysRevApplied.18.064076>



[//]: # (reference links)

   [paper1]: <https://link.aps.org/doi/10.1103/PhysRevApplied.14.024076>[paper1]
