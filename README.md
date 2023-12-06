#  Magnetic field reconstruction

Magrec is a package for the reconstruction of the source quantity from the measured magnetic field. 
The source quantity can be the magnetisation or the current density in 2 dimensions.
The task is completed by a untrained physics informed neural networks that learn on the fly on each new single image.


### Table of Contents
* [1. Installation and Requirements](#1-Installation and Requirements)
  * [1.1. Required Libraries](#11-Required Libraries)
  * [1.2. Installation](#12-installation)
  * [1.3. GPU Processing](#13-gpu-processing)
* [2. Usage](#2-Usage)
  * [2.1. Data format](#21-Data format)
  * [2.2. Measurements parameters](#22- Measurements parameters)
  * [2.3. Networks](#23-Networks)
  * [2.4. Training parameters](#13-raining parameters)
* [3. Admin](#3-Admin)
  * [3.1 Citation](#31-Documentation)
  * [3.2. Documentation](#32-Documentation)
  * [3.3. Collaboration](#33-Contributors)
  * [3.4. License](#34-License)
  * [3.5. Contributors](#35-Contributors)



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



##  2. Usage

Simple examples of use can be found in the example notebooks test notebooks (E.g. Test Magnetisation Reconstruction)


### 2.1. Data format

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

### 2.2 Data manipulation

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

![Example of actions](images/actions.png)


### 2.3 Networks 
Different networks are available depending of the reconstruction task(magnetisation or current density)

```
1) generator_CNN : Convolutionnal network for magnetization

optional arguments:
  --Size            size of the network (1,2,3)
  --ImageSize		size of the image
  --kernel  		size of the kernel
  --stride 			size of the stride


2)generator_MLP : Fully connected network for magnetization

optional arguments:
  --Size            size of the image


3)generator_CNN_J : Fully connected for current density

optional arguments:
  --Size            size of the network (1,2,3)
  --ImageSize		size of the image
  --kernel  		size of the kernel
  --stride 			size of the stride
```


### 2.4. Training parameters
the training can be controlled with the following options :
```
	--['mlp']=True for fully connected network
	--['LossFunction']= L1 or L2 
	--['Magnetization']= True for a given magnetization to compare with
	--['IntegerOnly']= True to output only integer values 
	--['PositiveMagnetisationOnly']= True to output only postive values
	--['PrintLossValue']= True to print loss values
	--['Epochs']= Number of epochs
```

## 3. Admin

#### 3.1. Citation
If you are publishing scientific results, mentioning this package in your methods description is the least you can do as good scientific practice. 

You should cite our paper : 
Dubois, A. E. E., Broadway, D. A., Stark, A., Tschudin, M. A., Healey, A. J., Huber, S. D., Tetienne, J.-P., Greplova, E., & Maletinsky, P. (2022). Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources. Physical Review Applied, 18(6), 064076. <https://doi.org/10.1103/PhysRevApplied.18.064076>

### 3.2. Documentation

Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements.

Untrained Physically Informed Neural Network for Image Reconstruction of Magnetic Field Sources.

### 3.3 Collaboration

If you wanna collaborate or have questions please contact one of the contributors.

For AI related questions : Adrien Dubois (adr.dubois@gmail.com)

For physics related questions : David Broadway (davidaaron.broadway@unibas.ch)

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
