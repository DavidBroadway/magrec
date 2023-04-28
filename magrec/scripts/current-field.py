import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import image

# Load the two input images
input1 = image.imread('input1.png')
input2 = image.imread('input2.png')

# Get the shape of the input images
shape1 = input1.shape
shape2 = input2.shape

# Create the XY planes
x1 = np.linspace(-1, 1, shape1[1])
y1 = np.linspace(-1, 1, shape1[0])
X1, Y1 = np.meshgrid(x1, y1)
Z1 = np.zeros((shape1[0], shape1[1]))

x2 = np.linspace(-1, 1, shape2[1])
y2 = np.linspace(-1, 1, shape2[0])
X2, Y2 = np.meshgrid(x2, y2)
Z2 = np.ones((shape2[0], shape2[1]))

# Create the figure and the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first XY plane with the distorted input1 image
ax.plot_surface(X1, Y1, Z1, facecolors=input1, alpha=0.8)

# Plot the second XY plane with the distorted input2 image
ax.plot_surface(X2, Y2, Z2, facecolors=input2, alpha=0.8)

# Set the plot limits and labels
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
