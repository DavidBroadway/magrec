import torch
from magrec.misc.load import load_matlab_data
from magrec import __datapath__
import numpy as np
from magrec.misc.plot import plot_n_components
from magrec.nn.arch import BnCNN, GioCNN, GioUNet

from magrec.prop.Fourier import FourierTransform2d
from magrec.prop.Propagator import AxisProjectionPropagator

def load_B_NV():
    # mat = load_matlab_data(__datapath__ / "ExperimentalData" / "Harvard" / "XY_Bnv_Bxyz_Jxy_SChenJJ_data2.mat")
    mat = load_matlab_data(__datapath__ / "experimental" / "Harvard" / "XY_Bnv_Bxyz_Jxy_SChenJJ_data2.mat")

    Bx = mat['BX_map'].T
    By = mat['BY_map'].T
    Bz = mat['BZ_map'].T
    Bnv = mat['Bnv_map'].T

    B = np.stack((Bx, By, Bz, Bnv), axis=0) * 1e6  # Convert to μT
    B = torch.tensor(B, dtype=torch.float32)
    return B[[-1], ...]

# Notes:
# Upon thinking about `extend()` again, the extension does not guarantee 
# the true current density is the one obtained from B_z.

##
# Data constants

dx = 0.0169  # μm
dy = 0.0292

# theta = 35
# phi = 30

theta = -54.7  # degrees
phi = -150  # degrees

B_NV = load_B_NV()

# TODO: Rewrite so as to not use FourierTransform2d, instead
# have a better understanding of Fourier transforms and use parameters W, H (width and length)
shape = B_NV.shape[-2:]

ft = FourierTransform2d(shape, dx, dy, real_signal=False)
k_x, k_y = ft.kx_vector, ft.ky_vector
k = ft.k_matrix

##
# Load and initialize data and objects
B_NV = load_B_NV()

net = GioCNN(n_channels_in=1, n_channels_out=1)
grid = torch.meshgrid(torch.linspace(0, 1, 20), torch.linspace(0, 1, 20))
mse = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

def test_module():
    B_z_extended = extend(B_NV)
    B_x_extended, B_y_extended = get_B_x_y(B_z_extended)
    B_NV_extended = get_B_NV(B_x_extended, B_y_extended, B_z_extended, theta, phi)
    plot_n_components(B_NV_extended - extend(B_NV), show=True, cmap="bwr")
    plot_n_components((B_x_extended, B_y_extended, B_z_extended), show=True, cmap="RdBu_r")
    plot_n_components(get_field_of_view(B_NV_extended), show=True, cmap="RdBu_r")

# Define training step
def train_step():
    B_z = net(B_NV.unsqueeze(0)).squeeze(0)

    B_z_extended = extend(B_z)

    # depends on k = 0 components
    B_x_extended, B_y_extended = get_B_x_y(B_z_extended)

    B_NV_extended = get_B_NV(B_x_extended, B_y_extended, B_z_extended, theta, phi)

    B_NV_estimate = get_field_of_view(B_NV_extended)

    loss = mse(B_NV_estimate, B_NV)
    loss.backward()
    optimizer.step()
    return loss, B_x_extended, B_y_extended, B_z_extended, B_NV_extended
    
def train(n_epochs=100, print_every=10):
    try:
        for epoch in range(n_epochs):
            loss, B_x_extended, B_y_extended, B_z_extended, B_NV_extended = train_step()
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: loss = {loss.item()}")
    except KeyboardInterrupt:
        pass
        
    return net, B_x_extended, B_y_extended, B_z_extended, B_NV_extended
    
    
def extend(B_z):
    """Use replication and mirror rule to obtain 
    2x2 image from a 1x1 lower left piece of a tensor."""
    B_r = torch.flip(B_z, dims=(-2,))
    B_c = torch.flip(B_z, dims=(-1,))
    B_rc = torch.flip(B_z, dims=(-1, -2))
    out = torch.cat((torch.cat((B_z, B_r), dim=-2),
                     torch.cat((B_c, B_rc), dim=-2)), dim=-1)
    return out

def get_B_x_y(B_z, b_x_0=0.0, b_y_0=0.0):
    """Use connection between the components in the Fourier
    space to calculate B_x and B_y from the B_z component."""
    ft = FourierTransform2d(B_z.shape, dx, dy, real_signal=False)
    k_x, k_y = ft.kx_vector, ft.ky_vector
    k = ft.k_matrix
    
    b_z = ft.forward(B_z, dim=(-2, -1))
    
    M_x = - 1j * k_x[:, None] / k
    M_y = - 1j * k_y[None, :] / k
    
    M_x[0, 0] = 1
    M_y[0, 0] = 1
    
    b_y = M_x * b_z
    b_x = M_y * b_z
    
    b_y[0, 0] = b_y_0
    b_x[0, 0] = b_x_0
    
    B_y = ft.backward(b_y, dim=(-2, -1))
    B_x = ft.backward(b_x, dim=(-2, -1))
    return B_y.real, B_x.real

def get_B_NV(B_x, B_y, B_z, theta, phi):
    """Project B_x, B_y, B_z onto the NV-axis to obtain B_NV."""
    proj = AxisProjectionPropagator(theta=theta, phi=phi)
    B_NV = proj.project(torch.cat((B_x, B_y, B_z), 0)).unsqueeze(0)
    return B_NV

def get_field_of_view(B_NV_extended):
    """Extract lower left piece from the tensor that correspond 
    to the original field of view."""
    return B_NV_extended[..., :shape[0], :shape[1]]