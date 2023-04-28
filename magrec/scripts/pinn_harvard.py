from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision

# import module root if this file called as a script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from magrec.misc.plot import plot_n_components
from magrec.misc.load import load_matlab_data
from magrec.prop.Propagator import AxisProjectionPropagator, CurrentPropagator2d
from magrec import __datapath__

mat = load_matlab_data(__datapath__ / "experimental" / "Harvard" / "XY_Bnv_Bxyz_Jxy_SChenJJ_data2.mat")

Bx = mat['BX_map'].T
By = mat['BY_map'].T
Bz = mat['BZ_map'].T
Bnv = mat['Bnv_map'].T

B = np.stack((Bx, By, Bz, Bnv), axis=0) * 1e6  # Convert to μT
B = torch.from_numpy(B).float()
W, H = B.shape[-2:]

crop = torchvision.transforms.CenterCrop([W, H])

dx = 0.0169  # μm
dy = 0.0292

height = 0.015          # μm
layer_thickness = 0.030 # μm

theta = 54.7            # degrees
# theta = -35.0  # predicted by David
phi = 30.0              # degrees

# Projection object to compute the projection of the vector onto the defined axis
proj = AxisProjectionPropagator(theta=theta, phi=phi)
prop = CurrentPropagator2d(source_shape=(2*W, 2*H),
                           dx=dx,
                           dy=dy,
                           height=height,
                           layer_thickness=layer_thickness)

# PDE
# div(J) = 0

# compute a mesh of points with shape (2, H, W) in torch
xs = torch.linspace(0, 1, 2*W, requires_grad=True)
ys = torch.linspace(0, 1, 2*H, requires_grad=True)
mesh = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=0)

def model(J, mesh):
    """Prepare synthetic model that takes J and gives some result."""
    B = prop.B_from_J(J)
    B_NV = proj(B)
    return crop(B)

# Prepare a network to train

# Number of channels (dimensions) in the input image
n_channels_in = 2
# Number of channels (dimensions) in the output image
n_channels_out = 2  # same dimensions as J

# net = torch.nn.Sequential(
#     torch.nn.Conv2d(n_channels_in, 20, 3, 1, 1),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 20, 3, 1, 1),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, 20, 3, 1, 1),
#     torch.nn.ReLU(),
#     torch.nn.Conv2d(20, n_channels_out, 3, 1, 1)
#     )

# create a fully connected network 
net = torch.nn.Sequential(
    torch.nn.Linear(2*W*2*H, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 2*W*H),
)

mse = torch.nn.MSELoss()

def div(J, mesh):
    """Functions to calculate divergence of J wrt mesh."""
    # J is a tensor of shape (2, H, W)
    # mesh is a tensor of shape (2, H, W)
    # we want to compute the divergence of J at each point in mesh
    # to do this we take the partial derivatives of J with respect to x and y
    # and then sum them together
    dJ = torch.autograd.grad(J, mesh, grad_outputs=torch.ones_like(J),
        create_graph=True, retain_graph=True)
    return dJ[0][0, :, :] + dJ[0][1, :, :]

def loss_fn(J):
    """Calculate loss. `alpha` is a hyperparameter."""
    div_J = div(J, mesh)
    div_loss = mse(div_J, torch.zeros_like(div_J))
    field_loss = mse(model(J, mesh)[[-1], :, :], B[[-1], :, :])
    return field_loss, div_loss

def main(field_weight=1.0, div_weight=1.0):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Show the target field
    # plot_n_components(B.detach().numpy(), title=r"Target $B$", show=True)
    # plt.show(block=True)
    
    # Print a table header for printing the loss
    # format epoch_n, loss, div_loss, field_loss and align them to the right
    print("{:10} {:10} {:10} {:10}".format("Epoch", "Total", "Div", "Field"))
    for i in range(1000):
        optimizer.zero_grad()
        J = net(mesh.reshape(1, 2*H*2*W)).reshape(2, H, W)
        field_loss, div_loss = loss_fn(J)
        field_loss = field_loss * field_weight
        div_loss = div_loss * div_weight
        loss = field_loss + div_loss
        print("{:<10} {:10.4e} {:10.4e} {:10.4e}".format(i, loss.item(), div_loss.item(), field_loss.item()))
        loss.backward()
        optimizer.step()
        
    # Show the results
    fig = plot_n_components(model(J, mesh).detach().numpy(), title=r"Model predicted $B$", show=True)
    plt.show(block=True)


if __name__ == "__main__":
    main(field_weight=1e0, div_weight=1e6)