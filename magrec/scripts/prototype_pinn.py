import torch

# PDE
# div(J) = 0

# compute a mesh of points with shape (2, H, W) in torch
xs = torch.linspace(0, 1, 20, requires_grad=True)
ys = torch.linspace(0, 1, 20, requires_grad=True)
mesh = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=0)

# pretend this is a non-invertable function
# we would like to construct net(mesh) such that model(net(mesh)) = B
# model is identity, then net should learn `func`
def func(mesh: torch.Tensor) -> torch.Tensor:
    return torch.sum(mesh ** 2, dim=-3)

def model(J, mesh):
    """Prepare synthetic model that takes J and gives some result."""
    return torch.sum(J, dim=-3)

with torch.no_grad():
    B = func(mesh)

# Prepare a network to train

# Number of channels (dimensions) in the input image
n_channels_in = 2
# Number of channels (dimensions) in the output image
n_channels_out = 2  # same dimensions as J

net = torch.nn.Sequential(
    torch.nn.Conv2d(n_channels_in, 20, 3, 1, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(20, 20, 3, 1, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(20, 20, 3, 1, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(20, n_channels_out, 3, 1, 1)
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

def loss_fn(J, alpha=1.0):
    """Calculate loss. `alpha` is a hyperparameter."""
    div_J = div(J, mesh)
    term1 = mse(div_J, torch.zeros_like(div_J))
    term2 = mse(model(J, mesh), B)
    return term1 + alpha * term2

# Create a simple optimization loop
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    J = net(mesh)
    loss = loss_fn(J)
    print("Loss at step {}: {:.5f}".format(i, loss.item()))
    loss.backward()
    optimizer.step()
