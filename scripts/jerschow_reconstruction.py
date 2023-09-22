# reconstruction of current with torchphysics
import deepxde as dde
import numpy as np
import pytorch_lightning as pl
import scipy.interpolate
import torch

import magrec
from magrec import __datapath__
from magrec.misc.plot import plot_n_components, plot_vector_field_2d
from magrec.misc.sampler import GridSampler, RectangleSampler, Sampler, StaticSampler
from magrec.nn.callbacks import ModuleCheckpoint, WeightSaveCallback
from magrec.nn.models import FourierFeatures2dCurrent
from magrec.nn.solver import OptimizerSetting, Solver
from magrec.prop.conditions import PINNCondition
from magrec.prop.Propagator import AxisProjectionPropagator, CurrentPropagator2d


def zero_region_sample_fn(n_points):
    """Function to get points from the two vertical region Os.

      ..code::

      Y
      ^
    3 │ ┌───────┐       ┌───────┐
      │ │       │       │       │
      │ │       │       │       │
    2 │ │       │       │       │
      │ │   I   │       │  II   │
      │ │       │       │       │
    1 │ │       │       │       │
      │ │       │       │       │
      │ │       │       │       │
    0 │ └───────┘       └───────┘
      └─────────────────────────────>
        0       1       2       3  X

    """
    region_n_points = n_points // 2  # ratios of the regions' areas are 3:1:1:3
    region1_points = RectangleSampler.sample_rectangular_region(
        region_n_points, origin=[0, 0], diagonal=[1, 3]
    )
    region2_points = RectangleSampler.sample_rectangular_region(
        region_n_points, origin=[2, 0], diagonal=[3, 3]
    )
    points = torch.cat((region1_points, region2_points), dim=0)
    return points

def get_B_on_grid_from_file(path):
    data = np.loadtxt(path, delimiter=",")
    
    # Get coordinates of the measurement points
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    # Find unique values of the coordinates
    x_positions = np.unique(x_coords)
    y_positions = np.unique(y_coords)

    # Compute average spacing between measurement points
    dx_avg = np.mean(np.diff(x_positions))
    dy_avg = np.mean(np.diff(y_positions))
    # Generate a regular grid of points, if measurements are on regular grid, x_grid and y_grid are the same as x_coords and y_coords, but reshaped
    grid_x, grid_y = np.mgrid[
        x_positions.min() : x_positions.max() + dx_avg : dx_avg,
        y_positions.min() : y_positions.max() + dy_avg : dy_avg,
    ]

    # QUESTION: Where do these x, y positions come from? A measurement?
    
    # OBSERVATION: Upon inspecting B_x, B_y, B_z values with
    # magrec.misc.plot.plot_n_components(data[:, 2:5])
    # I noticed that the coordinate system in the sense of this package
    # differs from the one in the dataset. In the package xy-plane is the
    # plane of the current distribution, and the measurement plane is above. 
    # 
    # From these considerations follows that there must be a permutation of the dataset
    # to the field structure expected in the reconstruction:
    # B_x → B_x, B_y → B_z, B_z → -B_y

    Bx = data[:, 2]
    By = data[:, 4]
    Bz = -data[:, 3]

    Bx = scipy.interpolate.griddata((x_coords, y_coords), Bx, (grid_x, grid_y), method='cubic')
    By = scipy.interpolate.griddata((x_coords, y_coords), By, (grid_x, grid_y), method='cubic')
    Bz = scipy.interpolate.griddata((x_coords, y_coords), Bz, (grid_x, grid_y), method='cubic')

    B = torch.tensor(np.array([Bx, By, Bz]), dtype=torch.float32)
    return B

field = get_B_on_grid_from_file(__datapath__ / "Jerschow" / "Sine_wire.txt")
background = get_B_on_grid_from_file(__datapath__ / "Jerschow" / "Sine_wire_blank.txt")

values = field - background

height = 12000  # μm
layer_thickness = 100  # μm
dx = 2000  # in μm
dy = 2000

W, H = values.shape[-2:]


def grid_sample_fn(n_points=None):
    """Function to sample in a regular grid of shape (nx_points, ny_points) from the entire region ROAs."""
    points = GridSampler.sample_grid(
        nx_points=3 * W, ny_points=3 * H, origin=[0, 0], diagonal=[3, 3]
    )
    return points


zero_sampler = Sampler(
    zero_region_sample_fn, batch_n_points=10**3, cached_n_points=10**6
)

grid_sampler = StaticSampler(grid_sample_fn)

prop = CurrentPropagator2d(
    source_shape=(3 * W, 3 * H),
    dx=dx,
    dy=dy,
    height=height,
    layer_thickness=layer_thickness,
    real_signal=False,
)


def zero_residual(J, x):
    """Residual function for the zero current reconstruction."""
    # error is the current itself, 0 is expected
    err = J
    return err


def data_residual(J, x):
    """Residual function for the data condition."""
    J_grid = GridSampler.pts_to_grid(J, 3 * W, 3 * H)
    values_hat = prop(J_grid)[..., W:-W, H:-H]
    err = values - values_hat
    return err


# define a model that maps from x ∊ R² to J ∊ R², div J = 0 by construction
module = FourierFeatures2dCurrent(ff_sigmas=[(1, 10), (0.01, 10), (0.5, 10)])

zero_cond = PINNCondition(
    module=module,
    sampler=zero_sampler,
    residual_fn=zero_residual,
    name="zero_cond",
    weight=1e5,  # add weight to balance out the data_cond 
)

data_cond = PINNCondition(
    module=module,
    sampler=grid_sampler,
    residual_fn=data_residual,
    name="data_cond",
)


class Solver(magrec.nn.solver.Solver):
    def on_train_start(self):
        super().on_train_start()
        target_figure = plot_n_components(
            values, labels=[r"B_{x}", r"B_{y}", r"B_{z}"], cmap="bwr"
        )
        self.logger.experiment.add_figure(tag="target", figure=target_figure)

    # This can be extracted to callbacks
    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        with torch.inference_mode(False):
            pts = GridSampler.sample_grid(3 * W, 3 * H, origin=[0, 0], diagonal=[3, 3])
            pts.requires_grad_(True)
            y_hat = module(pts).detach()
            y_hat_grid = GridSampler.pts_to_grid(y_hat, 3 * W, 3 * H)
            pts.requires_grad_(False)
            # Clear the gradients of be doomed
            # (by excessive memory consumption)
            # For some reason, dde does not clear the gradients, which is expected
            # for caching. It does so for training (somehow?), but not for validation.
            dde.grad.clear()

        curr_fig = plot_n_components(
            y_hat_grid,
            labels=[r"J_x", r"J_y"],
            cmap="bwr",
        )
        self.logger.experiment.add_figure(
            tag="val/current_distribution",
            figure=curr_fig,
            global_step=self.global_step,
        )

        flow_fig = plot_vector_field_2d(y_hat_grid, cmap="plasma")
        self.logger.experiment.add_figure(
            tag="val/current_flow", figure=flow_fig, global_step=self.global_step
        )

        values_hat = prop(y_hat_grid).real
        mag_fig = plot_n_components(
            values_hat,
            labels=[r"B_x", r"B_y", r"B_z", r"B_{NV}"],
            cmap="bwr",
        )
        self.logger.experiment.add_figure(
            tag="val/magnetic_field", figure=mag_fig, global_step=self.global_step
        )


conditions = [data_cond, zero_cond]
optim = OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)

solver = Solver(
    train_conditions=conditions,
    optimizer_setting=optim,
)
# ckpt = WeightSaveCallback(model=module, name="current", check_interval=1)
ckpt = ModuleCheckpoint(
    module=module,
    name="current",
    every_n_train_steps=10,
    monitor="train/data_cond",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="step={step}-data_cond={train/data_cond:.2f}",
    auto_insert_metric_name=False,
)

trainer = pl.Trainer(
    accelerator="cpu",
    max_steps=20000,
    logger=pl.loggers.TensorBoardLogger("logs/jerschow/"),
    benchmark=True,
    log_every_n_steps=5,
    val_check_interval=200,
    callbacks=[ckpt],
)

trainer.fit(solver)
