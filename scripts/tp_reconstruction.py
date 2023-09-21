# reconstruction of current with torchphysics
import numpy as np
import torch
import deepxde as dde

import pytorch_lightning as pl
import magrec
from magrec.misc.plot import (
    plot_n_components,
    plot_vector_field_2d,
)

from magrec.nn.models import FourierFeatures2dCurrent
from magrec.nn.solver import Solver, OptimizerSetting
from magrec.misc.sampler import (
    GridSampler,
    Sampler,
    RectangleSampler,
    StaticSampler,
)

# from magrec.nn.callbacks import FieldPlotCallback, CurrentPlotCallback, CurrentFlowPlotCallback
from magrec.prop.Propagator import AxisProjectionPropagator, CurrentPropagator2d
from magrec.prop.conditions import PINNCondition
from magrec import __datapath__

"""
   Regions R, O, A:
   
   ..code::
   
      Y
      ^
    3 │ ┌───────┬───────┬───────┐
      │ │       │   O   │       │
      │ │       │       │       │
    2 │ │       ├───────┤       │
      │ │   R   │       │   R   │
      │ │       │   A   │       │
    1 │ │       ├───────┤       │
      │ │       │       │       │
      │ │       │   O   │       │
    0 │ └───────┴───────┴───────┘
      └─────────────────────────────>
        0       1       2       3  X

"""


def const_region_sample_fn(n_points):
    """Function to get points from the outer region around the measurement region Rs and Os.

    ..code::

      Y
      ^
    3 │ ┌───────┬───────┬───────[3, 3]
      │ │       │  IV   │       │
      │ │       │       │       │
    2 │ │       ├───────┤       │
      │ │   I   │       │   II  │
      │ │       │       │       │
    1 │ │       ├───────┤       │
      │ │       │       │       │
      │ │       │  III  │       │
    0 │[0, 0]───┴───────┴───────┘
      └─────────────────────────────>
        0       1       2       3  X

    """
    region_n_points = n_points // 8  # ratios of the regions' areas are 3:1:1:3
    region1_points = RectangleSampler.sample_rectangular_region(
        region_n_points * 3, origin=[0, 0], diagonal=[1, 3]
    )
    region2_points = RectangleSampler.sample_rectangular_region(
        region_n_points * 3, origin=[2, 0], diagonal=[3, 3]
    )
    region3_points = RectangleSampler.sample_rectangular_region(
        region_n_points, origin=[1, 0], diagonal=[2, 1]
    )
    region4_points = RectangleSampler.sample_rectangular_region(
        region_n_points, origin=[1, 0], diagonal=[2, 1]
    )
    points = torch.cat(
        (region1_points, region2_points, region3_points, region4_points), dim=0
    )
    return points


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


Bx = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "Bx.txt")
By = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "By.txt")
Bz = np.loadtxt(__datapath__ / "ExperimentalData" / "NbWire" / "Bz.txt")
B = torch.tensor(np.array([Bx, By, Bz]), dtype=torch.float32)

theta = 54.7  # degrees
phi = 45.0  # degrees
height = 0.015  # μm
layer_thickness = 0.030  # μm
dx = 0.408  # in μm
dy = 0.408

proj = AxisProjectionPropagator(theta, phi)
B_NV = proj(B)
values = B_NV
W, H = values.shape[-2:]


def grid_sample_fn(n_points=None):
    """Function to sample in a regular grid of shape (nx_points, ny_points) from the entire region ROAs."""
    points = GridSampler.sample_grid(
        nx_points=3 * W, ny_points=3 * H, origin=[0, 0], diagonal=[3, 3]
    )
    return points


const_sampler = Sampler(
    const_region_sample_fn, batch_n_points=10**2, cached_n_points=10**3
)

zero_sampler = Sampler(
    const_region_sample_fn, batch_n_points=10**2, cached_n_points=10**3
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


def const_residual(J, x):
    """Residual function for the constant current reconstruction."""
    # computes a tensor of derivatives of J_x, J_y with respect to x, y
    err = dde.grad.jacobian(J, x)
    return err


def zero_residual(J, x):
    """Residual function for the zero current reconstruction."""
    # error is the current itself, 0 is expected
    err = J
    return err


def data_residual(J, x):
    """Residual function for the data condition."""
    J_grid = GridSampler.pts_to_grid(J, 3 * W, 3 * H)
    values_hat = prop(J_grid)[..., W:-W, H:-H]
    err = values - proj(values_hat)
    return err


# define a model that maps from x ∊ R² to J ∊ R², div J = 0 by construction
module = FourierFeatures2dCurrent(ff_sigmas=[(1, 10), (0.5, 10), (2, 10)])

const_cond = PINNCondition(
    module=module,
    sampler=const_sampler,
    residual_fn=const_residual,
    name="const_cond",
)

zero_cond = PINNCondition(
    module=module,
    sampler=zero_sampler,
    residual_fn=zero_residual,
    name="zero_cond",
)

data_cond = PINNCondition(
    module=module,
    sampler=grid_sampler,
    residual_fn=data_residual,
    name="data_cond",
)

conditions = [const_cond, data_cond, zero_cond]
optim = OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.01)


class Solver(magrec.nn.solver.Solver):
    def on_train_start(self):
        super().on_train_start()
        target_figure = plot_n_components(values, labels=[r"B_{NV}"], cmap="bwr")
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

        writer = self.logger.experiment

        curr_fig = plot_n_components(
            y_hat_grid,
            labels=[r"J_x", r"J_y"],
            cmap="bwr",
            zoom_in_region=((20, 20), (30, 30)),
        )
        # plot_to_tensorboard(
        #     writer, curr_fig, tag="val/current distribution", step=self.global_step
        # )
        self.logger.experiment.add_figure(tag="val/current distribution", figure=curr_fig, global_step=self.global_step)

        flow_fig = plot_vector_field_2d(
            y_hat_grid, cmap="plasma", zoom_in_region=((20, 20), (30, 30))
        )
        # plot_to_tensorboard(
        #     writer, flow_fig, tag="val/current flow", step=self.global_step
        # )
        self.logger.experiment.add_figure(tag="val/current flow", figure=flow_fig, global_step=self.global_step)

        values_hat = prop(y_hat_grid).real
        mag_field = torch.cat([values_hat, proj(values_hat).unsqueeze(0)], dim=0)
        mag_fig = plot_n_components(
            mag_field,
            labels=[r"B_x", r"B_y", r"B_z", r"B_{NV}"],
            cmap="bwr",
            zoom_in_region=((20, 20), (30, 30)),
        )
        # plot_to_tensorboard(
        #     writer, mag_fig, tag="val/magnetic field", step=self.global_step
        # )
        self.logger.experiment.add_figure(tag="val/magnetic field", figure=mag_fig, global_step=self.global_step)


solver = Solver(train_conditions=conditions, optimizer_setting=optim,)
checkpointer = pl.callbacks.ModelCheckpoint(monitor="val/data_cond", save_top_k=1, every_n_train_steps=10, save_last=True)

trainer = pl.Trainer(
    accelerator="cpu",
    max_steps=20,
    logger=pl.loggers.TensorBoardLogger("logs/"),
    benchmark=True,
    log_every_n_steps=5,
    val_check_interval=20,
    callbacks=[checkpointer],
)

trainer.fit(solver)
