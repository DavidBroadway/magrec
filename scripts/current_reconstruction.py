import os

import pytorch_lightning as L
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import torch
import torch.nn.functional as F
from magrec.misc.plot import plot_n_components

from magrec.nn.models import FourierFeaturesPINN
from magrec.prop.conditions import Condition
from magrec.prop.Propagator import AxisProjectionPropagator, CurrentPropagator2d
from magrec import __datapath__, __logpath__

import numpy as np

import deepxde as dde


class MagneticFieldObservations2d(Condition):
    """
    Gives a condition on a 2d set of points and corresponding magnetic field observations,
    where points and observations are connected through a propagator `prop`.
    """

    def __init__(
        self, values, dx, dy, height, layer_thickness, theta, phi, name, weight
    ) -> None:
        super().__init__(name)
        self.values = values  # shape (n, W, H) where n is the number of components of the magnetic field
        W, H = values.shape[-2:]
        self.prop = CurrentPropagator2d(
            source_shape=(3 * W, 3 * H),
            dx=dx,
            dy=dy,
            height=height,
            layer_thickness=layer_thickness,
            real_signal=False,
        )

        self.pts_sample = torch.cartesian_prod(
            torch.linspace(-1, 2, 3 * W), torch.linspace(-1, 2, 3 * H)
        ).reshape(-1, 2)
        self.weight = weight
        self.proj = AxisProjectionPropagator(theta, phi)

    def sample(self):
        """
        Gives a set of points x where to evaluate the current density. The set of points is larger than the set of points self.x
        since the current density outside the measurement region contributes to the magnetic field inside the measurement region.
        """
        return self.pts_sample

    def loss(self, x, y_hat):
        error = self.error(x, y_hat)
        return F.mse_loss(error, torch.zeros_like(error)) * self.weight

    def error(self, x, y_hat):
        error = self.propagate(x, y_hat) - self.values
        return error

    def propagate(self, x, y_hat):
        W, H = self.values.shape[-2:]
        y_hat_grid = y_hat.reshape(-1, 3 * W, 3 * H)
        values_hat = self.proj(self.prop(y_hat_grid).real)
        return values_hat[..., W : 2 * W, H : 2 * H]

    def validate(self, x, y_hat, logger, step):
        W, H = self.values.shape[-2:]
        y_hat_grid = y_hat.reshape(-1, 3 * W, 3 * H)

        curr_fig = plot_n_components(y_hat_grid, labels=[r"J_x", r"J_y"], cmap="bwr")
        logger.add_figure(tag="val/current distribution", figure=curr_fig, global_step=step)

        values_hat = self.prop(y_hat_grid).real
        mag_field = torch.cat([values_hat, self.proj(values_hat).unsqueeze(0)], dim=0)

        mag_fig = plot_n_components(
            mag_field, labels=[r"target", r"B_x", r"B_y", r"B_z", r"B_{NV}"], cmap="bwr"
        )
        logger.add_figure(tag="val/magnetic field", figure=mag_fig, global_step=step)

    def in_region(self, x) -> bool:
        pass


class ZeroCurrent(Condition):
    """Condition to require zero current in a region."""

    def __init__(self) -> None:
        super().__init__(name="zero_current")

    def sample(self, n=1000):
        """Samples points in the region where the current should be zero."""
        # Current is zero for x in [-1, 0] and [1, 2] and y in [-1, 1]
        # We sample n points in from those regions. Because they are
        # disjoint and of equal area, we can sample n // 2 points from each region.
        return torch.cat(
            [
                torch.rand(n // 2, 2) * torch.tensor([1, 3]) - torch.tensor([1, 1]),
                torch.rand(n // 2, 2) * torch.tensor([1, 3]) + torch.tensor([1, -1]),
            ],
            dim=0,
        )

    def loss(self, x, y_hat):
        return F.mse_loss(y_hat, torch.zeros_like(y_hat))


class ZeroDerivative(Condition):
    def __init__(self) -> None:
        super().__init__(name="zero_derivative")

    def sample(self, n=1000):
        """Samples points in the region where the derivative of the current should be zero."""
        return torch.cat(
            [
                torch.rand(n // 2, 2) * torch.tensor([1, 3]) - torch.tensor([1, 1]),
                torch.rand(n // 2, 2) * torch.tensor([1, 3]) + torch.tensor([1, -1]),
                torch.rand(n // 4, 2) + torch.tensor([0, 1]),
                torch.rand(n // 4, 2) - torch.tensor([0, 1]),
            ],
            dim=0,
        )

    def loss(self, x, y_hat):
        # Both derivatives should be zero in the region where we expect constant current
        djx_dr = dde.grad.jacobian(y_hat, x, i=0)
        djy_dr = dde.grad.jacobian(y_hat, x, i=1)
        return F.mse_loss(djx_dr, torch.zeros_like(djx_dr)) + F.mse_loss(
            djy_dr, torch.zeros_like(djy_dr)
        )


class MagneticField2dDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        # Load the data
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

        B_NV = AxisProjectionPropagator(theta, phi).project(B)
        values = B_NV

        conditions = [
            MagneticFieldObservations2d(
                values,
                dx=dx,
                dy=dy,
                height=height,
                layer_thickness=layer_thickness,
                theta=theta,
                phi=phi,
                name="observe_B",
                weight=1e6,
            ),
            ZeroCurrent(),
            ZeroDerivative()
        ]

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, conditions):
                self.conditions = conditions

            def __getitem__(self, index):
                batch = []
                for condition in self.conditions:
                    batch.append((condition.sample(), condition))
                return batch

            def __len__(self):
                return 1

        self.dataset = Dataset(conditions)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.dataset, batch_size=None)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.dataset, batch_size=None)


def main():
    # Prepare a logger at /logs/
    tensorboard_logger = pl_loggers.TensorBoardLogger("logs/")
    # Create a datamodule which contains data and appropriate conditions
    # for the model to calculate loss against
    datamodule = MagneticField2dDataModule()
    model = FourierFeaturesPINN(ff_sigmas=[(1, 20), (2, 20), (0.5, 20), (5, 20)])
    trainer = L.Trainer(
        logger=tensorboard_logger,
        accelerator="cpu",
        max_epochs=2000,
        check_val_every_n_epoch=50,
        log_every_n_steps=1,
        callbacks=[L.callbacks.ModelCheckpoint(monitor="total_loss", dirpath="logs/", save_on_train_epoch_end=True)]
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
