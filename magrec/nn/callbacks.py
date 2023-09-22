import pytorch_lightning as pl
import torch


class WeightSaveCallback(pl.Callback):
    """
    A callback to save the weights of a model during training. Can save
    the model weights before, during and after training. During training, only
    the model with minimal loss will be saved.

    Parameters
    ----------
    model : torch.nn.Module
        The model of which the weights should be saved.
    path : str
        The relative path of the saved weights.
    name : str
        A name that will become part of the file name of the saved weights.
    check_interval : int
        The callback will check for minimal loss every check_interval
        iterations. If negative, no weights will be saved during training.
    save_initial_model : False
        Whether the model should be saved before training as well.
    save_final_model: True
        Whether the model should always be saved after the last iteration.
    """

    def __init__(
        self,
        model,
        name,
        check_interval,
        path=None,
        save_initial_model=False,
        save_final_model=True,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.name = name
        self.check_interval = check_interval
        self.save_initial_model = save_initial_model
        self.save_final_model = save_final_model

        self.current_loss = float("inf")

    def on_train_start(self, trainer, pl_module):
        self.path = trainer.log_dir if self.path is None else self.path
        self.path = self.path + "/checkpoints"
        if self.save_initial_model:
            torch.save(
                self.model, 
                self.path + "/" + self.name + "_init.pt"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (self.check_interval > 0 and batch_idx > 0) and (
            (batch_idx - 1) % self.check_interval == 0
        ):
            if trainer.logged_metrics["train/loss"] < self.current_loss:
                self.current_loss = trainer.logged_metrics["train/loss"]
                torch.save(
                    self.model,
                    self.path + "/" + self.name + "_min_loss.pt",
                )

    def on_train_end(self, trainer, pl_module):
        if self.save_final_model:
            torch.save(
                self.model,
                self.path + "/checkpoints/" + self.name + "_final.pt",
            )


class ModuleCheckpoint(pl.callbacks.ModelCheckpoint):
    
    def __init__(self, module, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.name = name
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint.update({self.name: self.module.state_dict()})


class ProgressPlotsCallback(pl.Callback):
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        return super().on_validation_end(trainer, pl_module)
