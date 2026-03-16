"""Hydra-driven training entry point."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import lightning as L
from lightning.fabric.plugins.io import TorchCheckpointIO
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)


class TrustedCheckpointIO(TorchCheckpointIO):
    """
    TorchCheckpointIO with weights_only=False.

    PyTorch 2.6+ defaults to weights_only=True, which blocks any type not
    in its allowlist — including OmegaConf's ListConfig/DictConfig that
    Lightning stores in checkpoints via save_hyperparameters().

    Since we are loading our own checkpoints (not arbitrary third-party
    files), weights_only=False is safe here. This is the Lightning-supported
    way to override checkpoint loading behaviour without patching internals.
    """

    def load_checkpoint(self, path, map_location=None, weights_only=True):
        return super().load_checkpoint(
            path, map_location=map_location, weights_only=False
        )


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Instantiate data module
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    # Get dataset info for model construction
    train_ds = datamodule.train_ds
    model_kwargs = {}
    if train_ds is not None:
        model_kwargs["lat_weights"] = train_ds.lat_weights
        # Infer spatial_shape from the dataset so the model graph matches
        # the actual grid — avoids hardcoding it in the config.
        model_kwargs["spatial_shape"] = train_ds.spatial_shape

    # Instantiate model
    model = instantiate(cfg.model, **model_kwargs)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer — use TrustedCheckpointIO to handle PyTorch 2.6+ weights_only
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=cfg.hardware.output_dir,
        log_every_n_steps=10,
        plugins=[TrustedCheckpointIO()],
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    if datamodule.test_ds is not None:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
