"""Linear baseline LightningModule — per-pixel linear regression."""

import numpy as np
import torch
import torch.nn as nn
import lightning as L


class LinearBaseline(L.LightningModule):
    """Per-pixel linear model: each grid point gets an independent linear map.

    This is the simplest possible weather forecast model — it learns a separate
    linear transformation for each spatial location, mapping input channels to
    output channels.
    """

    def __init__(
        self,
        n_channels: int = 2,
        spatial_shape: tuple[int, int] = (32, 64),
        lr: float = 1e-3,
        lat_weights: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lat_weights"])
        H, W = spatial_shape
        # Independent linear map per grid point
        self.weight = nn.Parameter(
            torch.eye(n_channels)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, H, W)
            .clone()
        )
        self.bias = nn.Parameter(torch.zeros(n_channels, H, W))

        if lat_weights is not None:
            self.register_buffer("lat_weights", torch.from_numpy(lat_weights).float())
        else:
            self.register_buffer(
                "lat_weights",
                torch.ones(H, dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Per-pixel linear: out_c = sum_c'(weight[c,c',h,w] * x[c']) + bias[c,h,w]
        out = torch.einsum("cdhw,bdhw->bchw", self.weight, x) + self.bias
        return out

    def _lat_weighted_mse(self, pred, target):
        """Latitude-weighted MSE loss."""
        # pred, target: (B, C, H, W)
        sq_err = (pred - target) ** 2  # (B, C, H, W)
        # Weight by latitude (broadcast over B, C, W)
        weighted = sq_err * self.lat_weights[None, None, :, None]
        return weighted.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self._lat_weighted_mse(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self._lat_weighted_mse(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self._lat_weighted_mse(pred, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
