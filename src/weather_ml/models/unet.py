"""Simple 2D UNet LightningModule for spatial weather prediction."""

import numpy as np
import torch
import torch.nn as nn
import lightning as L


class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(L.LightningModule):
    """Small 2D UNet for weather prediction.

    Encoder-decoder with skip connections. Uses circular padding in longitude
    (via padding_mode on convolutions would be ideal, but we keep it simple
    here with standard padding).

    Architecture: 3-level UNet with channel progression base_ch → 2x → 4x.
    """

    def __init__(
        self,
        n_channels: int = 2,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        spatial_shape: tuple[int, int] = (32, 64),
        lat_weights: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["lat_weights"])
        bc = base_channels

        # Encoder
        self.enc1 = DoubleConv(n_channels, bc)
        self.enc2 = DoubleConv(bc, bc * 2)
        self.enc3 = DoubleConv(bc * 2, bc * 4)

        # Bottleneck
        self.bottleneck = DoubleConv(bc * 4, bc * 4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 4, 2, stride=2)
        self.dec3 = DoubleConv(bc * 8, bc * 2)  # concat skip
        self.up2 = nn.ConvTranspose2d(bc * 2, bc * 2, 2, stride=2)
        self.dec2 = DoubleConv(bc * 4, bc)  # concat skip
        self.up1 = nn.ConvTranspose2d(bc, bc, 2, stride=2)
        self.dec1 = DoubleConv(bc * 2, bc)  # concat skip

        # Output
        self.out_conv = nn.Conv2d(bc, n_channels, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Latitude weights
        H = spatial_shape[0]
        if lat_weights is not None:
            self.register_buffer(
                "lat_weights", torch.from_numpy(lat_weights).float()
            )
        else:
            self.register_buffer(
                "lat_weights", torch.ones(H, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)  # (B, bc, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 2bc, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 4bc, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 4bc, H/8, W/8)

        # Decoder path with skip connections
        d3 = self.up3(b)  # (B, 4bc, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)  # (B, 2bc, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)  # (B, bc, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Residual connection: predict the *change*
        return x + self.out_conv(d1)

    def _lat_weighted_mse(self, pred, target):
        """Latitude-weighted MSE loss."""
        sq_err = (pred - target) ** 2
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
