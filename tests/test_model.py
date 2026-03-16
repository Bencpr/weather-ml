"""Tests for weather ML models."""

import numpy as np
import pytest
import torch

from weather_ml.models.linear import LinearBaseline
from weather_ml.models.unet import SimpleUNet
from weather_ml.models.random_forest import RandomForestBaseline


@pytest.fixture
def sample_batch():
    """Create a sample batch of weather data."""
    torch.manual_seed(42)
    B, C, H, W = 4, 2, 32, 64
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    return x, y


@pytest.fixture
def lat_weights():
    """Latitude weights for 32-lat grid."""
    lats = np.linspace(90, -90, 32)
    w = np.cos(np.deg2rad(lats)).astype(np.float32)
    w /= w.mean()
    return w


class TestLinearBaseline:
    def test_forward_shape(self, sample_batch):
        model = LinearBaseline(n_channels=2, spatial_shape=(32, 64))
        x, _ = sample_batch
        out = model(x)
        assert out.shape == x.shape

    def test_training_step(self, sample_batch):
        model = LinearBaseline(n_channels=2, spatial_shape=(32, 64))
        loss = model.training_step(sample_batch, 0)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_identity_init(self, sample_batch):
        """Model should start near identity (residual ≈ input)."""
        model = LinearBaseline(n_channels=2, spatial_shape=(32, 64))
        x, _ = sample_batch
        out = model(x)
        # Output should be close to input due to identity weight init
        assert torch.allclose(out, x, atol=0.01)

    def test_lat_weighted_loss(self, sample_batch, lat_weights):
        model = LinearBaseline(
            n_channels=2, spatial_shape=(32, 64), lat_weights=lat_weights
        )
        loss = model.training_step(sample_batch, 0)
        assert loss.item() > 0

    def test_configure_optimizers(self):
        model = LinearBaseline(n_channels=2, spatial_shape=(32, 64), lr=0.01)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["lr"] == 0.01


class TestSimpleUNet:
    def test_forward_shape(self, sample_batch):
        model = SimpleUNet(n_channels=2, base_channels=16, spatial_shape=(32, 64))
        x, _ = sample_batch
        out = model(x)
        assert out.shape == x.shape

    def test_training_step(self, sample_batch):
        model = SimpleUNet(n_channels=2, base_channels=16, spatial_shape=(32, 64))
        loss = model.training_step(sample_batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_residual_connection(self, sample_batch):
        """With untrained model, output should be close to input (residual)."""
        model = SimpleUNet(n_channels=2, base_channels=16, spatial_shape=(32, 64))
        x, _ = sample_batch
        out = model(x)
        # The residual output should be in the same ballpark as input
        diff = (out - x).abs().mean()
        assert diff < 5.0  # rough sanity check

    def test_validation_step(self, sample_batch):
        model = SimpleUNet(n_channels=2, base_channels=16, spatial_shape=(32, 64))
        loss = model.validation_step(sample_batch, 0)
        assert loss.item() > 0

    def test_configure_optimizers(self):
        model = SimpleUNet(
            n_channels=2, base_channels=16, spatial_shape=(32, 64), lr=0.001
        )
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)

    def test_different_base_channels(self, sample_batch):
        """UNet should work with various base channel widths."""
        for bc in [8, 16, 64]:
            model = SimpleUNet(n_channels=2, base_channels=bc, spatial_shape=(32, 64))
            x, _ = sample_batch
            out = model(x)
            assert out.shape == x.shape


class TestRandomForestBaseline:
    """Uses a tiny synthetic torch Dataset for fit/predict/evaluate."""

    @pytest.fixture
    def tiny_dataset(self):
        """Minimal dataset: list-like of (x, y) tensor pairs."""
        torch.manual_seed(0)

        class _DS:
            def __init__(self):
                self.data = [
                    (torch.randn(2, 8, 16), torch.randn(2, 8, 16)) for _ in range(20)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return _DS()

    def test_fit_and_predict_shape(self, tiny_dataset):
        model = RandomForestBaseline(n_estimators=5, max_depth=3)
        model.fit(tiny_dataset)
        x = torch.randn(3, 2, 8, 16).numpy()
        pred = model.predict(x)
        assert pred.shape == (3, 2, 8, 16)

    def test_evaluate_returns_rmse(self, tiny_dataset):
        model = RandomForestBaseline(n_estimators=5, max_depth=3)
        model.fit(tiny_dataset)
        metrics = model.evaluate(tiny_dataset)
        assert "rmse" in metrics
        assert "rmse_per_channel" in metrics
        assert metrics["rmse"] > 0
        assert len(metrics["rmse_per_channel"]) == 2

    def test_evaluate_with_lat_weights(self, tiny_dataset):
        model = RandomForestBaseline(n_estimators=5, max_depth=3)
        model.fit(tiny_dataset)
        weights = np.cos(np.deg2rad(np.linspace(90, -90, 8))).astype(np.float32)
        weights /= weights.mean()
        metrics = model.evaluate(tiny_dataset, lat_weights=weights)
        assert metrics["rmse"] > 0

    def test_subsample_limit(self, tiny_dataset):
        model = RandomForestBaseline(n_estimators=5, max_depth=3)
        model.fit(tiny_dataset, max_samples=100)
        x = torch.randn(1, 2, 8, 16).numpy()
        pred = model.predict(x)
        assert pred.shape == (1, 2, 8, 16)
