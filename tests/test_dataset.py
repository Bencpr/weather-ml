"""Tests for WeatherBench dataset and data module."""

import numpy as np
import pytest
import torch
import xarray as xr

from weather_ml.data.dataset import WeatherBenchDataset, WeatherBenchDataModule


@pytest.fixture
def synthetic_ds():
    """Create a small synthetic xarray Dataset mimicking WeatherBench."""
    np.random.seed(42)
    n_times = 100
    n_lat = 32
    n_lon = 64
    import pandas as pd

    times = pd.date_range("2020-01-01", periods=n_times, freq="6h").values
    lats = np.linspace(90, -90, n_lat)
    lons = np.linspace(0, 354.375, n_lon)

    ds = xr.Dataset(
        {
            "z": (
                ["time", "lat", "lon"],
                np.random.randn(n_times, n_lat, n_lon).astype(np.float32),
            ),
            "t": (
                ["time", "lat", "lon"],
                np.random.randn(n_times, n_lat, n_lon).astype(np.float32) * 10 + 250,
            ),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


class TestWeatherBenchDataset:
    def test_length(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        # With 6h lead time and 6h resolution, lead_steps=1, so n_samples = 100 - 1 = 99
        assert len(ds) == 99

    def test_shapes(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        x, y = ds[0]
        assert x.shape == (2, 32, 64)  # (C, H, W)
        assert y.shape == (2, 32, 64)
        assert x.dtype == torch.float32

    def test_normalization_stats(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        assert ds.mean.shape == (1, 2, 1, 1)
        assert ds.std.shape == (1, 2, 1, 1)
        # Std should be positive
        assert (ds.std > 0).all()

    def test_normalized_output_reasonable(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        x, _ = ds[0]
        # Normalized data should roughly be in [-5, 5] range
        assert x.abs().max() < 10.0

    def test_lat_weights(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        assert ds.lat_weights.shape == (32,)
        # Mean weight should be ~1.0 (normalized)
        assert abs(ds.lat_weights.mean() - 1.0) < 0.01
        # Equatorial weights > polar weights
        assert ds.lat_weights[16] > ds.lat_weights[0]

    def test_custom_variables(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6, variables=["z"])
        x, y = ds[0]
        assert x.shape == (1, 32, 64)

    def test_n_channels(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        assert ds.n_channels == 2

    def test_spatial_shape(self, synthetic_ds):
        ds = WeatherBenchDataset(synthetic_ds, lead_time_hours=6)
        assert ds.spatial_shape == (32, 64)


class TestWeatherBenchDataModule:
    """Tests use synthetic data spanning 2020-01-01 to ~2020-01-25 (100 steps × 6h)."""

    def _make_dm(self, tmp_path, synthetic_ds, batch_size=8):
        synthetic_ds.to_netcdf(tmp_path / "test_data.nc")
        # Split the 25-day synthetic range: train first 17 days, val next 4, test last 4
        return WeatherBenchDataModule(
            data_dir=str(tmp_path),
            batch_size=batch_size,
            num_workers=0,
            train_years="2020-01-01/2020-01-17",
            val_years="2020-01-18/2020-01-21",
            test_years="2020-01-22/2020-01-26",
        )

    def test_setup_with_synthetic_data(self, synthetic_ds, tmp_path):
        dm = self._make_dm(tmp_path, synthetic_ds)
        dm.setup("fit")

        assert dm.train_ds is not None
        assert dm.val_ds is not None
        assert len(dm.train_ds) > 0

    def test_dataloader_batch_shape(self, synthetic_ds, tmp_path):
        dm = self._make_dm(tmp_path, synthetic_ds, batch_size=4)
        dm.setup("fit")

        batch = next(iter(dm.train_dataloader()))
        x, y = batch
        assert x.shape[0] == 4  # batch size
        assert x.shape[1] == 2  # channels
