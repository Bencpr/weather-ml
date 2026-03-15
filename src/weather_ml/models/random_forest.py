"""RandomForest baseline — pixel-wise regression using sklearn."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestBaseline:
    """Per-pixel RandomForest: flattens spatial dims, treats each pixel as a sample.

    Each pixel's C input channels are mapped to C output channels independently.
    This is a non-parametric baseline — no gradient descent, just tree ensembles.

    Usage:
        model = RandomForestBaseline(n_estimators=50)
        model.fit(train_dataset)
        preds = model.predict(x)  # x: (B, C, H, W) numpy array
        rmse = model.evaluate(test_dataset)
    """

    def __init__(self, n_estimators: int = 50, max_depth: int | None = 10, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.rf: RandomForestRegressor | None = None

    def _flatten(self, x: np.ndarray, y: np.ndarray | None = None):
        """Reshape (B, C, H, W) → (B*H*W, C) for sklearn."""
        B, C, H, W = x.shape
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        if y is not None:
            y_flat = y.transpose(0, 2, 3, 1).reshape(-1, C)
            return x_flat, y_flat
        return x_flat

    def fit(self, dataset, max_samples: int = 50_000) -> "RandomForestBaseline":
        """Fit on a WeatherBenchDataset. Subsamples pixels to keep training fast."""
        xs, ys = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.stack(xs)  # (N, C, H, W)
        Y = np.stack(ys)
        X_flat, Y_flat = self._flatten(X, Y)

        # Subsample for tractability
        if len(X_flat) > max_samples:
            idx = np.random.default_rng(42).choice(len(X_flat), max_samples, replace=False)
            X_flat, Y_flat = X_flat[idx], Y_flat[idx]

        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self.rf.fit(X_flat, Y_flat)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict from (B, C, H, W) numpy array → (B, C, H, W)."""
        B, C, H, W = x.shape
        x_flat = self._flatten(x)
        y_flat = self.rf.predict(x_flat)  # (B*H*W, C)
        return y_flat.reshape(B, H, W, C).transpose(0, 3, 1, 2)

    def evaluate(self, dataset, lat_weights: np.ndarray | None = None) -> dict[str, float]:
        """Compute RMSE (optionally latitude-weighted) over a dataset."""
        xs, ys = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.stack(xs)
        Y = np.stack(ys)
        P = self.predict(X)

        sq_err = (P - Y) ** 2  # (N, C, H, W)
        if lat_weights is not None:
            sq_err = sq_err * lat_weights[None, None, :, None]

        mse_per_channel = sq_err.mean(axis=(0, 2, 3))
        rmse_per_channel = np.sqrt(mse_per_channel)
        return {
            "rmse": float(np.sqrt(sq_err.mean())),
            "rmse_per_channel": rmse_per_channel.tolist(),
        }
