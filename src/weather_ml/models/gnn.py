"""
Graph Neural Network weather model: Encoder → Processor → Decoder.

Architecture mirrors Anemoi's anemoi-models EncProcDec design.

    Data grid (B, C, H, W)
        ↓  reshape → nodes: (B·N, C),  N = H·W
        ↓  concat positional features  → (B·N, C+4)
        ↓  Encoder  [MLP: C+4 → hidden_dim]
    Hidden nodes (B·N, hidden_dim)
        ↓  Processor  [N_layers × GraphConvLayer]
        │  each layer: double-residual message passing
        │  + skip connection across the full processor stack
        ↓  Decoder  [MLP: hidden_dim → C]
        ↓  + residual (x input): model predicts corrections, not full state
    Output (B, C, H, W)

GraphConvLayer — Anemoi double-residual convention:
    edge_new = MLP([x_src ‖ x_dst ‖ e]) + e        ← edge residual
    node_agg = scatter_sum(edge_new, dst)
    node_new = MLP([x ‖ node_agg])      + x        ← node residual

Both residuals make each layer learn a *correction* rather than a full
transformation — same principle as ResNets. Critical for stable training
with 6–16 layers; without residuals, gradients vanish.

Batching strategy:
    The graph topology is identical for every sample in the batch.
    Rather than rebuilding it B times, we:
      1. Replicate node features B times: (B·N, *)
      2. Offset edge_index per sample: edges for sample b use nodes [b·N, (b+1)·N)
    This avoids PyG's Batch API and keeps the code self-contained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf

from weather_ml.data.graph import grid_to_graph


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------


def _to_python(obj):
    """Recursively convert OmegaConf containers to plain Python types.

    Walks the entire structure so mixed dicts (some plain, some OmegaConf)
    are fully converted. Required for PyTorch 2.6+ weights_only checkpoint
    loading which rejects any non-allowlisted type.
    """
    if OmegaConf.is_config(obj):
        obj = OmegaConf.to_container(obj, resolve=True, throw_on_missing=False)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_python(v) for v in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    return obj


class GraphConvLayer(nn.Module):
    """
    One message-passing step with Anemoi's double-residual convention.

    WHY LayerNorm instead of BatchNorm:
        In graph message passing, the effective 'batch' is N nodes, not B samples.
        LayerNorm normalises each node's features independently — stable even
        with small B or variable-size graphs.

    WHY SiLU instead of ReLU:
        Smooth everywhere — no dead neurons. Weather variables span negative
        anomalies (cold, low pressure) — ReLU would kill those gradients.
    """

    def __init__(self, node_dim: int, edge_dim: int, mlp_hidden: int):
        super().__init__()

        # f([x_src ‖ x_dst ‖ e]) → δe,  then e_new = δe + e
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, edge_dim),
        )

        # g([x ‖ agg_e]) → δx,  then x_new = δx + x
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, node_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # (total_nodes, node_dim)
        edge_index: torch.Tensor,  # (2, total_edges)
        edge_attr: torch.Tensor,  # (total_edges, edge_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # Edge update with residual
        edge_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_new = self.edge_mlp(edge_in) + edge_attr

        # Scatter-sum: aggregate incoming edge messages per destination node
        agg = x.new_zeros(x.size(0), edge_attr.size(-1))
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_new), edge_new)

        # Node update with residual
        x_new = self.node_mlp(torch.cat([x, agg], dim=-1)) + x

        return x_new, edge_new


# ---------------------------------------------------------------------------
# Encoder-Processor-Decoder
# ---------------------------------------------------------------------------


class GraphWeatherModel(L.LightningModule):
    """
    GNN weather forecasting model: Encoder → Processor → Decoder.

    Drop-in replacement for SimpleUNet / LinearBaseline — same (B, C, H, W)
    interface, same Lightning API, same Hydra-instantiable signature.

    Args:
        n_vars        number of atmospheric variables (C in the batch)
        hidden_dim    node embedding size in the processor
        edge_dim      edge feature size (lifted from 2D raw feats)
        n_layers      number of GraphConvLayer steps
        mlp_ratio     MLP hidden width = hidden_dim * mlp_ratio
        spatial_shape (H, W) grid shape — inferred from dataset in train.py
        k             KNN neighbours per node
        lr            AdamW learning rate
        weight_decay  AdamW weight decay
        lat_weights   optional (H,) or (H*W,) cos(lat) weights from dataset
    """

    def __init__(
        self,
        n_vars: int = 2,
        hidden_dim: int = 128,
        edge_dim: int = 32,
        n_layers: int = 6,
        mlp_ratio: int = 2,
        spatial_shape: tuple[int, int] = (32, 64),
        k: int = 9,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lat_weights: np.ndarray | None = None,
    ):
        super().__init__()

        # Coerce to tuple — Hydra passes spatial_shape as a ListConfig
        spatial_shape = tuple(spatial_shape)

        self.save_hyperparameters(ignore=["lat_weights"])

        H, W = spatial_shape
        mlp_hidden = hidden_dim * mlp_ratio

        # ---- Build graph once from the grid spec ----
        lats_1d = np.linspace(90, -90, H, dtype=np.float32)
        lons_1d = np.linspace(0, 360 - 360 / W, W, dtype=np.float32)
        graph = grid_to_graph(lats_1d, lons_1d, k=k)
        self._graph = graph  # Python ref — not moved to device

        # Register as buffers → auto-moved to device by Lightning
        self.register_buffer("edge_index", graph.edge_index)  # (2, E)
        self.register_buffer("edge_attr_raw", graph.edge_attr)  # (E, 2)
        self.register_buffer("node_pos", graph.node_pos_feats)  # (N, 4)

        # lat_weights: accept (H,) from ERA5Dataset and tile to (H*W,).
        # Condition: tile whenever the array is 1-D and shorter than H*W —
        # avoids relying on H being correct at this point (could differ from
        # the yaml default if spatial_shape was overridden from the dataset).
        if lat_weights is not None:
            lw = np.asarray(lat_weights, dtype=np.float32).ravel()
            if len(lw) != H * W:
                lw = np.repeat(lw, W)  # (H,) → (H*W,)
            self.register_buffer("lat_weights", torch.from_numpy(lw))
        else:
            self.register_buffer("lat_weights", graph.lat_weights)  # (N,)

        # ---- Encoder: (n_vars + 4 pos feats) → hidden_dim ----
        self.encoder = nn.Sequential(
            nn.Linear(n_vars + 4, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- Edge encoder: raw 2-channel feats → edge_dim ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
        )

        # ---- Processor: N_layers of message passing ----
        self.processor = nn.ModuleList(
            [GraphConvLayer(hidden_dim, edge_dim, mlp_hidden) for _ in range(n_layers)]
        )

        # ---- Decoder: hidden_dim → n_vars ----
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, n_vars),
        )

    # ------------------------------------------------------------------
    # Batching helper
    # ------------------------------------------------------------------

    def _batch_graph(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replicate single-graph topology for a batch of B independent samples.

        Offsets edge_index so sample b uses nodes [b·N, (b+1)·N).
        Tiles and encodes edge_attr B times.
        """
        N = self._graph.num_nodes
        E = self.edge_index.shape[1]
        dev = self.edge_index.device

        offsets = torch.arange(B, device=dev).mul(N).view(B, 1, 1)
        ei = self.edge_index.unsqueeze(0).expand(B, -1, -1) + offsets
        edge_index_b = ei.reshape(2, B * E)

        ea = self.edge_attr_raw.unsqueeze(0).expand(B, -1, -1).reshape(B * E, 2)
        edge_attr_b = self.edge_encoder(ea)  # (B·E, edge_dim)

        return edge_index_b, edge_attr_b

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  (B, C, H, W)  normalised input fields

        Returns:
            out  (B, C, H, W)  = x + decoder(processor(encoder(x)))
        """
        B, C, H, W = x.shape
        N = H * W

        # Grid → flat nodes
        x_nodes = x.view(B, C, N).permute(0, 2, 1).reshape(B * N, C)

        # Append positional features
        pos = self.node_pos.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 4)
        h = self.encoder(torch.cat([x_nodes, pos], dim=-1))  # (B·N, hidden_dim)

        # Processor with skip connection
        edge_index_b, edge_attr_b = self._batch_graph(B)
        h_skip = h.clone()
        for layer in self.processor:
            h, edge_attr_b = layer(h, edge_index_b, edge_attr_b)
        h = h + h_skip

        # Decode + residual
        delta = self.decoder(h).reshape(B, N, C).permute(0, 2, 1).view(B, C, H, W)
        return x + delta

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _lat_weighted_mse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        H, W = self.hparams.spatial_shape
        lw = self.lat_weights.view(H, W).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return ((pred - target) ** 2 * lw).mean()

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._lat_weighted_mse(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._lat_weighted_mse(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", torch.sqrt(loss), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self._lat_weighted_mse(self(x), y)
        self.log("test_loss", loss)
        self.log("test_rmse", torch.sqrt(loss))
        return loss

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # PyTorch 2.6+ loads checkpoints with weights_only=True by default.
        # save_hyperparameters() may store OmegaConf objects (ListConfig,
        # DictConfig) mixed with plain Python types. OmegaConf.create() alone
        # does not reliably convert mixed-type structures, so we walk the whole
        # tree recursively and convert every node to native Python.
        if "hyper_parameters" in checkpoint:
            checkpoint["hyper_parameters"] = _to_python(checkpoint["hyper_parameters"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=self.hparams.lr * 0.01,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
