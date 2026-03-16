"""
Tests for graph construction (graph/builder.py) and the GNN model (models/gnn.py).

Each test is focused and has a docstring explaining *what* property is being
checked and *why* it matters — useful to re-read before the Météo France interview.
"""

import numpy as np
import pytest
import torch

from weather_ml.data.graph import (
    latlon_to_xyz,
    build_graph,
    grid_to_graph,
)
from weather_ml.models.gnn import GraphConvLayer, GraphWeatherModel


# ============================================================
# latlon_to_xyz
# ============================================================


class TestLatLonToXyz:
    def test_equator_prime_meridian(self):
        """(0°, 0°) → (1, 0, 0) on the unit sphere."""
        xyz = latlon_to_xyz(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(xyz[0], [1, 0, 0], atol=1e-6)

    def test_north_pole(self):
        """(90°, any lon) → (0, 0, 1). z = sin(90°) = 1."""
        xyz = latlon_to_xyz(np.array([90.0]), np.array([0.0]))
        np.testing.assert_allclose(xyz[0], [0, 0, 1], atol=1e-6)

    def test_unit_norm(self):
        """All output vectors must lie on the unit sphere (‖xyz‖ = 1)."""
        lats = np.linspace(-90, 90, 50, dtype=np.float32)
        lons = np.linspace(0, 360, 50, dtype=np.float32)
        xyz = latlon_to_xyz(lats, lons)
        norms = np.linalg.norm(xyz, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dateline_continuity(self):
        """
        lon=-179 and lon=+179 should be nearby in 3D space.
        This is the key property that makes 3D KNN correct across the date-line.
        In lat/lon space their distance appears to be ~358° — completely wrong.
        """
        a = latlon_to_xyz(np.array([0.0]), np.array([-179.0]))
        b = latlon_to_xyz(np.array([0.0]), np.array([+179.0]))
        dist_3d = float(np.linalg.norm(a - b))
        assert dist_3d < 0.1, f"3D date-line distance should be tiny, got {dist_3d:.4f}"

    def test_pole_compression(self):
        """
        Near the poles, large lon differences correspond to tiny physical distances.
        3D KNN handles this correctly; lat/lon KNN would over-estimate the distance.
        """
        near_pole_0 = latlon_to_xyz(np.array([89.0]), np.array([0.0]))
        near_pole_170 = latlon_to_xyz(np.array([89.0]), np.array([170.0]))
        equator_0 = latlon_to_xyz(np.array([0.0]), np.array([0.0]))
        equator_170 = latlon_to_xyz(np.array([0.0]), np.array([170.0]))
        dist_pole = float(np.linalg.norm(near_pole_170 - near_pole_0))
        dist_eq = float(np.linalg.norm(equator_170 - equator_0))
        assert dist_pole < dist_eq, "170° lon gap should be smaller near the pole"


# ============================================================
# build_graph
# ============================================================


@pytest.fixture(scope="module")
def small_graph():
    """4×8 regular grid → 32 nodes, k=4 neighbours."""
    lats = np.linspace(75, -75, 4, dtype=np.float32)
    lons = np.linspace(0, 315, 8, dtype=np.float32)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    return build_graph(lat_g.ravel(), lon_g.ravel(), k=4)


class TestBuildGraph:
    def test_node_count(self, small_graph):
        assert small_graph.num_nodes == 32

    def test_edge_count(self, small_graph):
        """32 nodes × 4 neighbours = 128 directed edges."""
        assert small_graph.num_edges == 32 * 4

    def test_no_self_loops(self, small_graph):
        """KNN result should never include a node as its own neighbour."""
        src, dst = small_graph.edge_index
        assert (src != dst).all()

    def test_node_pos_shape(self, small_graph):
        assert small_graph.node_pos_feats.shape == (32, 4)

    def test_sin_cos_identity(self, small_graph):
        """sin²(lat)+cos²(lat) = 1 and sin²(lon)+cos²(lon) = 1 for every node."""
        f = small_graph.node_pos_feats.numpy()
        np.testing.assert_allclose(f[:, 0] ** 2 + f[:, 1] ** 2, 1.0, atol=1e-5)
        np.testing.assert_allclose(f[:, 2] ** 2 + f[:, 3] ** 2, 1.0, atol=1e-5)

    def test_edge_attr_shape(self, small_graph):
        assert small_graph.edge_attr.shape == (32 * 4, 2)

    def test_edge_attr_in_unit_interval(self, small_graph):
        """Both normalised distance channels should be in [0, 1]."""
        ea = small_graph.edge_attr.numpy()
        assert ea.min() >= -1e-6
        assert ea.max() <= 1.0 + 1e-6

    def test_lat_weights_mean_one(self, small_graph):
        """
        lat_weights are normalised so their mean = 1.
        This ensures that the latitude-weighted loss has the same scale
        regardless of grid resolution.
        """
        lw = small_graph.lat_weights.numpy()
        assert abs(lw.mean() - 1.0) < 0.01

    def test_grid_ref_dist_positive(self, small_graph):
        """grid_reference_distance is used as normalisation — must be > 0."""
        assert small_graph.grid_ref_dist > 0.0


class TestGridToGraph:
    def test_hw_stored(self):
        """grid_to_graph should store H and W for reshape operations."""
        g = grid_to_graph(
            np.linspace(90, -90, 8, dtype=np.float32),
            np.linspace(0, 337.5, 16, dtype=np.float32),
            k=4,
        )
        assert g.H == 8
        assert g.W == 16
        assert g.num_nodes == 128


# ============================================================
# GraphConvLayer
# ============================================================


@pytest.fixture
def ring_graph():
    """5-node ring: 0→1→2→3→4→0 plus reverses = 10 directed edges."""
    fwd = [(i, (i + 1) % 5) for i in range(5)]
    bwd = [(b, a) for a, b in fwd]
    edges = fwd + bwd
    src = torch.tensor([e[0] for e in edges])
    dst = torch.tensor([e[1] for e in edges])
    return torch.stack([src, dst])


class TestGraphConvLayer:
    def test_output_shapes(self, ring_graph):
        layer = GraphConvLayer(node_dim=16, edge_dim=8, mlp_hidden=32)
        x = torch.randn(5, 16)
        e = torch.randn(10, 8)
        x_new, e_new = layer(x, ring_graph, e)
        assert x_new.shape == (5, 16)
        assert e_new.shape == (10, 8)

    def test_node_residual(self, ring_graph):
        """
        With zero-initialised output layer, node_new should equal node_old.
        Verifies the '+x' residual path in isolation.
        """
        layer = GraphConvLayer(node_dim=16, edge_dim=8, mlp_hidden=32)
        torch.nn.init.zeros_(layer.node_mlp[-1].weight)
        torch.nn.init.zeros_(layer.node_mlp[-1].bias)
        x = torch.randn(5, 16)
        e = torch.randn(10, 8)
        x_new, _ = layer(x, ring_graph, e)
        torch.testing.assert_close(x_new, x, atol=1e-5, rtol=0)

    def test_edge_residual(self, ring_graph):
        """
        With zero-initialised edge MLP output, edge_new should equal edge_attr.
        Verifies the '+e' residual path.
        """
        layer = GraphConvLayer(node_dim=16, edge_dim=8, mlp_hidden=32)
        torch.nn.init.zeros_(layer.edge_mlp[-1].weight)
        torch.nn.init.zeros_(layer.edge_mlp[-1].bias)
        x = torch.randn(5, 16)
        e = torch.randn(10, 8)
        _, e_new = layer(x, ring_graph, e)
        torch.testing.assert_close(e_new, e, atol=1e-5, rtol=0)

    def test_gradients_flow_through_both_paths(self, ring_graph):
        """Gradients must reach both node and edge inputs (no dead paths)."""
        layer = GraphConvLayer(node_dim=16, edge_dim=8, mlp_hidden=32)
        x = torch.randn(5, 16, requires_grad=True)
        e = torch.randn(10, 8, requires_grad=True)
        x_new, _ = layer(x, ring_graph, e)
        x_new.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
        assert e.grad is not None and e.grad.abs().sum() > 0


# ============================================================
# GraphWeatherModel — integration
# ============================================================


@pytest.fixture(scope="module")
def tiny_model():
    """Very small GNN for fast CPU tests."""
    return GraphWeatherModel(
        n_vars=2,
        hidden_dim=32,
        edge_dim=8,
        n_layers=2,
        spatial_shape=(8, 16),
        k=4,
        lr=1e-3,
    )


@pytest.fixture
def batch_8x16():
    torch.manual_seed(0)
    B, C, H, W = 2, 2, 8, 16
    return torch.randn(B, C, H, W), torch.randn(B, C, H, W)


class TestGraphWeatherModel:
    def test_forward_shape(self, tiny_model, batch_8x16):
        """Output shape must equal input shape — same as UNet."""
        x, _ = batch_8x16
        out = tiny_model(x)
        assert out.shape == x.shape

    def test_residual_at_init(self, tiny_model, batch_8x16):
        """
        At initialisation, decoder weights are small → delta ≈ 0 → out ≈ x.
        The residual path (x + delta) should keep the output close to input.
        """
        x, _ = batch_8x16
        out = tiny_model(x)
        diff = (out - x).abs().mean().item()
        assert diff < 10.0, f"Initial delta unexpectedly large: {diff:.4f}"

    def test_training_step_scalar_loss(self, tiny_model, batch_8x16):
        loss = tiny_model.training_step(batch_8x16, 0)
        assert loss.ndim == 0 and loss.item() > 0

    def test_lat_weights_shape(self, tiny_model):
        """lat_weights should cover all N = H*W nodes."""
        assert tiny_model.lat_weights.shape == (8 * 16,)

    def test_lat_weights_equatorial_heavier(self, tiny_model):
        """
        cos(0°) = 1 > cos(90°) = 0, so equatorial weights must exceed polar.
        After normalisation to mean=1, equatorial rows still have higher weights.
        """
        lw = tiny_model.lat_weights.view(8, 16)  # (H, W)
        # Row 0 = 90°N (pole), row H//2 = ~0° (equator)
        assert lw[8 // 2].mean() > lw[0].mean()

    def test_lat_weights_from_h_array(self):
        """
        WeatherBenchDataset.lat_weights has shape (H,).
        The model should accept it and tile to (H*W,) without error.
        """
        H, W = 8, 16
        lats = np.linspace(90, -90, H, dtype=np.float32)
        lw_1d = np.cos(np.deg2rad(lats)).astype(np.float32)
        lw_1d /= lw_1d.mean()
        model = GraphWeatherModel(
            n_vars=2,
            hidden_dim=16,
            edge_dim=8,
            n_layers=1,
            spatial_shape=(H, W),
            k=4,
            lat_weights=lw_1d,
        )
        assert model.lat_weights.shape == (H * W,)

    def test_different_batch_sizes(self, tiny_model):
        """Batching via edge_index offsets must work for any B."""
        for B in [1, 2, 4]:
            x = torch.randn(B, 2, 8, 16)
            out = tiny_model(x)
            assert out.shape == (B, 2, 8, 16)

    def test_batch_graph_offsets(self, tiny_model):
        """
        For B=3, the batched edge_index should cover nodes [0, 3·N)
        with no out-of-bounds index.
        """
        B = 3
        N = tiny_model._graph.num_nodes
        E = tiny_model._graph.num_edges
        ei, ea = tiny_model._batch_graph(B)
        assert ei.shape == (2, B * E)
        assert ea.shape == (B * E, tiny_model.hparams.edge_dim)
        assert int(ei.min()) >= 0
        assert int(ei.max()) < B * N

    def test_configure_optimizers_returns_adamw(self, tiny_model):
        import lightning as L

        tiny_model.trainer = L.Trainer(
            max_epochs=10, fast_dev_run=True, enable_checkpointing=False
        )
        result = tiny_model.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert "lr_scheduler" in result

    def test_graph_node_count(self, tiny_model):
        assert tiny_model._graph.num_nodes == 8 * 16

    def test_buffers_share_device(self, tiny_model):
        """edge_index and node_pos must be on the same device."""
        assert tiny_model.edge_index.device == tiny_model.node_pos.device
