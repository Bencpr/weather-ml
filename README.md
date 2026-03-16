# weather-ml

Hands-on project on weather data and ML/GNN forecasting, built as preparation for working with [Anemoi](https://anemoi.readthedocs.io/en/latest/) (ECMWF's data-driven forecasting framework).

## Structure

```
src/weather_ml/
├── data/
│   ├── dataset.py       ERA5Dataset / ERA5DataModule (xarray → PyTorch)
│   ├── download.py      ERA5 download via earthkit-data (CDS API)
│   └── graph.py         Graph construction: KNN on sphere, node/edge features
├── models/
│   ├── gnn.py           Encoder-Processor-Decoder GNN (Anemoi-style)
│   ├── unet.py          2D UNet baseline
│   ├── linear.py        Per-pixel linear baseline
│   └── random_forest.py Random forest baseline
└── train.py             Hydra entry point

configs/
├── config.yaml          Defaults
├── data/
│   ├── era5_cds_dm.yaml         Full Lightning Datamodule config for ERA5 data
│   └── era5_sample_cds_dm.yaml  Sample Lightning Datamodule config for ERA5 data
├── model/
│   ├── gnn.yaml
│   ├── unet.yaml
│   └── linear_baseline.yaml
├── training/default.yaml
└── hardware/local.yaml
```

## Setup

```bash
uv venv -p 3.12

# Install (CPU-only torch, resolved automatically by uv)
uv sync --all-extras

# torch-cluster for faster KNN (optional — sklearn fallback is used otherwise)
# example: uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html

# ecCodes system library — required for reading GRIB files from ERA5
sudo apt install libeccodes-dev   # Ubuntu/WSL
brew install eccodes              # macOS
```

## Data

Data is downloaded from ERA5 reanalysis via [earthkit-data](https://github.com/ecmwf/earthkit-data) and the Copernicus Climate Data Store (CDS).

**One-time CDS setup:**
1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) (free)
2. Create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-KEY
```

**Download:**
```bash
# ~100MB — geopotential 500hPa + temperature 850hPa for 2020 at 5.625°
python -m weather_ml.data.download --years 2020 --grid 5.625 5.625

# Higher resolution (larger files)
python -m weather_ml.data.download --years 2018 2019 2020 --grid 1.0 1.0
```

Variables downloaded by default: `z500` (geopotential at 500 hPa) and `t850` (temperature at 850 hPa). Edit `DEFAULT_VARIABLES` in `data/download.py` to add more.

## Training

```bash
# GNN (Anemoi-style Encoder-Processor-Decoder)
python -m weather_ml.train model=gnn

# Small test
python -m weather_ml.train model=gnn \
  data.train_years="2020-01-01/2020-09-30" \
  data.val_years="2020-10-01/2020-11-30" \
  data.test_years="2020-12-01/2020-12-31"

# UNet baseline
python -m weather_ml.train model=unet

# Override any config value
python -m weather_ml.train model=gnn training.max_epochs=50 training.lr=5e-4

# Higher resolution data config
python -m weather_ml.train model=gnn data=era5_cds
```

## GNN architecture

`models/gnn.py` implements the Encoder-Processor-Decoder design from Anemoi's [anemoi-models](https://github.com/ecmwf/anemoi-models):

```
Data grid (B, C, H, W)
    ↓  reshape → nodes (B·N, C),  N = H·W
    ↓  concat positional features (sin/cos lat/lon)
    ↓  Encoder  [MLP: C+4 → hidden_dim]
    ↓  Processor  [N layers × GraphConvLayer with double residual]
    ↓  + skip connection
    ↓  Decoder  [MLP: hidden_dim → C]
    ↓  + residual  (model predicts corrections, not full state)
Output (B, C, H, W)
```

The graph is built in `data/graph.py` following [anemoi-graphs](https://github.com/ecmwf/anemoi-graphs) conventions:

- KNN in 3D Cartesian coordinates — handles date-line wraparound and pole singularity correctly
- Node features: `(sin_lat, cos_lat, sin_lon, cos_lon)`
- Edge features: normalised distance (unit = `grid_reference_distance`)
- Latitude weights: `cos(lat)` normalised to mean = 1

## Tests

```bash
uv run pytest
uv run pytest tests/test_gnn.py -v   # graph + GNN tests only
```

## Notebooks

| Notebook | Content |
|---|---|
| `01_xarray_basics.ipynb` | xarray fundamentals, ERA5 data exploration |
| `02_data_pipeline.ipynb` | Dataset, DataModule, batch shapes |
| `03_training_analysis.ipynb` | Loss curves, predictions vs ground truth |

## Dependencies

Key packages and their role in the Anemoi ecosystem:

| Package | Role |
|---|---|
| `earthkit-data` | ERA5 ingestion (same stack as anemoi-datasets) |
| `torch` + `lightning` | Model training (same as anemoi-training) |
| `hydra-core` | Config management (same as anemoi-training) |
| `torch-cluster` | Fast KNN for graph construction (same as anemoi-graphs) |
| `scikit-learn` | KNN fallback when torch-cluster is unavailable |
| `xarray` | NetCDF / Zarr data handling |
