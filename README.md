# Homelab Prometheus Multivariate Anomaly Detection

Forked from Striim Labs' TranAD Multivariate Anomaly Detection prototype and adapted to real Prometheus telemetry from a self-hosted media server.

This project uses TranAD, a transformer-based multivariate time-series anomaly detection model, to learn normal server behavior from Prometheus metrics and flag abnormal CPU, memory, and network patterns.

The project extends the original repository with a new data source, a custom Prometheus preprocessing pipeline, homelab-specific model artifacts, synthetic anomaly evaluation, and a FastAPI scoring endpoint for on-demand telemetry windows.

## Overview

The original Striim Labs repository demonstrates multivariate anomaly detection on the Server Machine Dataset. My extension applies the same TranAD-based workflow to a fresh streaming-style domain: homelab infrastructure monitoring.

The final pipeline is:

```
Prometheus / Node Exporter
-> query_range csv_export
-> homelab preprocessing
-> train/test split
-> synthetic anomaly injection
-> train-set min/max normalization
-> TranAD training
-> POT threshold evaluation
-> FastAPI /score endpoint
```

The goal is to show how a research-style anomaly detection model can be adapted into a practical infrastructure monitoring workflow for real server telemetry.

## What I Built

I adapted the repo to monitor a self-hosted media server using Prometheus metrics.

Main additions and changes:
- Added a Prometheus metric export pipeline that queries `query_range` and saves telemetry to CSV
- Added a new homelab preprocessing script: `src/preprocess_homelab.py`
- Replaced the original SMD preprocessing flow with a Prometheus-based preprocessing flow
- Generated Striim-compatible `.npy` files for a new machine: `media-server`
- Trained TranAD on real media-server telemetry
- Injected controlled synthetic CPU, memory, and network anomalies into the test set
- Evaluated the model using POT thresholding
- Adapted the FastAPI scoring endpoint to support requests using `machine="media-server"`
- Preserved compatibility with the original SMD-style `store_id` / `device_id` request format

## Data Source: Prometheus Homelab Telemetry

Prometheus is running on my media server and collects Node Exporter metrics. I wrote a Python script that queries Prometheus using the `query_range` API and saves the resulting time-series data to:
`data/prometheus/raw/metrics.csv`

The current model uses four features:
```
FEATURE_COLUMNS = [
  "media_cpu_percent",
  "media_memory_percent",
  "media_network_receive_bytes",
  "media_network_transmit_bytes"
]
```
PromQL queries:
```
"media_cpu_percent": """
100 - (avg(rate(node_cpu_seconds_total{job="node",mode="idle"}[5m])) * 100)
"""

"media_memory_percent": """
(1 - (node_memory_MemAvailable_bytes{job="node"} / node_memory_MemTotal_bytes{job="node"})) * 100
"""

"media_network_receive_bytes": """
rate(node_network_receive_bytes_total{job="node",device="eth0"}[5m])
"""

"media_network_transmit_bytes": """
rate(node_network_transmit_bytes_total{job="node",device="eth0"}[5m])
"""
```
These features represent CPU utilization, memory pressure, network receive throughput, and network transmit throughput.

## Project Structure

```
multivariate-anomaly-detection/
│
├── code/                                  # Numbered scripts -- the canonical workflow
│   ├── 0_verify_setup.py                  # Download data, verify environment
│   ├── 1_train_model.py                   # Train TranAD model
│   ├── 2_evaluate_model.py                # Evaluate baseline or best, print metrics
│   ├── 3_streaming_app.py                 # FastAPI real-time scoring API (Docker)
│   └── 4_grid_sweep.py                    # Sweep hyperparams, retrain best to models/tranad/best/ (Not used)
│
├── notebooks/                             # Interactive walkthroughs (motivation + reasoning)
│   ├── data_exploration.ipynb             # EDA: 38-feature telemetry, anomaly patterns
│   └── model_design.ipynb                 # Architecture walkthrough, scoring, attribution
│
├── src/                                   # Reusable library code
│   ├── model.py                           # TranADConfig, TranADNet, transformer layers
│   ├── train.py                           # Shared training loop (used by 1_ and 4_)
│   ├── scorer.py                          # Scoring, POT thresholding, attribution
│   ├── preprocess.py                      # Data download, normalization, loading
│   ├── registry.py                        # Per-device model loading + caching
│   ├── spot.py                            # SPOT algorithm (extreme value theory)
│   ├── schemas.py                         # Pydantic v2 request/response models
│   └── utils.py                           # Sliding window, device selection
│
├── data/
|    ├── prometheus/
|      └── raw/
|        └── metrics.csv                   # Raw prometheus export
|    |
|    └── homelab/
|      └── processed/                      # gitignored
|        ├── media-server_train.npy
|        ├── media-server_test.npy
|        ├── media-server_test_labels.npy
|        ├── media-server_interp_labels.npy
|        ├── media-server_norm_params.npy
|        └── media-server_metadata.json
│
├── models/tranad/                         # Prebuilt reference (never overwritten)
│   └── media-server/                      # Trained homelab TranAD artifacts
│
├── samples/
│   ├── score_request_homelab.json # Normal API scoring request
│   └── score_request_homelab2.json # Anomalous API scoring request
|
├── striim/                                # Striim integration: TQL, Open Processor, build scripts
├── docker-compose.rest.yml                # Single-node REST deployment
├── docker-compose.demo.yml                # Two-cluster demo deployment
├── Dockerfile                             # Production container
├── pyproject.toml                         # Python dependencies
├── STRIIM.md                              # Striim pipeline setup guide
└── TECHNICAL.md                           # Detailed technical reference
```

## Preprocessing

The new preprocessing script is:
`src/preprocess_homelab.py`

It converts raw Prometheus CSV data into the same style of model-ready arrays expected by the original repository.

The script performs the following steps:
1. Reads `data/prometheus/raw/metrics.csv`
2. Parses timestamp
3. Sorts rows chronologically
4. Selects the four feature columns
5. Replaces `inf` with `NaN`
6. Interpolates missing values
7. Splits data into 70% train and 30% test
8. Injects controlled synthetic anomalies into test set
9. Normalizes train/test data using train-set min/max values
10. Saves Striim-compatible `.npy` files

Successful preprocessing output:
```
Saved processed homelab data to data/homelab/processed
Train shape: (2823, 4)
Test shape: (1210, 4)
Labels shape: (1210,)
Interpretation labels shape: (1210, 4)
```

Generated Files:
```
data/homelab/processed/media-server_train.npy
data/homelab/processed/media-server_test.npy
data/homelab/processed/media-server_test_labels.npy
data/homelab/processed/media-server_interp_labels.npy
data/homelab/processed/media-server_norm_params.npy
data/homelab/processed/media-server_metadata.json
```

## Synthetic Anomaly Evaluation

Because this homelab data does not yet include labeled real incidents, I created a controlled evaluation setup by injecting synthetic anomalies into the test split.

Injected anomalies:
| Anomaly Type    | Feature                | Injection                     |
| --------------- | ---------------------- | ----------------------------- |
| CPU spike       | `media_cpu_percent`    | multiplied by 3 for 12 rows   |
| Memory pressure | `media_memory_percent` | multiplied by 1.6 for 24 rows | 
| Network burst   | receive/transmit bytes | multiplied by 10 for 18 rows  |

Total injected anomalous rows: `54`

This creates a way to test whether TranAD can detect abnormal deviations from normal media-server behavior

## Training

The model was trained on:
`data/homelab/processed/media-server_train.npy`

Training configuration:
```
Machine: media-server
Training samples: 2823
Features: 4
Epochs: 30
Device: CPU
```

Training loss improved significantly:
| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1     | 0.213795   | 0.133830        |
| 30    | 0.006765   | 0.002150        |

This indicates that TranAD learned to reconstruct normal media-server behavior.

Model artifacts are saved under:
`models/tranad/media-server/`

## Evaluation

Evaluation command:

```
uv run py .\code\2_evaluate_model.py --machine media-server --data-dir .\data\homelab\processed
```

Evaluation Loaded:

```
Loaded model for media-server: 4 features, window_size=10
Data: train=(2823, 4), test=(1210, 4), labels=(1210, 4)
```

The model used POT thresholding:

```text
Threshold: 0.313845
```

Final evaluation results:

| Metric    |  Value |
| --------- | -----: |
| F1        | 0.8485 |
| Precision | 0.9333 |
| Recall    | 0.7778 |
| ROC/AUC   | 0.8876 |

Confusion matrix:

| Result          | Count |
| --------------- | ----: |
| True Positives  |    42 |
| True Negatives  |  1153 |
| False Positives |     3 |
| False Negatives |    12 |

Since I injected 54 anomalous rows total, the model detected 42 out of 54 anomalous rows with only 3 false positives.

This is a strong first result for the homelab adaptation because the model achieved high precision while still detecting most of the injected CPU, memory, and network anomalies.

Evaluation outputs are saved to:

```text
models/tranad/media-server/eval_results.json
models/tranad/media-server/attribution_results.json
```

## Feature Mapping

The model uses four input dimensions:

| Dimension | Prometheus Feature             |
| --------- | ------------------------------ |
| `dim_0`   | `media_cpu_percent`            |
| `dim_1`   | `media_memory_percent`         |
| `dim_2`   | `media_network_receive_bytes`  |
| `dim_3`   | `media_network_transmit_bytes` |

## Attribution Results

The evaluation found three anomaly segments:

| Segment     | Main Attribution          | Interpretation               |
| ----------- | ------------------------- | ---------------------------- |
| `[545-568]` | `dim_1`                   | memory pressure              |
| `[848-865]` | `dim_3`                   | network transmit burst       |
| `[868-868]` | `dim_2`, `dim_0`, `dim_3` | mixed network / CPU activity |

The model clearly detected the memory pressure anomaly and network burst anomaly. The CPU spike was harder to detect, which likely explains most of the false negatives.

## FastAPI Scoring Demo

I also adapted the repository's FastAPI scoring app:

```text
code/3_streaming_app.py
```

The app now loads the homelab model and processed data directories:

```text
Model dir: models/tranad
Data dir: data/homelab/processed
```

Start the API server:

```powershell
uv run python code/3_streaming_app.py
```

Successful startup:

```text
Starting TranAD Anomaly Detection Server
Device: cpu
Model dir: models/tranad
Data dir: data/homelab/processed
Available devices: ['media-server']
Uvicorn running on http://0.0.0.0:8000
```

The original repository expected SMD-style identifiers:

```json
{
  "store_id": 1,
  "device_id": 1,
  "data": [...]
}
```

I updated the scoring schema so the endpoint also supports homelab-style machine identifiers:

```json
{
  "machine": "media-server",
  "data": [...]
}
```

This allows `/score` to work with the original SMD request format and the new homelab request format.

## Normal API Request

Sample file:

```text
samples/score_request_homelab.json
```

Run:

```powershell
curl.exe -X POST "http://localhost:8000/score" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/score_request_homelab.json" | python -m json.tool
```

Result summary:

```json
{
  "machine": "media-server",
  "store_id": 0,
  "device_id": 0,
  "timestamp": "demo-normal",
  "filename": "normal_prometheus_like_sample",
  "n_timesteps": 10,
  "n_features": 4,
  "threshold": 0.313844674971563,
  "n_anomalies": 0,
  "anomaly_ratio": 0.0,
  "anomaly_segments": [],
  "scoring_mode": "phase2_only",
  "threshold_method": "pot"
}
```

The normal Prometheus-like telemetry window stayed below the learned POT threshold and returned zero anomalies.

## Anomalous API Request

Sample file:

```text
samples/score_request_homelab_anomaly.json
```

Run:

```powershell
curl.exe -X POST "http://localhost:8000/score" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/score_request_homelab_anomaly.json" | python -m json.tool
```

Result summary:

```json
{
  "machine": "media-server",
  "store_id": 0,
  "device_id": 0,
  "timestamp": "demo-anomaly",
  "filename": "prometheus_like_with_spike",
  "n_timesteps": 10,
  "n_features": 4,
  "threshold": 0.313844674971563,
  "n_anomalies": 3,
  "anomaly_ratio": 0.3,
  "scoring_mode": "phase2_only",
  "threshold_method": "pot"
}
```

The anomalous request produced one detected anomaly segment:

```json
{
  "segment_start": 6,
  "segment_end": 8,
  "segment_length": 3,
  "peak_score": 0.479451,
  "peak_timestamp": 8,
  "mean_score": 0.439586
}
```

Per-timestep anomaly scores:

| Timestep |    Score | Anomaly |
| -------: | -------: | ------- |
|        6 | 0.372404 | true    |
|        7 | 0.466903 | true    |
|        8 | 0.479451 | true    |

This demonstrates that the API can distinguish a normal telemetry window from a modified anomalous telemetry window.

## Reproducing the Homelab Workflow

### 1. Install dependencies

```powershell
uv sync
```

### 2. Export Prometheus metrics

Generate the raw Prometheus CSV:

```text
data/prometheus/raw/metrics.csv
```

### 3. Preprocess homelab data

```powershell
uv run python src/preprocess_homelab.py
```

### 4. Train TranAD

```powershell
uv run python code/1_train_model.py --machine media-server --data-dir data/homelab/processed
```

### 5. Evaluate TranAD

```powershell
uv run py .\code\2_evaluate_model.py --machine media-server --data-dir .\data\homelab\processed
```

### 6. Start the API server

```powershell
uv run python code/3_streaming_app.py
```

### 7. Score a normal request

```powershell
curl.exe -X POST "http://localhost:8000/score" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/score_request_homelab.json" | python -m json.tool
```

### 8. Score an anomalous request

```powershell
curl.exe -X POST "http://localhost:8000/score" `
  -H "Content-Type: application/json" `
  --data-binary "@samples/score_request_homelab_anomaly.json" | python -m json.tool
```

## Detection Methodology

TranAD is a reconstruction-based anomaly detection model for multivariate time series. The model learns to reconstruct normal telemetry windows. When a new window is difficult to reconstruct, the reconstruction error increases. High reconstruction error is treated as anomalous behavior.

This project uses POT thresholding to convert continuous anomaly scores into binary anomaly labels. Instead of using a fixed manual threshold, POT calibrates the threshold from the tail of the score distribution.

The scoring API returns:

* Per-timestep anomaly scores
* Binary anomaly decisions
* Contiguous anomaly segments
* Peak anomaly score
* Mean segment score
* Per-dimension attribution

This makes the output useful not only for detecting that something is wrong, but also for identifying which metric likely caused the anomaly.

## Limitations

Current limitations:

* The evaluation uses synthetic anomalies because I do not yet have labeled real homelab incidents.
* The current model is trained on one machine: `media-server`.
* The current feature set only includes CPU, memory, network receive, and network transmit metrics.
* The API performs on-demand scoring, but it is not yet connected to Kafka or Spark.
* CPU spike detection was weaker than memory and network anomaly detection.
* The model was trained on a limited amount of Prometheus history. (2 weeks)

## Future Work

Future improvements:

* Add disk I/O and filesystem metrics
* Add Unraid NAS metrics once enough historical data is available
* Add Docker/container-level metrics
* Add Grafana dashboards for visualizing anomaly scores
* Compare TranAD against simpler baselines such as Isolation Forest
* Connect Prometheus polling to Kafka
* Add Spark Structured Streaming for continuous processing
* Collect real incident labels from homelab outages or performance issues

## License

BSD-3-Clause. See `LICENSE`.

