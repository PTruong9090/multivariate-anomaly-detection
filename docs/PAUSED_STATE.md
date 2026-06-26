# Homelab TranAD Monitoring - Paused State

## Current Status

This project is currently paused at a working MVP stage

The system curently supports:

- Live Prometheus metric collection from media-server
- TranAD FastAPI scoring endpoint
- Prometheus exporter for model outputs
- Grafana dashboard for:
  - current anomaly status
  - anomaly ratio
  - anomalous timesteps
  - TranAD score vs threshold
  - raw CPU, memory, network receive, and network transmit metrics
  - API/exporter reliability


## Current Live Architecture

Prometheus/node-exporter metrics → anomaly exporter → TranAD API → Prometheus custom metrics → Grafana dashboard

## Current Features

Model input features:

- media_cpu_percent
- media_memory_percent
- media_network_receive_bytes
- media_network_transmit_bytes

## Known Limitations

- NAS/Unraid disk metrics are not included in the trained model yet
- Model trained only on media-server behavior
- Threshold may need recalibration after more real-world data

## Next Steps When Resuming

1. Add anomaly event logging to CSV or SQLite
2. Add Unraid/NAS disk read, disk write, filesytem usage, temperature, and network metrics
3. Retrain model with expanded homelab feature set
