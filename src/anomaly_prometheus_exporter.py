import argparse
import time
from datetime import datetime, timezone
import json

from prometheus_client import Gauge, Counter, start_http_server

from anomaly_exporter import (
    DEFAULT_CONFIG_PATH,
    load_config,
    get_feature_items,
    run_pipeline,
    print_score_summary,
)

FEATURE_COLUMNS = [
    "media_cpu_percent",
    "media_memory_percent",
    "media_network_receive_bytes",
    "media_network_transmit_bytes",
]

TRANAD_SCORE = Gauge(
    "homelab_tranad_score",
    "Maximum TranAD anomaly score for the latest telemetry window",
    ["machine"],
)

TRANAD_THRESHOLD = Gauge(
    "homelab_tranad_threshold",
    "TranAD anomaly threshold loaded from scorer state",
    ["machine"],
)

TRANAD_ANOMALY = Gauge(
    "homelab_tranad_anomaly",
    "Whether the latest telemetry window is anomalous: 1=yes, 0=no",
    ["machine"],
)

TRANAD_ANOMALY_RATIO = Gauge(
    "homelab_tranad_anomaly_ratio",
    "Fraction of anomalous timesteps in the latest telemetry window",
    ["machine"],
)

TRANAD_ANOMALIES = Gauge(
    "homelab_tranad_anomalies",
    "Number of anomalous timesteps in the latest telemetry window",
    ["machine"],
)

TRANAD_API_UP = Gauge(
    "homelab_tranad_scoring_api_up",
    "Whether the FastAPI scoring endpoint is reachable: 1=yes, 0=no",
    ["machine"],
)

TRANAD_LAST_SUCCESS = Gauge(
    "homelab_tranad_last_success_timestamp_seconds",
    "Unix timestamp of the last successful live anomaly score",
    ["machine"],
)

FEATURE_MEAN = Gauge(
    "homelab_feature_mean",
    "Mean feature value in the latest telemetry window",
    ["machine", "feature"],
)

FEATURE_LATEST = Gauge(
    "homelab_feature_latest",
    "Latest feature value in the latest telemetry window",
    ["machine", "feature"],
)

EXPORTER_FAILURES = Counter(
    "homelab_tranad_exporter_failures_total",
    "Total exporter failures",
    ["machine", "stage"],
)

def get_max_score(score_res):
    per_timestep = score_res.get('per_timestep', [])

    scores = []

    for point in per_timestep:
        if isinstance(point, dict) and point.get('score') is not None:
            scores.append(float(point['score']))

        elif isinstance(point, (int, float)):
            scores.append(float(point))

    return max(scores) if scores else 0.0


def update_metrics(config, window_df, score_res):
    machine = config.get('machine', 'media-server')
    feature_cols = [name for name, _ in get_feature_items(config)]

    max_score = get_max_score(score_res)
    threshold = float(score_res.get('threshold', 0))
    n_anomalies = int(score_res.get('n_anomalies', 0))
    anomaly_ratio = float(score_res.get('anomaly_ratio', 0))

    TRANAD_SCORE.labels(machine=machine).set(max_score)
    TRANAD_THRESHOLD.labels(machine=machine).set(threshold)
    TRANAD_ANOMALY.labels(machine=machine).set(1 if n_anomalies > 0 else 0)
    TRANAD_ANOMALY_RATIO.labels(machine=machine).set(anomaly_ratio)
    TRANAD_ANOMALIES.labels(machine=machine).set(n_anomalies)

    TRANAD_API_UP.labels(machine=machine).set(1)
    TRANAD_LAST_SUCCESS.labels(machine=machine).set(time.time())

    for feature in feature_cols:
        FEATURE_MEAN.labels(machine=machine, feature=feature).set(
            float(window_df[feature].mean())
        )

        FEATURE_LATEST.labels(machine=machine, feature=feature).set(
            float(window_df[feature].iloc[-1])
        )


def run_export_once(config, print_json=False, quiet=False):
    window_df, score_res = run_pipeline(config)

    update_metrics(config, window_df, score_res)

    if not quiet:
        print_score_summary(score_res)

    if print_json:
        print(json.dumps(score_res, indent=4))

    return score_res


def main():
    parser = argparse.ArgumentParser(
        description="Expose live TranAD anomaly scores as prometheus metrics"
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to homelab live config YAML"
    )

    parser.add_argument(
        "metrics-port",
        default=9108,
        help="Port for Prometheus /metrics endpoint"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Scoring interval in seconds"
    )

    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full score response each loop"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print score summary each loop"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if config is None:
        raise RuntimeError(f"Failed to load config: {args.config}")
    
    machine = config.get("machine", "media-server")
    interval = args.interval or int(config.get("interval_seconds", 300))

    start_http_server(args.metric_port, addr=args.listen_addr)

    print("=" * 60)
    print("TranAD Prometheus exporter started")
    print(f"Metrics endpoint: http://{args.listen_addr}:{args.metrics_port}/metrics")
    print(f"Machine:        {machine}")
    print(f"Interval:       {interval} seconds")
    print("=" * 60)

    while True:
        try:
            print()
            print(f"[{datetime.now(timezone.utc).isoformat()}] Running live score...")

            run_export_once(
                config=config,
                print_json=args.print_json,
                quiet=args.quiet,
            )

        except Exception as exc:
            TRANAD_API_UP.labels(machine=machine).set(0)
            EXPORTER_FAILURES.labels(machine=machine, stage="live_score").inc()

            print(f"[Error] {exc}")

        time.sleep(interval)


if __name__ == "__main__":
    main()