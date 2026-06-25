import yaml
import requests
import json
import time
import pandas as pd
from functools import reduce 
import os
import argparse

DEFAULT_CONFIG_PATH = "./configs/homelab_live.yaml"


def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

            if not config:
                print(f"Config file: '{file_path}' is empty")
                return None
        
            return config
        
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        return None
        
    except yaml.YAMLError as exc:
        print(f"Syntax Error: Failed to parse YAML structure")
        print(f"    Details: {exc}")
        return None


def get_feature_items(config):
    '''
    Supports either:

    features:
        media_cpu_percent: promql

    or:

    features:
        - name: media_cpu_percent
          query: promql
    '''

    features = config["features"]

    if isinstance(features, dict):
        return list(features.items())
    
    if isinstance(features, list):
        items = []
        for feature in features:
            name = feature['name']
            promql = feature['query']

            if not promql:
                raise ValueError(f"Feature '{name}' is missing query/promql")
            
            items.append((name, promql))
        
        return items
    
    raise TypeError("config['features'] must be a dict or a list")


def query_metric(prometheus_url, name, query, start, end, step='5m'):
    res = requests.get(
        f"{prometheus_url}/api/v1/query_range",
            params = {
                "query": query,
                "start": start,
                "end": end,
                "step": step,
            },
            timeout=30,
    )

    res.raise_for_status()
    data = res.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed for {name}: {data}")
    
    result = data["data"]["result"]

    if not result:
        print(f"[WARN] No data returned for {name}")
        return None

    values = result[0]["values"]
    
    df = pd.DataFrame(values, columns=["timestamp", name])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True)
    df[name] = pd.to_numeric(df[name], errors="coerce")

    return df

def query_recent_metrics(config):
    prometheus_url = config['prometheus_url']
    features = config['features']
    lookback_minutes = int(config.get("lookback_mins", 120))
    step = config.get('step', '5m')

    end = int(time.time())
    start = end - lookback_minutes * 60

    dfs = []

    for name, promql in get_feature_items(config):
        print(f"Querying {name}...")

        df = query_metric(
            prometheus_url, 
            name, 
            promql, 
            start, 
            end,
            step
        )
        
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No metrics were exported. Check Prometheus and PromQL queries.")

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="inner"),
        dfs
    )

    merged = merged.sort_values("timestamp")
    merged = merged.dropna()

    os.makedirs("./data/prometheus/raw", exist_ok=True)
    merged.to_csv("./data/prometheus/raw/latest_metrics.csv", index=False)

    print("Saved metrics.csv")

    return merged


def build_latest_window(config, df):
    features_col = [name for name, _ in get_feature_items(config)]
    window_size = int(config.get('window_size', 10))

    complete_df = df.dropna(subset=features_col)

    if len(complete_df) < window_size:
        raise RuntimeError(f"Need at least 10 complete rows, only found {len(complete_df)}")
    
    window_df = complete_df.tail(window_size)
    window = window_df[features_col].astype(float).values.tolist()

    print("Latest scoring window:")
    print(window_df)

    return window_df, window


def score_window(config, window_df, window):
    score_url = config.get('score_url', "http://localhost:8000/score")

    payload = {
        "machine": config.get('machine', 'media-server'),
        "store_id": int(config.get("store_id", 1)),
        "device_id": int(config.get("device_id", 1)),
        "timestamp": str(window_df["timestamp"].iloc[-1]),
        "filename": "live_prometheus_window",
        "data": window,
        "include_attribution": True,
        "include_per_timestep": True,
        "scoring_mode": config.get("scoring_mode", "phase2_only"),
    }

    res = requests.post(score_url, json=payload, timeout=30)

    if res.status_code != 200:
        print("Status code:", res.status_code)
        print("Response text:")
        print(res.text)
        res.raise_for_status()

    return res.json()


def print_score_summary(res):
    timestamp = res.get('timestamp')
    n_anomalies = res.get('n_anomalies', 0)
    anomaly_ratio = res.get('anomaly_ratio', 0.0)
    threshold = res.get('threshold')

    per_timestep = res.get('per_timestep', [])
    scores = [point['score'] for point in per_timestep] if per_timestep else []
    max_score = max(scores) if scores else None

    print()
    print("=" * 60)

    if n_anomalies > 0:
        print("ALERT: anomaly detected")

    else:
        print("NORMAL: no anomaly detected")

    print(f"Timestamp:      {timestamp}")
    print(f"Anomalies:      {n_anomalies}")
    print(f"Anomaly ratio:  {anomaly_ratio}")
    print(f"Threshold:      {threshold}")

    if max_score is not None:
        print(f"Max score:      {max_score}")

    if res.get('anomaly_segments'):
        print(f"Segments:       {res['anomaly_segments']}")

    if res.get('dimension_means'):
        print(f"Dimension means: {res['dimension_means']}")

    print()
    print("=" * 60)


def run_pipeline(config):
    df = query_recent_metrics(config)
    window_df, window = build_latest_window(config, df)
    res = score_window(config, window_df, window)

    return window_df, res


def run_once(config):
    window_df, res = run_pipeline(config)
    print_score_summary(res)

    return res


def main():
    parser = argparse.ArgumentParser(description="Live Prometheus anomaly exporter")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to homelab live config YAML",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously score every interval"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Loop interval in seconds",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON response",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    
    interval = args.interval or int(config.get("interval_seconds", 300))

    if args.loop:
        while True:
            try:
                res = run_once(config)

                if args.print_json:
                    print(json.dumps(res, indent=4))

            except Exception as exc:
                print(f"[ERROR] {exc}")

            time.sleep(interval)

    else:
        res = run_once(config)

        if args.print_json:
            print(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
