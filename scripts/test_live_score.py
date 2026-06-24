import yaml
import requests
import json
import time
import pandas as pd
from functools import reduce 
import os

PATH = './configs/homelab_live.yaml'
END = int(time.time())
START = END - 14 * 24 * 60 * 60

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


def query_all_metrics(config, start, end):
    prometheus_url = config['prometheus_url']
    features = config['features']

    dfs = []

    for feature in features:
        name = feature['name']
        promql = feature['query']

        print(f"Querying {name}...")

        df = query_metric(prometheus_url, name, promql, start, end)
        
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
    merged.to_csv("./data/prometheus/raw/metrics.csv", index=False)

    print(merged.head())
    print("Saved metrics.csv")

    return merged


def score_latest_window(config, df):
    features_col = [feature['name'] for feature in config['features']]
    score_url = config['score_url']

    complete_df = df.dropna(subset=features_col)

    if len(complete_df) < 10:
        raise RuntimeError(f"Need at least 10 complete rows, only found {len(complete_df)}")
    
    window_df = complete_df.tail(10)

    window = window_df[features_col].astype(float).values.tolist()

    # print(str(window_df["timestamp"].iloc[-1]))

    payload = {
        "machine": config['machine'],
        "store_id": 1,
        "device_id": 1,
        "timestamp": str(window_df["timestamp"].iloc[-1]),
        "filename": "live_prometheus_window",
        "data": window,
        "include_attribution": True,
        "include_per_timestep": True,
        "scoring_mode": "phase2_only"
    }

    response = requests.post(score_url, json=payload, timeout=30)
    response.raise_for_status()

    res = response.json()

    print(json.dumps(res, indent=4))
    return res


if __name__ == "__main__":
    config = load_config(PATH)
    if config:
        print("Config loaded successfully!")

    else:
        print("Issue loading config")
        exit()

    df = query_all_metrics(config, START, END)
    print("testing \n", df)
    score_latest_window(config, df)






