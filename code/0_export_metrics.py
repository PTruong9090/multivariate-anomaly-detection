import requests
import time
import pandas as pd
from functools import reduce 
import os
from dotenv import load_dotenv


load_dotenv()
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL")

metrics = {
    "media_cpu_percent": '''100 - (avg(rate(node_cpu_seconds_total{job="node",mode="idle"}[5m])) * 100)''',

    "media_memory_percent": '''(1 - (node_memory_MemAvailable_bytes{job="node"} / node_memory_MemTotal_bytes{job="node"})) * 100''',

    "media_network_receive_bytes": '''rate(node_network_receive_bytes_total{job="node",device="eth0"}[5m])''',

    "media_network_transmit_bytes": '''rate(node_network_transmit_bytes_total{job="node",device="eth0"}[5m])''',

    # "unraid_disk_read_bytes": '''sum(rate(node_disk_read_bytes_total{job="unraid"}[5m]))''',

    # "unraid_disk_write_bytes": '''sum(rate(node_disk_written_bytes_total{job="unraid"}[5m]))''',

    # "unraid_disk_usage_percent": '''(1 - (node_filesystem_avail_bytes{job="unraid",mountpoint="/mnt/disk1"} / node_filesystem_size_bytes{job="unraid",mountpoint="/mnt/disk1"})) * 100''',
}


end = int(time.time())
start = end - 14 * 24 * 60 * 60


def query_metric(name, query, start, end, step='5m'):
    res = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
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

def main():
    dfs = []

    for name, promql in metrics.items():
        print(f"Querying {name}...")
        df = query_metric(name, promql, start, end)
        
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

    merged.to_csv("./data/prometheus/raw/metrics.csv", index=False)

    print(merged.head())
    print("Saved metrics.csv")

if __name__ == "__main__":
    main()