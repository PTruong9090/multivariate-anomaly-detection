from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "media_cpu_percent",
    "media_memory_percent",
    "media_network_receive_bytes",
    "media_network_transmit_bytes",
]

def minmax_normalize(train: np.ndarray, test: np.ndarray):
    min_vals = np.min(train, axis=0)
    max_vals = np.max(train, axis=0)

    train_norm = (train - min_vals) / (max_vals - min_vals + 1e-4)
    test_norm = (test - min_vals) / (max_vals - min_vals + 1e-4)

    return train_norm, test_norm, min_vals, max_vals

def inject_synthetic_anomalies(test_df: pd.DataFrame):
    '''
    Adds simple snythethic anomalies so evaluation has labels
        Returns:
            modified test dataframe
            test_labels: shape(N,)
            interp_labels: shape (N, num_features)
    '''

    test_df = test_df.copy()

    n = len(test_df)
    f = len(FEATURE_COLUMNS)

    test_labels = np.zeros(n)
    interp_labels = np.zeros((n, f))

    # CPU spike anomaly
    start = int(n * 0.20)
    end = start + 12
    test_df.loc[test_df.index[start:end], 'media_cpu_percent'] *= 3
    test_df["media_cpu_percent"] = test_df["media_cpu_percent"].clip(upper=100)
    test_labels[start:end] = 1
    interp_labels[start:end, FEATURE_COLUMNS.index("media_cpu_percent")] = 1

    # Memory pressure anomaly
    start = int(n * 0.45)
    end = start + 24
    test_df.loc[test_df.index[start:end], 'media_memory_percent'] *= 1.6
    test_df["media_memory_percent"] = test_df["media_memory_percent"].clip(upper=100)
    test_labels[start:end] = 1
    interp_labels[start:end, FEATURE_COLUMNS.index("media_memory_percent")] = 1

    # Network burst anomaly
    start = int(n * 0.70)
    end = start + 18
    test_df.loc[test_df.index[start:end], 'media_network_receive_bytes'] *= 10
    test_df.loc[test_df.index[start:end], 'media_network_transmit_bytes'] *= 10
    test_labels[start:end] = 1
    interp_labels[start:end, FEATURE_COLUMNS.index('media_network_receive_bytes')] = 1
    interp_labels[start:end, FEATURE_COLUMNS.index('media_network_transmit_bytes')] = 1

    return test_df, test_labels, interp_labels

def preprocess_homelab(
        csv_path: str = "data/prometheus/raw/metrics.csv",
        output_dir: str = "data/homelab/processed",
        machine_name: str = "media-server",
):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df[['timestamp'] + FEATURE_COLUMNS].copy()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].interpolate(limit_directions='both')
    df = df.dropna()

    # Split into train/test
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    test_df, test_labels, interp_labels = inject_synthetic_anomalies(test_df)

    train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    test_raw = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    train_norm, test_norm, min_vals, max_vals = minmax_normalize(train_raw, test_raw)

    np.save(output_dir / f"{machine_name}_train.npy", train_norm)
    np.save(output_dir / f"{machine_name}_test.npy", test_norm)
    np.save(output_dir / f"{machine_name}_test_labels.npy", test_labels)
    np.save(output_dir / f"{machine_name}_interp_labels.npy", interp_labels)
    np.save(output_dir / f"{machine_name}_norm_params.npy", np.stack([min_vals, max_vals]))

    metadata = {
        "machine_name": machine_name,
        "n_train_rows": len(train_norm),
        "n_test_rows": len(test_norm),
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "source_csv": str(csv_path),
    }

    pd.Series(metadata).to_json(output_dir / f"{machine_name}_metadata.json", indent=2)

    print(f"Saved processed homelab data to {output_dir}")
    print(f"Train shape: {train_norm.shape}")
    print(f"Test shape: {test_norm.shape}")
    print(f"Labels shape: {test_labels.shape}")
    print(f"Interpretation labels shape: {interp_labels.shape}")


if __name__ == "__main__":
    preprocess_homelab()