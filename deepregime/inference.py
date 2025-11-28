from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from deepregime.models.transformer_regime import RegimeTransformer

REGIME_PATH = Path("data/regimes/spy_vix_regimes_kmeans.parquet")


def prepare_inference_data(
    seq_len: int = 60,
    feature_cols: List[str] = None,
    train_fraction: float = 0.8,
):
    """
    Turn the full regime-labeled time series into sliding windows
    that the model can run on for inference.

    Returns:
      X_windows: (num_windows, seq_len, num_features)
      y_labels: (num_windows,) - true regime for each window
      dates:    (num_windows,) - date of last day in each window
    """
    df = pd.read_parquet(REGIME_PATH)

    # 🔧 1) FLATTEN MultiIndex columns IF present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Default features = same ones we used in training
    if feature_cols is None:
        feature_cols = ["ret_1d", "ret_5d", "ret_20d", "rv_5d", "rv_20d", "vix_level"]

    # 🔧 Sanity check: do we actually have these columns?
    required = feature_cols + ["regime_kmeans"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("\n[prepare_inference_data] Available columns:")
        print(list(df.columns))
        raise KeyError(f"Missing expected columns: {missing}")

    # If there's a 'date' column, use it as index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.sort_index()

    # Drop rows with missing data
    df = df.dropna(subset=required)

    # Use train portion only to compute mean/std, like in training
    n = len(df)
    split_idx = int(n * train_fraction)
    df_train = df.iloc[:split_idx]

    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1.0)

    # Standardize full history
    df[feature_cols] = (df[feature_cols] - means) / stds

    features = df[feature_cols].values.astype(np.float32)
    labels = df["regime_kmeans"].values.astype(np.int64)
    dates = df.index.to_numpy()

    # Build sliding windows
    windows = []
    window_labels = []
    window_dates = []

    num_windows = len(features) - seq_len + 1
    if num_windows <= 0:
        raise ValueError("Not enough data for the given seq_len.")

    for start in range(num_windows):
        end = start + seq_len
        x_window = features[start:end]
        y_label = labels[end - 1]
        y_date = dates[end - 1]

        windows.append(x_window)
        window_labels.append(y_label)
        window_dates.append(y_date)

    X_windows = np.stack(windows, axis=0)
    y_labels = np.array(window_labels)
    dates = np.array(window_dates)

    return X_windows, y_labels, dates


def load_trained_model(
    num_features: int,
    num_regimes: int,
    state_path: str = "best_regime_transformer.pt",
    device: torch.device = None,
) -> RegimeTransformer:
    """
    Rebuild the same model architecture and load trained weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RegimeTransformer(
        num_features=num_features,
        num_regimes=num_regimes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference_last_n(
    n_last: int = 100,
    seq_len: int = 60,
    feature_cols: List[str] = None,
    train_fraction: float = 0.8,
):
    """
    Return a clean DataFrame for the last `n_last` windows with:
      - date
      - true_regime
      - pred_regime
      - prob_regime_* columns
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_windows, y_labels, dates = prepare_inference_data(
        seq_len=seq_len,
        feature_cols=feature_cols,
        train_fraction=train_fraction,
    )

    num_windows, seq_len_real, num_features = X_windows.shape
    num_regimes = int(y_labels.max()) + 1

    model = load_trained_model(
        num_features=num_features,
        num_regimes=num_regimes,
        device=device,
    )

    # Take last n_last windows
    n_last = min(n_last, num_windows)
    X_last = X_windows[-n_last:]
    y_last = y_labels[-n_last:]
    dates_last = dates[-n_last:]

    X_tensor = torch.from_numpy(X_last).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)

    data = {
        "date": dates_last,
        "true_regime": y_last,
        "pred_regime": preds,
    }

    for k in range(num_regimes):
        data[f"prob_regime_{k}"] = probs[:, k]

    df_out = pd.DataFrame(data)
    return df_out
