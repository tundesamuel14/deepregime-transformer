from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


REGIME_PATH = Path("data/regimes/spy_vix_regimes_kmeans.parquet")


class RegimeSequenceDataset(Dataset):
    """
    Goal: take the regime-labeled time series and turn it into
    sliding windows that the Transformer can learn from.

    Each item:
      X: (seq_len, num_features)  - past seq_len days of features
      y: scalar int               - regime label on the last day
    """

    def __init__(
        self,
        seq_len: int = 60,
        feature_cols: List[str] = None,
        train: bool = True,
        train_fraction: float = 0.8,
    ):
        df = pd.read_parquet(REGIME_PATH)

        # 🔧 IMPORTANT: flatten MultiIndex columns if present
        # Your columns look like ('ret_1d', ''), ('regime_kmeans', ''), ...
        # This turns them into 'ret_1d', 'regime_kmeans', ...
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Default features if not provided
        if feature_cols is None:
            feature_cols = ["ret_1d", "ret_5d", "ret_20d", "rv_5d", "rv_20d", "vix_level"]

        self.feature_cols = feature_cols
        self.seq_len = seq_len

        # Sanity check: make sure all required columns exist
        missing = [c for c in feature_cols + ["regime_kmeans"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns in regime DataFrame: {missing}")

        # Drop rows with missing features/labels
        df = df.dropna(subset=feature_cols + ["regime_kmeans"])

        # Standardize features (z-score) using all data in this split
        self.feature_means = df[feature_cols].mean()
        self.feature_stds = df[feature_cols].std().replace(0, 1.0)
        df[feature_cols] = (df[feature_cols] - self.feature_means) / self.feature_stds

        # Convert to numpy arrays
        features = df[feature_cols].values.astype(np.float32)
        labels = df["regime_kmeans"].values.astype(np.int64)

        # Time-based train/val split
        n = len(df)
        split_idx = int(n * train_fraction)
        if train:
            features = features[:split_idx]
            labels = labels[:split_idx]
        else:
            features = features[split_idx:]
            labels = labels[split_idx:]

        self.features = features
        self.labels = labels

        # Number of sliding windows
        self.num_windows = len(self.features) - self.seq_len + 1
        if self.num_windows <= 0:
            raise ValueError("Not enough data for the given seq_len.")

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        idx = starting index of the window in the time series.
        We return a window of length seq_len and the label at the last day.
        """
        start = idx
        end = idx + self.seq_len

        x_window = self.features[start:end]  # (seq_len, num_features)
        y_label = self.labels[end - 1]       # scalar

        x_tensor = torch.from_numpy(x_window)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        return x_tensor, y_tensor


def create_dataloaders(
    seq_len: int = 60,
    batch_size: int = 32,
    feature_cols: List[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Helper to create train and validation DataLoaders.
    """
    train_dataset = RegimeSequenceDataset(
        seq_len=seq_len,
        feature_cols=feature_cols,
        train=True,
    )
    val_dataset = RegimeSequenceDataset(
        seq_len=seq_len,
        feature_cols=feature_cols,
        train=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
