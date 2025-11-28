from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from deepregime.models.transformer_regime import RegimeTransformer
from deepregime.inference import prepare_inference_data, load_trained_model


def run_inference_full(
    seq_len: int = 60,
    feature_cols: List[str] = None,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    """
    Run the trained model on ALL available windows and return:

      date, true_regime, pred_regime, prob_regime_*
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_windows, y_labels, dates = prepare_inference_data(
        seq_len=seq_len,
        feature_cols=feature_cols,
        train_fraction=train_fraction,
    )

    num_windows, _, num_features = X_windows.shape
    num_regimes = int(y_labels.max()) + 1

    model = load_trained_model(
        num_features=num_features,
        num_regimes=num_regimes,
        device=device,
    )

    X_tensor = torch.from_numpy(X_windows).to(device)
    with torch.no_grad():
        logits = model(X_tensor)                    # (num_windows, num_regimes)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)

    data = {
        "date": dates,
        "true_regime": y_labels,
        "pred_regime": preds,
    }
    for k in range(num_regimes):
        data[f"prob_regime_{k}"] = probs[:, k]

    df_out = pd.DataFrame(data)
    return df_out
