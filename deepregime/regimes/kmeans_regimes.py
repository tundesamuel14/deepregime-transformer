# deepregime/regimes/kmeans_regimes.py

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

FEATURE_DIR = Path("data/features")
REGIME_DIR = Path("data/regimes")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_DIR / "spy_vix_features.parquet")
    return df


def assign_kmeans_regimes(df: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
    """
    Run k-means on a subset of features and assign regime labels.
    """
    df = df.copy()

    feature_cols = ["ret_1d", "ret_20d", "rv_20d", "vix_level"]
    X = df[feature_cols]

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init="auto")
    df["regime_kmeans"] = kmeans.fit_predict(X)

    return df


def main():
    REGIME_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    df_reg = assign_kmeans_regimes(df, n_regimes=3)
    df_reg.to_parquet(REGIME_DIR / "spy_vix_regimes_kmeans.parquet")
    print(f"Saved regimes to {REGIME_DIR / 'spy_vix_regimes_kmeans.parquet'} with shape {df_reg.shape}")


if __name__ == "__main__":
    main()
