# deepregime/features/build_features.py

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")


def load_spy_and_vix():
    spy = pd.read_parquet(RAW_DIR / "spy.parquet")
    vix = pd.read_parquet(RAW_DIR / "vix.parquet")

    spy = spy.rename(columns={"Adj Close": "spy_close"})
    vix = vix.rename(columns={"Adj Close": "vix_close"})

    spy["date"] = pd.to_datetime(spy["Date"])
    vix["date"] = pd.to_datetime(vix["Date"])

    spy = spy.set_index("date").sort_index()
    vix = vix.set_index("date").sort_index()

    # Align on common dates
    df = spy.join(vix[["vix_close"]], how="inner")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add returns, rolling volatility, and simple derived features.
    """
    df = df.copy()

    # Daily returns
    df["ret_1d"] = df["spy_close"].pct_change()

    # Longer horizon returns
    df["ret_5d"] = df["spy_close"].pct_change(5)
    df["ret_20d"] = df["spy_close"].pct_change(20)

    # Realized volatility (rolling)
    df["rv_5d"] = df["ret_1d"].rolling(5).std() * np.sqrt(252)
    df["rv_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

    # VIX as implied vol proxy
    df["vix_level"] = df["vix_close"]

    # Drop initial NaNs
    df = df.dropna()
    return df


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_spy_and_vix()
    df_feat = add_basic_features(df)

    df_feat.to_parquet(FEATURE_DIR / "spy_vix_features.parquet")
    print(f"Saved features to {FEATURE_DIR / 'spy_vix_features.parquet'} with shape {df_feat.shape}")


if __name__ == "__main__":
    main()
