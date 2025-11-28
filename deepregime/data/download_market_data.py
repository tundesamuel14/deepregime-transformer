# deepregime/features/build_features.py

from pathlib import Path
import pandas as pd
import numpy as np

# project root (deepregime-transformer/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"


def load_spy_and_vix() -> pd.DataFrame:
    """
    Load SPY and VIX data from parquet and align them on dates.
    Handles yfinance column naming (date vs Date, Adj Close vs Close).
    """
    spy = pd.read_parquet(RAW_DIR / "spy.parquet")
    vix = pd.read_parquet(RAW_DIR / "vix.parquet")

    # Ensure we have a proper datetime index named "date" for both
    if "date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.set_index("date").sort_index()
    else:
        spy.index = pd.to_datetime(spy.index)
        spy.index.name = "date"

    if "date" in vix.columns:
        vix["date"] = pd.to_datetime(vix["date"])
        vix = vix.set_index("date").sort_index()
    else:
        vix.index = pd.to_datetime(vix.index)
        vix.index.name = "date"

    # Handle price column name differences (Adj Close vs Close)
    spy_price_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
    vix_price_col = "Adj Close" if "Adj Close" in vix.columns else "Close"

    spy = spy.rename(columns={spy_price_col: "spy_close"})
    vix = vix.rename(columns={vix_price_col: "vix_close"})

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

    out_path = FEATURE_DIR / "spy_vix_features.parquet"
    df_feat.to_parquet(out_path)
    print(f"Saved features to {out_path} with shape {df_feat.shape}")


if __name__ == "__main__":
    main()
