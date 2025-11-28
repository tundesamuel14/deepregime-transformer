# deepregime/data/download_market_data.py

import pandas as pd
import yfinance as yf
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_asset(symbol: str, start: str = "2005-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV data for a symbol using yfinance.
    """
    df = yf.download(symbol, start=start, end=end)
    df = df.rename_axis("date").reset_index()
    return df


def save_parquet(df: pd.DataFrame, name: str) -> None:
    """
    Save a DataFrame to data/raw as parquet.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / f"{name}.parquet")


def main():
    # Example: SPY (S&P 500 ETF) and VIX
    market_symbols = {
        "spy": "SPY",
        "vix": "^VIX",
    }

    for name, ticker in market_symbols.items():
        print(f"Downloading {name} ({ticker})...")
        df = download_asset(ticker)
        save_parquet(df, name)
        print(f"Saved to {DATA_DIR / f'{name}.parquet'}")


if __name__ == "__main__":
    main()
