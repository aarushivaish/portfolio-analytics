import pandas as pd
import yfinance as yf

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
]
BENCHMARK = "^NSEI"  # Nifty 50

def download_prices(tickers=TICKERS, start="2016-01-01", end=None):
    raw = yf.download(tickers + [BENCHMARK], start=start, end=end,
                  auto_adjust=True, progress=False, threads=False)
    return raw["Close"]

def clean_prices(prices, max_missing=0.05):
    # drop tickers with too much missing data
    missing = prices.isna().mean()
    dropped = missing[missing > max_missing].index.tolist()
    prices = prices.drop(columns=dropped)
    # fill only small gaps (holidays), never long ones
    prices = prices.ffill(limit=3).dropna()
    return prices, dropped

def validate(prices):
    rets = prices.pct_change().dropna()
    checks = {
        "n_days": len(prices),
        "n_assets": prices.shape[1],
        "remaining_nans": int(prices.isna().sum().sum()),
        "non_positive_prices": int((prices <= 0).sum().sum()),
        "daily_moves_over_20pct": int((rets.abs() > 0.20).sum().sum()),
    }
    return checks, rets[(rets.abs() > 0.20).any(axis=1)]