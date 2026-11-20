import os
import warnings
from typing import List
import pandas as pd
import finpy_tse as fpy

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = [
    "فولاد", "فملی", "فارس", "شپنا", "شستا",
    "خودرو", "خساپا", "وبملت", "وتجارت",
    "دعبید", "غکورش", "عیار"
]


def fetch_tse_prices(
    tickers: List[str] = DEFAULT_TICKERS,
    start_date: str = "1397-01-01",
    end_date: str = "1403-12-29",
    cache_path: str = "data/tse_prices.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    دریافت قیمت پایانی تعدیل‌شده نمادها از بورس تهران با قابلیت کش محلی.
    روزهای توقف نماد با آخرین قیمت معتبر (ffill) پر می‌شوند تا در محاسبه بازده
    دچار نشت داده یا حذف روزهای معاملاتی بازار نشویم.
    """
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached data from {cache_path}...")
        return pd.read_csv(cache_path, index_col=0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    price_series = {}

    for ticker in tickers:
        print(f"Fetching: {ticker} ...")
        try:
            df = fpy.Get_Price_History(
                stock=ticker,
                start_date=start_date,
                end_date=end_date,
                ignore_date=False,
                adjust_price=True,
            )
            if df is not None and not df.empty and "Adj Final" in df.columns:
                price_series[ticker] = df["Adj Final"]
            else:
                print(f"  [Warning] No valid data returned for {ticker}")
        except Exception as exc:
            print(f"  [Error] Failed fetching {ticker}: {exc}")

    prices_df = pd.DataFrame(price_series).sort_index()

    # حذف روزهایی که بیش از نصف نمادها معامله نشده‌اند، سپس پر کردن توقف‌های مقطعی
    min_valid = max(1, len(price_series) // 2)
    prices_df = prices_df.dropna(thresh=min_valid)
    prices_df = prices_df.ffill().dropna()

    prices_df.to_csv(cache_path)
    print(f"Saved clean dataset to {cache_path} | Shape: {prices_df.shape}")
    return prices_df


def compute_returns(prices_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """محاسبه بازده روزانه (ساده برای تجمیع پرتفوی، لگاریتمی برای مدل‌های آماری)."""
    if method == "log":
        import numpy as np
        return np.log(prices_df / prices_df.shift(1)).dropna()
    return prices_df.pct_change().dropna()


if __name__ == "__main__":
    df_prices = fetch_tse_prices()
    print(df_prices.tail())
    print(f"\nDate range: {df_prices.index[0]} -> {df_prices.index[-1]}")