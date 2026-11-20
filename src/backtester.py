from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.data_loader import compute_returns, fetch_tse_prices
from src.optimizers import (
    equal_weight,
    hierarchical_risk_parity,
    markowitz_ledoit_wolf,
    markowitz_min_var,
)
from src.risk_metrics import evaluate_portfolio


def run_walk_forward_backtest(
    returns_df: pd.DataFrame,
    lookback: int = 242,
    rebalance_freq: int = 60,
    round_trip_fee: float = 0.0125,
    max_weight: float = 0.35,
    rf_annual: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    اجرای بک‌تست خارج از نمونه (Walk-Forward) برای ۴ استراتژی تخصیص دارایی.
    - بین تاریخ‌های بازتنظیم، وزن‌ها به‌صورت طبیعی با بازده روزانه دارایی‌ها تغییر می‌کنند.
    - در تاریخ بازتنظیم، کارمزد معاملات متناسب با گردش سبد (Turnover) کسر می‌شود.
    """
    strategies = {
        "Equal-Weight (1/N)": lambda window: equal_weight(window),
        "Markowitz (Min-Var)": lambda window: markowitz_min_var(window, max_weight=max_weight),
        "Ledoit-Wolf (Min-Var)": lambda window: markowitz_ledoit_wolf(window, max_weight=max_weight),
        "HRP (Hierarchical)": lambda window: hierarchical_risk_parity(window)[0],
    }

    n_days, n_assets = returns_df.shape
    if n_days <= lookback:
        raise ValueError(f"Not enough data ({n_days} days) for lookback={lookback}.")

    oos_dates = returns_df.index[lookback:]
    one_way_fee = round_trip_fee / 2.0

    daily_port_returns: Dict[str, list] = {name: [] for name in strategies}
    weight_history: Dict[str, list] = {name: [] for name in strategies}
    turnovers: Dict[str, list] = {name: [] for name in strategies}

    current_weights: Dict[str, np.ndarray] = {}

    for step, t in enumerate(range(lookback, n_days)):
        window = returns_df.iloc[t - lookback : t]
        day_ret = returns_df.iloc[t].values

        for name, opt_fn in strategies.items():
            cost = 0.0
            if step % rebalance_freq == 0:
                target_w = opt_fn(window)
                if name in current_weights:
                    turnover = float(np.sum(np.abs(target_w - current_weights[name])))
                else:
                    turnover = 1.0  # خرید اولیه پرتفوی
                cost = turnover * one_way_fee
                turnovers[name].append(turnover)
                current_weights[name] = target_w.copy()
                weight_history[name].append((returns_df.index[t], target_w.copy()))

            w = current_weights[name]
            port_ret = float(np.dot(w, day_ret)) - cost
            daily_port_returns[name].append(port_ret)

            # به‌روزرسانی وزن‌ها پس از نوسان قیمت در پایان روز (Weight Drift)
            drifted_w = w * (1.0 + day_ret)
            drifted_sum = drifted_w.sum()
            if drifted_sum > 0:
                current_weights[name] = drifted_w / drifted_sum

    returns_out = pd.DataFrame(daily_port_returns, index=oos_dates)

    summary_rows = {}
    weights_dfs = {}
    for name in strategies:
        avg_to = float(np.mean(turnovers[name][1:])) if len(turnovers[name]) > 1 else 0.0
        summary_rows[name] = evaluate_portfolio(
            returns_out[name], rf_annual=rf_annual, avg_turnover=avg_to
        )
        dates_w = [item[0] for item in weight_history[name]]
        mat_w = np.vstack([item[1] for item in weight_history[name]])
        weights_dfs[name] = pd.DataFrame(mat_w, index=dates_w, columns=returns_df.columns)

    summary_df = pd.DataFrame(summary_rows).T
    return returns_out, summary_df, weights_dfs


if __name__ == "__main__":
    prices = fetch_tse_prices()
    rets = compute_returns(prices)

    port_rets, summary, _ = run_walk_forward_backtest(rets)
    print("\n=== Out-of-Sample Walk-Forward Backtest Results (TSE) ===")
    print(summary.to_string())