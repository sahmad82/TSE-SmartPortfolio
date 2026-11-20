from typing import Dict
import numpy as np
import pandas as pd

TRADING_DAYS_TSE = 242


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """محاسبه سری زمانی افت سرمایه از سقف (Drawdown)."""
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    return (wealth / running_max) - 1.0


def evaluate_portfolio(
    returns: pd.Series,
    rf_annual: float = 0.30,
    avg_turnover: float = 0.0,
) -> Dict[str, float]:
    """محاسبه جامع شاخص‌های بازده، ریسک و دنباله توزیع (Tail Risk)."""
    n_days = len(returns)
    wealth = (1.0 + returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    cagr = float((wealth.iloc[-1] ** (TRADING_DAYS_TSE / n_days)) - 1.0)

    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_TSE))
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_TSE) - 1.0
    excess_returns = returns - rf_daily

    sharpe = float((excess_returns.mean() / returns.std(ddof=1)) * np.sqrt(TRADING_DAYS_TSE))

    downside = excess_returns[excess_returns < 0]
    downside_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-9
    sortino = float((excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_TSE))

    dd_series = compute_drawdown_series(returns)
    max_dd = float(dd_series.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # ارزش در معرض خطر (VaR) و ارزش در معرض خطر شرطی (CVaR 95%)
    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())

    # شبیه‌سازی مونت‌کارلو برای CVaR 95%
    rng = np.random.default_rng(42)
    sim_returns = rng.normal(loc=returns.mean(), scale=returns.std(ddof=1), size=10_000)
    mc_var_95 = float(np.percentile(sim_returns, 5))
    mc_cvar_95 = float(sim_returns[sim_returns <= mc_var_95].mean())

    return {
        "Total Return (%)": round(total_return * 100, 2),
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 3),
        "Hist VaR 95% (%)": round(var_95 * 100, 2),
        "Hist CVaR 95% (%)": round(cvar_95 * 100, 2),
        "MC CVaR 95% (%)": round(mc_cvar_95 * 100, 2),
        "Avg Rebalance Turnover (%)": round(avg_turnover * 100, 2),
    }