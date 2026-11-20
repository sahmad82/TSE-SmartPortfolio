from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def equal_weight(returns_df: pd.DataFrame) -> np.ndarray:
    """پرتفوی مبنا با وزن برابر (1/N)."""
    n_assets = returns_df.shape[1]
    return np.ones(n_assets, dtype=float) / n_assets


def _solve_min_variance(cov_matrix: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """حل بهینه‌سازی حداقل واریانس با قید عدم فروش استقراضی (Long-Only)."""
    n_assets = cov_matrix.shape[0]
    init_weights = np.ones(n_assets, dtype=float) / n_assets
    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # مقیاس‌دهی ماتریس کوواریانس برای همگرایی دقیق‌تر الگوریتم SLSQP
    scaled_cov = cov_matrix * 1e4

    result = minimize(
        fun=lambda w: float(w @ scaled_cov @ w),
        x0=init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    if not result.success:
        return init_weights

    weights = np.clip(result.x, 0.0, max_weight)
    return weights / weights.sum()


def markowitz_min_var(returns_df: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """بهینه‌سازی کلاسیک مارکوئیتز با ماتریس کوواریانس نمونه."""
    cov = returns_df.cov().values
    return _solve_min_variance(cov, max_weight=max_weight)


def markowitz_ledoit_wolf(returns_df: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """بهینه‌سازی مارکوئیتز با اصلاح آماری ماتریس کوواریانس (Ledoit-Wolf Shrinkage)."""
    lw = LedoitWolf().fit(returns_df.values)
    return _solve_min_variance(lw.covariance_, max_weight=max_weight)


def get_quasi_diag(link: np.ndarray) -> List[int]:
    """مرحله دوم HRP: شبه‌قطری‌سازی (مرتب‌سازی برگ‌های درخت خوشه‌بندی)."""
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    num_items = link[-1, 3]

    while max(sort_ix) >= num_items:
        expanded = []
        for idx in sort_ix:
            if idx >= num_items:
                row = idx - num_items
                expanded.extend([link[row, 0], link[row, 1]])
            else:
                expanded.append(idx)
        sort_ix = expanded

    return sort_ix


def _get_cluster_var(cov: np.ndarray, cluster_items: List[int]) -> float:
    """محاسبه واریانس یک خوشه با وزن‌دهی معکوس واریانس."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    inv_diag = 1.0 / np.maximum(np.diag(cov_slice), 1e-12)
    ivp_weights = inv_diag / inv_diag.sum()
    return float(ivp_weights @ cov_slice @ ivp_weights)


def _get_rec_bipart(cov: np.ndarray, sort_ix: List[int]) -> np.ndarray:
    """مرحله سوم HRP: دو نیم کردن بازگشتی و تخصیص وزن بر اساس ریسک خوشه‌ها."""
    weights = np.ones(cov.shape[0], dtype=float)
    clusters = [sort_ix]

    while len(clusters) > 0:
        clusters = [
            c[start:end]
            for c in clusters
            for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1]
            var0 = _get_cluster_var(cov, c0)
            var1 = _get_cluster_var(cov, c1)
            alpha = 1.0 - var0 / (var0 + var1) if (var0 + var1) > 0 else 0.5
            weights[c0] *= alpha
            weights[c1] *= (1.0 - alpha)

    return weights / weights.sum()


def hierarchical_risk_parity(
    returns_df: pd.DataFrame, linkage_method: str = "single"
) -> Tuple[np.ndarray, Dict]:
    """
    الگوریتم Hierarchical Risk Parity (HRP):
    ۱. محاسبه ماتریس فاصله از همبستگی و خوشه‌بندی سلسله‌مراتبی
    ۲. شبه‌قطری‌سازی ماتریس کوواریانس
    ۳. دو نیم کردن بازگشتی (Recursive Bisection)
    """
    cov = returns_df.cov().values
    corr = returns_df.corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)

    dist_matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    condensed_dist = squareform(dist_matrix, checks=False)
    link = linkage(condensed_dist, method=linkage_method)

    sort_ix = get_quasi_diag(link)
    weights = _get_rec_bipart(cov, sort_ix)

    artifacts = {
        "linkage": link,
        "sort_ix": sort_ix,
        "corr": corr,
    }
    return weights, artifacts