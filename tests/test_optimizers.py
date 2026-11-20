import numpy as np
import pandas as pd

from src.optimizers import (
    equal_weight,
    hierarchical_risk_parity,
    markowitz_ledoit_wolf,
    markowitz_min_var,
)


def _sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = rng.normal(0.001, 0.02, size=(200, 6))
    cols = ["A", "B", "C", "D", "E", "F"]
    return pd.DataFrame(data, columns=cols)


def test_weights_sum_to_one_and_non_negative():
    rets = _sample_returns()
    w_eq = equal_weight(rets)
    w_mvo = markowitz_min_var(rets, max_weight=0.4)
    w_lw = markowitz_ledoit_wolf(rets, max_weight=0.4)
    w_hrp, _ = hierarchical_risk_parity(rets)

    for w in (w_eq, w_mvo, w_lw, w_hrp):
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)
        assert np.all(w >= -1e-8)


def test_max_weight_constraint_respected():
    rets = _sample_returns()
    w_mvo = markowitz_min_var(rets, max_weight=0.30)
    w_lw = markowitz_ledoit_wolf(rets, max_weight=0.30)
    assert np.max(w_mvo) <= 0.30 + 1e-6
    assert np.max(w_lw) <= 0.30 + 1e-6