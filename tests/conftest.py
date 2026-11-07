from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(
    n: int = 400,
    *,
    seed: int = 7,
    drift: float = 0.0003,
    volatility: float = 0.012,
    start: str = "2023-01-02",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n)
    close = 10.0 * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    index = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def sample_panel() -> dict[str, pd.DataFrame]:
    return {
        f"sh60000{i}": make_frame(seed=i + 1, drift=0.0002 + i * 0.00005)
        for i in range(6)
    }

