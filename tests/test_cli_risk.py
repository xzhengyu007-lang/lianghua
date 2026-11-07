from __future__ import annotations

import argparse

import pytest

from quant_system.cli import _risk_from_args
from quant_system.config import RiskConfig


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_default_risk_matches_config_default() -> None:
    risk = _risk_from_args(_ns())
    assert risk == RiskConfig()


def test_target_vol_accepts_fraction() -> None:
    risk = _risk_from_args(_ns(target_vol=0.12))
    assert risk.target_annual_vol == pytest.approx(0.12)
    assert risk.vol_lookback == RiskConfig().vol_lookback


def test_target_vol_accepts_percentage() -> None:
    risk = _risk_from_args(_ns(target_vol=12))
    assert risk.target_annual_vol == pytest.approx(0.12)
