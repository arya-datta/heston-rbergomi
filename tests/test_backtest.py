"""Tests for the OpenBB-driven backtest harness.

We replace `load_option_chain` with a stub that returns a deterministic
synthetic chain — same shape as a real OpenBB pull but with prices generated
from a known Heston parameter set. This isolates the backtest logic
(filtering, calibration, warm-starting, aggregation) from network/vendor
flakiness.

The synthetic chain is built ONCE at module load (it's expensive — it prices
a full Heston surface) and re-used across days with small perturbations so
warm-start saves time.
"""

import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volengine.backtesting import backtest_engine as bte
from volengine.backtesting.data_loader import OptionChainSnapshot
from volengine.backtesting.backtest_engine import (
    BacktestConfig,
    DayResult,
    run_backtest,
)
from volengine.models.heston import HestonParameters, heston_vanilla_price


def _synthetic_chain(snapshot_date: dt.date, spot: float = 100.0,
                     bump: float = 0.0) -> pd.DataFrame:
    """Build a single-day SPY-like option chain priced from a fixed Heston surface.

    `bump` shifts the v0 parameter so successive days look slightly different,
    exercising the warm-start codepath without making calibration impossible.
    """
    p = HestonParameters(kappa=1.5, theta=0.04 + bump, xi=0.5,
                         rho=-0.7, v0=0.04 + bump)
    rows = []
    for dte_days in [30, 60, 120, 240]:
        T = dte_days / 365.25
        expiration = snapshot_date + dt.timedelta(days=dte_days)
        for k_log in np.linspace(-0.10, 0.10, 7):
            K = float(spot * np.exp(k_log))
            mid = float(heston_vanilla_price(K, T, spot, 0.05, 0.013, p))
            # Symmetric synthetic bid/ask around mid.
            half_spread = max(0.05, mid * 0.005)
            rows.append(dict(
                date=snapshot_date,
                expiration=expiration,
                strike=K,
                type="call",
                bid=mid - half_spread,
                ask=mid + half_spread,
                mid=mid,
                open_interest=500,
                implied_vol=np.nan,        # force IV inversion path
                underlying_price=spot,
                dte_years=T,
            ))
    return pd.DataFrame(rows)


@pytest.fixture
def mock_loader(monkeypatch):
    """Patch load_option_chain to return synthetic chains keyed by date.

    Each day gets a tiny v0 bump (~25 bps in vol terms over a 5-day window)
    so the surface drifts realistically.
    """
    base = dt.date(2024, 1, 2)

    def fake_load(symbol, date, *args, **kwargs):
        days_since = (date - base).days
        bump = 0.0005 * days_since           # small param drift
        chain = _synthetic_chain(date, spot=100.0, bump=bump)
        return OptionChainSnapshot(
            symbol=symbol, date=date, spot=100.0, r=0.05, q=0.013, chain=chain,
        )

    monkeypatch.setattr(bte, "load_option_chain", fake_load)
    return fake_load


def _short_cfg(tmp_path: Path, **overrides) -> BacktestConfig:
    """3-trading-day backtest into a tmp dir. Override any field for a test."""
    defaults = dict(
        symbol="SYNTH",
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 4),
        r=0.05, q=0.013,
        provider="yfinance",
        cache_dir=tmp_path / "cache",
        save_dir=tmp_path / "results",
        warm_start=True,
        rbergomi_n_paths=2000,             # tiny for test speed
        rbergomi_n_steps_per_year=40,
        # Minimum-viable DE budgets so the test takes ~30s, not ~30min.
        # The synthetic surface is built FROM a Heston model and is so
        # well-behaved that 3 DE iterations + L-BFGS-B is plenty.
        heston_de_maxiter=3,
        heston_de_popsize=5,
        rbergomi_de_maxiter=3,
        rbergomi_de_popsize=5,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


@pytest.mark.slow
def test_backtest_produces_valid_day_results(mock_loader, tmp_path):
    """End-to-end: 3 days, mocked chains, both models calibrate, day rows are finite."""
    cfg = _short_cfg(tmp_path)
    result = run_backtest(cfg)

    assert isinstance(result.days, list)
    assert len(result.days) >= 2, "expected at least 2 trading days in window"
    for day in result.days:
        assert isinstance(day, DayResult)
        assert day.n_quotes >= 10, f"too few quotes on {day.date}: {day.n_quotes}"
        assert day.spot == pytest.approx(100.0)
        assert np.isfinite(day.heston_rmse)
        assert day.heston_rmse < 0.01      # synthetic, should reprice nearly exactly
        assert np.isfinite(day.rbergomi_rmse)
        assert -1.0 < day.heston_rho < 1.0
        assert 0.0 < day.rbergomi_H < 0.5
        assert np.isfinite(day.model_risk_spread_bps)


@pytest.mark.slow
def test_backtest_warm_start_skips_global_stage(mock_loader, tmp_path):
    """Warm-start mode should produce results that aren't worse than cold-start.

    We're not asserting on timings (flaky), just on quality: warm-start should
    fit at least as well as cold-start on a smooth, slowly-drifting surface.
    """
    cfg_warm = _short_cfg(tmp_path / "warm", warm_start=True)
    cfg_cold = _short_cfg(tmp_path / "cold", warm_start=False)

    r_warm = run_backtest(cfg_warm)
    r_cold = run_backtest(cfg_cold)

    warm_rmse = np.median([d.heston_rmse for d in r_warm.days if np.isfinite(d.heston_rmse)])
    cold_rmse = np.median([d.heston_rmse for d in r_cold.days if np.isfinite(d.heston_rmse)])
    # Warm-start should be no worse than cold-start by more than 10 bps.
    assert warm_rmse <= cold_rmse + 0.001


@pytest.mark.slow
def test_backtest_writes_parquet(mock_loader, tmp_path):
    cfg = _short_cfg(tmp_path)
    run_backtest(cfg)
    parquet_path = tmp_path / "results" / "daily_results.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert {"date", "heston_rmse", "rbergomi_rmse", "model_risk_spread_bps"}.issubset(df.columns)
    assert len(df) >= 2


def test_backtest_config_trading_days_generator():
    """Sanity: weekday range generator skips weekends."""
    cfg = BacktestConfig(symbol="X", start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 7))
    days = cfg.trading_days()
    # Mon Jan 1 through Fri Jan 5 (Jan 1 is observed New Year's day but still a weekday).
    assert all(d.weekday() < 5 for d in days)
    assert len(days) == 5
