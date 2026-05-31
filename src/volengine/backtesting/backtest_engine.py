"""Rolling-window backtesting engine for Heston / rBergomi.

For each trading day in the window:
  1) Load + filter the option chain via the OpenBB loader.
  2) Invert market mid prices to implied vols.
  3) Calibrate Heston (warm-started from previous day's params if available).
  4) Calibrate rBergomi (likewise).
  5) Record per-day RMSE, params, ATM-skew fit, and out-of-sample reprice
     errors on a held-out portfolio.
  6) Optionally save figures / parquet for later inspection.

The harness is intentionally synchronous and single-threaded — the bottleneck
is rBergomi MC, which already parallelises within numpy/numba; running days in
parallel would conflict with vendor rate-limits and provides little gain.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from volengine.backtesting.data_loader import (
    OptionChainSnapshot,
    filter_for_calibration,
    load_option_chain,
)
from volengine.calibration import (
    IVQuote,
    calibrate_heston,
    calibrate_rbergomi,
)
from volengine.calibration.heston_calibrator import HestonCalibrationResult
from volengine.calibration.rbergomi_calibrator import RBergomiCalibrationResult
from volengine.models.heston import HestonParameters, heston_vanilla_price
from volengine.models.rbergomi import RBergomiParameters, rbergomi_price
from volengine.surfaces.implied_vol import implied_vol

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    symbol: str
    start: dt.date
    end: dt.date
    r: float = 0.05
    q: float = 0.015
    provider: str = "yfinance"
    cache_dir: Path = Path("data/cache")

    # Quote filtering.
    min_dte_days: int = 7
    max_dte_days: int = 365
    min_oi: int = 50
    max_spread_frac: float = 0.25
    moneyness_band: tuple[float, float] = (0.85, 1.15)

    # Calibration.
    rbergomi_n_paths: int = 15_000
    rbergomi_n_steps_per_year: int = 80
    warm_start: bool = True
    # DE budgets — exposed so tests (and fast-iteration users) can dial down.
    heston_de_maxiter: int = 80
    heston_de_popsize: int = 20
    rbergomi_de_maxiter: int = 40
    rbergomi_de_popsize: int = 15

    # Reporting.
    save_dir: Path | None = Path("results/backtest")

    def trading_days(self) -> list[dt.date]:
        """Generate weekday dates in [start, end] inclusive (no holiday calendar)."""
        days = pd.bdate_range(self.start, self.end).date.tolist()
        return list(days)


@dataclass
class DayResult:
    """Per-day result row."""
    date: dt.date
    n_quotes: int
    spot: float
    heston_rmse: float
    heston_kappa: float
    heston_theta: float
    heston_xi: float
    heston_rho: float
    heston_v0: float
    heston_feller_ok: bool
    rbergomi_rmse: float
    rbergomi_H: float
    rbergomi_eta: float
    rbergomi_rho: float
    rbergomi_xi0: float
    model_risk_spread_bps: float    # mean absolute price diff on held-out portfolio (bps of spot)
    note: str = ""


@dataclass
class BacktestResult:
    """Aggregated backtest output."""
    config: BacktestConfig
    days: list[DayResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(d) for d in self.days])

    def save(self) -> None:
        if self.config.save_dir is None:
            return
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(save_dir / "daily_results.parquet")


def _quotes_from_snapshot(snapshot: OptionChainSnapshot, cfg: BacktestConfig) -> list[IVQuote]:
    """Build IVQuote list from a filtered, IV-inverted snapshot."""
    df = filter_for_calibration(
        snapshot,
        min_dte_days=cfg.min_dte_days,
        max_dte_days=cfg.max_dte_days,
        min_oi=cfg.min_oi,
        max_spread_frac=cfg.max_spread_frac,
        moneyness_band=cfg.moneyness_band,
        flag="call",
    )
    quotes: list[IVQuote] = []
    for _, row in df.iterrows():
        # Trust the provider's IV if present; otherwise invert from mid.
        iv = float(row["implied_vol"]) if np.isfinite(row.get("implied_vol", np.nan)) else float("nan")
        if not np.isfinite(iv) or iv <= 0:
            iv = implied_vol(
                price=float(row["mid"]), S=snapshot.spot, K=float(row["strike"]),
                T=float(row["dte_years"]), r=snapshot.r, q=snapshot.q, flag="call",
            )
        if not np.isfinite(iv) or iv <= 0:
            continue
        # Weight by 1 / (bid-ask half-spread) so tight quotes dominate.
        spread = max(float(row["ask"] - row["bid"]), 1e-4)
        w = 1.0 / spread
        quotes.append(IVQuote(K=float(row["strike"]), T=float(row["dte_years"]),
                              iv_mkt=iv, weight=w))
    return quotes


def _model_risk_spread(
    heston: HestonParameters,
    rbergomi: RBergomiParameters,
    quotes: list[IVQuote],
    S0: float, r: float, q: float,
    rbergomi_n_paths: int,
    rbergomi_n_steps_per_year: int,
) -> float:
    """Mean absolute price diff (in bps of spot) between models, over the quote set.

    Uses the same quote universe as calibration as a proxy for a fully held-out
    portfolio. For a stricter measure, swap in a separately specified vanilla
    book — the structure here is identical.
    """
    if not quotes:
        return float("nan")
    diffs = []
    by_T: dict[float, list[IVQuote]] = {}
    for qte in quotes:
        by_T.setdefault(qte.T, []).append(qte)
    for T, group in by_T.items():
        Ks = np.array([g.K for g in group])
        try:
            h_prices = np.atleast_1d(heston_vanilla_price(Ks, T, S0, r, q, heston, flag="call"))
            n_steps = max(20, int(np.ceil(rbergomi_n_steps_per_year * T)))
            rb_prices = rbergomi_price(
                Ks, T, S0, r, q, rbergomi,
                n_paths=rbergomi_n_paths, n_steps=n_steps, seed=12345,
            )
            rb_prices = np.atleast_1d(rb_prices)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            # Numerical pricing failure for this maturity (e.g. degenerate
            # params, Cholesky failure in the hybrid scheme). Skip the maturity
            # but let unexpected errors (bugs, MemoryError, KeyboardInterrupt)
            # propagate rather than silently swallowing them.
            logger.warning("model-risk pricing failed at T=%.3f: %s", T, e)
            continue
        diffs.extend(np.abs(h_prices - rb_prices).tolist())
    if not diffs:
        return float("nan")
    return float(np.mean(diffs) / S0 * 1e4)


def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    """Run the full rolling-window backtest defined by `cfg`.

    Returns
    -------
    BacktestResult with one DayResult per trading day. Failed days carry
    NaN-filled rows with `note` explaining the failure mode.
    """
    result = BacktestResult(config=cfg)
    prev_heston: HestonParameters | None = None
    prev_rbergomi: RBergomiParameters | None = None

    for day in cfg.trading_days():
        try:
            snapshot = load_option_chain(
                symbol=cfg.symbol, date=day,
                r=cfg.r, q=cfg.q, provider=cfg.provider,
                cache_dir=cfg.cache_dir,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("[%s] data load failed: %s", day, e)
            result.days.append(_empty_day_result(day, note=f"data load: {e}"))
            continue

        quotes = _quotes_from_snapshot(snapshot, cfg)
        if len(quotes) < 10:
            logger.warning("[%s] only %d quotes after filtering — skipping", day, len(quotes))
            result.days.append(_empty_day_result(day, n_quotes=len(quotes), spot=snapshot.spot,
                                                  note="too few quotes"))
            continue

        try:
            h_res = calibrate_heston(
                quotes, S0=snapshot.spot, r=snapshot.r, q=snapshot.q,
                initial=prev_heston if cfg.warm_start else None,
                skip_global=cfg.warm_start and prev_heston is not None,
                de_maxiter=cfg.heston_de_maxiter,
                de_popsize=cfg.heston_de_popsize,
            )
            rb_res = calibrate_rbergomi(
                quotes, S0=snapshot.spot, r=snapshot.r, q=snapshot.q,
                initial=prev_rbergomi if cfg.warm_start else None,
                skip_global=cfg.warm_start and prev_rbergomi is not None,
                n_paths=cfg.rbergomi_n_paths,
                n_steps_per_year=cfg.rbergomi_n_steps_per_year,
                de_maxiter=cfg.rbergomi_de_maxiter,
                de_popsize=cfg.rbergomi_de_popsize,
            )
        except Exception as e:
            logger.warning("[%s] calibration failed: %s", day, e)
            result.days.append(_empty_day_result(day, n_quotes=len(quotes), spot=snapshot.spot,
                                                  note=f"calib: {e}"))
            continue

        spread_bps = _model_risk_spread(
            h_res.params, rb_res.params, quotes,
            S0=snapshot.spot, r=snapshot.r, q=snapshot.q,
            rbergomi_n_paths=cfg.rbergomi_n_paths,
            rbergomi_n_steps_per_year=cfg.rbergomi_n_steps_per_year,
        )

        result.days.append(_pack_day(day, snapshot, len(quotes), h_res, rb_res, spread_bps))
        prev_heston, prev_rbergomi = h_res.params, rb_res.params

    if cfg.save_dir is not None:
        result.save()
    return result


def _pack_day(
    day: dt.date,
    snapshot: OptionChainSnapshot,
    n_quotes: int,
    h: HestonCalibrationResult,
    rb: RBergomiCalibrationResult,
    spread_bps: float,
) -> DayResult:
    return DayResult(
        date=day, n_quotes=n_quotes, spot=snapshot.spot,
        heston_rmse=h.rmse_vol_points,
        heston_kappa=h.params.kappa, heston_theta=h.params.theta,
        heston_xi=h.params.xi, heston_rho=h.params.rho, heston_v0=h.params.v0,
        heston_feller_ok=h.feller_ok,
        rbergomi_rmse=rb.rmse_vol_points,
        rbergomi_H=rb.params.H, rbergomi_eta=rb.params.eta,
        rbergomi_rho=rb.params.rho, rbergomi_xi0=float(rb.params.xi0),
        model_risk_spread_bps=spread_bps,
    )


def _empty_day_result(day: dt.date, n_quotes: int = 0, spot: float = float("nan"), note: str = "") -> DayResult:
    return DayResult(
        date=day, n_quotes=n_quotes, spot=spot,
        heston_rmse=float("nan"),
        heston_kappa=float("nan"), heston_theta=float("nan"),
        heston_xi=float("nan"), heston_rho=float("nan"), heston_v0=float("nan"),
        heston_feller_ok=False,
        rbergomi_rmse=float("nan"),
        rbergomi_H=float("nan"), rbergomi_eta=float("nan"),
        rbergomi_rho=float("nan"), rbergomi_xi0=float("nan"),
        model_risk_spread_bps=float("nan"),
        note=note,
    )
