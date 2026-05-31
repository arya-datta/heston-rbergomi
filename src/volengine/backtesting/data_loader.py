"""OpenBB data loader for option chain snapshots, with on-disk caching.

OpenBB unifies vendor APIs behind `openbb.obb.derivatives.options.chains`. The
function accepts a `provider` argument routing to e.g. Yahoo, Tradier, CBOE
delayed, or paid vendors (Intrinio, FMP). We default to Yahoo for the
free path; production users will configure their own provider via OpenBB.

Returned chains are normalized into a single tidy DataFrame with one row per
(expiration, strike, type) and uniform column names regardless of provider:

    date, expiration, strike, type, bid, ask, mid, open_interest, implied_vol,
    underlying_price, dte_years
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/cache")
_BUSINESS_YEAR_DAYS = 252

# Provider capability registry: which providers return TRUE historical chains
# vs. always-current snapshots. yfinance is a "live-only" provider — passing
# date != today still returns today's chain, which silently corrupts a
# backtest cache if we trust the user-supplied date as the cache key.
_HISTORICAL_CHAIN_PROVIDERS = frozenset({"intrinio", "fmp", "polygon"})


def provider_supports_history(provider: str) -> bool:
    """Whether `provider` returns historical option chains (vs. always-current)."""
    return provider.lower() in _HISTORICAL_CHAIN_PROVIDERS


@dataclass
class OptionChainSnapshot:
    """A single trading day's option chain snapshot for one symbol.

    Attributes
    ----------
    symbol : underlying ticker (e.g., "SPX", "AAPL").
    date   : observation date (the date the quotes were captured).
    spot   : underlying price at snapshot time.
    r, q   : risk-free rate and dividend yield in use for IV inversion.
    chain  : tidy DataFrame of quotes (see module docstring).
    """
    symbol: str
    date: dt.date
    spot: float
    r: float
    q: float
    chain: pd.DataFrame


def _cache_path(symbol: str, date: dt.date, cache_dir: Path, provider: str) -> Path:
    """Stable cache path; one parquet per (symbol, date, provider).

    The provider is part of the key because two providers will quote different
    chains for the same (symbol, date) — different liquidity, different
    coverage of weeklies, different IV inversion conventions. Mixing them in
    one cache silently corrupts backtests.
    """
    return cache_dir / f"{symbol.upper()}_{date.isoformat()}_{provider.lower()}.parquet"


def _normalize_openbb_chain(
    raw: pd.DataFrame,
    snapshot_date: dt.date,
    spot: float,
) -> pd.DataFrame:
    """Harmonize column names across OpenBB providers."""
    # OpenBB returns slightly different column names across providers; normalize.
    rename_map = {
        "expiration": "expiration",
        "strike": "strike",
        "option_type": "type",
        "bid": "bid",
        "ask": "ask",
        "open_interest": "open_interest",
        "implied_volatility": "implied_vol",
        "underlying_price": "underlying_price",
    }
    df = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    # Ensure required columns exist; fill missing with NaN.
    for col in ("bid", "ask", "open_interest", "implied_vol"):
        if col not in df.columns:
            df[col] = np.nan
    if "underlying_price" not in df.columns:
        df["underlying_price"] = spot

    df["mid"] = 0.5 * (df["bid"].astype(float) + df["ask"].astype(float))
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["dte_years"] = df["expiration"].apply(
        lambda d: max((d - snapshot_date).days, 0) / 365.25
    )
    df["date"] = snapshot_date
    df["type"] = df["type"].str.lower().str[0].map({"c": "call", "p": "put"})
    cols = ["date", "expiration", "strike", "type", "bid", "ask", "mid",
            "open_interest", "implied_vol", "underlying_price", "dte_years"]
    return df[cols].copy()


def load_option_chain(
    symbol: str,
    date: dt.date,
    spot: float | None = None,
    r: float = 0.05,
    q: float = 0.015,
    provider: str = "yfinance",
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> OptionChainSnapshot:
    """Load one day's option chain via OpenBB, with on-disk caching.

    Parameters
    ----------
    symbol : underlying ticker.
    date : snapshot date.
    spot : optional spot override (otherwise pulled from OpenBB equity data).
    r, q : rates used downstream for IV inversion; the loader carries them
           through but does not pull a curve here.
    provider : OpenBB provider routing key. Defaults to yfinance (free,
        intraday-resolution lacking but fine for daily snapshots).
    cache_dir : on-disk parquet cache. Each (symbol, date) is one file.
    refresh : if True, ignore cache and re-pull from OpenBB.

    Returns
    -------
    OptionChainSnapshot with normalized columns.

    Notes
    -----
    OpenBB intentionally exposes only the latest chain for many free
    providers — true historical chains require paid vendors (Intrinio, FMP)
    or third-party storage. We treat the daily file as the canonical archive.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Live-only providers (yfinance et al) return today's chain regardless of
    # the date argument. If the caller asks for a historical date through such
    # a provider, the cached file would be silently mislabeled — so we coerce
    # the cache key to today's date and warn loudly. Backtests against
    # historical dates must use a provider with real history (Intrinio, FMP).
    today = dt.date.today()
    if not provider_supports_history(provider) and date != today:
        warnings.warn(
            f"Provider '{provider}' does not support historical chains. "
            f"Requested date {date} is being cached under today's date "
            f"({today}) because the data is actually today's chain. "
            f"Use Intrinio/FMP/Polygon for true historical backtests.",
            stacklevel=2,
        )
        effective_date = today
    else:
        effective_date = date

    cache_path = _cache_path(symbol, effective_date, cache_dir, provider)

    if cache_path.exists() and not refresh:
        chain = pd.read_parquet(cache_path)
        spot_resolved = float(spot) if spot is not None else float(chain["underlying_price"].iloc[0])
        return OptionChainSnapshot(symbol=symbol, date=effective_date, spot=spot_resolved,
                                    r=r, q=q, chain=chain)

    # Lazy import: don't require openbb for tests that use cached data only.
    try:
        from openbb import obb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openbb is not installed. Install with `pip install openbb` or "
            "supply a cached parquet at " + str(cache_path)
        ) from e

    raw = obb.derivatives.options.chains(symbol=symbol, provider=provider).to_df()
    if spot is None:
        # Pull the most recent close as a spot proxy. On market holidays (and
        # for some free providers) the historical-equity endpoint raises on
        # empty data instead of returning an empty frame, so we treat any
        # failure as "fall back to chain's underlying_price field".
        spot_resolved: float | None = None
        try:
            eq = obb.equity.price.historical(
                symbol=symbol, start_date=date.isoformat(), end_date=date.isoformat(),
                provider=provider,
            ).to_df()
            if len(eq):
                spot_resolved = float(eq["close"].iloc[-1])
        except Exception:
            spot_resolved = None
        if spot_resolved is None:
            spot_resolved = float(raw["underlying_price"].iloc[0])
    else:
        spot_resolved = float(spot)

    chain = _normalize_openbb_chain(raw, snapshot_date=effective_date, spot=spot_resolved)
    chain.to_parquet(cache_path)
    return OptionChainSnapshot(symbol=symbol, date=effective_date, spot=spot_resolved,
                                r=r, q=q, chain=chain)


def filter_for_calibration(
    snapshot: OptionChainSnapshot,
    min_dte_days: int = 7,
    max_dte_days: int = 365,
    min_oi: int = 50,
    max_spread_frac: float = 0.25,
    moneyness_band: tuple[float, float] = (0.7, 1.3),
    flag: str = "call",
) -> pd.DataFrame:
    """Filter raw quotes for calibration-grade liquid quotes.

    Filters applied:
      - Right type (call by default; OTM puts are usually preferable on the
        downside but for simple calibration we stick to calls).
      - DTE in [min_dte_days, max_dte_days].
      - Open interest >= min_oi.
      - Bid > 0, ask > bid, mid > 0.
      - Bid-ask spread / mid <= max_spread_frac.
      - Strike/spot ratio in moneyness_band.

    The filter is intentionally conservative — junk quotes destroy
    calibrators long before they help.
    """
    df = snapshot.chain
    spot = snapshot.spot
    today = snapshot.date

    dte_days = (df["expiration"] - today).apply(lambda d: d.days)
    keep = (
        (df["type"] == flag)
        & (dte_days >= min_dte_days)
        & (dte_days <= max_dte_days)
        & (df["open_interest"].fillna(0) >= min_oi)
        & (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["mid"] > 0)
        & ((df["ask"] - df["bid"]) / df["mid"].clip(lower=1e-6) <= max_spread_frac)
        & (df["strike"] / spot >= moneyness_band[0])
        & (df["strike"] / spot <= moneyness_band[1])
    )
    return df.loc[keep].reset_index(drop=True)
