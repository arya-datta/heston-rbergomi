# Architecture

The volengine package is organized as eight layered scopes, each a sub-package
of `volengine`. Layering is strict: lower layers know nothing about higher
layers. New work should add to the highest layer that can accommodate it.

```
                        ┌─────────────────────────────────────┐
                        │ 8. backtesting   (OpenBB harness)   │  CLI: volengine-backtest
                        └─────────────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 7. neural        (NN calibration)   │  optional: [neural] extra
                        └─────────────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 6. analysis      (ATM-skew study)   │
                        └─────────────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 5. products      (Euros + var swap) │
                        └─────────────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 4. calibration   (DE + L-BFGS-B)    │
                        └─────────────────────────────────────┘
        ┌────────────────────────────┐ ┌────────────────────────────┐
        │ 3a. models.heston          │ │ 3b. models.rbergomi        │
        │     CF + Carr-Madan + QE   │ │     hybrid scheme + MC     │
        └────────────────────────────┘ └────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 2. surfaces      (BS, IV, SVI)      │
                        └─────────────────────────────────────┘
                        ┌─────────────────────────────────────┐
                        │ 1. (NumPy, SciPy, pandas baseline)  │
                        └─────────────────────────────────────┘
```

## Layer-by-layer

### 1. Foundation — NumPy / SciPy / pandas

External; not part of the package. Everything else assumes NumPy 1.24+ semantics
and SciPy ≥ 1.11. NumPy 2.0 compatibility was added in v0.1.0 (the project
uses `np.trapezoid` with a fallback).

### 2. Surfaces — `volengine.surfaces`

Black-Scholes pricing, vega, and Brent IV inversion (`implied_vol.py`); raw-SVI
parameterization with no-butterfly-arbitrage and no-calendar-arbitrage checks
(`svi.py`). This layer is purely market-data plumbing — it doesn't know
about stochastic vol models.

**Depends on:** layer 1 only.
**Used by:** everything above.

### 3a. Heston model — `volengine.models.heston`

- `parameters.py` — `HestonParameters(kappa, theta, xi, rho, v0)` with a
  `feller_condition()` diagnostic.
- `characteristic_function.py` — Albrecher's *little Heston trap* form of
  `phi(u; T)`, avoiding the branch-cut bug of the original Heston (1993)
  formulation.
- `carr_madan.py` — FFT-based vanilla pricer; defaults to `N=8192, eta=0.15`
  for 4-decimal-place agreement with brute-force numerical integration.
- `qe_simulation.py` — Andersen (2008) Quadratic-Exponential Monte Carlo
  scheme with antithetic variates.

**Depends on:** layer 2 (for BS / IV inversion in tests, and parity in `heston_vanilla_price`).
**Used by:** calibration, products, analysis, backtesting.

### 3b. rough Bergomi model — `volengine.models.rbergomi`

- `parameters.py` — `RBergomiParameters(H, eta, rho, xi0)` with a callable
  forward-variance curve.
- `hybrid_scheme.py` — Bennedsen-Lunde-Pakkanen (2017) simulation of the
  Volterra integral `Y_t = ∫₀ᵗ (t-s)^{H-1/2} dZ_s` at MC rate `O(N^{-1/2})`.
- `pricing.py` — turns `(Y, Z)` into spot paths via the rBergomi SDE and
  prices vanillas by MC.

**Depends on:** layer 2.
**Used by:** calibration, products, analysis, backtesting, neural.

### 4. Calibration — `volengine.calibration`

Two-stage (global differential evolution → local L-BFGS-B) calibration of
both models in *IV space* (weighted RMSE of model vs. market implied vols).
The `objective.py` module defines the `IVQuote(K, T, iv_mkt, weight)` data
shape that flows through both calibrators.

Warm-starting from a previous day's parameters is supported (`skip_global=True`
with an `initial` argument), critical for backtest throughput.

**Depends on:** layers 2, 3a, 3b.
**Used by:** products (occasionally), analysis, backtesting.

### 5. Products — `volengine.products`

- `european.py` — thin wrappers calling either model's vanilla pricer.
- `variance_swap.py` — closed-form variance strike under Heston, integral of
  forward variance under rBergomi, and Demeterfi-Derman-Kamal-Zou (1999)
  model-free static replication.

**Depends on:** layers 2, 3a, 3b.
**Used by:** analysis, backtesting (for the held-out vanilla portfolio).

### 6. Analysis — `volengine.analysis`

- `atm_skew.py` — extracts the ATM-skew term structure from an SVI surface
  (analytic), from a Heston model (FFT-priced central difference), and from
  an rBergomi model (MC-priced central difference using *shared paths* for
  the three strikes — essential for noise reduction).
- `fit_skew_power_law` — fits `log ψ(T) = c + α log T`, implying `H = α + 1/2`.

**Depends on:** layers 2, 3a, 3b.
**Used by:** notebooks (the killer plot lives here).

### 7. Neural calibration (optional stretch) — `volengine.neural`

- `data_generation.py` — generates `(params, IV-grid)` training pairs from
  uniform-random parameter draws and rBergomi MC pricing.
- `nn_calibrator.py` — a small PyTorch MLP that learns the inverse map
  `IV-grid → params`.

Requires `pip install volengine[neural]` (adds torch). Without torch the
training-data utility still imports; only the actual trainer is gated.

**Depends on:** layers 2, 3b.
**Used by:** notebook 09.

### 8. Backtesting — `volengine.backtesting`

- `data_loader.py` — OpenBB chain pulls + filtering + parquet caching, with
  provider-history capability awareness (yfinance is treated as live-only).
- `backtest_engine.py` — rolling-window recalibration of both models with
  warm-starting, plus held-out model-risk reporting in bps of spot.
- `run_backtest.py` — argparse CLI runner; entry point `volengine-backtest`.

**Depends on:** everything below.
**Used by:** CLI users.

## Data flow

A single trading day inside the backtest moves through every layer:

```
  OpenBB chain ─► filter_for_calibration ─► IVQuote list ─┬─► calibrate_heston   ─► HestonParameters
                                                          │
                                                          └─► calibrate_rbergomi ─► RBergomiParameters
                                                                  │
                                          held-out vanilla book   │
                                              (same quotes)       ▼
                                                          _model_risk_spread
                                                                  │
                                                                  ▼
                                                             DayResult row
```

## Testing strategy

- Every numerical routine has a corresponding test that checks it against
  either a closed form, an alternative numerical method, or a published
  benchmark.
- `pytest.mark.slow` flags tests > 10s; the fast subset (`pytest -m 'not slow'`)
  runs in under 10 seconds for CI on push.
- Network-bound tests live behind `pytest.mark.live` and are deselected by
  default.
- The backtest is tested via a `monkeypatch`-injected `load_option_chain` that
  returns a synthetic chain, isolating the harness from vendor flakiness.

## Extension points

| If you want to … | Touch this file |
|---|---|
| Add a new vendor provider | `data_loader.py` (add to `_HISTORICAL_CHAIN_PROVIDERS` if relevant) |
| Add a new product (e.g. up-and-out call) | new file under `products/` |
| Replace bumped Greeks with AAD | new module under `models/heston/` and `models/rbergomi/` |
| Add SSVI parameterization | new module under `surfaces/`, mirror `svi.py` interface |
| Add a third model (Bates, Heston-Hull-White) | new sub-package `models/<name>/` |
| Replace the flat xi0 with a curve | extend `RBergomiParameters.xi0` typing (it already accepts a callable) |
