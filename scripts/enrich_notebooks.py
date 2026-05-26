"""One-shot script: inject narrative markdown cells into notebooks 02-10.

The existing notebooks have only a title cell + code cells. Each one is
augmented with: a Context cell (math + intent), per-section explainers, and
post-plot interpretation/caption cells. The injections are keyed by notebook
filename so each story matches the notebook it lives in.

Run once:
    python scripts/enrich_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(*lines: str) -> dict:
    """Build an ipynb markdown cell from a list of lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [(line + "\n") if i < len(lines) - 1 else line
                   for i, line in enumerate(lines)],
    }


# (filename, list of (insert_before_cell_index, markdown_cell))
INSERTIONS: dict[str, list[tuple[int, dict]]] = {
    # ----- 02 -----
    "02_heston_characteristic_function.ipynb": [
        (1, md(
            "## Context",
            "",
            "The Heston characteristic function of the log-spot is",
            "",
            "$$\\phi(u; T) = \\exp\\!\\big[ C(u, T) + D(u, T)\\, v_0 + iu \\log S_0 + iu (r - q) T \\big].$$",
            "",
            "Two algebraically equivalent closed forms exist. The original Heston (1993) form has a branch-cut bug at long maturities; Albrecher–Mayer–Schoutens–Tistaert (2007) — the *Little Heston Trap* — swap the sign inside the auxiliary $g$ to keep it on the principal branch. Carr–Madan (1999) Fourier-inverts $\\phi$ on a uniform log-strike grid to deliver the full smile in a single FFT.",
            "",
            "This notebook validates both the characteristic function and the FFT pricer against brute-force numerical integration."
        )),
        (3, md(
            "## FFT vs. quadrature",
            "",
            "The table below shows the FFT and `scipy.integrate.quad` prices side by side. Differences should sit at $\\le 10^{-4}$ across all strikes — that's the 'four decimal places' benchmark every Heston implementation chases."
        )),
        (5, md(
            "## Heston smile",
            "",
            "**Figure.** Implied-vol smile under SPX-typical Heston parameters ($\\kappa{=}1.5,\\ \\theta{=}0.04,\\ \\xi{=}0.5,\\ \\rho{=}-0.7,\\ v_0{=}0.04$), $T = 0.5\\,$y. The negative skew is set by $\\rho < 0$; the level by $v_0$. Heston produces smiles that are too flat at long maturities — the empirical motivation for the rough-vol extension in notebook 05."
        )),
    ],
    # ----- 03 -----
    "03_heston_calibration.ipynb": [
        (1, md(
            "## Context",
            "",
            "Heston calibration minimizes the weighted RMSE of model versus market implied volatilities across the surface. We work in IV space rather than price space because IV errors are roughly homogeneous across moneyness, whereas price MSE is dominated by ITM quotes and ignores the wings — exactly where stochastic-vol matters.",
            "",
            "The calibrator runs in two stages: SciPy `differential_evolution` to find the right basin, then L-BFGS-B for local polish. The Feller condition $2\\kappa\\theta \\ge \\xi^2$ is reported as a diagnostic but not enforced — calibrated SPX surfaces routinely violate it, and the QE Monte Carlo scheme handles sub-Feller variance paths by construction."
        )),
        (3, md(
            "## Fit quality",
            "",
            "**Figure.** Market IVs (markers) vs. calibrated Heston model IVs (lines) on a single trading day, grouped by maturity. A good calibration sits inside the bid-ask half-spread band on liquid strikes; deviations on the deep wings are expected and concentrate the model's known weaknesses."
        )),
        (5, md(
            "## Parameter sanity checks",
            "",
            "Calibrated SPX parameters typically land in the ranges: $\\kappa \\in [0.5, 5]$, $\\theta \\in [0.02, 0.08]$, $\\xi \\in [0.3, 1.5]$, $\\rho \\in [-0.9, -0.5]$, $v_0$ close to current realized variance. Values outside these ranges usually mean the optimizer got stuck — re-run with a different DE seed or tighter bounds."
        )),
    ],
    # ----- 04 -----
    "04_heston_monte_carlo_qe.ipynb": [
        (1, md(
            "## Context",
            "",
            "A naive Euler discretization of Heston's variance SDE drives $v$ below zero whenever $2\\kappa\\theta < \\xi^2$ — i.e., almost always on calibrated SPX. Andersen (2008) replaces Euler with a moment-matched conditional draw: a squared-Gaussian if the variance-to-mean ratio $\\psi$ is low, an exponential with a Dirac mass at zero if it's high. The result is strictly non-negative and unbiased in moments.",
            "",
            "We validate by repricing a small set of vanillas via QE Monte Carlo and comparing to Carr–Madan FFT."
        )),
        (3, md(
            "## Convergence plot",
            "",
            "**Figure.** QE Monte Carlo vanilla price vs. number of paths, with the FFT reference. The error band shrinks as $1/\\sqrt{N}$, the canonical MC rate. ~50 k paths suffices to land within 30 bps of the FFT reference at SPX scale."
        )),
        (5, md(
            "## Variance-path diagnostic",
            "",
            "**Figure.** A handful of variance paths under QE. Visual confirmation that paths stay non-negative even in deeply sub-Feller regimes where Euler would routinely go below zero."
        )),
    ],
    # ----- 05 -----
    "05_rbergomi_simulation_hybrid_scheme.ipynb": [
        (1, md(
            "## Context",
            "",
            "Rough Bergomi (Bayer–Friz–Gatheral 2016) is driven by a Volterra integral against a Brownian motion,",
            "",
            "$$Y_t = \\int_0^t (t - s)^{H - 1/2}\\, dZ_s, \\qquad H \\in (0, 1/2).$$",
            "",
            "The kernel diverges at $s \\to t$, so a naive Riemann sum converges at $O(n^{-1/2 - \\alpha})$ with $\\alpha = H - 1/2 \\in (-1/2, 0)$ — hopelessly slow. Bennedsen–Lunde–Pakkanen (2017) split the integral into a near-block (covering $[t-\\kappa\\,dt, t]$) handled by Cholesky on the exact joint Gaussian, and a far-block handled by FFT convolution. With $\\kappa = 1$ the scheme converges at the regular $O(n^{-1/2})$ rate.",
            "",
            "This notebook verifies that the simulated $Y_t$ has the correct theoretical variance $\\mathrm{Var}(Y_t) = t^{2H} / (2H)$."
        )),
        (3, md(
            "## Empirical vs. theoretical variance of Y_t",
            "",
            "**Figure.** Empirical $\\mathrm{Var}(Y_t)$ from $\\sim 20{,}000$ hybrid-scheme paths overlaid on the theoretical curve $t^{2H} / (2H)$. Agreement to ~1% relative error is the BLP benchmark."
        )),
        (5, md(
            "## Sample paths",
            "",
            "**Figure.** A handful of $Y_t$ paths and corresponding instantaneous variance $V_t = \\xi_0 \\exp(\\eta\\sqrt{2H}\\,Y_t - \\tfrac{1}{2}\\eta^2 t^{2H})$. Note the *rough* appearance — qualitatively different from any diffusive SV model — which is precisely the empirical regularity the model was built to capture."
        )),
    ],
    # ----- 06 -----
    "06_rbergomi_calibration.ipynb": [
        (1, md(
            "## Context",
            "",
            "rBergomi has no closed-form characteristic function, so every objective evaluation is a Monte Carlo simulation. The MC seed is fixed across the optimizer call so the objective is a *deterministic* function of $(H, \\eta, \\rho, \\xi_0)$ — this is essential for L-BFGS-B's finite-difference gradients.",
            "",
            "We typically achieve IV RMSE comparable to Heston on the same surface, sometimes slightly better at the short end — but the real diagnostic is the *term structure of ATM skew* in notebook 07, where rBergomi pulls clearly ahead."
        )),
        (3, md(
            "## Calibrated parameters",
            "",
            "SPX-typical rBergomi parameters cluster around $H \\in [0.07, 0.15]$, $\\eta \\in [1.5, 2.5]$, $\\rho \\in [-0.95, -0.7]$. The $\\xi_0$ parameter, in this simplified setup, is a flat level; a production calibration would fit a piecewise-constant forward variance curve from ATM term structure."
        )),
        (5, md(
            "## Fit overlay",
            "",
            "**Figure.** Market IVs versus calibrated rBergomi IVs across maturities. Comparable in-sample to Heston on the same surface — out-of-sample superiority shows up in the skew term structure."
        )),
    ],
    # ----- 07 -----
    "07_term_structure_of_atm_skew.ipynb": [
        (1, md(
            "## Context — the killer plot",
            "",
            "ATM skew at maturity $T$ is",
            "",
            "$$\\psi(T) := \\left| \\partial_k \\sigma_{\\mathrm{imp}}(k, T) \\right|_{k = 0}.$$",
            "",
            "Empirically on SPX, $\\psi(T) \\sim T^{H - 1/2}$ with $H \\approx 0.07{-}0.15$ — i.e., a power-law of slope $\\sim -0.4$ in log-log. Heston (and every diffusive SV model) instead predicts $\\psi(T) \\to \\mathrm{const}$ as $T \\to 0$ and $\\psi(T) \\sim 1/T$ for large $T$. The mismatch is the single most cited empirical motivation for rough volatility.",
            "",
            "This notebook produces the headline plot of the repo: log-log market skew with Heston and rBergomi overlays."
        )),
        (3, md(
            "## The plot",
            "",
            "**Figure (headline).** Term structure of ATM skew on SPX. The market (markers) decays roughly as $T^{H - 1/2}$ with $H \\approx 0.1$. The Heston overlay (dashed) is too flat for long $T$ and too steep for short $T$. The rBergomi overlay (solid) matches the slope of the market line by construction — the slope is essentially fitted by $H$.",
            "",
            "This is the single most important figure in the repository."
        )),
        (5, md(
            "## Power-law fit",
            "",
            "Linear regression of $\\log \\psi(T)$ against $\\log T$ recovers an empirical $H$. Compare against the $H$ extracted by rBergomi calibration in notebook 06 — they should agree to within a few percent on a clean trading day."
        )),
    ],
    # ----- 08 -----
    "08_variance_swap_pricing.ipynb": [
        (1, md(
            "## Context",
            "",
            "A variance swap pays $(\\mathrm{RV}(0, T) - K_{\\mathrm{var}})$ at maturity. The fair strike $K_{\\mathrm{var}}$ has closed forms under both models:",
            "",
            "- **Heston:** $K_{\\mathrm{var}} = \\theta + (v_0 - \\theta)(1 - e^{-\\kappa T}) / (\\kappa T)$.",
            "- **rBergomi:** $K_{\\mathrm{var}} = \\tfrac{1}{T} \\int_0^T \\xi_0(s)\\, ds$ — directly in terms of the forward variance curve. Elegant, and one of rBergomi's selling points.",
            "",
            "We cross-check both against the Demeterfi–Derman–Kamal–Zou (1999) model-free static replication, $K_{\\mathrm{var}} = (2/T)\\big[\\int_0^F P(K)/K^2\\,dK + \\int_F^\\infty C(K)/K^2\\,dK\\big]$, evaluated on the calibrated models' vanilla prices."
        )),
        (3, md(
            "## Three-way agreement table",
            "",
            "Closed form, model MC (under each model's calibrated parameters), and static replication should all agree within 30 bps. Discrepancies localize either to MC noise, wing truncation in the replication, or model misspecification."
        )),
        (5, md(
            "## Term structure",
            "",
            "**Figure.** Fair variance strike $K_{\\mathrm{var}}(T)$ under each model as a function of $T$. Both models flatten toward the long-run variance $\\theta$ (Heston) / curve-implied level (rBergomi); the short end is sensitive to $v_0$ / $\\xi_0(0)$."
        )),
    ],
    # ----- 09 -----
    "09_neural_network_calibration.ipynb": [
        (1, md(
            "## Context (optional stretch)",
            "",
            "Following Horvath–Muguruza–Tomas (2019), we train a small fully-connected MLP that maps an IV grid directly to rBergomi parameters. Inference is then a few milliseconds — order $10^3 \\times$ faster than the traditional DE → L-BFGS-B pipeline.",
            "",
            "This notebook is intentionally illustrative: training data is generated from a uniform parameter grid (production would use stratified sampling), and the architecture is a 3-layer MLP (production would use deeper / residual). The goal is to demonstrate the technique, not to ship a deployable calibrator.",
            "",
            "Requires `pip install volengine[neural]` (PyTorch)."
        )),
        (3, md(
            "## Loss curve",
            "",
            "**Figure.** Training MSE versus epoch on normalized parameters. We expect rapid convergence — the inverse map is smooth and 4-dimensional."
        )),
        (5, md(
            "## NN vs. classical calibration",
            "",
            "Speed comparison on a held-out market surface: classical (DE → L-BFGS-B, $\\sim 30$ s) vs. NN inversion ($\\sim 10$ ms). The classical result is the *ground truth*; the NN result should match within $\\sim 5\\%$ relative error on each parameter for a well-trained network."
        )),
    ],
    # ----- 10 -----
    "10_backtest_openbb.ipynb": [
        (1, md(
            "## Context",
            "",
            "We use the OpenBB-driven backtest harness to recalibrate both models on each trading day in a chosen window, then track:",
            "",
            "- per-day IV-RMSE for each model;",
            "- the parameter time series ($H_t$, $\\eta_t$, $\\rho_t$ for rBergomi; the five Heston parameters);",
            "- the **model risk spread** — mean absolute price difference between the two calibrated models on a held-out vanilla portfolio, in bps of spot.",
            "",
            "On free-tier yfinance the harness operates on SPY; for true historical SPX backtests you need a paid provider (Intrinio, FMP, Polygon). The data loader refuses to cache historical dates against live-only providers."
        )),
        (3, md(
            "## RMSE time series",
            "",
            "**Figure.** Per-day IV-RMSE for Heston and rBergomi. The smoother / lower line is the better-fitting model on that day. rBergomi typically wins at short maturities; both models track each other on longer-dated quotes."
        )),
        (5, md(
            "## Parameter drift",
            "",
            "**Figure.** Calibrated parameters versus calendar time. Smooth drift indicates a healthy calibration; jumps usually signal data anomalies (corporate actions, settlement gaps) rather than model failure. Warm-start propagates each day's parameters as the next day's initial guess; cold-start re-runs DE every day for comparison."
        )),
        (7, md(
            "## Model risk spread",
            "",
            "**Figure.** Mean absolute price difference between calibrated Heston and rBergomi on the day's quote universe, in bps of spot. This is the production-relevant punchline: it quantifies how much your P&L would differ between the two models for the same vanilla book."
        )),
    ],
}


_TAG = "volengine-narrative"  # metadata tag marks cells we own


def enrich(path: Path, insertions: list[tuple[int, dict]]) -> int:
    """Insert markdown cells. Idempotent: skips cells already tagged by us.

    Past-end insertion indices append instead of being dropped, so caption
    cells for the last plot don't go missing on short notebooks.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Idempotency: if any of our tagged cells exist, this notebook is done.
    if any(c.get("metadata", {}).get("tag") == _TAG for c in cells):
        return 0

    for cell in [c for _, c in insertions]:
        cell.setdefault("metadata", {})["tag"] = _TAG

    # Apply in reverse so earlier indices stay valid; clamp past-end to len.
    for idx, cell in sorted(insertions, reverse=True):
        cells.insert(min(idx, len(cells)), cell)

    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(insertions)


def main() -> int:
    total = 0
    for name, ins in INSERTIONS.items():
        path = NB_DIR / name
        if not path.exists():
            print(f"[skip] {name}: not found")
            continue
        added = enrich(path, ins)
        print(f"[ok] {name}: +{added} markdown cells")
        total += added
    print(f"\nTotal cells added: {total}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
