import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MarkowitzResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


def _validate_inputs(mu: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mu.shape != (3,):
        raise ValueError("mu must be a length-3 vector (expected returns for 3 assets).")
    if cov.shape != (3, 3):
        raise ValueError("cov must be a 3x3 covariance matrix.")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("cov must be symmetric.")

    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> MarkowitzResult:
    w = np.asarray(weights, dtype=float)
    exp_ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))

    if vol <= 0:
        sharpe = -np.inf
    else:
        sharpe = (exp_ret - rf) / vol

    return MarkowitzResult(
        weights=w,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=float(sharpe),
    )


def optimize_sharpe_unconstrained(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> MarkowitzResult:
    """
    Analytical tangency portfolio (allows shorting):
      w* ~ Sigma^{-1}(mu - rf*1), normalized so sum(w)=1.
    """
    mu, cov = _validate_inputs(mu, cov)

    ones = np.ones(3)
    excess = mu - rf * ones

    try:
        raw = np.linalg.solve(cov, excess)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov is singular or ill-conditioned; cannot solve analytically.") from exc

    denom = ones @ raw
    if abs(denom) < 1e-14:
        raise ValueError("Degenerate solution: normalization denominator is ~0.")

    w = raw / denom
    return portfolio_stats(w, mu, cov, rf)


def optimize_sharpe_long_only(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    grid_size: int = 2001,
) -> MarkowitzResult:
    """
    Long-only Sharpe optimizer for exactly 3 assets using a dense simplex grid.

    Constraints:
      - w_i >= 0
      - sum_i w_i = 1
    """
    mu, cov = _validate_inputs(mu, cov)

    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")

    best = MarkowitzResult(
        weights=np.array([1.0, 0.0, 0.0]),
        expected_return=mu[0],
        volatility=float(np.sqrt(cov[0, 0])),
        sharpe_ratio=-np.inf,
    )

    grid = np.linspace(0.0, 1.0, grid_size)
    for w1 in grid:
        max_w2 = 1.0 - w1
        # Keep same resolution for the second dimension.
        n2 = max(2, int(round(max_w2 * (grid_size - 1))) + 1)
        w2_values = np.linspace(0.0, max_w2, n2)

        for w2 in w2_values:
            w3 = 1.0 - w1 - w2
            w = np.array([w1, w2, w3])
            stats = portfolio_stats(w, mu, cov, rf)
            if stats.sharpe_ratio > best.sharpe_ratio:
                best = stats

    return best


def optimize_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
    grid_size: int = 2001,
) -> MarkowitzResult:
    """Convenience wrapper."""
    if long_only:
        return optimize_sharpe_long_only(mu=mu, cov=cov, rf=rf, grid_size=grid_size)
    return optimize_sharpe_unconstrained(mu=mu, cov=cov, rf=rf)


if __name__ == "__main__":
    # Example annualized inputs for 3 assets.
    mu_ex = np.array([0.08, 0.12, 0.10])
    vol = np.array([0.15, 0.22, 0.18])
    corr = np.array([
        [1.0, 0.30, 0.45],
        [0.30, 1.0, 0.40],
        [0.45, 0.40, 1.0],
    ])
    cov_ex = np.outer(vol, vol) * corr
    rf_ex = 0.02

    res_long_only = optimize_sharpe(mu_ex, cov_ex, rf=rf_ex, long_only=True, grid_size=1501)
    res_unconstrained = optimize_sharpe(mu_ex, cov_ex, rf=rf_ex, long_only=False)

    np.set_printoptions(precision=6, suppress=True)
    print("Long-only optimum:")
    print("weights       =", res_long_only.weights)
    print("return        =", round(res_long_only.expected_return, 6))
    print("volatility    =", round(res_long_only.volatility, 6))
    print("sharpe        =", round(res_long_only.sharpe_ratio, 6))

    print("\nUnconstrained optimum:")
    print("weights       =", res_unconstrained.weights)
    print("return        =", round(res_unconstrained.expected_return, 6))
    print("volatility    =", round(res_unconstrained.volatility, 6))
    print("sharpe        =", round(res_unconstrained.sharpe_ratio, 6))
