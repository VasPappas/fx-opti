import math
from typing import Tuple

import numpy as np


def _read_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            return value
        except ValueError:
            print("Please enter a valid number.")


def _read_vol(prompt: str) -> float:
    while True:
        value = _read_float(prompt)
        if value < 0:
            print("Volatility must be non-negative.")
            continue
        return value


def _read_corr(prompt: str) -> float:
    while True:
        value = _read_float(prompt)
        if value < -1 or value > 1:
            print("Correlation must be between -1 and 1.")
            continue
        return value


def min_variance_weights(vol_eurusd: float, vol_usdjpy: float, corr_eurusd_usdjpy: float) -> Tuple[float, float]:
    """
    Returns minimum-variance weights for EUR and JPY in a fully-invested FX basket.

    We model returns in USD terms:
      r_EUR = r(EUR/USD)
      r_JPY = r(JPY/USD) = -r(USD/JPY)

    So corr(r_EUR, r_JPY) = -corr(r(EUR/USD), r(USD/JPY)).
    """
    sigma_eur = vol_eurusd
    sigma_jpy = vol_usdjpy
    corr_eur_jpy = -corr_eurusd_usdjpy

    cov = np.array(
        [
            [sigma_eur**2, corr_eur_jpy * sigma_eur * sigma_jpy],
            [corr_eur_jpy * sigma_eur * sigma_jpy, sigma_jpy**2],
        ],
        dtype=float,
    )

    ones = np.ones(2)
    inv_cov = np.linalg.inv(cov)
    weights = inv_cov @ ones / (ones @ inv_cov @ ones)

    return float(weights[0]), float(weights[1])


def portfolio_volatility(weight_eur: float, weight_jpy: float, vol_eurusd: float, vol_usdjpy: float, corr_eurusd_usdjpy: float) -> float:
    sigma_eur = vol_eurusd
    sigma_jpy = vol_usdjpy
    corr_eur_jpy = -corr_eurusd_usdjpy

    cov = np.array(
        [
            [sigma_eur**2, corr_eur_jpy * sigma_eur * sigma_jpy],
            [corr_eur_jpy * sigma_eur * sigma_jpy, sigma_jpy**2],
        ],
        dtype=float,
    )
    w = np.array([weight_eur, weight_jpy], dtype=float)
    var = float(w @ cov @ w)
    return math.sqrt(max(var, 0.0))


def main() -> None:
    print("Markowitz FX Optimizer (USD base)")
    print("Input volatilities in decimal form (example: 0.12 for 12%).")

    vol_eurusd = _read_vol("EUR/USD volatility: ")
    vol_usdjpy = _read_vol("USD/JPY volatility: ")
    corr = _read_corr("Correlation between EUR/USD and USD/JPY: ")

    if vol_eurusd == 0 and vol_usdjpy == 0:
        print("Both volatilities are zero. Any weights are minimum-variance.")
        return

    if vol_eurusd == 0:
        print("Minimum-variance solution: 100% EUR, 0% JPY, 0% USD")
        return

    if vol_usdjpy == 0:
        print("Minimum-variance solution: 0% EUR, 100% JPY, 0% USD")
        return

    w_eur, w_jpy = min_variance_weights(vol_eurusd, vol_usdjpy, corr)
    port_vol = portfolio_volatility(w_eur, w_jpy, vol_eurusd, vol_usdjpy, corr)

    print("\nMinimum-variance weights (fully invested in non-USD currencies):")
    print(f"EUR weight: {w_eur:.4f}")
    print(f"JPY weight: {w_jpy:.4f}")
    print("USD weight: 0.0000")
    print(f"Portfolio volatility: {port_vol:.4%}")


if __name__ == "__main__":
    main()
