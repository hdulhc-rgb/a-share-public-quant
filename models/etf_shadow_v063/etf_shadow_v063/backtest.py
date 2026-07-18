from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .core import EPS, TRADING_DAYS, one_way_turnover


def vectorized_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return returns.mul(weights.reindex(returns.columns), axis=1).sum(axis=1)


def event_loop_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    values: list[float] = []
    aligned = weights.reindex(returns.columns).to_numpy(dtype=float)
    for row in returns.to_numpy(dtype=float):
        values.append(float(np.dot(row, aligned)))
    return pd.Series(values, index=returns.index, name="portfolio_return")


def differential_check(returns: pd.DataFrame, weights: pd.Series, tolerance: float) -> dict[str, object]:
    vector = vectorized_returns(returns, weights)
    event = event_loop_returns(returns, weights)
    divergence = float((vector - event).abs().max())
    return {"engine_a": "vectorized", "engine_b": "event_loop", "max_abs_divergence": divergence, "tolerance": tolerance, "status": "PASS" if divergence <= tolerance else "FAIL"}


def metrics(returns: pd.Series, turnover: float, cost_bps: float) -> dict[str, float]:
    net = returns.copy()
    if len(net):
        net.iloc[0] -= turnover * cost_bps / 10_000.0
    wealth = (1.0 + net).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = max(len(net) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if 1.0 + total_return > 0 else -1.0
    annual_volatility = float(net.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(net) > 1 else 0.0
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    losses = -net.to_numpy(dtype=float)
    threshold = float(np.quantile(losses, 0.95))
    cvar = float(losses[losses >= threshold - 1e-15].mean())
    drawdown_losses = -drawdown.to_numpy(dtype=float)
    dd_threshold = float(np.quantile(drawdown_losses, 0.95))
    cdar = float(drawdown_losses[drawdown_losses >= dd_threshold - 1e-15].mean())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "cvar_95_daily": cvar,
        "cdar_95": cdar,
        "one_way_turnover": turnover,
        "transaction_cost_return": turnover * cost_bps / 10_000.0,
    }


def stability_score(weight_history: list[pd.Series]) -> dict[str, float]:
    if len(weight_history) < 2:
        return {"median_l1_distance": 0.0, "max_l1_distance": 0.0, "stable_region_share": 1.0}
    distances: list[float] = []
    for left, right in zip(weight_history[:-1], weight_history[1:]):
        distances.append(float((left - right).abs().sum()))
    return {
        "median_l1_distance": float(np.median(distances)),
        "max_l1_distance": float(np.max(distances)),
        "stable_region_share": float(np.mean(np.asarray(distances) <= 0.20)),
    }

