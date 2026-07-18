from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from .core import EPS, ResearchClosed, normalize_long_only


INVERSE_VOLATILITY_FLOOR_ANNUAL = 0.005


@dataclass
class ChallengerResult:
    name: str
    weights: pd.Series | None
    status: str
    method: str
    message: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)


def _slsqp(name: str, assets: list[str], objective: Callable[[np.ndarray], float], x0: np.ndarray) -> ChallengerResult:
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(assets),
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 600, "ftol": 1e-12, "disp": False},
    )
    if not result.success or not np.isfinite(result.fun):
        return ChallengerResult(name, None, "SOLVER_FAILED", "SLSQP", str(result.message), {"iterations": int(getattr(result, "nit", -1))})
    weights = normalize_long_only(result.x, assets)
    return ChallengerResult(name, weights, "OK", "SLSQP", str(result.message), {"objective": float(result.fun), "iterations": int(result.nit)})


def equal_weight(returns: pd.DataFrame) -> ChallengerResult:
    assets = list(returns.columns)
    return ChallengerResult("equal_weight", normalize_long_only(np.ones(len(assets)), assets), "OK", "CLOSED_FORM")


def inverse_volatility(returns: pd.DataFrame) -> ChallengerResult:
    raw_daily_vol = returns.std(ddof=1)
    if raw_daily_vol.isna().any():
        return ChallengerResult("inverse_volatility", None, "NOT_EVALUABLE", "CLOSED_FORM", "MISSING_VOLATILITY")
    daily_floor = INVERSE_VOLATILITY_FLOOR_ANNUAL / np.sqrt(252.0)
    floored_assets = raw_daily_vol.index[raw_daily_vol < daily_floor].tolist()
    effective_vol = raw_daily_vol.clip(lower=daily_floor)
    return ChallengerResult(
        "inverse_volatility",
        normalize_long_only((1.0 / effective_vol).to_numpy(), returns.columns),
        "OK",
        "CLOSED_FORM_WITH_PREREGISTERED_VOLATILITY_FLOOR",
        diagnostics={
            "volatility_floor_annual": INVERSE_VOLATILITY_FLOOR_ANNUAL,
            "floored_assets": floored_assets,
        },
    )


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items]
    diag = np.diag(sub.to_numpy(dtype=float))
    inv_diag = 1.0 / np.maximum(diag, EPS)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub.to_numpy(dtype=float) @ weights)


def hierarchical_risk_parity(returns: pd.DataFrame) -> ChallengerResult:
    cov = returns.cov()
    corr = returns.corr().fillna(0.0).clip(-1.0, 1.0)
    np.fill_diagonal(corr.values, 1.0)
    if not np.isfinite(cov.to_numpy()).all() or not np.isfinite(corr.to_numpy()).all():
        return ChallengerResult("hrp", None, "NOT_EVALUABLE", "HRP", "NON_FINITE_COVARIANCE")
    distance = np.sqrt(np.maximum(0.0, (1.0 - corr.to_numpy(dtype=float)) / 2.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    sorted_assets = [returns.columns[i] for i in order]
    weights = pd.Series(1.0, index=sorted_assets)
    clusters = [sorted_assets]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_var, right_var = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - left_var / max(left_var + right_var, EPS)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return ChallengerResult("hrp", normalize_long_only(weights.reindex(returns.columns).to_numpy(), returns.columns), "OK", "SCIPY_HIERARCHICAL")


def _cvar_loss(weights: np.ndarray, matrix: np.ndarray, alpha: float = 0.95) -> float:
    losses = -(matrix @ weights)
    threshold = float(np.quantile(losses, alpha))
    tail = losses[losses >= threshold - 1e-15]
    return float(tail.mean()) if len(tail) else float(losses.max())


def risk_budget_cvar(returns: pd.DataFrame, alpha: float = 0.95) -> ChallengerResult:
    assets = list(returns.columns)
    matrix = returns.to_numpy(dtype=float)
    n = len(assets)
    x0 = np.full(n, 1.0 / n)

    def objective(weights: np.ndarray) -> float:
        base = max(_cvar_loss(weights, matrix, alpha), EPS)
        step = 1e-5
        gradient = np.empty(n)
        for index in range(n):
            perturbed = weights.copy()
            perturbed[index] += step
            gradient[index] = (_cvar_loss(perturbed, matrix, alpha) - base) / step
        contribution = weights * gradient
        target = base / n
        return float(np.mean(((contribution - target) / base) ** 2) + 1e-3 * base)

    result = _slsqp("risk_budget_cvar", assets, objective, x0)
    result.diagnostics["alpha"] = alpha
    return result


def _cdar(weights: np.ndarray, matrix: np.ndarray, alpha: float = 0.95) -> float:
    portfolio_returns = matrix @ weights
    wealth = np.cumprod(1.0 + portfolio_returns)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / np.maximum(peaks, EPS)
    threshold = float(np.quantile(drawdowns, alpha))
    tail = drawdowns[drawdowns >= threshold - 1e-15]
    return float(tail.mean()) if len(tail) else 0.0


def min_cdar(returns: pd.DataFrame, alpha: float = 0.95) -> ChallengerResult:
    assets = list(returns.columns)
    matrix = returns.to_numpy(dtype=float)
    x0 = np.full(len(assets), 1.0 / len(assets))
    result = _slsqp("min_cdar", assets, lambda w: _cdar(w, matrix, alpha), x0)
    result.diagnostics["alpha"] = alpha
    return result


def shrunk_mean_risk(returns: pd.DataFrame, risk_aversion: float = 8.0) -> ChallengerResult:
    assets = list(returns.columns)
    matrix = returns.to_numpy(dtype=float)
    estimator = LedoitWolf().fit(matrix)
    covariance = estimator.covariance_
    raw_mean = matrix.mean(axis=0)
    shrunk_mean = 0.35 * raw_mean + 0.65 * raw_mean.mean()
    x0 = np.full(len(assets), 1.0 / len(assets))

    def objective(weights: np.ndarray) -> float:
        return float(-(shrunk_mean @ weights) + risk_aversion * (weights @ covariance @ weights))

    result = _slsqp("shrunk_mean_risk", assets, objective, x0)
    result.diagnostics.update({"risk_aversion": risk_aversion, "covariance_shrinkage": float(estimator.shrinkage_), "mean_shrinkage_to_grand_mean": 0.65})
    return result


CHALLENGERS: dict[str, Callable[[pd.DataFrame], ChallengerResult]] = {
    "equal_weight": equal_weight,
    "inverse_volatility": inverse_volatility,
    "hrp": hierarchical_risk_parity,
    "risk_budget_cvar": risk_budget_cvar,
    "min_cdar": min_cdar,
    "shrunk_mean_risk": shrunk_mean_risk,
}


def build_challenger(name: str, returns: pd.DataFrame) -> ChallengerResult:
    if name not in CHALLENGERS:
        raise ResearchClosed(f"UNKNOWN_CHALLENGER:{name}")
    return CHALLENGERS[name](returns)
