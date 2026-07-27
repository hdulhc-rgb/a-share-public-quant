from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from etf_shadow_v063.backtest import vectorized_returns
from etf_shadow_v063.core import ResearchClosed

from .data_contract import SKFOLIO_PIN


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _skfolio_cross_check(returns: pd.DataFrame) -> dict[str, Any]:
    if importlib.util.find_spec("skfolio") is None:
        return {
            "status": "DEPENDENCY_MISSING",
            "version": None,
            "required_version": SKFOLIO_PIN,
            "executed": False,
        }
    version = _version("skfolio")
    if version != SKFOLIO_PIN:
        return {
            "status": "VERSION_MISMATCH",
            "version": version,
            "required_version": SKFOLIO_PIN,
            "executed": False,
        }
    try:
        from skfolio.optimization import HierarchicalRiskParity

        model = HierarchicalRiskParity()
        model.fit(returns)
        weights = np.asarray(model.weights_, dtype=float)
        passed = bool(
            weights.shape == (returns.shape[1],)
            and np.isfinite(weights).all()
            and (weights >= -1e-12).all()
            and abs(float(weights.sum()) - 1.0) <= 1e-10
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "version": version,
            "required_version": SKFOLIO_PIN,
            "executed": True,
            "check": "HIERARCHICAL_RISK_PARITY_LONG_ONLY_WEIGHT_IDENTITY",
            "weight_sum": float(weights.sum()),
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
        }
    except Exception as error:  # pragma: no cover - optional dependency path
        return {
            "status": "EXECUTION_FAILED",
            "version": version,
            "required_version": SKFOLIO_PIN,
            "executed": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _vectorbt_cross_check(returns: pd.DataFrame, weights: pd.Series) -> dict[str, Any]:
    if importlib.util.find_spec("vectorbt") is None:
        return {
            "status": "DEPENDENCY_MISSING",
            "version": None,
            "executed": False,
            "required": False,
        }
    version = _version("vectorbt")
    try:
        import vectorbt as vbt

        portfolio_returns = vectorized_returns(returns, weights)
        synthetic_price = pd.concat(
            [
                pd.Series([1.0], index=[returns.index[0] - pd.Timedelta(days=1)]),
                (1.0 + portfolio_returns).cumprod(),
            ]
        )
        portfolio = vbt.Portfolio.from_holding(synthetic_price, init_cash=1.0)
        vectorbt_total = float(portfolio.total_return())
        expected_total = float((1.0 + portfolio_returns).prod() - 1.0)
        divergence = abs(vectorbt_total - expected_total)
        return {
            "status": "PASS" if divergence <= 1e-10 else "FAIL",
            "version": version,
            "executed": True,
            "required": False,
            "total_return_divergence": divergence,
            "tolerance": 1e-10,
        }
    except Exception as error:  # pragma: no cover - optional dependency path
        return {
            "status": "EXECUTION_FAILED",
            "version": version,
            "executed": True,
            "required": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_external_validation(
    returns: pd.DataFrame,
    weights: pd.Series,
    profile: str,
) -> dict[str, object]:
    if profile not in {"shadow", "promotion"}:
        raise ResearchClosed("EXTERNAL_VALIDATION_PROFILE_INVALID")
    sample = returns.iloc[-min(len(returns), 756) :]
    skfolio_result = _skfolio_cross_check(sample)
    vectorbt_result = _vectorbt_cross_check(sample, weights.reindex(sample.columns))
    promotion_passed = skfolio_result["status"] == "PASS"
    if profile == "promotion" and not promotion_passed:
        raise ResearchClosed(f"SKFOLIO_PROMOTION_VALIDATION_FAILED:{skfolio_result['status']}")
    return {
        "profile": profile,
        "skfolio": skfolio_result,
        "vectorbt": vectorbt_result,
        "skfolio_required_for_promotion": True,
        "vectorbt_optional": True,
        "promotion_gate_passed": promotion_passed,
        "silent_fallback": False,
    }
