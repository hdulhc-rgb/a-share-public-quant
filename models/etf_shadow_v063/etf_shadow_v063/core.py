from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_VERSION = "0.6.3"
TRADING_DAYS = 252
EPS = 1e-12


class ResearchClosed(RuntimeError):
    """Raised when a research invariant fails and the run must close."""


@dataclass(frozen=True)
class ResearchPolicy:
    annual_tracking_error_max: float = 0.04
    one_way_turnover_max: float = 0.10
    transaction_cost_bps: float = 10.0
    min_train_observations: int = 252
    test_observations: int = 63
    walk_forward_step: int = 63
    cpcv_folds: int = 6
    cpcv_test_folds: int = 2
    cpcv_embargo_observations: int = 5
    parameter_budget: int = 24
    differential_tolerance: float = 1e-10
    long_only: bool = True
    fully_invested: bool = True
    no_leverage: bool = True
    research_only: bool = True


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: object) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_returns(returns: pd.DataFrame, as_of: pd.Timestamp, minimum: int) -> None:
    if returns.empty or returns.shape[1] < 2:
        raise ResearchClosed("RETURN_PANEL_EMPTY_OR_TOO_NARROW")
    if len(returns) < minimum:
        raise ResearchClosed(f"INSUFFICIENT_HISTORY:{len(returns)}<{minimum}")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ResearchClosed("INDEX_NOT_DATETIME")
    if returns.index.tz is not None:
        returns.index = returns.index.tz_convert(None)
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ResearchClosed("RETURN_INDEX_NOT_STRICTLY_INCREASING")
    if returns.columns.has_duplicates:
        raise ResearchClosed("DUPLICATE_ASSET_COLUMNS")
    if returns.index.max() > as_of.tz_localize(None):
        raise ResearchClosed("AS_OF_VIOLATION")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ResearchClosed("NON_FINITE_RETURN")
    if (returns.abs() >= 1.0).any().any():
        raise ResearchClosed("IMPLAUSIBLE_DAILY_RETURN")


def normalize_long_only(weights: Sequence[float], assets: Sequence[str]) -> pd.Series:
    series = pd.Series(np.asarray(weights, dtype=float), index=list(assets), dtype=float)
    if not np.isfinite(series.to_numpy()).all():
        raise ResearchClosed("NON_FINITE_WEIGHT")
    if (series < -1e-10).any():
        raise ResearchClosed("SHORT_WEIGHT_PROPOSED")
    series = series.clip(lower=0.0)
    total = float(series.sum())
    if total <= EPS:
        raise ResearchClosed("ALL_CASH_OR_ZERO_WEIGHT_PROPOSAL")
    series /= total
    if abs(float(series.sum()) - 1.0) > 1e-10:
        raise ResearchClosed("WEIGHT_NORMALIZATION_FAILED")
    return series


def align_weights(weights: Mapping[str, float] | pd.Series, assets: Sequence[str]) -> pd.Series:
    series = pd.Series(dict(weights), dtype=float).reindex(list(assets)).fillna(0.0)
    return normalize_long_only(series.to_numpy(), assets)


def one_way_turnover(previous: pd.Series, target: pd.Series) -> float:
    left, right = previous.align(target, join="outer", fill_value=0.0)
    return 0.5 * float((right - left).abs().sum())


def drift_weights(previous: pd.Series, asset_returns: pd.Series) -> pd.Series:
    aligned_returns = asset_returns.reindex(previous.index)
    if aligned_returns.isna().any():
        raise ResearchClosed("MISSING_EXECUTION_RETURN")
    grown = previous * (1.0 + aligned_returns)
    total = float(grown.sum())
    if total <= EPS:
        raise ResearchClosed("DRIFTED_PORTFOLIO_NON_POSITIVE")
    return grown / total


def cap_turnover(previous: pd.Series, requested: pd.Series, cap: float) -> tuple[pd.Series, dict[str, float | bool]]:
    requested_turnover = one_way_turnover(previous, requested)
    scale = 1.0 if requested_turnover <= cap + 1e-12 else cap / requested_turnover
    actual = previous + scale * (requested - previous)
    actual = normalize_long_only(actual.to_numpy(), previous.index)
    actual_turnover = one_way_turnover(previous, actual)
    if actual_turnover > cap + 1e-9:
        raise ResearchClosed("TURNOVER_CAP_NOT_CONSERVED")
    return actual, {
        "requested_turnover": requested_turnover,
        "actual_turnover": actual_turnover,
        "turnover_scale": scale,
        "turnover_binding": bool(scale < 1.0 - 1e-12),
    }


def annual_tracking_error(weights: pd.Series, benchmark: pd.Series, covariance: pd.DataFrame) -> float:
    delta = (weights - benchmark).reindex(covariance.index).fillna(0.0).to_numpy(dtype=float)
    cov = covariance.reindex(index=covariance.index, columns=covariance.index).to_numpy(dtype=float)
    variance = max(float(delta @ cov @ delta), 0.0)
    return float(np.sqrt(variance * TRADING_DAYS))


def cap_tracking_error(target: pd.Series, benchmark: pd.Series, covariance: pd.DataFrame, cap: float) -> tuple[pd.Series, dict[str, float | bool]]:
    before = annual_tracking_error(target, benchmark, covariance)
    if before <= cap + 1e-12:
        return target, {"tracking_error_before": before, "tracking_error_after": before, "tracking_error_scale": 1.0, "tracking_error_binding": False}
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        candidate = benchmark + mid * (target - benchmark)
        if annual_tracking_error(candidate, benchmark, covariance) <= cap:
            low = mid
        else:
            high = mid
    constrained = normalize_long_only((benchmark + low * (target - benchmark)).to_numpy(), target.index)
    after = annual_tracking_error(constrained, benchmark, covariance)
    if after > cap + 1e-8:
        raise ResearchClosed("TRACKING_ERROR_CAP_NOT_CONSERVED")
    return constrained, {"tracking_error_before": before, "tracking_error_after": after, "tracking_error_scale": low, "tracking_error_binding": True}


def next_tradable_time(calendar: pd.DatetimeIndex, signal_time: pd.Timestamp) -> pd.Timestamp:
    future = calendar[calendar > signal_time]
    if len(future) == 0:
        raise ResearchClosed("NO_NEXT_TRADABLE_TIME")
    result = pd.Timestamp(future[0])
    if result <= signal_time:
        raise ResearchClosed("LOOKAHEAD_TIMING_INVARIANT_FAILED")
    return result


def data_fingerprint(returns: pd.DataFrame, source_path: Path | None, as_of: pd.Timestamp, source_id: str) -> dict[str, object]:
    csv_bytes = returns.to_csv(index=True, float_format="%.12g").encode("utf-8")
    return {
        "source_id": source_id,
        "source_path": str(source_path.resolve()) if source_path else "SYNTHETIC_FIXTURE",
        "source_file_sha256": sha256_file(source_path) if source_path else None,
        "normalized_panel_sha256": sha256_bytes(csv_bytes),
        "as_of": as_of.isoformat(),
        "min_date": returns.index.min().isoformat(),
        "max_date": returns.index.max().isoformat(),
        "rows": int(len(returns)),
        "columns": list(map(str, returns.columns)),
        "adjustment": "TOTAL_RETURN_REQUIRED_FOR_PRODUCTION; SYNTHETIC_IN_DEMO",
        "point_in_time": True,
    }


def create_run_directory(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    if run_dir.exists() and (run_dir / "run_manifest.json").exists():
        raise ResearchClosed("COMPLETED_RUN_ID_ALREADY_EXISTS")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def policy_dict(policy: ResearchPolicy) -> dict[str, object]:
    return asdict(policy)

