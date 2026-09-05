from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from etf_shadow_v063.core import ResearchClosed, sha256_file


MODEL_VERSION = "0.6.3-data.2"
SCHEMA_VERSION = "2.0"
MIN_OBSERVATIONS = 315
MAX_STALENESS_CALENDAR_DAYS = 7
QDII_ASSETS = {"US_SP500", "US_NASDAQ100"}
QDII_SOURCE_COMPONENTS = {"market", "underlying", "fx", "premium"}
SKFOLIO_PIN = "0.20.1"
OFFICIAL_SOURCE_TYPES = {
    "official_total_return_index",
    "official_fund_adjusted_nav",
    "official_exchange_nav",
    "official_distribution_reconstruction",
}
LOCAL_PROXY_MODES = {
    "POINT_IN_TIME_EFFECTIVE_DATED",
    "CURRENT_FORWARD_ONLY",
}


def load_return_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ResearchClosed("RETURN_PANEL_REQUIRES_DATE_COLUMN")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.set_index("date").sort_index()
    if frame.empty:
        raise ResearchClosed("RETURN_PANEL_EMPTY")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise ResearchClosed(f"RETURN_PANEL_NON_NUMERIC:{error}") from error
    return frame


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchClosed(f"DATA2_MANIFEST_UNREADABLE:{error}") from error
    if not isinstance(manifest, dict):
        raise ResearchClosed("DATA2_MANIFEST_NOT_OBJECT")
    return manifest


def _required_timestamp(value: object, reason: str) -> pd.Timestamp:
    if value is None or not str(value).strip():
        raise ResearchClosed(reason)
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResearchClosed(reason) from error
    if pd.isna(timestamp):
        raise ResearchClosed(reason)
    return timestamp


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ResearchClosed(f"DATA2_MANIFEST_PATH_ESCAPE:{relative}")
    return candidate


def _validate_panel_shape(
    returns: pd.DataFrame,
    as_of: pd.Timestamp,
    minimum_observations: int = MIN_OBSERVATIONS,
) -> None:
    cutoff = pd.Timestamp(as_of).normalize()
    if len(returns) < minimum_observations:
        raise ResearchClosed("DATA2_RETURN_PANEL_TOO_SHORT")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ResearchClosed("DATA2_RETURN_PANEL_DATE_IDENTITY_FAILED")
    if returns.columns.has_duplicates or not len(returns.columns):
        raise ResearchClosed("DATA2_RETURN_PANEL_ASSET_IDENTITY_FAILED")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ResearchClosed("DATA2_RETURN_PANEL_NON_FINITE")
    if np.abs(values).max() >= 1.0:
        raise ResearchClosed("DATA2_RETURN_PANEL_IMPOSSIBLE_RETURN")
    if returns.index.max().normalize() > cutoff:
        raise ResearchClosed("DATA2_RETURN_PANEL_FUTURE_DATED")
    staleness = int((cutoff - returns.index.max().normalize()).days)
    if staleness < 0 or staleness > MAX_STALENESS_CALENDAR_DAYS:
        raise ResearchClosed("DATA2_RETURN_PANEL_STALE")


def validate_official_economic_manifest(
    returns: pd.DataFrame,
    returns_path: Path,
    manifest_path: Path,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    """Validate a frozen economic-return panel backed by official raw evidence.

    The validator intentionally does not prescribe one provider API. It requires an
    immutable raw packet per model asset and records the official source class. The
    transformation from those raw levels into returns must be deterministic and
    represented by the panel hash.
    """

    _validate_panel_shape(returns, as_of)
    manifest = _read_manifest(manifest_path)
    cutoff = pd.Timestamp(as_of).normalize()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResearchClosed("ECONOMIC_MANIFEST_SCHEMA_UNSUPPORTED")
    if manifest.get("demo") is not False:
        raise ResearchClosed("ECONOMIC_MANIFEST_MARKED_DEMO")
    if _required_timestamp(manifest.get("as_of"), "ECONOMIC_MANIFEST_AS_OF_MISSING").normalize() != cutoff:
        raise ResearchClosed("ECONOMIC_MANIFEST_AS_OF_MISMATCH")
    if manifest.get("source_snapshot_frozen") is not True:
        raise ResearchClosed("ECONOMIC_SOURCE_SNAPSHOT_NOT_FROZEN")

    panel = manifest.get("return_panel") or {}
    if panel.get("return_kind") != "official_economic_total_return":
        raise ResearchClosed("ECONOMIC_RETURN_KIND_NOT_OFFICIAL")
    if panel.get("sha256") != sha256_file(returns_path):
        raise ResearchClosed("ECONOMIC_RETURN_PANEL_HASH_MISMATCH")
    if int(panel.get("rows", -1)) != len(returns):
        raise ResearchClosed("ECONOMIC_RETURN_PANEL_ROW_COUNT_MISMATCH")
    if list(panel.get("columns") or []) != list(returns.columns):
        raise ResearchClosed("ECONOMIC_RETURN_PANEL_ASSET_IDENTITY_MISMATCH")
    if panel.get("min_date") != returns.index.min().date().isoformat():
        raise ResearchClosed("ECONOMIC_RETURN_PANEL_MIN_DATE_MISMATCH")
    if panel.get("max_date") != returns.index.max().date().isoformat():
        raise ResearchClosed("ECONOMIC_RETURN_PANEL_MAX_DATE_MISMATCH")

    evidence = manifest.get("asset_evidence")
    if not isinstance(evidence, list):
        raise ResearchClosed("ECONOMIC_ASSET_EVIDENCE_MISSING")
    evidence_assets = [str(item.get("asset")) for item in evidence if isinstance(item, dict)]
    if len(evidence_assets) != len(set(evidence_assets)) or set(evidence_assets) != set(returns.columns):
        raise ResearchClosed("ECONOMIC_ASSET_EVIDENCE_COVERAGE_MISMATCH")

    root = manifest_path.parent
    provider_classes: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise ResearchClosed("ECONOMIC_ASSET_EVIDENCE_INVALID")
        asset = str(item.get("asset"))
        source_type = str(item.get("source_type"))
        if source_type not in OFFICIAL_SOURCE_TYPES:
            raise ResearchClosed(f"ECONOMIC_SOURCE_NOT_OFFICIAL:{asset}")
        if item.get("point_in_time") is not True:
            raise ResearchClosed(f"ECONOMIC_SOURCE_NOT_POINT_IN_TIME:{asset}")
        source_url = str(item.get("source_url", ""))
        if not source_url.startswith("https://"):
            raise ResearchClosed(f"ECONOMIC_SOURCE_URL_INVALID:{asset}")
        available_at = _required_timestamp(
            item.get("available_at"), f"ECONOMIC_SOURCE_AVAILABLE_AT_MISSING:{asset}"
        )
        if available_at.tzinfo is not None:
            available_at = available_at.tz_convert(None)
        if available_at.normalize() > cutoff:
            raise ResearchClosed(f"ECONOMIC_SOURCE_AVAILABLE_AFTER_AS_OF:{asset}")
        if available_at.normalize() < returns.index.max().normalize():
            raise ResearchClosed(f"ECONOMIC_SOURCE_SNAPSHOT_PREDATES_PANEL:{asset}")
        relative = str(item.get("raw_path", ""))
        raw_path = _safe_relative_file(root, relative)
        if not raw_path.is_file() or item.get("raw_sha256") != sha256_file(raw_path):
            raise ResearchClosed(f"ECONOMIC_RAW_SOURCE_HASH_MISMATCH:{asset}")
        provider_classes[asset] = source_type

    gates = manifest.get("gates") or {}
    required_gates = (
        "official_source_identity_passed",
        "point_in_time_passed",
        "raw_packet_frozen_passed",
        "economic_return_panel_passed",
    )
    if not all(gates.get(key) is True for key in required_gates):
        raise ResearchClosed("ECONOMIC_MANIFEST_GATE_NOT_ATTESTED")
    return {
        "status": "PASS",
        "data_grade": "A",
        "return_kind": "official_economic_total_return",
        "adjustment": "official_economic_total_return",
        "manifest_sha256": sha256_file(manifest_path),
        "panel_sha256": sha256_file(returns_path),
        "provider_classes": provider_classes,
        "providers": provider_classes,
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "production_return_panel_gate_passed": True,
        "promotion_eligible": True,
    }


def validate_local_proxy_manifest(
    local_returns: pd.DataFrame,
    returns_path: Path,
    manifest_path: Path,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    _validate_panel_shape(local_returns, as_of, minimum_observations=1)
    manifest = _read_manifest(manifest_path)
    cutoff = pd.Timestamp(as_of).normalize()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResearchClosed("LOCAL_PROXY_MANIFEST_SCHEMA_UNSUPPORTED")
    if _required_timestamp(manifest.get("as_of"), "LOCAL_PROXY_MANIFEST_AS_OF_MISSING").normalize() != cutoff:
        raise ResearchClosed("LOCAL_PROXY_MANIFEST_AS_OF_MISMATCH")
    if manifest.get("point_in_time") is not True:
        raise ResearchClosed("LOCAL_PROXY_NOT_POINT_IN_TIME")
    mode = str(manifest.get("construction_mode"))
    if mode not in LOCAL_PROXY_MODES:
        raise ResearchClosed("LOCAL_PROXY_CONSTRUCTION_MODE_INVALID")
    if manifest.get("panel_sha256") != sha256_file(returns_path):
        raise ResearchClosed("LOCAL_PROXY_PANEL_HASH_MISMATCH")
    model_assets = list(manifest.get("model_assets") or [])
    if set(model_assets) != set(local_returns.columns):
        raise ResearchClosed("LOCAL_PROXY_ASSET_IDENTITY_MISMATCH")
    effective_from = _required_timestamp(
        manifest.get("effective_from"), "LOCAL_PROXY_EFFECTIVE_DATE_MISSING"
    ).normalize()
    if effective_from > cutoff:
        raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_AFTER_AS_OF")
    if local_returns.index.min().normalize() < effective_from:
        raise ResearchClosed("LOCAL_PROXY_BACKCAST_BEFORE_EFFECTIVE_DATE")
    if int(manifest.get("constituent_count", 0)) <= 0:
        raise ResearchClosed("LOCAL_PROXY_CONSTITUENT_COUNT_INVALID")
    if not str(manifest.get("component_packet_sha256", "")):
        raise ResearchClosed("LOCAL_PROXY_COMPONENT_PACKET_HASH_MISSING")
    history_evidence_count = 0
    if mode == "POINT_IN_TIME_EFFECTIVE_DATED":
        history_evidence = manifest.get("history_evidence")
        if not isinstance(history_evidence, list):
            raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_EVIDENCE_MISSING")
        if len(history_evidence) != int(manifest["constituent_count"]):
            raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_EVIDENCE_COUNT_MISMATCH")
        component_ids: set[str] = set()
        for item in history_evidence:
            if not isinstance(item, dict):
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_EVIDENCE_INVALID")
            component_id = str(item.get("component_id", ""))
            if not component_id or component_id in component_ids:
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_COMPONENT_ID_INVALID")
            component_ids.add(component_id)
            if item.get("point_in_time") is not True:
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_NOT_POINT_IN_TIME")
            if not str(item.get("source_url", "")).startswith("https://"):
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_SOURCE_URL_INVALID")
            available_at = _required_timestamp(
                item.get("available_at"), "LOCAL_PROXY_EFFECTIVE_HISTORY_AVAILABLE_AT_MISSING"
            )
            if available_at.tzinfo is not None:
                available_at = available_at.tz_convert(None)
            if available_at.normalize() > cutoff:
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_FUTURE_DATA")
            raw_path = _safe_relative_file(
                manifest_path.parent, str(item.get("raw_path", ""))
            )
            if not raw_path.is_file() or item.get("raw_sha256") != sha256_file(raw_path):
                raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_RAW_HASH_MISMATCH")
        gates = manifest.get("history_gates") or {}
        if not all(
            gates.get(key) is True
            for key in (
                "effective_dated_weights_passed",
                "constituent_survivorship_passed",
                "raw_packet_frozen_passed",
            )
        ):
            raise ResearchClosed("LOCAL_PROXY_EFFECTIVE_HISTORY_GATE_NOT_ATTESTED")
        history_evidence_count = len(history_evidence)
    return {
        "status": "PASS",
        "construction_mode": mode,
        "effective_from": effective_from.isoformat(),
        "model_assets": model_assets,
        "constituent_count": int(manifest["constituent_count"]),
        "history_evidence_count": history_evidence_count,
        "panel_sha256": sha256_file(returns_path),
        "historical_backtest_eligible": mode == "POINT_IN_TIME_EFFECTIVE_DATED",
        "promotion_eligible": mode == "POINT_IN_TIME_EFFECTIVE_DATED",
    }


def apply_local_proxy(
    base_returns: pd.DataFrame,
    local_returns: pd.DataFrame,
    attestation: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = base_returns.copy()
    assets = list(attestation.get("model_assets") or [])
    unknown = sorted(set(assets) - set(result.columns))
    if unknown:
        raise ResearchClosed(f"LOCAL_PROXY_UNKNOWN_MODEL_ASSET:{unknown}")
    effective_from = pd.Timestamp(attestation["effective_from"]).normalize()
    if (
        bool(attestation.get("historical_backtest_eligible"))
        and effective_from > result.index.min().normalize()
    ):
        raise ResearchClosed("LOCAL_PROXY_HISTORICAL_COVERAGE_START_GAP")
    required_dates = result.index[result.index.normalize() > effective_from]
    replaced_rows: dict[str, int] = {}
    for asset in assets:
        available_dates = local_returns.index[local_returns[asset].notna()]
        missing_dates = required_dates.difference(available_dates)
        if len(missing_dates):
            preview = ",".join(date.date().isoformat() for date in missing_dates[:5])
            suffix = "..." if len(missing_dates) > 5 else ""
            raise ResearchClosed(
                f"LOCAL_PROXY_FORWARD_COVERAGE_GAP:{asset}:{preview}{suffix}"
            )
        overlap = result.index.intersection(local_returns.index)
        if len(overlap):
            result.loc[overlap, asset] = local_returns.loc[overlap, asset]
        replaced_rows[asset] = int(len(overlap))
    return result, {
        "status": "PASS",
        "construction_mode": attestation["construction_mode"],
        "model_assets": assets,
        "replaced_rows": replaced_rows,
        "required_forward_rows": int(len(required_dates)),
        "coverage_through": (
            required_dates.max().date().isoformat() if len(required_dates) else None
        ),
        "historical_backtest_eligible": bool(attestation["historical_backtest_eligible"]),
        "no_pre_effective_backcast": True,
        "complete_forward_coverage": True,
    }


def validate_qdii_decomposition(
    path: Path,
    manifest_path: Path,
    as_of: pd.Timestamp,
    execution_returns: pd.DataFrame,
    minimum_rows_per_asset: int = 252,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    manifest = _read_manifest(manifest_path)
    cutoff = pd.Timestamp(as_of).normalize()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResearchClosed("QDII_MANIFEST_SCHEMA_UNSUPPORTED")
    if _required_timestamp(manifest.get("as_of"), "QDII_MANIFEST_AS_OF_MISSING").normalize() != cutoff:
        raise ResearchClosed("QDII_MANIFEST_AS_OF_MISMATCH")
    if manifest.get("point_in_time") is not True:
        raise ResearchClosed("QDII_MANIFEST_NOT_POINT_IN_TIME")
    if manifest.get("source_snapshot_frozen") is not True:
        raise ResearchClosed("QDII_SOURCE_SNAPSHOT_NOT_FROZEN")
    if manifest.get("panel_sha256") != sha256_file(path):
        raise ResearchClosed("QDII_DECOMPOSITION_PANEL_HASH_MISMATCH")
    if set(manifest.get("assets") or []) != QDII_ASSETS:
        raise ResearchClosed("QDII_MANIFEST_ASSET_COVERAGE_MISMATCH")
    if manifest.get("construction_formula") != "MULTIPLICATIVE_RETURN_IDENTITY":
        raise ResearchClosed("QDII_CONSTRUCTION_FORMULA_INVALID")
    raw_evidence = manifest.get("raw_evidence")
    if not isinstance(raw_evidence, list):
        raise ResearchClosed("QDII_RAW_EVIDENCE_MISSING")
    observed_pairs: set[tuple[str, str]] = set()
    raw_snapshot_dates: list[pd.Timestamp] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise ResearchClosed("QDII_RAW_EVIDENCE_INVALID")
        asset = str(item.get("asset"))
        component = str(item.get("component"))
        pair = (asset, component)
        if pair in observed_pairs:
            raise ResearchClosed("QDII_RAW_EVIDENCE_DUPLICATE_KEY")
        observed_pairs.add(pair)
        if asset not in QDII_ASSETS or component not in QDII_SOURCE_COMPONENTS:
            raise ResearchClosed("QDII_RAW_EVIDENCE_IDENTITY_MISMATCH")
        if item.get("point_in_time") is not True:
            raise ResearchClosed(f"QDII_RAW_EVIDENCE_NOT_POINT_IN_TIME:{asset}:{component}")
        if not str(item.get("source_url", "")).startswith("https://"):
            raise ResearchClosed(f"QDII_RAW_EVIDENCE_URL_INVALID:{asset}:{component}")
        available_at = _required_timestamp(
            item.get("available_at"),
            f"QDII_RAW_EVIDENCE_AVAILABLE_AT_MISSING:{asset}:{component}",
        )
        if available_at.tzinfo is not None:
            available_at = available_at.tz_convert(None)
        if available_at.normalize() > cutoff:
            raise ResearchClosed(f"QDII_RAW_EVIDENCE_FUTURE:{asset}:{component}")
        raw_snapshot_dates.append(available_at.normalize())
        raw_path = _safe_relative_file(manifest_path.parent, str(item.get("raw_path", "")))
        if not raw_path.is_file() or item.get("raw_sha256") != sha256_file(raw_path):
            raise ResearchClosed(f"QDII_RAW_EVIDENCE_HASH_MISMATCH:{asset}:{component}")
    expected_pairs = {
        (asset, component)
        for asset in QDII_ASSETS
        for component in QDII_SOURCE_COMPONENTS
    }
    if observed_pairs != expected_pairs:
        raise ResearchClosed("QDII_RAW_EVIDENCE_COVERAGE_MISMATCH")
    gates = manifest.get("gates") or {}
    if not all(
        gates.get(key) is True
        for key in (
            "source_identity_passed",
            "point_in_time_passed",
            "raw_packet_frozen_passed",
            "decomposition_panel_passed",
        )
    ):
        raise ResearchClosed("QDII_MANIFEST_GATE_NOT_ATTESTED")

    frame = pd.read_csv(path)
    required = {
        "date",
        "asset",
        "market_return",
        "underlying_return_local",
        "fx_return",
        "premium_return",
        "fee_tracking_residual",
        "available_at",
    }
    if not required.issubset(frame.columns):
        raise ResearchClosed(f"QDII_DECOMPOSITION_SCHEMA_MISMATCH:{sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="raise", utc=True).dt.tz_convert(None)
    if frame.duplicated(["date", "asset"]).any():
        raise ResearchClosed("QDII_DECOMPOSITION_DUPLICATE_KEY")
    if any(snapshot < frame["date"].max() for snapshot in raw_snapshot_dates):
        raise ResearchClosed("QDII_RAW_SOURCE_SNAPSHOT_PREDATES_PANEL")
    if set(frame["asset"]) != QDII_ASSETS:
        raise ResearchClosed("QDII_DECOMPOSITION_ASSET_COVERAGE_MISMATCH")
    if not QDII_ASSETS.issubset(execution_returns.columns):
        raise ResearchClosed("QDII_EXECUTION_PANEL_ASSET_COVERAGE_MISMATCH")
    if frame["date"].max() > cutoff or frame["available_at"].max().normalize() > cutoff:
        raise ResearchClosed("QDII_DECOMPOSITION_FUTURE_DATA")
    if (frame["available_at"].dt.normalize() < frame["date"]).any():
        raise ResearchClosed("QDII_DECOMPOSITION_AVAILABLE_BEFORE_MARKET_DATE")
    numeric = [
        "market_return",
        "underlying_return_local",
        "fx_return",
        "premium_return",
        "fee_tracking_residual",
    ]
    values = frame[numeric].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values <= -1.0).any().any():
        raise ResearchClosed("QDII_DECOMPOSITION_INVALID_RETURN")
    reconstructed_log = (
        np.log1p(values["underlying_return_local"])
        + np.log1p(values["fx_return"])
        + np.log1p(values["premium_return"])
        + np.log1p(values["fee_tracking_residual"])
    )
    identity_error = np.log1p(values["market_return"]) - reconstructed_log
    max_error = float(identity_error.abs().max())
    if max_error > tolerance:
        raise ResearchClosed("QDII_DECOMPOSITION_IDENTITY_FAILED")

    execution = execution_returns.sort_index()
    required_dates = execution.index[-minimum_rows_per_asset:]
    if len(required_dates) < minimum_rows_per_asset:
        raise ResearchClosed("QDII_EXECUTION_PANEL_HISTORY_INSUFFICIENT")
    market_panel_errors: list[float] = []
    for asset in sorted(QDII_ASSETS):
        subset = frame.loc[frame["asset"] == asset].set_index("date").sort_index()
        missing_dates = required_dates.difference(subset.index)
        if len(missing_dates):
            preview = ",".join(date.date().isoformat() for date in missing_dates[:5])
            suffix = "..." if len(missing_dates) > 5 else ""
            raise ResearchClosed(
                f"QDII_DECOMPOSITION_FORWARD_COVERAGE_GAP:{asset}:{preview}{suffix}"
            )
        unknown_dates = subset.index.difference(execution.index)
        if len(unknown_dates):
            raise ResearchClosed(f"QDII_DECOMPOSITION_UNKNOWN_MARKET_DATE:{asset}")
        aligned_market = subset["market_return"].astype(float)
        aligned_execution = execution.loc[aligned_market.index, asset].astype(float)
        market_panel_errors.extend((aligned_market - aligned_execution).abs().tolist())
    max_market_panel_error = float(max(market_panel_errors, default=0.0))
    if max_market_panel_error > tolerance:
        raise ResearchClosed("QDII_DECOMPOSITION_MARKET_PANEL_MISMATCH")

    row_counts = frame.groupby("asset").size().to_dict()
    coverage_passed = all(int(row_counts.get(asset, 0)) >= minimum_rows_per_asset for asset in QDII_ASSETS)
    return {
        "status": "PASS" if coverage_passed else "INSUFFICIENT_HISTORY",
        "sha256": sha256_file(path),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_evidence_count": len(raw_evidence),
        "row_counts": {str(key): int(value) for key, value in row_counts.items()},
        "identity_max_abs_log_error": max_error,
        "identity_tolerance": tolerance,
        "market_panel_max_abs_error": max_market_panel_error,
        "required_coverage_start": required_dates.min().date().isoformat(),
        "required_coverage_end": required_dates.max().date().isoformat(),
        "promotion_eligible": coverage_passed,
    }


def dependency_evidence(profile: str) -> dict[str, object]:
    if profile not in {"shadow", "promotion"}:
        raise ResearchClosed("DATA2_PROFILE_INVALID")
    packages: dict[str, dict[str, object]] = {}
    for name in ("skfolio", "vectorbt"):
        installed = importlib.util.find_spec(name) is not None
        version: str | None = None
        if installed:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = "UNKNOWN"
        packages[name] = {"installed": installed, "version": version}
    skfolio_passed = bool(
        packages["skfolio"]["installed"]
        and packages["skfolio"]["version"] == SKFOLIO_PIN
    )
    return {
        "profile": profile,
        "packages": packages,
        "skfolio_required_for_promotion": True,
        "skfolio_required_version": SKFOLIO_PIN,
        "vectorbt_optional": True,
        "promotion_dependency_gate_passed": skfolio_passed,
        "silent_fallback": False,
    }


def assess_data2_grade(
    *,
    execution_gate_passed: bool,
    official_economic_attestation: Mapping[str, object] | None,
    local_proxy_attestation: Mapping[str, object] | None,
    qdii_attestation: Mapping[str, object] | None,
    dependency_attestation: Mapping[str, object],
    profile: str,
) -> dict[str, object]:
    gaps: list[str] = []
    if not execution_gate_passed:
        gaps.append("EXECUTION_RETURN_PANEL_GATE_FAILED")
    if official_economic_attestation is None:
        gaps.append("OFFICIAL_ECONOMIC_RETURN_PANEL_MISSING")
    if local_proxy_attestation is None:
        gaps.append("LOCAL_A_SHARE_PROXY_MISSING")
    elif not bool(local_proxy_attestation.get("historical_backtest_eligible")):
        gaps.append("LOCAL_A_SHARE_PROXY_FORWARD_ONLY")
    if qdii_attestation is None:
        gaps.append("QDII_DECOMPOSITION_MISSING")
    elif not bool(qdii_attestation.get("promotion_eligible")):
        gaps.append("QDII_DECOMPOSITION_HISTORY_INSUFFICIENT")
    if not bool(dependency_attestation.get("promotion_dependency_gate_passed")):
        gaps.append("SKFOLIO_VALIDATION_MISSING")

    promotion_gate = not gaps
    if profile == "promotion" and not promotion_gate:
        raise ResearchClosed(f"DATA2_PROMOTION_GATE_FAILED:{'|'.join(gaps)}")
    if not execution_gate_passed:
        raise ResearchClosed("DATA2_SHADOW_EXECUTION_GATE_FAILED")
    return {
        "status": "PASS_SHADOW" if gaps else "PASS_PROMOTION_ELIGIBLE",
        "data_grade": "A" if not gaps else "B",
        "shadow_gate_passed": True,
        "promotion_gate_passed": promotion_gate,
        "evidence_gaps": gaps,
        "economic_return_policy": (
            "OFFICIAL_ECONOMIC_PANEL" if official_economic_attestation is not None else "EXPLICIT_EXECUTION_PROXY_FALLBACK"
        ),
        "no_silent_fallback": True,
    }


def align_return_panels(
    economic_returns: pd.DataFrame,
    execution_returns: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(economic_returns.columns) != set(execution_returns.columns):
        raise ResearchClosed("DUAL_PANEL_ASSET_IDENTITY_MISMATCH")
    columns = list(execution_returns.columns)
    common = economic_returns.index.intersection(execution_returns.index).sort_values()
    economic = economic_returns.reindex(index=common, columns=columns).dropna(how="any")
    execution = execution_returns.reindex(index=economic.index, columns=columns).dropna(how="any")
    economic = economic.reindex(execution.index)
    _validate_panel_shape(economic, as_of)
    _validate_panel_shape(execution, as_of)
    if not economic.index.equals(execution.index):
        raise ResearchClosed("DUAL_PANEL_CALENDAR_ALIGNMENT_FAILED")
    return economic, execution


def return_basis_diagnostics(
    economic_returns: pd.DataFrame,
    execution_returns: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset in execution_returns.columns:
        economic = economic_returns[asset].astype(float)
        execution = execution_returns[asset].astype(float)
        difference = execution - economic
        log_gap = float((np.log1p(execution) - np.log1p(economic)).sum())
        correlation: float | None = None
        if float(execution.std()) > 0.0 and float(economic.std()) > 0.0:
            value = float(execution.corr(economic))
            correlation = value if np.isfinite(value) else None
        rows.append(
            {
                "asset": asset,
                "observations": int(len(difference)),
                "median_abs_daily_gap": float(difference.abs().median()),
                "p99_abs_daily_gap": float(difference.abs().quantile(0.99)),
                "max_abs_daily_gap": float(difference.abs().max()),
                "cumulative_log_gap": log_gap,
                "return_correlation": correlation,
            }
        )
    return pd.DataFrame(rows)
