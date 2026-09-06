from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .core import ResearchClosed


COLLECTOR_VERSION = "0.6.3-data.1"
SCHEMA_VERSION = "1.0"
RETURN_KIND = "distribution_adjusted_market_return_proxy"
PRIMARY_PROVIDER = "Eastmoney"
SECONDARY_PROVIDER = "Tencent"
PRIMARY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SECONDARY_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
AKSHARE_DOCUMENTATION = "https://akshare.akfamily.xyz/data/fund/fund_public.html"

MIN_RETURN_OBSERVATIONS = 315
MAX_STALENESS_CALENDAR_DAYS = 7
MIN_SECONDARY_OVERLAP = 252
MEDIAN_ABS_RETURN_DIFF_MAX = 0.0005
P99_ABS_RETURN_DIFF_MAX = 0.005
MAX_ABS_RETURN_DIFF_MAX = 0.03
CUMULATIVE_LOG_RETURN_GAP_MAX = 0.03
RETURN_CORRELATION_MIN = 0.98
LOW_VOLATILITY_DAILY_STD_MAX = 0.00075
LOW_VOLATILITY_P99_DIFF_MAX = 0.0015


class DataGateClosed(RuntimeError):
    """Raised after a production data contract fails closed."""


@dataclass(frozen=True)
class AssetProxy:
    asset: str
    security: str
    market_symbol: str
    name: str


PROXIES = (
    AssetProxy("A_SHARE", "510300", "sh510300", "沪深300ETF华泰柏瑞"),
    AssetProxy("US_SP500", "513500", "sh513500", "标普500ETF博时"),
    AssetProxy("US_NASDAQ100", "513100", "sh513100", "纳指ETF国泰"),
    AssetProxy("GOLD", "159937", "sz159937", "黄金ETF博时"),
    AssetProxy("CASH", "511880", "sh511880", "银华日利ETF"),
)


PrimaryFetcher = Callable[[AssetProxy, pd.Timestamp, pd.Timestamp, str], pd.DataFrame]
SecondaryFetcher = Callable[[AssetProxy, pd.Timestamp, pd.Timestamp], pd.DataFrame]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_text(
    url: str,
    params: Mapping[str, str],
    attempts: int = 3,
    timeout: float = 30.0,
) -> str:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                request_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; etf-shadow-research/0.6.3)"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except Exception as error:  # pragma: no cover - live network path
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise DataGateClosed(f"PUBLIC_SOURCE_UNAVAILABLE:{url}:{last_error}")


def _normalize_ohlc(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"date", "open", "close", "high", "low"}
    if not required.issubset(frame.columns):
        raise DataGateClosed(f"SOURCE_SCHEMA_MISMATCH:{label}:{sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    numeric = [column for column in ["open", "close", "high", "low", "volume", "amount"] if column in result]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if result.empty:
        raise DataGateClosed(f"SOURCE_EMPTY:{label}")
    if result[["open", "close", "high", "low"]].isna().any().any():
        raise DataGateClosed(f"SOURCE_NON_NUMERIC_OHLC:{label}")
    if (result[["open", "close", "high", "low"]] <= 0).any().any():
        raise DataGateClosed(f"SOURCE_NON_POSITIVE_OHLC:{label}")
    return result


def fetch_eastmoney(
    proxy: AssetProxy,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjustment: str,
) -> pd.DataFrame:
    if adjustment not in {"qfq", "unadjusted"}:
        raise ValueError(f"unsupported adjustment: {adjustment}")
    market_id = "1" if proxy.security.startswith(("5", "6")) else "0"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1" if adjustment == "qfq" else "0",
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "secid": f"{market_id}.{proxy.security}",
    }
    payload = json.loads(_request_text(PRIMARY_URL, params))
    data = payload.get("data") or {}
    rows = data.get("klines") or []
    parsed = [row.split(",") for row in rows]
    if not parsed:
        raise DataGateClosed(f"SOURCE_EMPTY:{PRIMARY_PROVIDER}:{proxy.security}:{adjustment}")
    frame = pd.DataFrame(
        parsed,
        columns=["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct", "change", "turnover"],
    )
    frame = _normalize_ohlc(frame, f"{PRIMARY_PROVIDER}:{proxy.security}:{adjustment}")
    return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)


def fetch_tencent(
    proxy: AssetProxy,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        params = {
            "_var": f"kline_dayqfq{year}",
            "param": f"{proxy.market_symbol},day,{year}-01-01,{year}-12-31,640,qfq",
            "r": "0.8205512681390605",
        }
        text = _request_text(SECONDARY_URL, params)
        separator = text.find("=")
        if separator < 0:
            raise DataGateClosed(f"SOURCE_SCHEMA_MISMATCH:{SECONDARY_PROVIDER}:{proxy.security}")
        payload = json.loads(text[separator + 1 :])
        security_payload = (payload.get("data") or {}).get(proxy.market_symbol) or {}
        rows = security_payload.get("qfqday") or security_payload.get("day") or []
        if rows:
            frames.append(
                pd.DataFrame(
                    [row[:6] for row in rows],
                    columns=["date", "open", "close", "high", "low", "amount"],
                )
            )
    if not frames:
        raise DataGateClosed(f"SOURCE_EMPTY:{SECONDARY_PROVIDER}:{proxy.security}:qfq")
    frame = _normalize_ohlc(pd.concat(frames, ignore_index=True), f"{SECONDARY_PROVIDER}:{proxy.security}:qfq")
    return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)


def _frame_integrity(frame: pd.DataFrame, start: pd.Timestamp, as_of: pd.Timestamp) -> dict[str, object]:
    returns = frame.set_index("date")["close"].pct_change(fill_method=None).dropna()
    return {
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "staleness_calendar_days": int((as_of - frame["date"].max()).days),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "null_close": int(frame["close"].isna().sum()),
        "non_positive_close": int(frame["close"].le(0).sum()),
        "largest_abs_daily_return": float(returns.abs().max()),
        "covers_requested_start": bool(frame["date"].min() <= start),
        "freshness_passed": bool(0 <= (as_of - frame["date"].max()).days <= MAX_STALENESS_CALENDAR_DAYS),
        "integrity_passed": bool(
            not frame["date"].duplicated().any()
            and not frame["close"].isna().any()
            and frame["close"].gt(0).all()
            and not returns.empty
            and returns.abs().lt(1.0).all()
        ),
    }


def _compare_adjusted_returns(primary: pd.DataFrame, secondary: pd.DataFrame) -> dict[str, object]:
    left = primary.set_index("date")["close"].pct_change(fill_method=None).rename("primary")
    right = secondary.set_index("date")["close"].pct_change(fill_method=None).rename("secondary")
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if joined.empty:
        return {
            "overlap_rows": 0,
            "median_abs_return_diff": None,
            "p99_abs_return_diff": None,
            "max_abs_return_diff": None,
            "cumulative_log_return_gap": None,
            "return_correlation": None,
            "low_volatility_exception": False,
            "secondary_total_return_gate_passed": False,
        }
    difference = (joined["primary"] - joined["secondary"]).abs()
    correlation = float(joined.corr().iloc[0, 1]) if len(joined) > 1 else float("nan")
    primary_std = float(joined["primary"].std(ddof=1))
    secondary_std = float(joined["secondary"].std(ddof=1))
    median_diff = float(difference.median())
    p99_diff = float(difference.quantile(0.99))
    max_diff = float(difference.max())
    cumulative_gap = float(
        abs(np.log1p(joined["primary"]).sum() - np.log1p(joined["secondary"]).sum())
    )
    low_volatility_exception = bool(
        max(primary_std, secondary_std) <= LOW_VOLATILITY_DAILY_STD_MAX
        and p99_diff <= LOW_VOLATILITY_P99_DIFF_MAX
    )
    correlation_passed = bool(np.isfinite(correlation) and correlation >= RETURN_CORRELATION_MIN)
    gate = bool(
        len(joined) >= MIN_SECONDARY_OVERLAP
        and median_diff <= MEDIAN_ABS_RETURN_DIFF_MAX
        and p99_diff <= P99_ABS_RETURN_DIFF_MAX
        and max_diff <= MAX_ABS_RETURN_DIFF_MAX
        and cumulative_gap <= CUMULATIVE_LOG_RETURN_GAP_MAX
        and (correlation_passed or low_volatility_exception)
    )
    return {
        "overlap_rows": int(len(joined)),
        "median_abs_return_diff": median_diff,
        "p99_abs_return_diff": p99_diff,
        "max_abs_return_diff": max_diff,
        "cumulative_log_return_gap": cumulative_gap,
        "return_correlation": correlation if np.isfinite(correlation) else None,
        "primary_daily_std": primary_std,
        "secondary_daily_std": secondary_std,
        "low_volatility_exception": low_volatility_exception,
        "secondary_total_return_gate_passed": gate,
    }


def _safe_relative_file(manifest_dir: Path, relative: str) -> Path:
    candidate = (manifest_dir / relative).resolve()
    root = manifest_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ResearchClosed(f"DATA_MANIFEST_PATH_ESCAPE:{relative}")
    return candidate


def build_production_panel(
    start: pd.Timestamp,
    as_of: pd.Timestamp,
    output_dir: Path,
    primary_fetcher: PrimaryFetcher = fetch_eastmoney,
    secondary_fetcher: SecondaryFetcher = fetch_tencent,
) -> Path:
    start = pd.Timestamp(start).normalize()
    as_of = pd.Timestamp(as_of).normalize()
    if start >= as_of:
        raise ValueError("start must be earlier than as_of")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    adjusted_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    raw_hashes: dict[str, str] = {}

    for proxy in PROXIES:
        primary_qfq = _normalize_ohlc(
            primary_fetcher(proxy, start, as_of, "qfq"),
            f"{PRIMARY_PROVIDER}:{proxy.security}:qfq",
        )
        primary_raw = _normalize_ohlc(
            primary_fetcher(proxy, start, as_of, "unadjusted"),
            f"{PRIMARY_PROVIDER}:{proxy.security}:unadjusted",
        )
        secondary_qfq = _normalize_ohlc(
            secondary_fetcher(proxy, start, as_of),
            f"{SECONDARY_PROVIDER}:{proxy.security}:qfq",
        )
        frames = {
            "eastmoney_qfq": primary_qfq,
            "eastmoney_unadjusted": primary_raw,
            "tencent_qfq": secondary_qfq,
        }
        for label, frame in frames.items():
            path = raw_dir / f"{proxy.asset}_{proxy.security}_{label}.csv"
            frame.to_csv(path, index=False)
            raw_hashes[str(path.relative_to(output_dir))] = sha256_file(path)

        primary_integrity = _frame_integrity(primary_qfq, start, as_of)
        secondary_integrity = _frame_integrity(secondary_qfq, start, as_of)
        raw_dates_match = bool(primary_qfq["date"].equals(primary_raw["date"]))
        adjustment_events = 0
        if raw_dates_match:
            ratio = primary_qfq["close"].to_numpy() / primary_raw["close"].to_numpy()
            adjustment_events = int(pd.Series(ratio).pct_change(fill_method=None).abs().gt(1e-8).sum())
        comparison = _compare_adjusted_returns(primary_qfq, secondary_qfq)
        asset_gate = bool(
            primary_integrity["integrity_passed"]
            and primary_integrity["freshness_passed"]
            and secondary_integrity["integrity_passed"]
            and secondary_integrity["freshness_passed"]
            and raw_dates_match
            and comparison["secondary_total_return_gate_passed"]
        )
        quality_rows.append(
            {
                "asset": proxy.asset,
                "security": proxy.security,
                "name": proxy.name,
                **{f"primary_{key}": value for key, value in primary_integrity.items()},
                "secondary_rows": secondary_integrity["rows"],
                "secondary_first_date": secondary_integrity["first_date"],
                "secondary_last_date": secondary_integrity["last_date"],
                "secondary_staleness_calendar_days": secondary_integrity["staleness_calendar_days"],
                "secondary_integrity_passed": secondary_integrity["integrity_passed"],
                "secondary_freshness_passed": secondary_integrity["freshness_passed"],
                "qfq_unadjusted_dates_match": raw_dates_match,
                "qfq_to_unadjusted_ratio_change_events": adjustment_events,
                **comparison,
                "asset_production_gate_passed": asset_gate,
            }
        )
        selected = primary_qfq[["date", "close"]].copy()
        selected["asset"] = proxy.asset
        adjusted_frames.append(selected)

    quality = pd.DataFrame(quality_rows)
    quality_path = output_dir / "data_quality.csv"
    quality.to_csv(quality_path, index=False)

    long_prices = pd.concat(adjusted_frames, ignore_index=True)
    prices = (
        long_prices.pivot(index="date", columns="asset", values="close")
        .reindex(columns=[proxy.asset for proxy in PROXIES])
        .dropna(how="any")
        .sort_index()
    )
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    returns_path = output_dir / "returns.csv"
    returns.rename_axis("date").to_csv(returns_path, float_format="%.12g")

    common_staleness = int((as_of - returns.index.max()).days) if not returns.empty else 10**9
    common_panel_gate = bool(
        len(returns) >= MIN_RETURN_OBSERVATIONS
        and returns.index.is_monotonic_increasing
        and not returns.index.has_duplicates
        and np.isfinite(returns.to_numpy(dtype=float)).all()
        and returns.abs().lt(1.0).all().all()
        and 0 <= common_staleness <= MAX_STALENESS_CALENDAR_DAYS
        and returns.index.max() <= as_of
    )
    asset_gate = bool(quality["asset_production_gate_passed"].all())
    production_gate = bool(asset_gate and common_panel_gate)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.date().isoformat(),
        "requested_start": start.date().isoformat(),
        "demo": False,
        "universe": [asdict(proxy) for proxy in PROXIES],
        "return_panel": {
            "path": returns_path.name,
            "sha256": sha256_file(returns_path),
            "rows": int(len(returns)),
            "columns": list(returns.columns),
            "min_date": returns.index.min().date().isoformat() if not returns.empty else None,
            "max_date": returns.index.max().date().isoformat() if not returns.empty else None,
            "staleness_calendar_days": common_staleness,
            "return_kind": RETURN_KIND,
            "adjustment": "qfq",
            "source_snapshot_frozen": True,
        },
        "sources": {
            "primary": {
                "provider": PRIMARY_PROVIDER,
                "endpoint": PRIMARY_URL,
                "adjustment": "qfq",
            },
            "secondary": {
                "provider": SECONDARY_PROVIDER,
                "endpoint": SECONDARY_URL,
                "adjustment": "qfq",
            },
            "adjustment_semantics_documentation": AKSHARE_DOCUMENTATION,
        },
        "verification_policy": {
            "minimum_secondary_overlap": MIN_SECONDARY_OVERLAP,
            "median_abs_return_diff_max": MEDIAN_ABS_RETURN_DIFF_MAX,
            "p99_abs_return_diff_max": P99_ABS_RETURN_DIFF_MAX,
            "max_abs_return_diff_max": MAX_ABS_RETURN_DIFF_MAX,
            "cumulative_log_return_gap_max": CUMULATIVE_LOG_RETURN_GAP_MAX,
            "return_correlation_min": RETURN_CORRELATION_MIN,
            "low_volatility_daily_std_max": LOW_VOLATILITY_DAILY_STD_MAX,
            "low_volatility_p99_diff_max": LOW_VOLATILITY_P99_DIFF_MAX,
            "maximum_staleness_calendar_days": MAX_STALENESS_CALENDAR_DAYS,
        },
        "quality": {
            "path": quality_path.name,
            "sha256": sha256_file(quality_path),
        },
        "raw_file_hashes": raw_hashes,
        "gates": {
            "asset_integrity_and_freshness_passed": asset_gate,
            "dual_source_adjusted_return_verification_passed": bool(
                quality["secondary_total_return_gate_passed"].all()
            ),
            "common_return_panel_passed": common_panel_gate,
            "production_return_panel_passed": production_gate,
        },
        "research_lock": True,
        "broker_connection": False,
        "orders_generated": False,
    }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not production_gate:
        raise DataGateClosed(f"PRODUCTION_RETURN_PANEL_GATE_FAILED:{manifest_path}")
    return manifest_path


def validate_production_manifest(
    returns: pd.DataFrame,
    returns_path: Path,
    manifest_path: Path,
    as_of: pd.Timestamp,
) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchClosed(f"DATA_MANIFEST_UNREADABLE:{error}") from error
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResearchClosed("DATA_MANIFEST_SCHEMA_UNSUPPORTED")
    if manifest.get("demo") is not False:
        raise ResearchClosed("PRODUCTION_DATA_MARKED_AS_DEMO")
    manifest_as_of = pd.Timestamp(manifest.get("as_of")).normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    if manifest_as_of != cutoff:
        raise ResearchClosed("DATA_MANIFEST_AS_OF_MISMATCH")
    gates = manifest.get("gates") or {}
    required_gates = (
        "asset_integrity_and_freshness_passed",
        "dual_source_adjusted_return_verification_passed",
        "common_return_panel_passed",
        "production_return_panel_passed",
    )
    if not all(gates.get(gate) is True for gate in required_gates):
        raise ResearchClosed("PRODUCTION_RETURN_PANEL_GATE_NOT_ATTESTED")
    panel = manifest.get("return_panel") or {}
    if panel.get("return_kind") != RETURN_KIND or panel.get("adjustment") != "qfq":
        raise ResearchClosed("RETURN_KIND_NOT_DISTRIBUTION_ADJUSTED")
    if panel.get("source_snapshot_frozen") is not True:
        raise ResearchClosed("SOURCE_SNAPSHOT_NOT_FROZEN")
    if panel.get("sha256") != sha256_file(returns_path):
        raise ResearchClosed("RETURN_PANEL_HASH_MISMATCH")
    if int(panel.get("rows", -1)) != len(returns):
        raise ResearchClosed("RETURN_PANEL_ROW_COUNT_MISMATCH")
    if list(panel.get("columns") or []) != list(returns.columns):
        raise ResearchClosed("RETURN_PANEL_ASSET_IDENTITY_MISMATCH")
    min_date = returns.index.min().date().isoformat()
    max_date = returns.index.max().date().isoformat()
    if panel.get("min_date") != min_date or panel.get("max_date") != max_date:
        raise ResearchClosed("RETURN_PANEL_DATE_IDENTITY_MISMATCH")
    staleness = int((cutoff - returns.index.max().normalize()).days)
    if staleness < 0 or staleness > MAX_STALENESS_CALENDAR_DAYS:
        raise ResearchClosed("RETURN_PANEL_STALE_OR_FUTURE_DATED")
    if int(panel.get("staleness_calendar_days", -1)) != staleness:
        raise ResearchClosed("RETURN_PANEL_STALENESS_MISMATCH")
    sources = manifest.get("sources") or {}
    primary = sources.get("primary") or {}
    secondary = sources.get("secondary") or {}
    if (
        primary.get("provider") != PRIMARY_PROVIDER
        or secondary.get("provider") != SECONDARY_PROVIDER
        or primary.get("provider") == secondary.get("provider")
        or primary.get("adjustment") != "qfq"
        or secondary.get("adjustment") != "qfq"
    ):
        raise ResearchClosed("DUAL_SOURCE_ADJUSTMENT_IDENTITY_MISMATCH")
    manifest_dir = manifest_path.parent
    quality = manifest.get("quality") or {}
    quality_path = _safe_relative_file(manifest_dir, str(quality.get("path", "")))
    if not quality_path.is_file() or quality.get("sha256") != sha256_file(quality_path):
        raise ResearchClosed("DATA_QUALITY_HASH_MISMATCH")
    raw_hashes = manifest.get("raw_file_hashes") or {}
    expected_raw_paths = {
        f"raw/{proxy.asset}_{proxy.security}_{suffix}.csv"
        for proxy in PROXIES
        for suffix in ("eastmoney_qfq", "eastmoney_unadjusted", "tencent_qfq")
    }
    if set(raw_hashes) != expected_raw_paths:
        raise ResearchClosed("RAW_SOURCE_PACKET_INCOMPLETE")
    for relative, expected_hash in sorted(raw_hashes.items()):
        raw_path = _safe_relative_file(manifest_dir, relative)
        if not raw_path.is_file() or sha256_file(raw_path) != expected_hash:
            raise ResearchClosed(f"RAW_SOURCE_HASH_MISMATCH:{relative}")

    # Do not trust attested booleans alone. Recompute the source and derivation
    # gates from the frozen raw packet that was just hash-verified.
    requested_start = pd.Timestamp(manifest.get("requested_start")).normalize()
    reconstructed_frames: list[pd.DataFrame] = []
    recomputed_quality: list[dict[str, object]] = []
    for proxy in PROXIES:
        prefix = f"raw/{proxy.asset}_{proxy.security}"
        primary_qfq = _normalize_ohlc(
            pd.read_csv(_safe_relative_file(manifest_dir, f"{prefix}_eastmoney_qfq.csv")),
            f"VERIFY:{PRIMARY_PROVIDER}:{proxy.security}:qfq",
        )
        primary_raw = _normalize_ohlc(
            pd.read_csv(
                _safe_relative_file(manifest_dir, f"{prefix}_eastmoney_unadjusted.csv")
            ),
            f"VERIFY:{PRIMARY_PROVIDER}:{proxy.security}:unadjusted",
        )
        secondary_qfq = _normalize_ohlc(
            pd.read_csv(_safe_relative_file(manifest_dir, f"{prefix}_tencent_qfq.csv")),
            f"VERIFY:{SECONDARY_PROVIDER}:{proxy.security}:qfq",
        )
        primary_integrity = _frame_integrity(primary_qfq, requested_start, cutoff)
        secondary_integrity = _frame_integrity(secondary_qfq, requested_start, cutoff)
        comparison = _compare_adjusted_returns(primary_qfq, secondary_qfq)
        dates_match = bool(primary_qfq["date"].equals(primary_raw["date"]))
        recomputed_gate = bool(
            primary_integrity["integrity_passed"]
            and primary_integrity["freshness_passed"]
            and secondary_integrity["integrity_passed"]
            and secondary_integrity["freshness_passed"]
            and dates_match
            and comparison["secondary_total_return_gate_passed"]
        )
        recomputed_quality.append(
            {
                "asset": proxy.asset,
                "gate": recomputed_gate,
                "comparison": comparison,
            }
        )
        selected = primary_qfq[["date", "close"]].copy()
        selected["asset"] = proxy.asset
        reconstructed_frames.append(selected)
    if not all(row["gate"] for row in recomputed_quality):
        raise ResearchClosed("RECOMPUTED_DUAL_SOURCE_DATA_GATE_FAILED")
    reconstructed_prices = (
        pd.concat(reconstructed_frames, ignore_index=True)
        .pivot(index="date", columns="asset", values="close")
        .reindex(columns=[proxy.asset for proxy in PROXIES])
        .dropna(how="any")
        .sort_index()
    )
    reconstructed_returns = reconstructed_prices.pct_change(fill_method=None).dropna(how="any")
    if (
        not reconstructed_returns.index.equals(returns.index)
        or list(reconstructed_returns.columns) != list(returns.columns)
        or not np.allclose(
            reconstructed_returns.to_numpy(dtype=float),
            returns.to_numpy(dtype=float),
            rtol=1e-10,
            atol=5e-12,
        )
    ):
        raise ResearchClosed("RETURN_PANEL_NOT_DERIVED_FROM_FROZEN_PRIMARY_SOURCE")
    quality_frame = pd.read_csv(quality_path)
    if (
        set(quality_frame.get("asset", pd.Series(dtype=str)))
        != {proxy.asset for proxy in PROXIES}
        or not quality_frame["asset_production_gate_passed"].astype(bool).all()
        or not quality_frame["secondary_total_return_gate_passed"].astype(bool).all()
    ):
        raise ResearchClosed("DATA_QUALITY_ATTESTATION_MISMATCH")
    return {
        "source_id": f"PUBLIC_DUAL_SOURCE_QFQ:{PRIMARY_PROVIDER}+{SECONDARY_PROVIDER}",
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "return_kind": RETURN_KIND,
        "adjustment": "qfq",
        "providers": [PRIMARY_PROVIDER, SECONDARY_PROVIDER],
        "production_return_panel_gate_passed": True,
        "research_lock": True,
    }
