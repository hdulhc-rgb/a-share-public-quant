from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


REQUIRED_COLUMNS = {
    "source_event_id", "observed_at", "available_at", "source_url", "underlying", "expiry",
    "option_type", "exercise_style", "pricing_model", "strike", "bid", "ask", "volume",
    "open_interest", "contract_multiplier", "currency", "underlying_price", "quote_time",
    "risk_free_rate", "dividend_yield", "vendor_iv",
}


@dataclass(frozen=True)
class QualityPolicy:
    max_quote_staleness_minutes: int = 30
    max_source_staleness_minutes: int = 60
    min_open_interest: int = 100
    min_volume: int = 10
    max_spread_to_mid: float = 0.15
    max_model_residual_to_mid: float = 0.02


class EvidenceError(RuntimeError):
    pass


def quantlib_available() -> bool:
    return bool(importlib.util.find_spec("QuantLib"))


def _year_fraction(as_of: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - as_of).total_seconds() / (365.0 * 86400.0), 1.0 / 3650.0)


def _bs_price(option_type: str, spot: float, strike: float, rate: float, dividend: float, volatility: float, maturity: float) -> float:
    if min(spot, strike, volatility, maturity) <= 0:
        raise EvidenceError("BLACK_SCHOLES_INPUT_NON_POSITIVE")
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility * volatility) * maturity) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    if option_type == "CALL":
        return spot * math.exp(-dividend * maturity) * norm.cdf(d1) - strike * math.exp(-rate * maturity) * norm.cdf(d2)
    return strike * math.exp(-rate * maturity) * norm.cdf(-d2) - spot * math.exp(-dividend * maturity) * norm.cdf(-d1)


def _european_implied_vol(option_type: str, price: float, spot: float, strike: float, rate: float, dividend: float, maturity: float) -> float:
    if price <= 0:
        raise EvidenceError("OPTION_PRICE_NON_POSITIVE")
    objective = lambda vol: _bs_price(option_type, spot, strike, rate, dividend, vol, maturity) - price
    try:
        return float(brentq(objective, 1e-6, 8.0, maxiter=300))
    except ValueError as error:
        raise EvidenceError(f"IMPLIED_VOL_NO_ROOT:{error}") from error


def _quantlib_implied_vol(row: pd.Series, price: float, as_of: pd.Timestamp) -> tuple[float, float]:
    if not quantlib_available():
        raise EvidenceError("QUANTLIB_DEPENDENCY_MISSING")
    import QuantLib as ql  # type: ignore

    calculation_date = ql.Date(as_of.day, as_of.month, as_of.year)
    expiry_ts = pd.Timestamp(row["expiry"])
    expiry_date = ql.Date(expiry_ts.day, expiry_ts.month, expiry_ts.year)
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()
    calendar = ql.HongKong(ql.HongKong.HKEx)
    spot = ql.QuoteHandle(ql.SimpleQuote(float(row["underlying_price"])))
    risk = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(row["risk_free_rate"]), day_count))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(row["dividend_yield"]), day_count))
    volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, 0.30, day_count))
    process = ql.BlackScholesMertonProcess(spot, dividend, risk, volatility)
    option_type = ql.Option.Call if row["option_type"] == "CALL" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type, float(row["strike"]))
    if row["exercise_style"] == "AMERICAN":
        exercise = ql.AmericanExercise(calculation_date, expiry_date)
        engine = ql.BinomialVanillaEngine(process, "crr", 801)
    else:
        exercise = ql.EuropeanExercise(expiry_date)
        engine = ql.AnalyticEuropeanEngine(process)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    implied = float(option.impliedVolatility(float(price), process, 1e-8, 500, 1e-6, 8.0))
    updated_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, implied, day_count))
    updated_process = ql.BlackScholesMertonProcess(spot, dividend, risk, updated_vol)
    if row["exercise_style"] == "AMERICAN":
        updated_engine = ql.BinomialVanillaEngine(updated_process, "crr", 801)
    else:
        updated_engine = ql.AnalyticEuropeanEngine(updated_process)
    option.setPricingEngine(updated_engine)
    theoretical = float(option.NPV())
    return implied, theoretical


def _bounds(row: pd.Series, maturity: float) -> tuple[float, float]:
    spot = float(row["underlying_price"])
    strike = float(row["strike"])
    rate = float(row["risk_free_rate"])
    dividend = float(row["dividend_yield"])
    if row["exercise_style"] == "EUROPEAN":
        discounted_spot = spot * math.exp(-dividend * maturity)
        discounted_strike = strike * math.exp(-rate * maturity)
        if row["option_type"] == "CALL":
            return max(0.0, discounted_spot - discounted_strike), discounted_spot
        return max(0.0, discounted_strike - discounted_spot), discounted_strike
    if row["option_type"] == "CALL":
        return max(0.0, spot - strike), spot
    return max(0.0, strike - spot), strike


def _timestamp(value: object, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise EvidenceError(f"{name}_MUST_BE_TIMEZONE_AWARE")
    return result.tz_convert("UTC")


def validate_chain(chain: pd.DataFrame, as_of: pd.Timestamp, policy: QualityPolicy = QualityPolicy()) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(chain.columns))
    if missing:
        raise EvidenceError(f"MISSING_COLUMNS:{','.join(missing)}")
    if as_of.tzinfo is None:
        raise EvidenceError("AS_OF_MUST_BE_TIMEZONE_AWARE")
    as_of = as_of.tz_convert("UTC")
    output_rows: list[dict[str, object]] = []
    for _, raw in chain.copy().iterrows():
        row = raw.copy()
        result = row.to_dict()
        errors: list[str] = []
        warnings: list[str] = []
        try:
            observed_at = _timestamp(row["observed_at"], "OBSERVED_AT")
            available_at = _timestamp(row["available_at"], "AVAILABLE_AT")
            quote_time = _timestamp(row["quote_time"], "QUOTE_TIME")
            expiry = _timestamp(row["expiry"], "EXPIRY")
            if available_at > as_of or quote_time > as_of or observed_at > as_of:
                errors.append("FUTURE_EVIDENCE")
            if available_at < observed_at:
                errors.append("AVAILABLE_BEFORE_OBSERVED")
            quote_staleness = (as_of - quote_time).total_seconds() / 60.0
            source_staleness = (as_of - available_at).total_seconds() / 60.0
            result["quote_staleness_minutes"] = quote_staleness
            result["source_staleness_minutes"] = source_staleness
            if quote_staleness > policy.max_quote_staleness_minutes:
                warnings.append("QUOTE_STALE")
            if source_staleness > policy.max_source_staleness_minutes:
                warnings.append("SOURCE_STALE")
            maturity = _year_fraction(as_of, expiry)
            result["dte"] = int(math.ceil(maturity * 365.0))
        except Exception as error:
            errors.append(str(error))
            maturity = 0.0

        option_type = str(row["option_type"]).upper()
        exercise_style = str(row["exercise_style"]).upper()
        pricing_model = str(row["pricing_model"]).upper()
        result["option_type"] = option_type
        result["exercise_style"] = exercise_style
        result["pricing_model"] = pricing_model
        if option_type not in {"CALL", "PUT"}:
            errors.append("INVALID_OPTION_TYPE")
        if exercise_style not in {"EUROPEAN", "AMERICAN"}:
            errors.append("INVALID_EXERCISE_STYLE")
        if str(row["currency"]).upper() != "HKD":
            errors.append("CURRENCY_NOT_HKD")

        for field in ["strike", "bid", "ask", "volume", "open_interest", "contract_multiplier", "underlying_price", "risk_free_rate", "dividend_yield"]:
            try:
                result[field] = float(row[field])
            except Exception:
                errors.append(f"NON_NUMERIC_{field.upper()}")
        bid = float(result.get("bid", np.nan))
        ask = float(result.get("ask", np.nan))
        if not np.isfinite(bid) or not np.isfinite(ask) or bid < 0 or ask <= 0:
            errors.append("INVALID_BID_ASK")
        if np.isfinite(bid) and np.isfinite(ask) and ask < bid:
            errors.append("CROSSED_MARKET")
        mid = (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) else np.nan
        spread_to_mid = (ask - bid) / mid if mid > 0 else np.nan
        result["mid"] = mid
        result["spread_to_mid"] = spread_to_mid
        if np.isfinite(spread_to_mid) and spread_to_mid > policy.max_spread_to_mid:
            warnings.append("SPREAD_WIDE")
        if float(result.get("open_interest", 0)) < policy.min_open_interest:
            warnings.append("OPEN_INTEREST_THIN")
        if float(result.get("volume", 0)) < policy.min_volume:
            warnings.append("VOLUME_THIN")

        lower = upper = np.nan
        if not errors and maturity > 0:
            lower, upper = _bounds(pd.Series(result), maturity)
            result["no_arbitrage_lower"] = lower
            result["no_arbitrage_upper"] = upper
            if mid < lower - 1e-9 or mid > upper + 1e-9:
                errors.append("MID_OUTSIDE_NO_ARBITRAGE_BOUNDS")

        iv_values: dict[str, float | str | None] = {"iv_bid": None, "iv_mid": None, "iv_ask": None, "theoretical_mid": None, "model_residual_to_mid": None}
        if not errors and maturity > 0:
            try:
                if exercise_style == "EUROPEAN" and pricing_model == "BLACK_SCHOLES_EUROPEAN":
                    for label, price in [("iv_bid", bid), ("iv_mid", mid), ("iv_ask", ask)]:
                        iv_values[label] = _european_implied_vol(option_type, price, float(result["underlying_price"]), float(result["strike"]), float(result["risk_free_rate"]), float(result["dividend_yield"]), maturity)
                    theoretical = _bs_price(option_type, float(result["underlying_price"]), float(result["strike"]), float(result["risk_free_rate"]), float(result["dividend_yield"]), float(iv_values["iv_mid"]), maturity)
                elif pricing_model == "QUANTLIB_EXPLICIT":
                    bid_iv, _ = _quantlib_implied_vol(pd.Series(result), bid, as_of)
                    mid_iv, theoretical = _quantlib_implied_vol(pd.Series(result), mid, as_of)
                    ask_iv, _ = _quantlib_implied_vol(pd.Series(result), ask, as_of)
                    iv_values.update({"iv_bid": bid_iv, "iv_mid": mid_iv, "iv_ask": ask_iv})
                else:
                    raise EvidenceError("UNSUPPORTED_EXPLICIT_PRICING_MODEL")
                iv_values["theoretical_mid"] = theoretical
                iv_values["model_residual_to_mid"] = (theoretical - mid) / mid if mid else None
                if not (float(iv_values["iv_bid"]) <= float(iv_values["iv_mid"]) <= float(iv_values["iv_ask"])):
                    errors.append("IV_BAND_NON_MONOTONIC")
                if abs(float(iv_values["model_residual_to_mid"])) > policy.max_model_residual_to_mid:
                    warnings.append("MODEL_RESIDUAL_HIGH")
            except EvidenceError as error:
                warnings.append(str(error))
        result.update(iv_values)

        if errors:
            quality_status = "REJECT"
        elif any(item in warnings for item in ["QUANTLIB_DEPENDENCY_MISSING", "UNSUPPORTED_EXPLICIT_PRICING_MODEL"]):
            quality_status = "NOT_EVALUABLE"
        elif warnings:
            quality_status = "THIN"
        else:
            quality_status = "PASS"
        result["quality_status"] = quality_status
        result["reason_codes"] = ";".join(errors + warnings)
        result["evidence_only"] = True
        result["auto_trade_allowed"] = False
        result["covered_call_auto_trigger_allowed"] = False
        output_rows.append(result)
    return pd.DataFrame(output_rows)


def parity_residuals(validated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    key_columns = ["underlying", "expiry", "strike"]
    for keys, group in validated.groupby(key_columns, dropna=False):
        calls = group[group["option_type"] == "CALL"]
        puts = group[group["option_type"] == "PUT"]
        if calls.empty or puts.empty:
            continue
        call = calls.iloc[0]
        put = puts.iloc[0]
        status = "NOT_EVALUABLE"
        residual = None
        reason = "AMERICAN_PARITY_NOT_EXACT" if call["exercise_style"] == "AMERICAN" else ""
        if call["exercise_style"] == "EUROPEAN" and call["quality_status"] in {"PASS", "THIN"} and put["quality_status"] in {"PASS", "THIN"}:
            as_of = _timestamp(call["quote_time"], "QUOTE_TIME")
            expiry = _timestamp(call["expiry"], "EXPIRY")
            maturity = _year_fraction(as_of, expiry)
            theoretical_right = float(call["underlying_price"]) * math.exp(-float(call["dividend_yield"]) * maturity) - float(call["strike"]) * math.exp(-float(call["risk_free_rate"]) * maturity)
            residual = float(call["mid"] - put["mid"] - theoretical_right)
            status = "OK"
        rows.append({"underlying": keys[0], "expiry": keys[1], "strike": keys[2], "parity_residual": residual, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")

