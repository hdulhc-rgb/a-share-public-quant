#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from meituan_options_v12.state_machine import (
    ExitAnchor,
    SaleState,
    apply_evidence_only_state_machine,
    counterfactual_covered_calls,
)
from meituan_options_v12.validator import (
    EvidenceError,
    QualityPolicy,
    file_sha256,
    parity_residuals,
    quantlib_available,
    validate_chain,
    write_json,
)


def synthetic_chain(as_of: pd.Timestamp) -> pd.DataFrame:
    """Deterministic, non-market fixture used only for contract tests."""

    rows = []
    spot = 100.0
    for strike in [80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
        for option_type in ["CALL", "PUT"]:
            intrinsic = max(spot - strike, 0.0) if option_type == "CALL" else max(strike - spot, 0.0)
            mid = intrinsic + 3.0 + 0.02 * abs(strike - spot)
            rows.append({
                "source_event_id": f"SYN-{strike:.0f}-{option_type}",
                "observed_at": (as_of - pd.Timedelta(minutes=4)).isoformat(),
                "available_at": (as_of - pd.Timedelta(minutes=3)).isoformat(),
                "source_url": "SYNTHETIC_FIXTURE_NOT_MARKET_DATA",
                "underlying": "SYNTHETIC_HK_STOCK",
                "expiry": (as_of + pd.Timedelta(days=90)).isoformat(),
                "option_type": option_type,
                "exercise_style": "EUROPEAN",
                "pricing_model": "BLACK_SCHOLES_EUROPEAN",
                "strike": strike,
                "bid": max(0.01, mid * 0.98),
                "ask": mid * 1.02,
                "volume": 200,
                "open_interest": 1000,
                "contract_multiplier": 100,
                "currency": "HKD",
                "underlying_price": spot,
                "quote_time": (as_of - pd.Timedelta(minutes=4)).isoformat(),
                "risk_free_rate": 0.025,
                "dividend_yield": 0.0,
                "vendor_iv": np.nan,
            })
    return pd.DataFrame(rows)


def synthetic_policy() -> tuple[SaleState, list[ExitAnchor]]:
    return (
        SaleState(
            price=100.0,
            protection_floor=90.0,
            hard_risk_count=0,
            base_sale_units=1000,
            risk_overlay_multiplier=1.0,
            lot_size=100,
        ),
        [ExitAnchor(90.0, 0), ExitAnchor(110.0, 1000), ExitAnchor(120.0, 1000)],
    )


def load_policy(path: Path) -> tuple[SaleState, list[ExitAnchor]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"price", "protection_floor", "anchors"}
    missing = sorted(required - set(payload))
    if missing:
        raise EvidenceError(f"POLICY_MISSING_FIELDS:{','.join(missing)}")
    state = SaleState(
        price=float(payload["price"]),
        protection_floor=float(payload["protection_floor"]),
        hard_risk_count=int(payload.get("hard_risk_count", 0)),
        base_sale_units=int(payload.get("base_sale_units", 0)),
        risk_overlay_multiplier=float(payload.get("risk_overlay_multiplier", 1.0)),
        lot_size=int(payload.get("lot_size", 1)),
    )
    anchors = [
        ExitAnchor(price=float(item["price"]), reference_units=int(item["reference_units"]))
        for item in payload["anchors"]
    ]
    if not anchors:
        raise EvidenceError("POLICY_REQUIRES_AT_LEAST_ONE_ANCHOR")
    return state, anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-safe v1.2 single-name options evidence validator")
    parser.add_argument("--chain-csv", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--as-of", default="2026-01-15T08:00:00+08:00")
    parser.add_argument("--output-dir", type=Path, default=Path("option_evidence_run_v12"))
    parser.add_argument("--fee-per-contract", type=float)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    as_of = pd.Timestamp(args.as_of)
    if as_of.tzinfo is None:
        print("FAILED_CLOSED: --as-of must include timezone", file=sys.stderr)
        return 2
    if not args.demo and (args.chain_csv is None or args.policy_json is None):
        print("FAILED_CLOSED: production runs require --chain-csv and --policy-json", file=sys.stderr)
        return 2

    try:
        if args.demo:
            chain = synthetic_chain(as_of)
            state, anchors = synthetic_policy()
        else:
            chain = pd.read_csv(args.chain_csv)
            state, anchors = load_policy(args.policy_json)
        validated = validate_chain(chain, as_of, QualityPolicy())
    except (EvidenceError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FAILED_CLOSED: {error}", file=sys.stderr)
        return 2

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    validated.to_csv(output / "validated_chain.csv", index=False)
    parity_residuals(validated).to_csv(output / "parity_residuals.csv", index=False)
    overall_quality = (
        "PASS"
        if len(validated) and (validated["quality_status"] == "PASS").all()
        else "NOT_EVALUABLE"
        if (validated["quality_status"] == "NOT_EVALUABLE").any()
        else "THIN_OR_REJECT"
    )
    invariant_report = apply_evidence_only_state_machine(
        state,
        overall_quality,
        option_evidence_requests_covered_call=True,
    )
    write_json(output / "invariant_report.json", invariant_report)
    counterfactual_covered_calls(validated, anchors, args.fee_per_contract).to_csv(
        output / "cc_counterfactual.csv",
        index=False,
    )
    snapshot = {
        "as_of": as_of.isoformat(),
        "source_file": str(args.chain_csv.resolve()) if args.chain_csv else "SYNTHETIC_FIXTURE",
        "source_sha256": file_sha256(args.chain_csv) if args.chain_csv else None,
        "policy_file_sha256": file_sha256(args.policy_json) if args.policy_json else None,
        "rows": int(len(validated)),
        "quality_counts": validated["quality_status"].value_counts(dropna=False).to_dict(),
        "quantlib": "AVAILABLE" if quantlib_available() else "DEPENDENCY_MISSING",
        "silent_model_fallback": False,
        "evidence_only": True,
        "covered_call_auto_trigger_allowed": False,
        "price_anchors_rewritten": False,
        "broker_connection": False,
        "orders_generated": False,
    }
    write_json(output / "evidence_snapshot.json", snapshot)
    required = [
        "validated_chain.csv",
        "parity_residuals.csv",
        "invariant_report.json",
        "cc_counterfactual.csv",
        "evidence_snapshot.json",
    ]
    manifest = {
        "version": "1.2.0-public-safe",
        "status": (
            "SHADOW_EVIDENCE_ONLY"
            if invariant_report["overall_invariant_status"] == "PASS"
            else "FAILED_CLOSED"
        ),
        "artifacts": [
            {
                "path": name,
                "sha256": file_sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in required
        ],
        "broker_connection": False,
        "orders_generated": False,
        "covered_call_auto_triggered": False,
    }
    write_json(output / "run_manifest.json", manifest)
    print(output)
    return 0 if manifest["status"] == "SHADOW_EVIDENCE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
