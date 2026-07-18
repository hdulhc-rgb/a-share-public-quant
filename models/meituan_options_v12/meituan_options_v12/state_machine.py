from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SaleState:
    """Runtime state supplied by a private policy layer.

    The public package deliberately provides no company, portfolio, price-floor,
    position-size, cost-basis, or vesting defaults.
    """

    price: float
    protection_floor: float
    hard_risk_count: int = 0
    base_sale_units: int = 0
    risk_overlay_multiplier: float = 1.0
    lot_size: int = 1


@dataclass(frozen=True)
class ExitAnchor:
    price: float
    reference_units: int


def apply_evidence_only_state_machine(
    state: SaleState,
    option_quality: str,
    option_evidence_requests_covered_call: bool = False,
) -> dict[str, object]:
    if state.price <= 0 or state.protection_floor <= 0 or state.lot_size <= 0:
        raise ValueError("STATE_PRICE_FLOOR_AND_LOT_MUST_BE_POSITIVE")
    if state.base_sale_units < 0 or state.hard_risk_count < 0:
        raise ValueError("STATE_COUNTS_MUST_BE_NON_NEGATIVE")

    floor_active = state.price < state.protection_floor and state.hard_risk_count < 2
    if floor_active:
        sale_units = 0
    else:
        sale_units = max(0, int(state.base_sale_units // state.lot_size) * state.lot_size)
        if sale_units > 0 and state.price >= state.protection_floor:
            sale_units = max(
                0,
                int((sale_units * state.risk_overlay_multiplier) // state.lot_size) * state.lot_size,
            )

    result = {
        "input_state": asdict(state),
        "option_quality": option_quality,
        "option_evidence_requests_covered_call": bool(option_evidence_requests_covered_call),
        "recommended_sale_units": sale_units,
        "covered_call_status": "HUMAN_REVIEW_ONLY" if option_quality == "PASS" else "DISABLED",
        "covered_call_auto_triggered": False,
        "price_anchors_rewritten": False,
        "protection_floor_effective": floor_active,
        "broker_connection": False,
        "order_payload": None,
    }
    result["invariants"] = {
        "OPTIONS_NEVER_AUTO_TRIGGER_CC": result["covered_call_auto_triggered"] is False,
        "OPTIONS_NEVER_REWRITE_PRICE_ANCHORS": result["price_anchors_rewritten"] is False,
        "CONFIGURED_FLOOR_EFFECTIVE_UNLESS_HARD_RISK_GE_2": (sale_units == 0) if floor_active else True,
        "RISK_OVERLAY_CANNOT_BREAK_CONFIGURED_FLOOR": (sale_units == 0) if floor_active else True,
        "NO_BROKER_OR_ORDER": result["broker_connection"] is False and result["order_payload"] is None,
    }
    result["overall_invariant_status"] = "PASS" if all(result["invariants"].values()) else "FAIL"
    return result


def counterfactual_covered_calls(
    validated: pd.DataFrame,
    anchors: Iterable[ExitAnchor],
    fee_per_contract: float | None = None,
) -> pd.DataFrame:
    """Build non-actionable covered-call scenarios for caller-supplied anchors."""

    rows: list[dict[str, object]] = []
    calls = validated[
        (validated["option_type"] == "CALL")
        & (validated["quality_status"].isin(["PASS", "THIN"]))
    ].copy()
    for anchor in anchors:
        eligible = calls.copy()
        if not eligible.empty:
            eligible["strike_distance"] = (eligible["strike"].astype(float) - anchor.price).abs()
            chosen = eligible.sort_values(["strike_distance", "spread_to_mid", "dte"]).iloc[0]
            multiplier = int(chosen["contract_multiplier"])
            contracts = anchor.reference_units // multiplier if anchor.reference_units > 0 else 0
            if fee_per_contract is None:
                net_premium = None
                status = "NOT_EVALUABLE"
                reason = "FEE_UNKNOWN"
            else:
                net_premium = float(chosen["mid"] * multiplier * contracts - fee_per_contract * contracts)
                status = "COUNTERFACTUAL_ONLY"
                reason = ""
            rows.append({
                "anchor_price": anchor.price,
                "reference_units": anchor.reference_units,
                "contract_multiplier": multiplier,
                "contracts": contracts,
                "expiry": chosen["expiry"],
                "strike": chosen["strike"],
                "bid": chosen["bid"],
                "ask": chosen["ask"],
                "mid": chosen["mid"],
                "quality_status": chosen["quality_status"],
                "net_premium": net_premium,
                "effective_exit_price": (
                    float(chosen["strike"]) + net_premium / anchor.reference_units
                    if net_premium is not None and anchor.reference_units
                    else None
                ),
                "status": status,
                "reason": reason,
                "actionable": False,
                "covered_call_auto_triggered": False,
            })
        else:
            rows.append({
                "anchor_price": anchor.price,
                "reference_units": anchor.reference_units,
                "status": "NOT_EVALUABLE",
                "reason": "NO_VALID_CALL_QUOTE",
                "actionable": False,
                "covered_call_auto_triggered": False,
            })
    return pd.DataFrame(rows)
