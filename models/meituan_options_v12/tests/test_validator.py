from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from meituan_options_v12.state_machine import (
    ExitAnchor,
    SaleState,
    apply_evidence_only_state_machine,
    counterfactual_covered_calls,
)
from meituan_options_v12.validator import validate_chain


def base_row() -> dict[str, object]:
    as_of = pd.Timestamp("2026-01-15T08:00:00+08:00")
    return {
        "source_event_id": "SYNTHETIC-TEST-1",
        "observed_at": (as_of - pd.Timedelta(minutes=5)).isoformat(),
        "available_at": (as_of - pd.Timedelta(minutes=4)).isoformat(),
        "source_url": "SYNTHETIC_TEST_FIXTURE",
        "underlying": "SYNTHETIC_HK_STOCK",
        "expiry": (as_of + pd.Timedelta(days=90)).isoformat(),
        "option_type": "CALL",
        "exercise_style": "EUROPEAN",
        "pricing_model": "BLACK_SCHOLES_EUROPEAN",
        "strike": 105.0,
        "bid": 5.90,
        "ask": 6.10,
        "volume": 100,
        "open_interest": 1000,
        "contract_multiplier": 100,
        "currency": "HKD",
        "underlying_price": 100.0,
        "quote_time": (as_of - pd.Timedelta(minutes=5)).isoformat(),
        "risk_free_rate": 0.025,
        "dividend_yield": 0.0,
        "vendor_iv": np.nan,
    }


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.as_of = pd.Timestamp("2026-01-15T08:00:00+08:00")

    def test_valid_quote_is_evidence_only(self) -> None:
        validated = validate_chain(pd.DataFrame([base_row()]), self.as_of)
        self.assertEqual(validated.iloc[0]["quality_status"], "PASS")
        self.assertTrue(validated.iloc[0]["evidence_only"])
        self.assertFalse(validated.iloc[0]["covered_call_auto_trigger_allowed"])

    def test_future_quote_rejected(self) -> None:
        row = base_row()
        row["quote_time"] = (self.as_of + pd.Timedelta(minutes=1)).isoformat()
        validated = validate_chain(pd.DataFrame([row]), self.as_of)
        self.assertEqual(validated.iloc[0]["quality_status"], "REJECT")
        self.assertIn("FUTURE_EVIDENCE", validated.iloc[0]["reason_codes"])

    def test_crossed_market_rejected(self) -> None:
        row = base_row()
        row["bid"], row["ask"] = 7.0, 6.0
        validated = validate_chain(pd.DataFrame([row]), self.as_of)
        self.assertEqual(validated.iloc[0]["quality_status"], "REJECT")

    def test_configured_floor_cannot_be_broken_by_options_or_overlay(self) -> None:
        state = SaleState(
            price=80.0,
            protection_floor=90.0,
            hard_risk_count=0,
            base_sale_units=1000,
            risk_overlay_multiplier=1.25,
            lot_size=100,
        )
        result = apply_evidence_only_state_machine(
            state,
            "PASS",
            option_evidence_requests_covered_call=True,
        )
        self.assertEqual(result["recommended_sale_units"], 0)
        self.assertFalse(result["covered_call_auto_triggered"])
        self.assertEqual(result["overall_invariant_status"], "PASS")

    def test_counterfactual_is_never_actionable(self) -> None:
        validated = validate_chain(pd.DataFrame([base_row()]), self.as_of)
        anchors = [ExitAnchor(90.0, 0), ExitAnchor(110.0, 1000)]
        table = counterfactual_covered_calls(validated, anchors, 5.0)
        self.assertFalse(table["actionable"].any())
        self.assertEqual(int(table.loc[table["anchor_price"] == 90.0, "reference_units"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
