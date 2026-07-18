from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_shadow_v063.backtest import differential_check
from etf_shadow_v063.challengers import inverse_volatility
from etf_shadow_v063.core import ResearchClosed, cap_turnover, next_tradable_time, normalize_long_only, validate_returns
from etf_shadow_v063.validation import anchored_walk_forward, combinatorial_purged_cv


class CoreInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assets = ["A", "B", "C"]
        self.index = pd.bdate_range("2024-01-02", periods=400)
        rng = np.random.default_rng(7)
        self.returns = pd.DataFrame(rng.normal(0, 0.01, (400, 3)), index=self.index, columns=self.assets)

    def test_next_tradable_time_is_strictly_later(self) -> None:
        signal = self.index[250]
        execution = next_tradable_time(self.index, signal)
        self.assertGreater(execution, signal)
        self.assertEqual(execution, self.index[251])

    def test_inverse_volatility_records_zero_volatility_floor(self) -> None:
        returns = pd.DataFrame({"RISK": [0.01, -0.01, 0.02, -0.02], "CASH": [0.0, 0.0, 0.0, 0.0]})
        result = inverse_volatility(returns)
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.diagnostics["floored_assets"], ["CASH"])
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)

    def test_as_of_violation_fails_closed(self) -> None:
        with self.assertRaises(ResearchClosed):
            validate_returns(self.returns, self.index[-2], 252)

    def test_turnover_cap_conserves_weights(self) -> None:
        previous = pd.Series([0.8, 0.1, 0.1], index=self.assets)
        requested = pd.Series([0.1, 0.8, 0.1], index=self.assets)
        actual, info = cap_turnover(previous, requested, 0.10)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=12)
        self.assertLessEqual(float(info["actual_turnover"]), 0.10 + 1e-12)
        self.assertTrue(info["turnover_binding"])

    def test_zero_weight_proposal_is_rejected(self) -> None:
        with self.assertRaises(ResearchClosed):
            normalize_long_only([0, 0, 0], self.assets)

    def test_differential_backtest(self) -> None:
        weights = pd.Series([0.2, 0.3, 0.5], index=self.assets)
        result = differential_check(self.returns.iloc[:50], weights, 1e-12)
        self.assertEqual(result["status"], "PASS")

    def test_splitters_do_not_overlap(self) -> None:
        wf = anchored_walk_forward(400, 252, 63, 63)
        self.assertTrue(wf)
        for split in wf:
            self.assertFalse(set(split.train).intersection(split.test))
            self.assertLess(max(split.train), min(split.test))
        cpcv = combinatorial_purged_cv(400, 6, 2, 5)
        self.assertTrue(cpcv)
        for split in cpcv:
            self.assertFalse(set(split.train).intersection(split.test))


if __name__ == "__main__":
    unittest.main()
