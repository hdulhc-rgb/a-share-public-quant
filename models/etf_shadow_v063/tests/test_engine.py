from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_shadow_v063.backtest import differential_check
from etf_shadow_v063.challengers import inverse_volatility
from etf_shadow_v063.core import ResearchClosed, ResearchPolicy, cap_turnover, next_tradable_time, normalize_long_only, validate_returns
from etf_shadow_v063.runner import run_research
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

    def test_live_current_cannot_change_historical_validation(self) -> None:
        benchmark = pd.Series([0.25, 0.35, 0.10, 0.15, 0.15], index=[
            "A_SHARE", "US_SP500", "US_NASDAQ100", "GOLD", "CASH",
        ])
        live_a = pd.Series([0.25, 0.35, 0.10, 0.15, 0.15], index=benchmark.index)
        live_b = pd.Series([0.15, 0.55, 0.10, 0.10, 0.10], index=benchmark.index)
        index = pd.bdate_range("2022-01-03", periods=400)
        rng = np.random.default_rng(20260727)
        market = rng.normal(0.0002, 0.008, len(index))
        returns = pd.DataFrame({
            "A_SHARE": 0.7 * market + rng.normal(0, 0.006, len(index)),
            "US_SP500": 0.9 * market + rng.normal(0, 0.004, len(index)),
            "US_NASDAQ100": 1.2 * market + rng.normal(0, 0.006, len(index)),
            "GOLD": -0.1 * market + rng.normal(0, 0.004, len(index)),
            "CASH": np.full(len(index), 0.00005),
        }, index=index)
        historical_artifacts = [
            "walk_forward_results.csv",
            "challenge_matrix.csv",
            "stability_regions.csv",
            "constraint_diagnostics.csv",
            "cpcv_results.csv",
            "candidate_filter_trace.jsonl",
            "shadow_replay.csv",
            "shadow_replay.jsonl",
            "performance_attribution.csv",
            "differential_backtest.json",
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_a = run_research(
                returns=returns,
                benchmark=benchmark,
                historical_anchor=benchmark,
                live_current=live_a,
                as_of=index[-1],
                output_root=root / "a",
                source_path=None,
                source_id="SYNTHETIC_POINT_IN_TIME_TEST",
                policy=ResearchPolicy(),
            )
            run_b = run_research(
                returns=returns,
                benchmark=benchmark,
                historical_anchor=benchmark,
                live_current=live_b,
                as_of=index[-1],
                output_root=root / "b",
                source_path=None,
                source_id="SYNTHETIC_POINT_IN_TIME_TEST",
                policy=ResearchPolicy(),
            )
            for filename in historical_artifacts:
                self.assertEqual(
                    (run_a / filename).read_bytes(),
                    (run_b / filename).read_bytes(),
                    filename,
                )
            snapshot_a = json.loads(
                (run_a / "decision_snapshot.json").read_text(encoding="utf-8")
            )
            snapshot_b = json.loads(
                (run_b / "decision_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot_a["historical_validation"], snapshot_b["historical_validation"])
            self.assertNotEqual(
                snapshot_a["challenger_actual_shadow_targets"],
                snapshot_b["challenger_actual_shadow_targets"],
            )
            self.assertFalse(snapshot_a["historical_validation"]["uses_live_current"])
            self.assertEqual(snapshot_a["signal_date"], index[-1].isoformat())


if __name__ == "__main__":
    unittest.main()
