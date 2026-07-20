from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_shadow_v063.core import ResearchPolicy
from etf_shadow_v063_data2.runner import run_research_data2


class Data2RunnerTests(unittest.TestCase):
    def test_dual_panel_run_keeps_research_lock_and_evidence(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=400)
        assets = ["A_SHARE", "US_SP500", "US_NASDAQ100", "GOLD", "CASH"]
        rng = np.random.default_rng(63)
        economic = pd.DataFrame(
            rng.normal(0.0002, 0.007, (len(index), len(assets))),
            index=index,
            columns=assets,
        )
        economic["CASH"] = 0.00005
        execution = economic.copy()
        execution["US_SP500"] += 0.0001
        benchmark = pd.Series([0.25, 0.35, 0.10, 0.15, 0.15], index=assets)
        current = benchmark.copy()
        assessment = {
            "status": "PASS_SHADOW",
            "data_grade": "B",
            "shadow_gate_passed": True,
            "promotion_gate_passed": False,
            "evidence_gaps": ["OFFICIAL_ECONOMIC_RETURN_PANEL_MISSING"],
            "economic_return_policy": "EXPLICIT_EXECUTION_PROXY_FALLBACK",
            "no_silent_fallback": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            run_dir = run_research_data2(
                economic_returns=economic,
                execution_returns=execution,
                benchmark=benchmark,
                current=current,
                as_of=index[-1],
                output_root=Path(temp),
                economic_source_path=None,
                execution_source_path=None,
                economic_source_id="SYNTHETIC_ECONOMIC",
                execution_source_id="SYNTHETIC_EXECUTION",
                policy=ResearchPolicy(),
                profile="shadow",
                data2_assessment=assessment,
                economic_source_attestation=None,
                execution_source_attestation={"return_kind": "SYNTHETIC"},
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            decision = json.loads((run_dir / "decision_snapshot.json").read_text(encoding="utf-8"))
            diagnostics = pd.read_csv(run_dir / "return_basis_diagnostics.csv").set_index("asset")
            self.assertEqual(manifest["integrity"], "PASS")
            self.assertEqual(manifest["data_grade"], "B")
            self.assertFalse(manifest["promotion_gate_passed"])
            self.assertTrue(manifest["research_lock"])
            self.assertFalse(manifest["orders_generated"])
            self.assertIn("LATEST_COMPLETE_WALK_FORWARD", decision["signal_date_semantics"])
            self.assertAlmostEqual(
                float(diagnostics.loc["US_SP500", "median_abs_daily_gap"]), 0.0001
            )
            self.assertTrue((run_dir / "external_validation.json").is_file())


if __name__ == "__main__":
    unittest.main()
