from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_shadow_v063.core import ResearchClosed, sha256_file
from etf_shadow_v063_data2.data_contract import (
    align_return_panels,
    apply_local_proxy,
    assess_data2_grade,
    return_basis_diagnostics,
    validate_local_proxy_manifest,
    validate_official_economic_manifest,
    validate_qdii_decomposition,
)


class Data2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2024-01-02", periods=420)
        self.as_of = self.index[-1]
        rng = np.random.default_rng(20260720)
        self.assets = ["A_SHARE", "US_SP500", "US_NASDAQ100", "GOLD", "CASH"]
        values = rng.normal(0.0002, 0.006, (len(self.index), len(self.assets)))
        values[:, -1] = rng.normal(0.00005, 0.00005, len(self.index))
        self.returns = pd.DataFrame(values, index=self.index, columns=self.assets)

    def _write_official_packet(self, root: Path) -> tuple[Path, Path]:
        returns_path = root / "economic_returns.csv"
        self.returns.rename_axis("date").to_csv(returns_path)
        evidence = []
        for asset in self.assets:
            raw_path = root / f"{asset}_official.csv"
            pd.DataFrame(
                {"date": self.index, "level": (1 + self.returns[asset]).cumprod()}
            ).to_csv(raw_path, index=False)
            evidence.append(
                {
                    "asset": asset,
                    "source_type": "official_total_return_index",
                    "source_url": f"https://official.example/{asset}",
                    "available_at": self.as_of.isoformat(),
                    "point_in_time": True,
                    "raw_path": raw_path.name,
                    "raw_sha256": sha256_file(raw_path),
                }
            )
        manifest = {
            "schema_version": "2.0",
            "demo": False,
            "as_of": self.as_of.date().isoformat(),
            "source_snapshot_frozen": True,
            "return_panel": {
                "return_kind": "official_economic_total_return",
                "sha256": sha256_file(returns_path),
                "rows": len(self.returns),
                "columns": self.assets,
                "min_date": self.index.min().date().isoformat(),
                "max_date": self.index.max().date().isoformat(),
            },
            "asset_evidence": evidence,
            "gates": {
                "official_source_identity_passed": True,
                "point_in_time_passed": True,
                "raw_packet_frozen_passed": True,
                "economic_return_panel_passed": True,
            },
        }
        manifest_path = root / "economic_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return returns_path, manifest_path

    def _write_qdii_packet(self, root: Path) -> tuple[Path, Path]:
        path = root / "qdii.csv"
        rows = []
        evidence = []
        for asset in ["US_SP500", "US_NASDAQ100"]:
            for date in self.index[-252:]:
                market = float(self.returns.loc[date, asset])
                rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "market_return": market,
                        "underlying_return_local": market,
                        "fx_return": 0.0,
                        "premium_return": 0.0,
                        "fee_tracking_residual": 0.0,
                        "available_at": f"{date.date().isoformat()}T23:59:00Z",
                    }
                )
            for component in ["market", "underlying", "fx", "premium"]:
                raw_path = root / f"{asset}_{component}.csv"
                raw_path.write_text("date,value\n2026-01-01,1.0\n", encoding="utf-8")
                evidence.append(
                    {
                        "asset": asset,
                        "component": component,
                        "source_url": f"https://official.example/{asset}/{component}",
                        "available_at": self.as_of.isoformat(),
                        "point_in_time": True,
                        "raw_path": raw_path.name,
                        "raw_sha256": sha256_file(raw_path),
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)
        manifest = {
            "schema_version": "2.0",
            "as_of": self.as_of.date().isoformat(),
            "point_in_time": True,
            "source_snapshot_frozen": True,
            "panel_sha256": sha256_file(path),
            "assets": ["US_SP500", "US_NASDAQ100"],
            "construction_formula": "MULTIPLICATIVE_RETURN_IDENTITY",
            "raw_evidence": evidence,
            "gates": {
                "source_identity_passed": True,
                "point_in_time_passed": True,
                "raw_packet_frozen_passed": True,
                "decomposition_panel_passed": True,
            },
        }
        manifest_path = root / "qdii_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest_path

    @staticmethod
    def _refresh_panel_hash(manifest_path: Path, panel_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["panel_sha256"] = sha256_file(panel_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_official_economic_packet_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            returns_path, manifest_path = self._write_official_packet(root)
            attestation = validate_official_economic_manifest(
                self.returns, returns_path, manifest_path, self.as_of
            )
            self.assertEqual(attestation["data_grade"], "A")
            self.assertTrue(attestation["promotion_eligible"])

    def test_tampered_official_raw_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            returns_path, manifest_path = self._write_official_packet(root)
            raw = root / "A_SHARE_official.csv"
            raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ResearchClosed, "ECONOMIC_RAW_SOURCE_HASH_MISMATCH"):
                validate_official_economic_manifest(
                    self.returns, returns_path, manifest_path, self.as_of
                )

    def test_current_local_proxy_cannot_backcast(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local = self.returns[["A_SHARE"]].iloc[-20:].copy()
            local["A_SHARE"] += 0.001
            local_path = root / "local_proxy.csv"
            local.rename_axis("date").to_csv(local_path)
            manifest = {
                "schema_version": "2.0",
                "as_of": self.as_of.date().isoformat(),
                "point_in_time": True,
                "construction_mode": "CURRENT_FORWARD_ONLY",
                "panel_sha256": sha256_file(local_path),
                "model_assets": ["A_SHARE"],
                "effective_from": local.index.min().date().isoformat(),
                "constituent_count": 2,
                "component_packet_sha256": hashlib.sha256(b"synthetic-components").hexdigest(),
            }
            manifest_path = root / "local_proxy_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            attestation = validate_local_proxy_manifest(
                local, local_path, manifest_path, self.as_of
            )
            overlaid, evidence = apply_local_proxy(self.returns, local, attestation)
            self.assertTrue(
                overlaid.loc[: local.index.min() - pd.Timedelta(days=1), "A_SHARE"].equals(
                    self.returns.loc[: local.index.min() - pd.Timedelta(days=1), "A_SHARE"]
                )
            )
            self.assertFalse(evidence["historical_backtest_eligible"])
            self.assertEqual(evidence["replaced_rows"]["A_SHARE"], 20)

    def test_local_proxy_row_before_effective_date_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local = self.returns[["A_SHARE"]].iloc[-20:].copy()
            local_path = root / "local_proxy.csv"
            local.rename_axis("date").to_csv(local_path)
            manifest = {
                "schema_version": "2.0",
                "as_of": self.as_of.date().isoformat(),
                "point_in_time": True,
                "construction_mode": "CURRENT_FORWARD_ONLY",
                "panel_sha256": sha256_file(local_path),
                "model_assets": ["A_SHARE"],
                "effective_from": (local.index.min() + pd.Timedelta(days=1)).date().isoformat(),
                "constituent_count": 2,
                "component_packet_sha256": hashlib.sha256(b"synthetic-components").hexdigest(),
            }
            manifest_path = root / "local_proxy_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ResearchClosed, "LOCAL_PROXY_BACKCAST"):
                validate_local_proxy_manifest(local, local_path, manifest_path, self.as_of)

    def test_local_proxy_missing_latest_common_date_fails_closed(self) -> None:
        local = self.returns[["A_SHARE"]].iloc[-20:-1].copy()
        attestation = {
            "construction_mode": "CURRENT_FORWARD_ONLY",
            "effective_from": self.returns.index[-21].date().isoformat(),
            "model_assets": ["A_SHARE"],
            "historical_backtest_eligible": False,
        }
        with self.assertRaisesRegex(
            ResearchClosed, "LOCAL_PROXY_FORWARD_COVERAGE_GAP:A_SHARE"
        ):
            apply_local_proxy(self.returns, local, attestation)

    def test_historical_local_proxy_cannot_start_mid_backtest(self) -> None:
        local = self.returns[["A_SHARE"]].iloc[-20:].copy()
        attestation = {
            "construction_mode": "POINT_IN_TIME_EFFECTIVE_DATED",
            "effective_from": self.returns.index[-21].date().isoformat(),
            "model_assets": ["A_SHARE"],
            "historical_backtest_eligible": True,
        }
        with self.assertRaisesRegex(
            ResearchClosed, "LOCAL_PROXY_HISTORICAL_COVERAGE_START_GAP"
        ):
            apply_local_proxy(self.returns, local, attestation)

    def test_qdii_decomposition_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest_path = self._write_qdii_packet(root)
            result = validate_qdii_decomposition(
                path, manifest_path, self.as_of, self.returns
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["promotion_eligible"])
            self.assertLessEqual(result["market_panel_max_abs_error"], 1e-10)
            frame = pd.read_csv(path)
            frame.loc[0, "market_return"] += 0.01
            frame.to_csv(path, index=False)
            self._refresh_panel_hash(manifest_path, path)
            with self.assertRaisesRegex(ResearchClosed, "QDII_DECOMPOSITION_IDENTITY_FAILED"):
                validate_qdii_decomposition(path, manifest_path, self.as_of, self.returns)

    def test_qdii_decomposition_must_match_execution_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest_path = self._write_qdii_packet(root)
            frame = pd.read_csv(path)
            frame.loc[0, "market_return"] += 0.001
            frame.loc[0, "underlying_return_local"] += 0.001
            frame.to_csv(path, index=False)
            self._refresh_panel_hash(manifest_path, path)
            with self.assertRaisesRegex(
                ResearchClosed, "QDII_DECOMPOSITION_MARKET_PANEL_MISMATCH"
            ):
                validate_qdii_decomposition(path, manifest_path, self.as_of, self.returns)

    def test_tampered_qdii_raw_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest_path = self._write_qdii_packet(root)
            raw = root / "US_SP500_fx.csv"
            raw.write_text(raw.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ResearchClosed, "QDII_RAW_EVIDENCE_HASH_MISMATCH"):
                validate_qdii_decomposition(path, manifest_path, self.as_of, self.returns)

    def test_shadow_grade_b_and_promotion_fail_closed(self) -> None:
        dependency = {"promotion_dependency_gate_passed": False}
        shadow = assess_data2_grade(
            execution_gate_passed=True,
            official_economic_attestation=None,
            local_proxy_attestation=None,
            qdii_attestation=None,
            dependency_attestation=dependency,
            profile="shadow",
        )
        self.assertEqual(shadow["data_grade"], "B")
        self.assertFalse(shadow["promotion_gate_passed"])
        with self.assertRaisesRegex(ResearchClosed, "DATA2_PROMOTION_GATE_FAILED"):
            assess_data2_grade(
                execution_gate_passed=True,
                official_economic_attestation=None,
                local_proxy_attestation=None,
                qdii_attestation=None,
                dependency_attestation=dependency,
                profile="promotion",
            )

    def test_dual_panel_alignment_and_basis_diagnostics(self) -> None:
        economic = self.returns.copy()
        execution = self.returns.copy()
        execution["US_SP500"] += 0.0001
        aligned_economic, aligned_execution = align_return_panels(
            economic, execution, self.as_of
        )
        diagnostics = return_basis_diagnostics(aligned_economic, aligned_execution)
        self.assertEqual(len(diagnostics), len(self.assets))
        gap = diagnostics.set_index("asset").loc["US_SP500", "median_abs_daily_gap"]
        self.assertAlmostEqual(float(gap), 0.0001)


if __name__ == "__main__":
    unittest.main()
