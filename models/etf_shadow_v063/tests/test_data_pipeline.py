from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_shadow_v063.core import ResearchClosed
from etf_shadow_v063.data_pipeline import (
    DataGateClosed,
    PROXIES,
    build_production_panel,
    fetch_tencent,
    tencent_request_windows,
    validate_production_manifest,
)


class ProductionDataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.bdate_range("2024-01-02", periods=420)
        self.start = self.index[0]
        self.as_of = self.index[-1]
        rng = np.random.default_rng(20260720)
        self.adjusted: dict[str, pd.DataFrame] = {}
        for number, proxy in enumerate(PROXIES):
            if proxy.asset == "CASH":
                daily = rng.normal(0.00005, 0.00008, len(self.index))
            else:
                daily = rng.normal(0.0002, 0.006 + number * 0.0005, len(self.index))
            close = (2.0 + number) * np.cumprod(1.0 + daily)
            self.adjusted[proxy.asset] = pd.DataFrame(
                {
                    "date": self.index,
                    "open": close * 0.999,
                    "close": close,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "volume": np.full(len(close), 1000 + number),
                    "amount": close * (1000 + number),
                }
            )

    def primary_fetcher(self, proxy, start, end, adjustment):
        frame = self.adjusted[proxy.asset].copy()
        if adjustment == "unadjusted":
            factor = np.ones(len(frame))
            factor[:210] = 0.97
            for column in ["open", "close", "high", "low"]:
                frame[column] = frame[column] / factor
        return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)

    def secondary_fetcher(self, proxy, start, end):
        frame = self.adjusted[proxy.asset].copy()
        for column in ["open", "close", "high", "low"]:
            frame[column] = frame[column] * 1.0001
        return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)

    def test_tencent_uses_bounded_two_year_windows(self) -> None:
        start = pd.Timestamp("2016-12-01")
        end = pd.Timestamp("2026-07-21")
        windows = tencent_request_windows(start, end)
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)
        for position, (window_start, window_end) in enumerate(windows):
            self.assertLessEqual(
                window_end,
                window_start + pd.DateOffset(years=2) - pd.Timedelta(days=1),
            )
            if position:
                self.assertEqual(window_start, windows[position - 1][1] + pd.Timedelta(days=1))

        def fake_request(url, params, **kwargs):
            requested_start = params["param"].split(",")[2]
            row = [requested_start, "1", "1", "1", "1", "100"]
            payload = {"data": {PROXIES[0].market_symbol: {"qfqday": [row]}}}
            return f"fixture={json.dumps(payload)}"

        with patch(
            "etf_shadow_v063.data_pipeline._request_text",
            side_effect=fake_request,
        ) as request:
            frame = fetch_tencent(PROXIES[0], start, end)
        self.assertEqual(len(request.call_args_list), 5)
        self.assertEqual(len(frame), 5)
        for call in request.call_args_list:
            self.assertEqual(call.kwargs["attempts"], 5)
            self.assertEqual(call.kwargs["timeout"], 45.0)

    def test_same_as_of_checkpoint_resumes_only_missing_sources(self) -> None:
        primary_calls: list[tuple[str, str]] = []
        secondary_calls: list[str] = []
        fail_gold_once = True

        def counting_primary(proxy, start, end, adjustment):
            primary_calls.append((proxy.asset, adjustment))
            return self.primary_fetcher(proxy, start, end, adjustment)

        def flaky_secondary(proxy, start, end):
            nonlocal fail_gold_once
            secondary_calls.append(proxy.asset)
            if proxy.asset == "GOLD" and fail_gold_once:
                fail_gold_once = False
                raise DataGateClosed("PUBLIC_SOURCE_UNAVAILABLE:fixture")
            return self.secondary_fetcher(proxy, start, end)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with self.assertRaisesRegex(DataGateClosed, "PUBLIC_SOURCE_UNAVAILABLE"):
                build_production_panel(
                    self.start,
                    self.as_of,
                    output,
                    primary_fetcher=counting_primary,
                    secondary_fetcher=flaky_secondary,
                )
            checkpoint = json.loads(
                (output / "collection_checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(checkpoint["completed_sources"]), 11)

            primary_calls.clear()
            secondary_calls.clear()
            manifest_path = build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=counting_primary,
                secondary_fetcher=flaky_secondary,
            )
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(
                primary_calls,
                [("CASH", "qfq"), ("CASH", "unadjusted")],
            )
            self.assertEqual(secondary_calls, ["GOLD", "CASH"])

    def test_tampered_resume_source_fails_before_refetch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=self.primary_fetcher,
                secondary_fetcher=self.secondary_fetcher,
            )
            raw_path = output / "raw" / "A_SHARE_510300_eastmoney_qfq.csv"
            raw = pd.read_csv(raw_path)
            raw.loc[0, "close"] *= 1.01
            raw.to_csv(raw_path, index=False)
            with self.assertRaisesRegex(
                DataGateClosed,
                "EXISTING_FINAL_PACKET_INVALID:RAW_SOURCE_HASH_MISMATCH",
            ):
                build_production_panel(
                    self.start,
                    self.as_of,
                    output,
                    primary_fetcher=self.primary_fetcher,
                    secondary_fetcher=self.secondary_fetcher,
                )

    def test_checkpoint_cannot_be_reused_for_a_different_as_of(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=self.primary_fetcher,
                secondary_fetcher=self.secondary_fetcher,
            )
            with self.assertRaisesRegex(
                DataGateClosed,
                "RESUME_CHECKPOINT_IDENTITY_MISMATCH",
            ):
                build_production_panel(
                    self.start,
                    self.as_of + pd.Timedelta(days=1),
                    output,
                    primary_fetcher=self.primary_fetcher,
                    secondary_fetcher=self.secondary_fetcher,
                )

    def test_build_and_validate_dual_source_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest_path = build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=self.primary_fetcher,
                secondary_fetcher=self.secondary_fetcher,
            )
            returns_path = output / "returns.csv"
            returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
            attestation = validate_production_manifest(
                returns,
                returns_path,
                manifest_path,
                self.as_of,
            )
            self.assertTrue(attestation["production_return_panel_gate_passed"])
            self.assertEqual(attestation["providers"], ["Eastmoney", "Tencent"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["gates"]["production_return_panel_passed"])
            self.assertEqual(len(manifest["raw_file_hashes"]), len(PROXIES) * 3)

            original_manifest = manifest_path.read_bytes()

            def unexpected_primary(*args, **kwargs):
                raise AssertionError("a complete verified packet must not refetch")

            def unexpected_secondary(*args, **kwargs):
                raise AssertionError("a complete verified packet must not refetch")

            resumed_manifest_path = build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=unexpected_primary,
                secondary_fetcher=unexpected_secondary,
            )
            self.assertEqual(resumed_manifest_path, manifest_path)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_tampered_return_panel_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest_path = build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=self.primary_fetcher,
                secondary_fetcher=self.secondary_fetcher,
            )
            returns_path = output / "returns.csv"
            frame = pd.read_csv(returns_path)
            frame.loc[0, "A_SHARE"] += 0.001
            frame.to_csv(returns_path, index=False)
            returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
            with self.assertRaisesRegex(ResearchClosed, "RETURN_PANEL_HASH_MISMATCH"):
                validate_production_manifest(
                    returns,
                    returns_path,
                    manifest_path,
                    self.as_of,
                )

    def test_secondary_disagreement_fails_data_gate(self) -> None:
        def disagreeing_secondary(proxy, start, end):
            frame = self.secondary_fetcher(proxy, start, end)
            if proxy.asset == "A_SHARE":
                frame.loc[300, ["open", "close", "high", "low"]] *= 1.25
            return frame

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with self.assertRaisesRegex(DataGateClosed, "PRODUCTION_RETURN_PANEL_GATE_FAILED"):
                build_production_panel(
                    self.start,
                    self.as_of,
                    output,
                    primary_fetcher=self.primary_fetcher,
                    secondary_fetcher=disagreeing_secondary,
                )
            manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["gates"]["production_return_panel_passed"])

    def test_tampered_raw_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            manifest_path = build_production_panel(
                self.start,
                self.as_of,
                output,
                primary_fetcher=self.primary_fetcher,
                secondary_fetcher=self.secondary_fetcher,
            )
            raw_path = output / "raw" / "A_SHARE_510300_eastmoney_qfq.csv"
            raw = pd.read_csv(raw_path)
            raw.loc[10, "close"] *= 1.02
            raw.to_csv(raw_path, index=False)
            returns_path = output / "returns.csv"
            returns = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
            with self.assertRaisesRegex(ResearchClosed, "RAW_SOURCE_HASH_MISMATCH"):
                validate_production_manifest(
                    returns,
                    returns_path,
                    manifest_path,
                    self.as_of,
                )


if __name__ == "__main__":
    unittest.main()
