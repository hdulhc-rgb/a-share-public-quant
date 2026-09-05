from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_public_snapshot as c


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)
        self.index = pd.bdate_range("2016-12-01", periods=440)
        self.calls = []
        rng = np.random.default_rng(73)
        self.frames = {}
        for i, proxy in enumerate(c.dp.PROXIES):
            r = rng.normal(0.0001, 0.0001 if proxy.asset == "CASH" else 0.006, len(self.index))
            price = (2 + i) * np.cumprod(1 + r)
            self.frames[proxy.asset] = pd.DataFrame(dict(date=self.index, open=price,
                close=price, high=price * 1.001, low=price * 0.999))

    def tearDown(self):
        self.temp.cleanup()

    def primary(self, proxy, start, end, adjustment):
        self.calls.append((start, end))
        frame = self.frames[proxy.asset]
        return frame.loc[frame.date.between(start, end)].copy()

    def secondary(self, proxy, start, end):
        return self.primary(proxy, start, end, "qfq")

    def run_collection(self, n=420, run_id="run1", **kwargs):
        cutoff = self.index[n - 1]
        return c.collect(self.state, (cutoff + pd.Timedelta(days=1)).date().isoformat(),
                         cutoff.date().isoformat(), run_id, primary=self.primary,
                         secondary=self.secondary, **kwargs)

    def test_success_packet_accepted_by_unchanged_model(self):
        r = self.run_collection()
        self.assertEqual(r["status"], "SUCCESS", r)
        c.validate_packet(c.resolve_success(self.state), self.index[419])

    def test_failed_source_terminal_and_last_success_preserved(self):
        self.run_collection()
        before = (self.state / "latest_success.json").read_bytes()
        self.primary = lambda *a: (_ for _ in ()).throw(c.dp.DataGateClosed("DATA_MISSING:HTTP_502"))
        r = self.run_collection(425, "run2")
        self.assertEqual(r["status"], "FAILED_CLOSED")
        checkpoint = json.loads((self.state / "runs/run2/incremental/collection_checkpoint.json").read_text())
        self.assertEqual(checkpoint["status"], "FAILED_CLOSED")
        self.assertEqual(before, (self.state / "latest_success.json").read_bytes())

    def test_incremental_matches_full_and_fetches_overlap_only(self):
        self.run_collection()
        self.calls = []
        r = self.run_collection(430, "run2")
        self.assertEqual(r["status"], "SUCCESS", r)
        self.assertEqual(r["run_path"], "WEEKLY_INCREMENTAL")
        self.assertTrue(all(start == self.index[419] - pd.Timedelta(days=28) for start, end in self.calls))
        incremental = (c.resolve_success(self.state) / "returns.csv").read_bytes()
        with tempfile.TemporaryDirectory() as second:
            original = self.state
            self.state = Path(second)
            full = self.run_collection(430, "fresh")
            self.assertEqual(full["status"], "SUCCESS", full)
            self.assertEqual(incremental, (c.resolve_success(self.state) / "returns.csv").read_bytes())
            self.state = original

    def test_qfq_revision_causes_full_rebuild_no_mixed_scale(self):
        self.run_collection()
        for frame in self.frames.values():
            for col in ("open", "close", "high", "low"):
                frame[col] *= 0.97
        r = self.run_collection(430, "revised")
        self.assertEqual(r["status"], "SUCCESS", r)
        self.assertEqual(r["run_path"], "FULL_REBUILD")
        self.assertIn("REVISION", r["full_rebuild_reason"])

    def test_source_conflict_cannot_publish_success(self):
        def conflicting(proxy, start, end):
            frame = self.primary(proxy, start, end, "qfq")
            for col in ("open", "close", "high", "low"):
                frame[col] *= np.exp(np.arange(len(frame)) * 0.001)
            return frame
        self.secondary = conflicting
        r = self.run_collection()
        self.assertEqual(r["status"], "FAILED_CLOSED")
        self.assertFalse((self.state / "latest_success.json").exists())

    def test_hash_tamper_is_not_treated_as_missing_base(self):
        self.run_collection()
        (c.resolve_success(self.state) / "returns.csv").write_text("bad")
        self.calls = []
        r = self.run_collection(430, "badbase")
        self.assertEqual(r["status"], "FAILED_CLOSED")
        self.assertEqual(self.calls, [])

    def test_same_completed_run_is_idempotent(self):
        first = self.run_collection()
        self.calls = []
        second = self.run_collection()
        self.assertEqual(first, second)
        self.assertEqual(self.calls, [])

    def test_same_data_new_run_does_not_recollect(self):
        self.run_collection()
        self.calls = []
        r = self.run_collection(run_id="run2")
        self.assertEqual(r["status"], "NO_ACTION")
        self.assertEqual(self.calls, [])

    def test_stale_cutoff_rejected_even_inside_legacy_seven_day_gate(self):
        self.primary_orig = self.primary
        self.primary = lambda p, s, e, a: self.primary_orig(p, s, min(e, self.index[418]), a)
        r = self.run_collection()
        self.assertEqual(r["status"], "FAILED_CLOSED")
        self.assertIn("CUTOFF", r["failure"])

    def test_bad_json_creates_terminal_receipt(self):
        self.primary = lambda *a: json.loads("{")
        r = self.run_collection()
        self.assertEqual(r["status"], "FAILED_CLOSED")
        self.assertIn("JSONDecodeError", r["failure"])

    def test_identity_mismatch_never_fetches(self):
        with patch.object(c, "verify_identity", side_effect=c.dp.DataGateClosed("MODEL_IDENTITY_MISMATCH")):
            r = self.run_collection()
        self.assertEqual(r["status"], "FAILED_CLOSED")
        self.assertEqual(self.calls, [])


class TransportTests(unittest.TestCase):
    def test_502_has_two_retries_even_legacy_requests_five(self):
        def failing(*a, **kw):
            raise urllib.error.HTTPError("https://example.invalid", 502, "bad", {}, None)
        t = c.Transport(opener=failing, sleeper=lambda _: None)
        with self.assertRaises(c.dp.DataGateClosed):
            t.request("https://example.invalid", {}, attempts=5, timeout=45)
        self.assertEqual(len(t.events), 3)

    def test_403_is_not_retried(self):
        def forbidden(*a, **kw):
            raise urllib.error.HTTPError("https://example.invalid", 403, "forbidden", {}, None)
        t = c.Transport(opener=forbidden, sleeper=lambda _: None)
        with self.assertRaises(c.dp.DataGateClosed):
            t.request("https://example.invalid", {})
        self.assertEqual(len(t.events), 1)

    def test_recoverable_network_failure_then_success(self):
        count = []
        def opener(*a, **kw):
            count.append(1)
            if len(count) == 1:
                raise urllib.error.URLError("temporary")
            return io.BytesIO(b'{"data": {}}')
        t = c.Transport(opener=opener, sleeper=lambda _: None)
        self.assertEqual(t.request("https://example.invalid", {}), '{"data": {}}')
        self.assertEqual(len(t.events), 2)


if __name__ == "__main__":
    unittest.main()
