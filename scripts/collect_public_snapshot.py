#!/usr/bin/env python3
"""Operational adapter around the immutable v0.6.3 data contract; no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/etf_shadow_v063"))
from etf_shadow_v063 import data_pipeline as dp

MODEL_COMMIT = "6ad8c97af188604cd83f0c5daada15eeda9e6aa7"
ADAPTER_VERSION = "public-collector-1.0"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    dp._write_json_atomic(path, value)


def verify_identity():
    expected = json.loads((ROOT / "scripts/public_collector_identity.json").read_text())
    if expected["model_commit"] != MODEL_COMMIT:
        raise dp.DataGateClosed("MODEL_IDENTITY_MISMATCH")
    for name, digest in expected["git_blob_hashes"].items():
        data = (ROOT / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != digest:
            raise dp.DataGateClosed("MODEL_IDENTITY_MISMATCH:" + name)


class Transport:
    def __init__(self, budget_seconds=480, opener=urllib.request.urlopen, sleeper=time.sleep):
        self.deadline = time.monotonic() + budget_seconds
        self.opener, self.sleeper = opener, sleeper
        self.events = []

    def request(self, url, params, **ignored_legacy_policy):
        # Override the legacy Tencent five-attempt policy, not the source parser.
        endpoint = urllib.parse.urlsplit(url).netloc
        for attempt in range(1, 4):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise dp.DataGateClosed("COLLECTION_DEADLINE_EXCEEDED")
            try:
                req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                             headers={"User-Agent": "Mozilla/5.0 etf-shadow-public/1.0"})
                with self.opener(req, timeout=min(20, remaining)) as response:
                    payload = response.read().decode("utf-8")
                self.events.append({"host": endpoint, "attempt": attempt, "state": "OK"})
                return payload
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code <= 599
                code = "HTTP_" + str(error.code)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                retryable, code = True, type(error).__name__
            self.events.append({"host": endpoint, "attempt": attempt, "state": code})
            if not retryable or attempt == 3:
                raise dp.DataGateClosed("DATA_MISSING:" + endpoint + ":" + code)
            self.sleeper(min(2 ** (attempt - 1), max(0, self.deadline - time.monotonic())))


def validate_packet(directory, required_cutoff=None):
    manifest_path = directory / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    returns_path = directory / "returns.csv"
    frame = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date")
    dp.validate_production_manifest(frame, returns_path, manifest_path, pd.Timestamp(manifest["as_of"]))
    if required_cutoff is not None and frame.index.max() != pd.Timestamp(required_cutoff):
        raise dp.DataGateClosed("STALE_OR_UNEXPECTED_COMMON_CUTOFF")
    return manifest


def resolve_success(state):
    pointer = state / "latest_success.json"
    if not pointer.exists():
        return None
    value = json.loads(pointer.read_text())
    if value.get("model_commit") != MODEL_COMMIT or value.get("status") != "SUCCESS":
        raise dp.DataGateClosed("SNAPSHOT_IDENTITY_MISMATCH")
    directory = dp._safe_relative_file(state, value["packet"])
    manifest = directory / "data_manifest.json"
    if dp.sha256_file(manifest) != value["manifest_sha256"]:
        raise dp.DataGateClosed("SNAPSHOT_HASH_MISMATCH")
    validate_packet(directory)
    return directory


class RevisionDetected(dp.DataGateClosed):
    pass


class IncrementalSources:
    def __init__(self, directory, primary, secondary):
        self.directory, self.primary, self.secondary = directory, primary, secondary
        self.manifest = validate_packet(directory)
        self.cutoff = pd.Timestamp(self.manifest["return_panel"]["max_date"])
        self.overlap_start = self.cutoff - pd.Timedelta(days=28)

    def merge(self, proxy, start, end, suffix, fetch):
        old = pd.read_csv(self.directory / f"raw/{proxy.asset}_{proxy.security}_{suffix}.csv",
                          parse_dates=["date"], float_precision="round_trip")
        new = dp._normalize_ohlc(fetch(proxy, self.overlap_start, end), "INCREMENTAL")
        old_overlap = old.loc[old.date >= self.overlap_start].set_index("date")
        new_overlap = new.loc[new.date <= self.cutoff].set_index("date")
        columns = [c for c in old_overlap if c in new_overlap]
        if (not old_overlap.index.equals(new_overlap.index)
                or not np.allclose(old_overlap[columns].to_numpy(float),
                                   new_overlap[columns].to_numpy(float), rtol=1e-10, atol=1e-10)):
            # In particular, never concatenate old-scale qfq with revised-scale qfq.
            raise RevisionDetected("HISTORICAL_REVISION_REQUIRES_FULL_REBUILD")
        return pd.concat([old.loc[old.date < self.overlap_start], new], ignore_index=True)

    def fetch_primary(self, proxy, start, end, adjustment):
        return self.merge(proxy, start, end, "eastmoney_" + adjustment,
                          lambda p, s, e: self.primary(p, s, e, adjustment))

    def fetch_secondary(self, proxy, start, end):
        return self.merge(proxy, start, end, "tencent_qfq", self.secondary)


def terminal_checkpoint(directory, reason):
    path = directory / dp.CHECKPOINT_FILENAME
    if path.exists():
        value = json.loads(path.read_text())
        value.update(status="FAILED_CLOSED", failure=reason, updated_at_utc=now())
        write_json(path, value)


def collect(state, as_of, required_cutoff, run_id, primary=dp.fetch_eastmoney,
            secondary=dp.fetch_tencent, force_full=False, transport=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("invalid run id")
    attempt_file = state / "attempts" / (run_id + ".json")
    if attempt_file.exists():
        previous = json.loads(attempt_file.read_text())
        if previous["as_of"] != as_of or previous["required_cutoff"] != required_cutoff:
            raise dp.DataGateClosed("RUN_ID_REUSE_CONFLICT")
        if previous["status"] != "RUNNING":
            return previous  # No repeated downloads or mutation for a completed run.
        raise dp.DataGateClosed("RUN_ALREADY_STARTED_OR_INTERRUPTED")
    started = time.monotonic()
    transport = transport or Transport()
    packet = state / "runs" / run_id / "incremental"
    receipt = dict(schema="PUBLIC_SNAPSHOT_ATTEMPT_V1", run_id=run_id, as_of=as_of,
                   required_cutoff=required_cutoff, model_commit=MODEL_COMMIT,
                   adapter_version=ADAPTER_VERSION, started_at=now(), status="RUNNING",
                   run_path="FULL_REBUILD", failure=None, orders_generated=False,
                   broker_connection=False, source_requests=[])
    write_json(attempt_file, receipt)
    write_json(state / "latest_attempt.json", receipt)
    stage = "MODEL_IDENTITY"
    try:
        verify_identity()
        if pd.Timestamp(required_cutoff) >= pd.Timestamp(as_of):
            raise dp.DataGateClosed("CUTOFF_MUST_PRECEDE_AS_OF")
        raw_primary, raw_secondary = primary, secondary
        primary = lambda p, s, e, a: raw_primary(p, s, min(e, pd.Timestamp(required_cutoff)), a)
        secondary = lambda p, s, e: raw_secondary(p, s, min(e, pd.Timestamp(required_cutoff)))
        stage = "BASE_PACKET_VALIDATION"
        prior = resolve_success(state)
        if prior:
            old = validate_packet(prior)
            if old["as_of"] == as_of and old["return_panel"]["max_date"] == required_cutoff:
                receipt.update(status="NO_ACTION", run_path="FAST_NO_DELTA", packet=str(prior.relative_to(state)))
                return receipt
            if pd.Timestamp(old["as_of"]) >= pd.Timestamp(as_of):
                raise dp.DataGateClosed("AS_OF_NOT_AFTER_PREVIOUS_SNAPSHOT")
        kwargs = dict(start=pd.Timestamp("2016-12-01"), as_of=pd.Timestamp(as_of), resume=False)
        stage = "DATA_COLLECTION"
        # This patch is scoped to this single-process adapter. Model files stay immutable.
        with patch.object(dp, "_request_text", transport.request):
            incremental = prior is not None and not force_full
            if incremental:
                sources = IncrementalSources(prior, primary, secondary)
                receipt["run_path"] = "WEEKLY_INCREMENTAL"
                try:
                    dp.build_production_panel(output_dir=packet, primary_fetcher=sources.fetch_primary,
                                              secondary_fetcher=sources.fetch_secondary, **kwargs)
                except RevisionDetected as error:
                    terminal_checkpoint(packet, str(error))
                    receipt["full_rebuild_reason"] = str(error)
                    incremental = False
            if not incremental:
                packet = state / "runs" / run_id / "full"
                receipt["run_path"] = "FULL_REBUILD"
                dp.build_production_panel(output_dir=packet, primary_fetcher=primary,
                                          secondary_fetcher=secondary, **kwargs)
        stage = "DATA_VALIDATION"
        manifest = validate_packet(packet, required_cutoff)
        pointer = dict(schema="PUBLIC_SNAPSHOT_POINTER_V1", status="SUCCESS", run_id=run_id,
                       model_commit=MODEL_COMMIT, adapter_version=ADAPTER_VERSION,
                       as_of=as_of, data_cutoff=manifest["return_panel"]["max_date"],
                       packet=str(packet.relative_to(state)),
                       manifest_sha256=dp.sha256_file(packet / "data_manifest.json"))
        receipt.update(status="SUCCESS", **{k: v for k, v in pointer.items() if k not in ("schema", "status")})
        stage = "SUCCESS_POINTER_COMMIT"
        write_json(state / "latest_success.json", pointer)
    except Exception as error:
        # All terminal paths persist a receipt, including JSON/schema/runtime errors.
        reason = (type(error).__name__ + ":" + str(error))[:600]
        terminal_checkpoint(packet, reason)
        classification = ("PARSE_FAILED" if isinstance(error, (json.JSONDecodeError, pd.errors.ParserError))
                          else "SOURCE_CONFLICT" if "RETURN_PANEL_GATE_FAILED" in reason
                          else "STALE" if "CUTOFF" in reason or "STALE" in reason
                          else "DATA_MISSING" if "DATA_MISSING" in reason
                          else "INTEGRITY_OR_RUNTIME_FAILURE")
        receipt.update(status="FAILED_CLOSED", failure=reason, failed_stage=stage,
                       failure_classification=classification)
    finally:
        receipt.update(finished_at=now(), elapsed_ms=round((time.monotonic() - started) * 1000),
                       source_requests=transport.events, retry_ceiling=2)
        write_json(attempt_file, receipt)
        write_json(state / "latest_attempt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--required-cutoff", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if pd.Timestamp(args.required_cutoff) >= pd.Timestamp(args.as_of):
        parser.error("required cutoff must precede the as-of date")
    result = collect(args.state_dir, args.as_of, args.required_cutoff, args.run_id, force_full=args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"SUCCESS", "NO_ACTION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
