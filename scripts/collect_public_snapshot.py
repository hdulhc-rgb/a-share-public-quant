#!/usr/bin/env python3
"""Operational adapter around the immutable v0.6.3 data contract; no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
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
ADAPTER_VERSION = "public-collector-1.1"


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
    def __init__(self, budget_seconds=480, opener=urllib.request.urlopen, sleeper=time.sleep,
                 curl_runner=None, minimum_interval=1.0):
        self.deadline = time.monotonic() + budget_seconds
        self.opener, self.sleeper = opener, sleeper
        self.curl_runner = curl_runner
        self.minimum_interval = minimum_interval
        self.last_request_at = {}
        self.open_hosts = set()
        self.events = []

    def request(self, url, params, **ignored_legacy_policy):
        # Override the legacy Tencent five-attempt policy, not the source parser.
        endpoint = urllib.parse.urlsplit(url).netloc
        request_id = hashlib.sha256(json.dumps([url, dict(params)], sort_keys=True).encode()).hexdigest()[:16]
        context = {k: str(params[k]) for k in ("secid", "fqt", "beg", "end", "param") if k in params}
        if endpoint in self.open_hosts:
            raise dp.DataGateClosed("DATA_MISSING:CIRCUIT_OPEN:" + endpoint)
        for attempt in range(1, 4):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise dp.DataGateClosed("COLLECTION_DEADLINE_EXCEEDED")
            wait = max(0, self.minimum_interval - (time.monotonic() - self.last_request_at.get(endpoint, 0)))
            if wait >= remaining:
                raise dp.DataGateClosed("COLLECTION_DEADLINE_EXCEEDED")
            self.sleeper(wait)
            client = "curl" if self.curl_runner is not None and attempt == 2 else "urllib"
            event = {"host": endpoint, "request_id": request_id, "params": context,
                     "attempt": attempt, "client": client}
            self.last_request_at[endpoint] = time.monotonic()
            retry_after = 0.0
            try:
                request_url = url + "?" + urllib.parse.urlencode(params)
                headers = {"User-Agent": "etf-shadow-public/1.1"}
                if client == "curl":
                    payload = self.curl_runner(request_url, headers, min(20, remaining))
                else:
                    req = urllib.request.Request(request_url, headers=headers)
                    with self.opener(req, timeout=min(20, remaining)) as response:
                        payload = response.read().decode("utf-8")
                self.events.append({**event, "state": "OK"})
                return payload
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code <= 599
                code = "HTTP_" + str(error.code)
                header = error.headers.get("Retry-After", "") if error.headers else ""
                if header:
                    try:
                        retry_after = max(0.0, float(header))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        try:
                            retry_after = max(0.0, (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError):
                            retry_after = 0.0
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                retryable, code = True, type(error).__name__
            self.events.append({**event, "state": code})
            if not retryable or attempt == 3:
                self.open_hosts.add(endpoint)
                raise dp.DataGateClosed("DATA_MISSING:" + endpoint + ":" + code)
            remaining = max(0, self.deadline - time.monotonic())
            if retry_after >= remaining:
                self.open_hosts.add(endpoint)
                raise dp.DataGateClosed("DATA_MISSING:RETRY_AFTER_EXCEEDS_BUDGET:" + endpoint)
            self.sleeper(min(max(2 ** (attempt - 1), retry_after), remaining))


def standard_curl(request_url, headers, timeout):
    """Same HTTPS endpoint/proxy policy; no shell, redirects, impersonation or TLS bypass."""
    command = ["curl", "--silent", "--show-error", "--proto", "=https", "--max-time", str(timeout),
               "--connect-timeout", str(min(10, timeout)), "--write-out", "\n%{http_code}"]
    for name, value in headers.items():
        command.extend(["--header", name + ": " + value])
    with tempfile.TemporaryDirectory() as directory:
        header_path = Path(directory) / "headers"
        command.extend(["--dump-header", str(header_path), request_url])
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout + 1)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise urllib.error.URLError(type(error).__name__) from error
        raw_headers = header_path.read_bytes() if header_path.exists() else b""
    if result.returncode:
        raise urllib.error.URLError("CURL_EXIT_" + str(result.returncode))
    body, _, status = result.stdout.rpartition(b"\n")
    code = int(status)
    if not 200 <= code < 300:
        from email import message_from_bytes
        blocks = [block for block in raw_headers.split(b"\r\n\r\n") if block.startswith(b"HTTP/")]
        response_headers = message_from_bytes(blocks[-1].partition(b"\r\n")[2]) if blocks else {}
        raise urllib.error.HTTPError(request_url, code, "curl HTTP response", response_headers, None)
    return body.decode("utf-8")


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


def source_specs(primary, secondary):
    for proxy in dp.PROXIES:
        prefix = f"raw/{proxy.asset}_{proxy.security}"
        for adjustment in ("qfq", "unadjusted"):
            yield (f"{prefix}_eastmoney_{adjustment}.csv",
                   f"{dp.PRIMARY_PROVIDER}:{proxy.security}:{adjustment}",
                   lambda s, e, p=proxy, a=adjustment: primary(p, s, e, a))
        yield (f"{prefix}_tencent_qfq.csv", f"{dp.SECONDARY_PROVIDER}:{proxy.security}:qfq",
               lambda s, e, p=proxy: secondary(p, s, e))


def import_partial_sources(state, destination, as_of, required_cutoff, run_id, receipt):
    """Recover same-as-of public snapshots into a NEW packet; never alter old runs."""
    start, cutoff = pd.Timestamp("2016-12-01"), pd.Timestamp(as_of)
    (destination / "raw").mkdir(parents=True, exist_ok=True)
    checkpoint_path, checkpoint = dp._load_or_create_checkpoint(destination, destination / "raw", start, cutoff, True)
    labels = {relative: label for relative, label, _ in source_specs(None, None)}
    files = sorted((state / "attempts").glob("*.json"), key=lambda p: p.name, reverse=True)
    recovered = []
    for file in files:
        old = json.loads(file.read_text())
        if (old.get("run_id") == run_id or old.get("as_of") != as_of
                or old.get("required_cutoff") != required_cutoff
                or old.get("model_commit") != MODEL_COMMIT
                or old.get("status") != "FAILED_CLOSED"
                or old.get("failure_classification") != "DATA_MISSING"):
            continue
        old_id = old.get("run_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", old_id):
            raise dp.DataGateClosed("RECOVERY_RUN_ID_INVALID")
        # v1.0 receipts did not carry packet paths; this is their documented layout.
        relative_packet = old.get("packet") or f"runs/{old_id}/full"
        previous = dp._safe_relative_file(state, relative_packet)
        previous_checkpoint_path = previous / dp.CHECKPOINT_FILENAME
        if not previous_checkpoint_path.exists():
            continue
        previous_checkpoint = json.loads(previous_checkpoint_path.read_text())
        identity = dp._checkpoint_identity(start, cutoff)
        if {k: previous_checkpoint.get(k) for k in identity} != identity:
            raise dp.DataGateClosed("RECOVERY_CHECKPOINT_IDENTITY_MISMATCH")
        for relative, entry in previous_checkpoint.get("completed_sources", {}).items():
            if relative not in labels:
                raise dp.DataGateClosed("RECOVERY_UNEXPECTED_SOURCE")
            if relative in checkpoint["completed_sources"]:
                continue
            observed = pd.Timestamp(entry["completed_at_utc"])
            age = (pd.Timestamp(now()) - observed).total_seconds()
            if age < 0 or age > 86400:
                continue
            frame = dp._load_checkpoint_source(previous, previous_checkpoint, relative, labels[relative], start, cutoff)
            if frame is None or frame.date.max() != pd.Timestamp(required_cutoff):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous / relative, target)
            checkpoint["completed_sources"][relative] = dict(entry)
            recovered.append({"source": relative, "origin_run_id": old_id,
                              "sha256": entry["sha256"], "original_acquired_at": entry["completed_at_utc"]})
    checkpoint.update(status="IN_PROGRESS", updated_at_utc=now())
    write_json(checkpoint_path, checkpoint)
    receipt["recovered_sources"] = recovered
    receipt["recovered_source_count"] = len(recovered)


def gather_sources(packet, start, as_of, primary, secondary, receipt):
    """A failed provider cannot prevent independent providers from completing."""
    (packet / "raw").mkdir(parents=True, exist_ok=True)
    path, checkpoint = dp._load_or_create_checkpoint(packet, packet / "raw", start, as_of, True)
    outcomes = []
    failures = []
    for relative, label, fetch in source_specs(primary, secondary):
        try:
            cached = dp._load_checkpoint_source(packet, checkpoint, relative, label, start, as_of)
            if cached is None:
                frame = fetch(start, as_of)
                dp._persist_checkpoint_source(packet, path, checkpoint, relative, label, frame, start, as_of)
            outcomes.append({"source": relative, "status": "CACHED" if cached is not None else "FETCHED"})
        except RevisionDetected:
            raise
        except Exception as error:
            # Data/identity conflicts are not retryable. Continue only independent
            # network failures; malformed cached bytes remain a hard gate failure.
            if not isinstance(error, dp.DataGateClosed) or "DATA_MISSING" not in str(error):
                raise
            failures.append({"source": relative, "error": str(error)[:300]})
            outcomes.append({"source": relative, "status": "MISSING"})
        receipt["source_outcomes"] = outcomes
        receipt["source_failures"] = failures
        receipt["completed_source_count"] = len(checkpoint["completed_sources"])
    if failures:
        raise dp.DataGateClosed("DATA_MISSING:INCOMPLETE_SOURCE_PACKET:" + str(len(failures)))


def build_collected_packet(packet, primary, secondary, receipt, kwargs):
    gather_sources(packet, kwargs["start"], kwargs["as_of"], primary, secondary, receipt)
    return dp.build_production_panel(output_dir=packet, primary_fetcher=primary,
                                    secondary_fetcher=secondary, **kwargs)


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
    transport = transport or Transport(curl_runner=standard_curl)
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
                packet = prior
                receipt.update(status="NO_ACTION", run_path="FAST_NO_DELTA", packet=str(prior.relative_to(state)))
                return receipt
            if pd.Timestamp(old["as_of"]) >= pd.Timestamp(as_of):
                raise dp.DataGateClosed("AS_OF_NOT_AFTER_PREVIOUS_SNAPSHOT")
        kwargs = dict(start=pd.Timestamp("2016-12-01"), as_of=pd.Timestamp(as_of), resume=True)
        stage = "DATA_COLLECTION"
        # This patch is scoped to this single-process adapter. Model files stay immutable.
        with patch.object(dp, "_request_text", transport.request):
            incremental = prior is not None and not force_full
            if incremental:
                sources = IncrementalSources(prior, primary, secondary)
                receipt["run_path"] = "WEEKLY_INCREMENTAL"
                try:
                    build_collected_packet(packet, sources.fetch_primary, sources.fetch_secondary, receipt, kwargs)
                except RevisionDetected as error:
                    terminal_checkpoint(packet, str(error))
                    receipt["full_rebuild_reason"] = str(error)
                    incremental = False
            if not incremental:
                packet = state / "runs" / run_id / "full"
                receipt["run_path"] = "FULL_REBUILD"
                import_partial_sources(state, packet, as_of, required_cutoff, run_id, receipt)
                build_collected_packet(packet, primary, secondary, receipt, kwargs)
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
                       source_requests=transport.events, retry_ceiling=2,
                       packet=str(packet.relative_to(state)),
                       adapter_sha256=dp.sha256_file(Path(__file__)))
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
