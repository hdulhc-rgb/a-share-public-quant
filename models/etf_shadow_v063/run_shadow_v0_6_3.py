#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from etf_shadow_v063.core import ResearchClosed, ResearchPolicy, align_weights, policy_dict
from etf_shadow_v063.data_pipeline import validate_production_manifest
from etf_shadow_v063.runner import run_research


def demo_returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260719)
    index = pd.bdate_range("2019-01-02", "2026-07-17")
    assets = ["A_SHARE", "SP500", "NASDAQ100", "GOLD", "CASH", "CHINA_DIVIDEND", "SEMICONDUCTOR"]
    market = rng.normal(0.00025, 0.0090, size=len(index))
    inflation = rng.normal(0.00005, 0.0040, size=len(index))
    regime = np.where((np.arange(len(index)) // 190) % 3 == 1, -0.00035, 0.00020)
    matrix = np.column_stack([
        0.65 * market + regime + rng.normal(0, 0.0080, len(index)),
        0.85 * market + rng.normal(0, 0.0045, len(index)),
        1.15 * market + rng.normal(0, 0.0070, len(index)),
        -0.10 * market + 0.65 * inflation + rng.normal(0, 0.0050, len(index)),
        np.full(len(index), 0.00008),
        0.45 * market + rng.normal(0, 0.0055, len(index)),
        1.25 * market + rng.normal(0, 0.0100, len(index)),
    ])
    return pd.DataFrame(np.clip(matrix, -0.18, 0.18), index=index, columns=assets)


def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ResearchClosed("RETURNS_CSV_REQUIRES_DATE_COLUMN")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.set_index("date").sort_index()
    return frame.astype(float)


def load_weight_file(path: Path, assets: list[str]) -> pd.Series:
    frame = pd.read_csv(path)
    if not {"asset", "weight"}.issubset(frame.columns):
        raise ResearchClosed("WEIGHT_FILE_REQUIRES_ASSET_WEIGHT")
    return align_weights(frame.set_index("asset")["weight"], assets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-asset ETF v0.6.3 shadow challenge engine")
    parser.add_argument("--returns-csv", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--current-weights", type=Path)
    parser.add_argument("--benchmark-weights", type=Path)
    parser.add_argument("--as-of", default="2026-07-17")
    parser.add_argument("--output-root", type=Path, default=Path("runs_v063"))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--optional-engines", choices=["record", "require"], default="record")
    parser.add_argument("--print-policy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = ResearchPolicy()
    if args.print_policy:
        print(json.dumps(policy_dict(policy), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        if not args.demo and args.returns_csv is None:
            raise ResearchClosed("PRODUCTION_RUN_REQUIRES_RETURNS_CSV")
        if not args.demo and args.data_manifest is None:
            raise ResearchClosed("PRODUCTION_RUN_REQUIRES_DATA_MANIFEST")
        returns = demo_returns() if args.demo else load_returns(args.returns_csv)
        assets = list(returns.columns)
        source_attestation = None
        if args.demo:
            # Synthetic fixtures deliberately avoid encoding any real portfolio weights.
            benchmark = align_weights(
                pd.Series(np.full(len(assets), 1.0 / len(assets)), index=assets),
                assets,
            )
            current = benchmark.copy()
        else:
            if args.current_weights is None or args.benchmark_weights is None:
                raise ResearchClosed(
                    "PRODUCTION_RUN_REQUIRES_CURRENT_AND_BENCHMARK_WEIGHTS"
                )
            source_attestation = validate_production_manifest(
                returns=returns,
                returns_path=args.returns_csv,
                manifest_path=args.data_manifest,
                as_of=pd.Timestamp(args.as_of),
            )
            benchmark = load_weight_file(args.benchmark_weights, assets)
            current = load_weight_file(args.current_weights, assets)
        run_dir = run_research(
            returns=returns,
            benchmark=benchmark,
            current=current,
            as_of=pd.Timestamp(args.as_of),
            output_root=args.output_root,
            source_path=args.returns_csv,
            source_id=(
                "SYNTHETIC_DEMO"
                if args.demo
                else str(source_attestation["source_id"])
            ),
            policy=policy,
            optional_engines=args.optional_engines,
            source_attestation=source_attestation,
        )
    except (ResearchClosed, OSError, ValueError) as error:
        print(f"FAILED_CLOSED: {error}", file=sys.stderr)
        return 2
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
