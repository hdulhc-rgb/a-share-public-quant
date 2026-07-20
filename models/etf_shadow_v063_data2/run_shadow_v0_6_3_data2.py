#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_ROOT = Path(__file__).resolve().parents[1]
FROZEN_V063_ROOT = MODELS_ROOT / "etf_shadow_v063"
if str(FROZEN_V063_ROOT) not in sys.path:
    sys.path.insert(0, str(FROZEN_V063_ROOT))

from etf_shadow_v063.core import ResearchClosed, ResearchPolicy, align_weights, policy_dict
from etf_shadow_v063.data_pipeline import validate_production_manifest
from etf_shadow_v063_data2.data_contract import (
    MODEL_VERSION,
    align_return_panels,
    apply_local_proxy,
    assess_data2_grade,
    dependency_evidence,
    load_return_panel,
    validate_local_proxy_manifest,
    validate_official_economic_manifest,
    validate_qdii_decomposition,
)
from etf_shadow_v063_data2.runner import run_research_data2


def demo_returns() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    index = pd.bdate_range("2019-01-02", "2026-07-17")
    assets = ["A_SHARE", "US_SP500", "US_NASDAQ100", "GOLD", "CASH"]
    market = rng.normal(0.00025, 0.0090, size=len(index))
    inflation = rng.normal(0.00005, 0.0040, size=len(index))
    matrix = np.column_stack(
        [
            0.65 * market + rng.normal(0, 0.0080, len(index)),
            0.85 * market + rng.normal(0, 0.0045, len(index)),
            1.15 * market + rng.normal(0, 0.0070, len(index)),
            -0.10 * market + 0.65 * inflation + rng.normal(0, 0.0050, len(index)),
            np.full(len(index), 0.00008),
        ]
    )
    return pd.DataFrame(np.clip(matrix, -0.18, 0.18), index=index, columns=assets)


def load_weight_file(path: Path, assets: list[str]) -> pd.Series:
    frame = pd.read_csv(path)
    if not {"asset", "weight"}.issubset(frame.columns):
        raise ResearchClosed("WEIGHT_FILE_REQUIRES_ASSET_WEIGHT")
    return align_weights(frame.set_index("asset")["weight"], assets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-asset ETF v0.6.3-data.2 dual-return shadow challenge engine"
    )
    parser.add_argument("--execution-returns-csv", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--economic-returns-csv", type=Path)
    parser.add_argument("--economic-manifest", type=Path)
    parser.add_argument("--local-proxy-returns-csv", type=Path)
    parser.add_argument("--local-proxy-manifest", type=Path)
    parser.add_argument("--qdii-decomposition-csv", type=Path)
    parser.add_argument("--qdii-decomposition-manifest", type=Path)
    parser.add_argument("--current-weights", type=Path)
    parser.add_argument("--benchmark-weights", type=Path)
    parser.add_argument("--as-of", default="2026-07-20")
    parser.add_argument("--output-root", type=Path, default=Path("runs_v063_data2"))
    parser.add_argument("--profile", choices=["shadow", "promotion"], default="shadow")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--print-policy", action="store_true")
    return parser.parse_args()


def _paired(left: Path | None, right: Path | None, label: str) -> None:
    if (left is None) != (right is None):
        raise ResearchClosed(f"{label}_REQUIRES_PANEL_AND_MANIFEST")


def main() -> int:
    args = parse_args()
    policy = ResearchPolicy()
    if args.print_policy:
        print(
            json.dumps(
                {"model_version": MODEL_VERSION, **policy_dict(policy)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        if args.demo and args.profile == "promotion":
            raise ResearchClosed("DEMO_CANNOT_USE_PROMOTION_PROFILE")
        _paired(args.economic_returns_csv, args.economic_manifest, "ECONOMIC_INPUT")
        _paired(args.local_proxy_returns_csv, args.local_proxy_manifest, "LOCAL_PROXY_INPUT")
        _paired(
            args.qdii_decomposition_csv,
            args.qdii_decomposition_manifest,
            "QDII_DECOMPOSITION_INPUT",
        )
        cutoff = pd.Timestamp(args.as_of).normalize()

        if args.demo:
            execution = demo_returns()
            economic = execution.copy()
            assets = list(execution.columns)
            benchmark = align_weights(
                pd.Series(np.full(len(assets), 1.0 / len(assets)), index=assets), assets
            )
            current = benchmark.copy()
            execution_attestation = {
                "return_kind": "SYNTHETIC_DEMO",
                "production_return_panel_gate_passed": False,
            }
            economic_attestation = None
            local_attestation = None
            local_evidence = None
            qdii_attestation = None
            dependencies = dependency_evidence("shadow")
            assessment = {
                "status": "SYNTHETIC_DEMO",
                "data_grade": "DEMO",
                "shadow_gate_passed": True,
                "promotion_gate_passed": False,
                "evidence_gaps": ["SYNTHETIC_DEMO_NOT_PRODUCTION_EVIDENCE"],
                "economic_return_policy": "SYNTHETIC_DEMO",
                "no_silent_fallback": True,
            }
            execution_source_path = None
            economic_source_path = None
            execution_source_id = "SYNTHETIC_DEMO"
            economic_source_id = "SYNTHETIC_DEMO"
        else:
            if args.execution_returns_csv is None or args.execution_manifest is None:
                raise ResearchClosed("PRODUCTION_RUN_REQUIRES_EXECUTION_PANEL_AND_MANIFEST")
            if args.current_weights is None or args.benchmark_weights is None:
                raise ResearchClosed("PRODUCTION_RUN_REQUIRES_CURRENT_AND_BENCHMARK_WEIGHTS")
            execution = load_return_panel(args.execution_returns_csv)
            execution_attestation = validate_production_manifest(
                returns=execution,
                returns_path=args.execution_returns_csv,
                manifest_path=args.execution_manifest,
                as_of=cutoff,
            )
            if args.economic_returns_csv is not None:
                economic = load_return_panel(args.economic_returns_csv)
                economic_attestation = validate_official_economic_manifest(
                    returns=economic,
                    returns_path=args.economic_returns_csv,
                    manifest_path=args.economic_manifest,
                    as_of=cutoff,
                )
                economic_source_path = args.economic_returns_csv
                economic_source_id = "OFFICIAL_ECONOMIC_TOTAL_RETURN"
            else:
                economic = execution.copy()
                economic_attestation = None
                economic_source_path = args.execution_returns_csv
                economic_source_id = "EXPLICIT_EXECUTION_PROXY_FALLBACK"

            local_attestation = None
            local_evidence = None
            if args.local_proxy_returns_csv is not None:
                local_returns = load_return_panel(args.local_proxy_returns_csv)
                local_attestation = validate_local_proxy_manifest(
                    local_returns=local_returns,
                    returns_path=args.local_proxy_returns_csv,
                    manifest_path=args.local_proxy_manifest,
                    as_of=cutoff,
                )
                execution, local_evidence = apply_local_proxy(
                    execution, local_returns, local_attestation
                )
                economic, _ = apply_local_proxy(economic, local_returns, local_attestation)

            economic, execution = align_return_panels(economic, execution, cutoff)
            qdii_attestation = (
                validate_qdii_decomposition(
                    args.qdii_decomposition_csv,
                    args.qdii_decomposition_manifest,
                    cutoff,
                    execution_returns=execution,
                )
                if args.qdii_decomposition_csv is not None
                else None
            )
            dependencies = dependency_evidence(args.profile)
            assessment = assess_data2_grade(
                execution_gate_passed=bool(
                    execution_attestation.get("production_return_panel_gate_passed")
                ),
                official_economic_attestation=economic_attestation,
                local_proxy_attestation=local_attestation,
                qdii_attestation=qdii_attestation,
                dependency_attestation=dependencies,
                profile=args.profile,
            )
            assets = list(execution.columns)
            benchmark = load_weight_file(args.benchmark_weights, assets)
            current = load_weight_file(args.current_weights, assets)
            execution_source_path = args.execution_returns_csv
            execution_source_id = str(execution_attestation["source_id"])

        run_dir = run_research_data2(
            economic_returns=economic,
            execution_returns=execution,
            benchmark=benchmark,
            current=current,
            as_of=cutoff,
            output_root=args.output_root,
            economic_source_path=economic_source_path,
            execution_source_path=execution_source_path,
            economic_source_id=economic_source_id,
            execution_source_id=execution_source_id,
            policy=policy,
            profile=args.profile,
            data2_assessment=assessment,
            economic_source_attestation=economic_attestation,
            execution_source_attestation=execution_attestation,
            local_proxy_evidence=local_evidence,
            qdii_evidence=qdii_attestation,
        )
    except (ResearchClosed, OSError, ValueError) as error:
        print(f"FAILED_CLOSED: {error}", file=sys.stderr)
        return 2
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
