#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local, forward-only model-asset return proxy without publishing constituents"
    )
    parser.add_argument("--component-prices-csv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cutoff = pd.Timestamp(args.as_of).normalize()
        prices = pd.read_csv(args.component_prices_csv)
        mapping = pd.read_csv(args.mapping_csv)
        required_prices = {"date", "asset", "close"}
        required_mapping = {"component_asset", "model_asset", "weight", "effective_from"}
        if not required_prices.issubset(prices.columns):
            raise ValueError(f"COMPONENT_PRICE_SCHEMA_MISMATCH:{sorted(required_prices - set(prices.columns))}")
        if not required_mapping.issubset(mapping.columns):
            raise ValueError(f"LOCAL_MAPPING_SCHEMA_MISMATCH:{sorted(required_mapping - set(mapping.columns))}")
        prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
        prices["close"] = pd.to_numeric(prices["close"], errors="raise")
        mapping["weight"] = pd.to_numeric(mapping["weight"], errors="raise")
        mapping["effective_from"] = pd.to_datetime(
            mapping["effective_from"], errors="raise"
        ).dt.normalize()
        if prices.duplicated(["date", "asset"]).any():
            raise ValueError("COMPONENT_PRICE_DUPLICATE_KEY")
        if mapping.duplicated(["component_asset", "model_asset"]).any():
            raise ValueError("LOCAL_MAPPING_DUPLICATE_COMPONENT")
        if (prices["date"] > cutoff).any() or (mapping["effective_from"] > cutoff).any():
            raise ValueError("LOCAL_PROXY_AS_OF_VIOLATION")
        if (prices["close"] <= 0).any() or (mapping["weight"] < 0).any():
            raise ValueError("LOCAL_PROXY_NON_POSITIVE_PRICE_OR_NEGATIVE_WEIGHT")
        missing = sorted(set(mapping["component_asset"]) - set(prices["asset"]))
        if missing:
            raise ValueError(f"LOCAL_PROXY_COMPONENT_MISSING:{missing}")
        weight_sums = mapping.groupby("model_asset")["weight"].sum()
        if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-10, rtol=0):
            raise ValueError("LOCAL_PROXY_WEIGHTS_NOT_FULLY_INVESTED")

        price_wide = prices.pivot(index="date", columns="asset", values="close").sort_index()
        component_returns = price_wide.pct_change(fill_method=None)
        output = pd.DataFrame(index=component_returns.index)
        effective_dates: dict[str, pd.Timestamp] = {}
        constituent_counts: dict[str, int] = {}
        for model_asset, group in mapping.groupby("model_asset", sort=True):
            dates = group["effective_from"].drop_duplicates()
            if len(dates) != 1:
                raise ValueError(f"LOCAL_PROXY_MIXED_EFFECTIVE_DATES:{model_asset}")
            effective_from = pd.Timestamp(dates.iloc[0]).normalize()
            components = list(group["component_asset"])
            weights = group.set_index("component_asset")["weight"].reindex(components)
            available = component_returns[components].dropna(how="any")
            available = available.loc[available.index > effective_from]
            output[model_asset] = available.mul(weights, axis=1).sum(axis=1)
            effective_dates[str(model_asset)] = effective_from
            constituent_counts[str(model_asset)] = len(components)
        output = output.dropna(how="any").sort_index()
        if output.empty:
            raise ValueError("LOCAL_PROXY_NO_FORWARD_OBSERVATIONS")
        if len(set(effective_dates.values())) != 1:
            raise ValueError("LOCAL_PROXY_MODEL_ASSETS_REQUIRE_COMMON_EFFECTIVE_DATE")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        panel_path = args.output_dir / "local_proxy_returns.csv"
        output.rename_axis("date").to_csv(panel_path, float_format="%.12g")
        packet_hash = hashlib.sha256(
            (
                sha256_file(args.component_prices_csv)
                + sha256_file(args.mapping_csv)
                + json.dumps(constituent_counts, sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": "2.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "as_of": cutoff.date().isoformat(),
            "point_in_time": True,
            "construction_mode": "CURRENT_FORWARD_ONLY",
            "effective_from": next(iter(effective_dates.values())).date().isoformat(),
            "model_assets": list(output.columns),
            "constituent_count": int(sum(constituent_counts.values())),
            "panel_sha256": sha256_file(panel_path),
            "component_packet_sha256": packet_hash,
            "source_price_packet_sha256": sha256_file(args.component_prices_csv),
            "mapping_packet_sha256": sha256_file(args.mapping_csv),
            "privacy": "CONSTITUENT_IDENTITIES_AND_WEIGHTS_REMAIN_LOCAL",
            "historical_backtest_eligible": False,
            "promotion_eligible": False,
            "research_lock": True,
        }
        manifest_path = args.output_dir / "local_proxy_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"FAILED_CLOSED: {error}", file=sys.stderr)
        return 2
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
