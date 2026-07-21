#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from etf_shadow_v063.data_pipeline import DataGateClosed, build_production_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dual-source, distribution-adjusted ETF return panel"
    )
    parser.add_argument("--start", default="2016-12-01")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="require an empty output directory instead of resuming a verified same-as-of checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_production_panel(
            start=pd.Timestamp(args.start),
            as_of=pd.Timestamp(args.as_of),
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )
    except (DataGateClosed, ValueError) as error:
        print(f"FAILED_CLOSED: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
