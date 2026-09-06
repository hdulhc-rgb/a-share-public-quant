from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class LocalProxyBuilderTests(unittest.TestCase):
    def test_builder_starts_strictly_after_snapshot(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            dates = pd.bdate_range("2026-07-06", periods=8)
            prices = []
            for component, start in [("COMPONENT_A", 1.0), ("COMPONENT_B", 2.0)]:
                for number, date in enumerate(dates):
                    prices.append(
                        {"date": date, "asset": component, "close": start * (1.01**number)}
                    )
            prices_path = workspace / "components.csv"
            pd.DataFrame(prices).to_csv(prices_path, index=False)
            mapping_path = workspace / "mapping.csv"
            pd.DataFrame(
                [
                    {
                        "component_asset": "COMPONENT_A",
                        "model_asset": "A_SHARE",
                        "weight": 0.8,
                        "effective_from": dates[2],
                    },
                    {
                        "component_asset": "COMPONENT_B",
                        "model_asset": "A_SHARE",
                        "weight": 0.2,
                        "effective_from": dates[2],
                    },
                ]
            ).to_csv(mapping_path, index=False)
            output = workspace / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "build_local_proxy.py"),
                    "--component-prices-csv",
                    str(prices_path),
                    "--mapping-csv",
                    str(mapping_path),
                    "--as-of",
                    dates[-1].date().isoformat(),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            panel = pd.read_csv(output / "local_proxy_returns.csv", parse_dates=["date"])
            manifest = json.loads(
                (output / "local_proxy_manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(panel["date"].min(), dates[2])
            self.assertEqual(manifest["construction_mode"], "CURRENT_FORWARD_ONLY")
            self.assertFalse(manifest["promotion_eligible"])
            self.assertNotIn("COMPONENT_A", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
