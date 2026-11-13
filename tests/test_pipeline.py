from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from thermal_tuning.pipeline import build_irimage_uav_manifest, process_batch


class FakeParser:
    def parse(self, image_path: Path):
        matrix = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        metadata = {"Latitude": 34.1, "Longitude": -117.2, "Altitude": 400.0}
        return matrix, metadata


class PipelineTests(unittest.TestCase):
    def test_process_batch_writes_matrix_stats_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "frame1.RJPG").write_bytes(b"fake")
            (input_dir / "frame2.RJPG").write_bytes(b"fake")

            results = process_batch(input_dir, output_dir, parser=FakeParser())

            self.assertEqual(len(results), 2)
            matrix = np.load(output_dir / "matrices" / "frame1.npy")
            self.assertEqual(matrix.dtype, np.float32)
            self.assertEqual(matrix.shape, (2, 2))

            stats_payload = json.loads((output_dir / "stats" / "frame1.json").read_text(encoding="utf-8"))
            self.assertEqual(stats_payload["stats"]["mean"], 25.0)

            summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["images_processed"], 2)

    def test_process_batch_threshold_mask(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "frame1.RJPG").write_bytes(b"fake")

            process_batch(input_dir, output_dir, threshold_c=25.0, parser=FakeParser())

            mask = np.load(output_dir / "masks" / "frame1_ge_25c.npy")
            np.testing.assert_array_equal(mask, np.array([[0, 0], [1, 1]], dtype=np.uint8))

    def test_build_irimage_manifest_uses_gps_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "frame1.RJPG").write_bytes(b"fake")

            results = process_batch(input_dir, output_dir, parser=FakeParser())
            manifest_path = build_irimage_uav_manifest(results, output_dir / "irimage-uav-manifest.csv")

            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["latitude"], "34.1")
            self.assertEqual(rows[0]["longitude"], "-117.2")


if __name__ == "__main__":
    unittest.main()
