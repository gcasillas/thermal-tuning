from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_irimage_uav_manifest, process_batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch thermal processing for DJI RJPG datasets")
    parser.add_argument("input_dir", type=Path, help="Folder containing RJPG images")
    parser.add_argument("output_dir", type=Path, help="Destination folder for outputs")
    parser.add_argument("--pattern", default="*.RJPG", help="Glob for thermal files (default: *.RJPG)")
    parser.add_argument("--threshold-c", type=float, default=None, help="Optional threshold in Celsius")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV path for IRimage-UAV/ImageJ import",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    results = process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        threshold_c=args.threshold_c,
    )
    if args.manifest:
        build_irimage_uav_manifest(results, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
