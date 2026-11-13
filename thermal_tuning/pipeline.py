from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np


class ThermalParser(Protocol):
    def parse(self, image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Return (temperature_matrix, metadata) for one thermal image."""


@dataclass(frozen=True)
class ImageResult:
    source: Path
    matrix_file: Path
    stats_file: Path
    metadata: dict[str, Any]
    stats: dict[str, float]


class DJIThermalSDKParser:
    """Best-effort parser that uses a community DJI thermal SDK wrapper if installed."""

    MATRIX_ATTRS = (
        "temperature",
        "temperature_map",
        "thermal_data",
        "get_temperature",
        "get_temperature_array",
        "get_numpy",
    )
    METADATA_ATTRS = ("metadata", "meta", "exif", "get_metadata")

    def parse(self, image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            from thermal_parser.thermal import Thermal  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "No compatible DJI thermal parser found. Install a wrapper such as "
                "SanNianYiSi/thermal_parser and ensure the DJI SDK runtime is available."
            ) from exc

        thermal_obj = Thermal(str(image_path))

        matrix_data: Any | None = None
        for attr_name in self.MATRIX_ATTRS:
            attr = getattr(thermal_obj, attr_name, None)
            if callable(attr):
                matrix_data = attr()
                break
            if attr is not None:
                matrix_data = attr
                break

        if matrix_data is None:
            raise RuntimeError(
                "Unable to extract thermal matrix from parser object. Supported attributes: "
                f"{', '.join(self.MATRIX_ATTRS)}"
            )

        matrix = np.asarray(matrix_data)
        if matrix.ndim != 2:
            raise RuntimeError(f"Expected a 2D temperature matrix, got shape {matrix.shape!r}")

        metadata: dict[str, Any] = {}
        for attr_name in self.METADATA_ATTRS:
            attr = getattr(thermal_obj, attr_name, None)
            if callable(attr):
                candidate = attr()
            else:
                candidate = attr
            if isinstance(candidate, dict):
                metadata = candidate
                break

        return matrix.astype(np.float32, copy=False), metadata


def _thermal_stats(matrix: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
        "p95": float(np.percentile(matrix, 95)),
    }


def _gps_value(metadata: dict[str, Any], *keys: str) -> Any:
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def process_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*.RJPG",
    threshold_c: float | None = None,
    parser: ThermalParser | None = None,
) -> list[ImageResult]:
    parser = parser or DJIThermalSDKParser()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "matrices"
    stats_dir = output_dir / "stats"
    masks_dir = output_dir / "masks"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(input_dir.rglob(pattern))
    results: list[ImageResult] = []
    for source in sources:
        matrix, metadata = parser.parse(source)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Parsed matrix for {source} is not 2D")

        matrix_path = matrices_dir / f"{source.stem}.npy"
        np.save(matrix_path, matrix)

        stats = _thermal_stats(matrix)
        if threshold_c is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            threshold_mask = (matrix >= threshold_c).astype(np.uint8)
            np.save(masks_dir / f"{source.stem}_ge_{threshold_c:g}c.npy", threshold_mask)
            stats["threshold_c"] = float(threshold_c)
            stats["pixels_ge_threshold"] = float(np.sum(threshold_mask))

        stats_path = stats_dir / f"{source.stem}.json"
        stats_payload = {
            "source": str(source),
            "matrix_file": str(matrix_path),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "stats": stats,
            "metadata": metadata,
        }
        stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

        results.append(
            ImageResult(
                source=source,
                matrix_file=matrix_path,
                stats_file=stats_path,
                metadata=metadata,
                stats=stats,
            )
        )

    summary = {
        "input_dir": str(input_dir),
        "pattern": pattern,
        "images_processed": len(results),
        "files": [str(result.source) for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def build_irimage_uav_manifest(results: Iterable[ImageResult], manifest_path: Path) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "matrix_file", "latitude", "longitude", "altitude"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "source": str(item.source),
                    "matrix_file": str(item.matrix_file),
                    "latitude": _gps_value(item.metadata, "latitude", "gpslatitude", "lat"),
                    "longitude": _gps_value(item.metadata, "longitude", "gpslongitude", "lon", "lng"),
                    "altitude": _gps_value(item.metadata, "altitude", "gpsaltitude", "alt"),
                }
            )
    return manifest_path
