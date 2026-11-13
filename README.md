# thermal-tuning

Minimal batch-first thermal processing workflow for DJI radiometric JPEG datasets.

## What this project now supports

1. **Automated batch processing pipelines**
   - Parse folders of `.RJPG` files in one pass.
   - Save each frame's full thermal matrix as `float32` NumPy (`.npy`) output.
2. **Matrix data extraction**
   - Compute thermal stats (`min`, `max`, `mean`, `std`, `p95`) per frame.
   - Optionally generate threshold masks (for hotspot/isotherm-style filtering).
3. **GIS & orthomosaic prep workflow**
   - Export an IRimage-UAV/ImageJ-friendly CSV manifest with source path + GPS fields.
   - Use the manifest and original frames in Fiji + IRimage-UAV to batch-generate 32-bit GeoTIFF products while preserving GPS EXIF.

## Usage

```bash
python -m thermal_tuning.cli /absolute/path/to/rjpg_folder /absolute/path/to/output \
  --pattern "*.RJPG" \
  --threshold-c 45 \
  --manifest /absolute/path/to/output/irimage-uav-manifest.csv
```

Outputs are written to:

- `output/matrices/*.npy` (2D `float32` temperature matrices)
- `output/stats/*.json` (stats + metadata)
- `output/masks/*.npy` (optional threshold masks)
- `output/summary.json`
- `output/irimage-uav-manifest.csv` (optional)

## Notes on DJI SDK wrappers

The default parser expects a compatible Python wrapper around DJI Thermal SDK libraries (for example, community wrappers such as `thermal_parser`).

If your environment is missing the wrapper or the required SDK runtime libraries, install/configure those first (Windows-native setups are commonly used for DJI DLL-based wrappers).
