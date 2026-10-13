# Input preparation

Run preparation commands from the project root.

## Meteorology

`python scripts/meteorology.py` reads local ERA5 NetCDF files or downloads a point
series, then writes a SOLWEIG `met.txt` table:

```console
python scripts/meteorology.py --download --latitude 53.35 --longitude -6.26 --start 2025-07-12 --end 2025-07-12 --output data/met.txt
```
## Morphology

Morphology preparation uses the Rasterio and Shapely dependencies:

```console
python scripts/morphology.py --latitude 22.28 --longitude 114.16 --width 2000 --height 2000 --resolution 1 --output data/morphology
```

Coordinates are WGS84 degrees; width, height and resolution are metres.
Local UTM is selected automatically; `--crs EPSG:2157` selects another
projected CRS. Fractional cell counts round up. The output comprises aligned
Float32 `DEM.tif`, `Building_DSM.tif` and `Trees.tif`: terrain, terrain plus
buildings, and above-ground canopy height respectively. Land cover is optional.

| Source | Use and licence |
| --- | --- |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Building footprints, height and level tags; ODbL. |
| [GHSL GHS-BUILT-H R2023A](https://data.jrc.ec.europa.eu/dataset/85005901-3a49-48dd-9d19-6261354f56fe) | 2018 average net building height at 100 m; European Commission reuse notice. |
| [Meta/WRI canopy-height v1](https://registry.opendata.aws/dataforgood-fb-forests/) | Above-ground canopy heights; CC BY 4.0. |
| [FABDEM V1-2](https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn) | Approximately 30 m terrain elevations; CC BY-NC-SA 4.0. |

After preparation, run`python scripts/input_check.py --case case.json` 
to check raster alignment, missing values and negative canopy heights. 

Downloads are cached in `.morphology_cache` beside the output, or at `--cache`.
Keep source downloads for reproducibility, particularly because OSM changes.
Dataset licences apply separately from the software.

## Tiling

```console
python -m src.tiling data/morphology --output data/tiles --core 1800 --halo 100
```

Core and halo sizes are pixels. VRTs reference windows of the source rasters,
which must remain available. `--format tif` creates independent copies.
Cores cover the input once; halos are clipped at its edges. Choose the halo
for the surrounding geometry; 100 pixels is not a guarantee against boundary
effects. `create_tiles` returns a list of input paths and `output_core` crops
in memory. 

## Automatic case preparation

`python run.py` combines acquisition, tiling and simulation. Set these
fields in a copy of `examples/case.json`:

| Field | Meaning |
| --- | --- |
| `case_type` | `open_data` prepares missing inputs; `local` uses supplied inputs. |
| `case_name` | Optional case label. |
| `case_size_m` | Side length of the square morphology domain. |
| `resolution_m` | Cell size in metres, default 1. |
| `core_area_m` | Side length of each retained core, in metres. |
| `buffer_area_m` | Halo width on each side, in metres. |
| `start_date`, `end_date` | Inclusive UTC dates in `YYYY-MM-DD` format. |
