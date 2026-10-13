from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from src import __version__
from src.runtime import configure_runtime

PATH_KEYS = {"building_dsm_path", "tree_path", "dem_path", "met_file",
             "building_geometry_path", "building_height_path", "landcover_path", "output_path"}
PREPARATION_KEYS = {"case_name", "case_type", "case_size_m", "core_area_m", "buffer_area_m",
                    "resolution_m", "start_date", "end_date"}


_MORPHOLOGY = {
    "building_dsm_path": "Building_DSM.tif", "tree_path": "Trees.tif", "dem_path": "DEM.tif",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SOLWEIG for the supplied rasters and meteorological time series.")
    parser.add_argument("--version", action="version", version=f"SOLWEIG-CUDA {__version__}")
    parser.add_argument("--case", type=Path, default=Path(__file__).resolve().parent / "examples/case.json",
                        help="JSON case file (default: examples/case.json beside run.py); paths are relative to this file")
    parser.add_argument("--output", type=Path, help="override case output_path; default: output beside the case")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="radiation precision")
    parser.add_argument("--output-core", type=int, nargs=4, metavar=("ROW", "COL", "ROWS", "COLS"), help="crop outputs to this window")
    parser.add_argument("--output-chunk-steps", type=int, help="maximum time steps per output file (default: 24)")
    parser.add_argument("--sync-output", action="store_true", help="disable asynchronous writing")
    parser.add_argument("--output-queue-size", type=int, help="maximum live output batches (default: 2)")
    parser.add_argument("--no-write", action="store_true", help="compute without writing files")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="print timestamped stage and daily progress")
    args = parser.parse_args(argv)
    case_path = args.case.resolve()
    case, settings = load_case(case_path)
    if args.output_core is not None:
        case["output_core"] = args.output_core
    if args.output_chunk_steps is not None:
        case["output_chunk_steps"] = args.output_chunk_steps
    if args.sync_output:
        case["async_output"] = False
    if args.output_queue_size is not None:
        case["output_queue_size"] = args.output_queue_size
    output = Path(case.pop("output_path", case_path.parent / "output"))
    output = (args.output or output).resolve()
    configure_runtime()
    if settings["case_type"] == "open_data":
        prepare_open_data(case_path, case, settings, progress=args.progress)
    import numpy as np
    from src.series import load_meteorology, run_timeseries
    if args.progress:
        print(f"[{datetime.now():%H:%M:%S}] Meteorology loading started", flush=True)
    forcing = load_meteorology(case["met_file"])
    if args.progress and "core_area_m" in settings:
        print(f"[{datetime.now():%H:%M:%S}] Spatial tiling started", flush=True)
    for label, options in _tile_jobs(case_path, case, settings):
        destination = output / label if "core_area_m" in settings else output
        if args.progress and "core_area_m" in settings:
            print(f"[{datetime.now():%H:%M:%S}] Tile {label} started", flush=True)
        run_timeseries(**{**options, "met_file": forcing}, output_path=destination,
                       dtype=np.dtype(args.dtype), no_write=args.no_write, progress=args.progress)
    if args.progress and not args.no_write:
        print(f"[{datetime.now():%H:%M:%S}] Results: {output}", flush=True)
    return 0


def load_case(path):
    path = Path(path).resolve()
    case = json.loads(path.read_text(encoding="utf-8-sig"))
    settings = {key: case.pop(key) for key in PREPARATION_KEYS if key in case}
    settings.setdefault("case_type", "local")
    settings.setdefault("resolution_m", 1)
    for key in PATH_KEYS & case.keys():
        value = case[key]
        if value is None:
            continue
        entries = value if isinstance(value, list) else [value]
        resolved = [str((path.parent / Path(entry).expanduser()).resolve()) for entry in entries]
        case[key] = resolved if isinstance(value, list) else resolved[0]
    case.setdefault("async_output", True)
    case.setdefault("output_queue_size", 2)
    case.setdefault("output_chunk_steps", 24)
    return case, settings


def prepare_open_data(case_path, case, settings, *, progress=False):
    morphology = [Path(case[key]) for key in _MORPHOLOGY]
    folder = morphology[0].parent
    present = [path.is_file() for path in morphology]
    if not all(present):
        if progress:
            print(f"[{datetime.now():%H:%M:%S}] Morphology preparation started", flush=True)
        size = settings["case_size_m"]
        from scripts.morphology import prepare_morphology
        prepare_morphology(latitude=case["latitude"], longitude=case["longitude"],
                           width=size, height=size, resolution=settings.get("resolution_m", 1),
                           output=folder, cache=Path(case_path).parent / "cache" / "morphology")
    met = Path(case["met_file"])
    if not met.is_file():
        if progress:
            print(f"[{datetime.now():%H:%M:%S}] Meteorology preparation started", flush=True)
        from scripts.meteorology import download_request, download_era5, prepare_met
        latitude, longitude = case["latitude"], case["longitude"]
        start, end = settings["start_date"], settings["end_date"]
        request = download_request(latitude, longitude, start, end)
        files = [download_era5(met.resolve().parent / "era5", request)]
        prepare_met(files, met, latitude=latitude, longitude=longitude,
                    start_date=start, end_date=end)


def _tile_jobs(case_path, case, settings):
    if "core_area_m" not in settings:
        return [(settings.get("case_name", "tile"), dict(case))]
    import rasterio
    names = {"building_dsm_path": "Building_DSM.tif", "tree_path": "Trees.tif",
             "dem_path": "DEM.tif"}
    if case.get("landcover_path"):
        names["landcover_path"] = "Landcover.tif"
    source_dir = Path(case["building_dsm_path"]).parent
    with rasterio.open(case["building_dsm_path"]) as src:
        counts = [settings[key] / src.res[0] for key in ("core_area_m", "buffer_area_m")]
    core, halo = (int(round(n)) for n in counts)
    from src.tiling import create_tiles
    tiles = create_tiles(source_dir, case_path.parent / "tiles", core=core, halo=halo)
    return [(tile["tile_id"], {**case, **{key: tile["inputs"][key] for key in names},
                                "output_core": tile["output_core"]}) for tile in tiles]


if __name__ == "__main__":
    raise SystemExit(main())
