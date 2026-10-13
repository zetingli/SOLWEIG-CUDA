import argparse
import calendar
from datetime import date
import json
import math
import operator
import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.series import Meteorology, _read_values, load_meteorology
from src.runtime import configure_runtime

PATH_KEYS = {"building_dsm_path", "tree_path", "dem_path", "met_file",
             "building_geometry_path", "building_height_path", "landcover_path", "output_path"}
PREPARATION_KEYS = {"case_name", "case_type", "case_size_m", "core_area_m", "buffer_area_m",
                    "resolution_m", "start_date", "end_date"}
_MORPHOLOGY = {
    "building_dsm_path": "Building_DSM.tif", "tree_path": "Trees.tif", "dem_path": "DEM.tif",
}


def check_weather(source):
    if isinstance(source, Meteorology):
        source = source.values
    if not isinstance(source, np.ndarray):
        if isinstance(source, (str, os.PathLike)):
            paths = [source]
        elif isinstance(source, (list, tuple)) and source:
            if not all(isinstance(path, (str, os.PathLike)) for path in source):
                raise TypeError("forcing file lists must contain only paths; use an ndarray for numeric data")
            paths = source
        else:
            raise TypeError("forcing must be a path, ndarray, nonempty ordered path list, or Meteorology")
        source = [Path(path).expanduser().resolve() for path in paths]
        if len(set(source)) != len(source):
            raise ValueError("forcing file list contains a duplicate path")
    values = _read_values(source)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 23:
        raise ValueError("forcing requires at least one row and 23 SOLWEIG columns in a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("forcing contains non-finite values")
    fields = values[:, :4]
    if not np.array_equal(fields, np.floor(fields)):
        raise ValueError("forcing year, day-of-year, hour and minute must be integers")
    if np.any((fields[:, 0] < 1) | (fields[:, 0] > 9999)):
        raise ValueError("forcing year must be between 1 and 9999")
    if np.any((fields[:, 1] < 1) | (fields[:, 1] > 366)):
        raise ValueError("forcing day-of-year must be valid for its calendar year")
    if np.any((fields[:, 2] < 0) | (fields[:, 2] > 23)):
        raise ValueError("forcing hour must be between 0 and 23")
    if np.any((fields[:, 3] < 0) | (fields[:, 3] > 59)):
        raise ValueError("forcing minute must be between 0 and 59")
    for year, doy in fields[:, :2].astype(np.int64):
        if doy > 365 + calendar.isleap(int(year)):
            raise ValueError(f"forcing day-of-year {doy} is invalid for calendar year {year}")
    for column, minimum, maximum, name in (
        (9, 0, np.inf, "wind"), (10, 0, 100, "relative humidity"),
        (11, -100, 100, "air temperature"), (12, 0.001, np.inf, "pressure"),
        (14, 0, np.inf, "global shortwave"),
    ):
        if np.any((values[:, column] < minimum) | (values[:, column] > maximum)):
            raise ValueError(f"forcing contains missing or invalid {name} values")
    forcing = load_meteorology(values)
    intervals = [(right - left).total_seconds() / 60
                 for left, right in zip(forcing.timestamps, forcing.timestamps[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("forcing timestamps must be strictly increasing with no duplicates")
    if intervals and any(interval != intervals[0] for interval in intervals[1:]):
        raise ValueError("forcing timestamps must have one constant timestep; missing samples or days are not supported")
    return forcing


def validate_preparation_options(raw):
    if not isinstance(raw, dict):
        raise ValueError("case.json must contain a JSON object")
    options = {key: raw[key] for key in PREPARATION_KEYS & raw.keys()}
    case_type = options.setdefault("case_type", "local")
    if case_type not in ("local", "open_data"):
        raise ValueError("case_type must be local or open_data")
    if "case_name" in options:
        name = options["case_name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("case_name must be a nonempty string")
    if case_type == "open_data" and "case_size_m" not in options:
        raise ValueError("case_type=open_data requires case_size_m")
    options.setdefault("resolution_m", 1)
    for key in ("case_size_m", "core_area_m", "buffer_area_m", "resolution_m"):
        if key not in options:
            continue
        value = options[key]
        try:
            finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (value < 0 if key == "buffer_area_m" else value <= 0):
            qualifier = "nonnegative" if key == "buffer_area_m" else "positive"
            raise ValueError(f"{key} must be a {qualifier} finite number")
    if ("core_area_m" in options) != ("buffer_area_m" in options):
        raise ValueError("core_area_m and buffer_area_m must be supplied together")
    for key in ("core_area_m", "buffer_area_m"):
        if key in options:
            pixels = options[key] / options["resolution_m"]
            if not math.isfinite(pixels) or not math.isclose(pixels, round(pixels), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{key} must be a whole number of resolution_m pixels")
    if ("start_date" in options) != ("end_date" in options):
        raise ValueError("start_date and end_date must be supplied together")
    if "start_date" in options:
        dates = []
        for key in ("start_date", "end_date"):
            value = options[key]
            try:
                parsed = date.fromisoformat(value) if isinstance(value, str) else None
            except ValueError:
                parsed = None
            if parsed is None or value != parsed.isoformat():
                raise ValueError(f"{key} must be a calendar date in YYYY-MM-DD format")
            dates.append(parsed)
        if dates[1] < dates[0]:
            raise ValueError("end_date must not precede start_date")
        if case_type == "open_data" and (isinstance(raw.get("utc_offset_hours"), bool)
                                          or raw.get("utc_offset_hours") != 0):
            raise ValueError("Open-data meteorology requires utc_offset_hours=0")
    return options


def check_options(case, *, dtype=np.float32, output_path=None, no_write=False):
    if np.dtype(dtype) not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64')
    for name, default in (('output_chunk_steps', 24), ('output_queue_size', 2)):
        value = case.get(name, default)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a positive integer')
        try:
            value = operator.index(value)
        except TypeError:
            raise ValueError(f'{name} must be a positive integer') from None
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(case.get('async_output', True), (bool, np.bool_)):
        raise ValueError('async_output must be boolean')
    for name, low, high in (('longitude', -180, 180), ('latitude', -90, 90),
                            ('utc_offset_hours', -24, 24)):
        value = case.get(name)
        if name in case and value is None and name != 'utc_offset_hours':
            raise ValueError(f'{name} must be a finite number between {low} and {high}')
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f'{name} must be a finite number between {low} and {high}')
    altitude = case.get('site_altitude_m', 3.)
    if isinstance(altitude, (bool, np.bool_)) or not isinstance(altitude, (int, float, np.number)) or not math.isfinite(altitude):
        raise ValueError('site_altitude_m must be finite')
    for name in ('minimum_wall_height_m', 'dsm_wall_threshold_m',
                 'default_building_height_m', 'level_height_m', 'minimum_building_height_m'):
        value = case.get(name, 1.)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be a positive finite number')
    for name in ('save_tmrt', 'save_svf', 'save_kup', 'save_kdown', 'save_lup', 'save_ldown', 'save_shadow'):
        if name in case and not isinstance(case[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be true or false')
    if case.get('wall_method', 'dsm') not in ('dsm', 'geometry'):
        raise ValueError('wall_method must be dsm or geometry')
    if case.get('wall_method') == 'geometry' and not all(
            case.get(key) for key in ('building_geometry_path', 'building_height_path')):
        raise ValueError('wall_method=geometry requires building_geometry_path and building_height_path')
    check_output_window(case.get('output_core'), None)
    if output_path is not None and not no_write:
        output = Path(output_path).resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f'Output must be a new or empty directory: {output}')


def check_output_window(core, shape, mask=None):
    if core is not None:
        if (not isinstance(core, (list, tuple)) or len(core) != 4
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in core)
                or min(core[:2]) < 0 or min(core[2:]) < 1):
            raise ValueError('output_core must be [row, col, rows, cols] with nonnegative offsets and positive dimensions')
        row, col, rows, cols = core
        if shape is not None and (row + rows > shape[0] or col + cols > shape[1]):
            raise ValueError('output_core must lie inside the computational raster')
        shape = (rows, cols)
    if mask is not None:
        values = np.asarray(mask)
        if values.dtype != np.dtype(bool):
            raise TypeError('output_valid_mask must have boolean dtype')
        if values.ndim != 2 or values.shape != tuple(shape):
            raise ValueError('output_valid_mask must be two-dimensional and match the cropped output shape')


def check_case(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    settings = validate_preparation_options(raw)
    case = {key: value for key, value in raw.items() if key not in PREPARATION_KEYS}
    required = {'building_dsm_path', 'tree_path', 'dem_path', 'met_file', 'longitude', 'latitude'}
    allowed = PATH_KEYS | required | {'minimum_wall_height_m', 'dsm_wall_threshold_m',
        'default_building_height_m', 'level_height_m', 'minimum_building_height_m',
        'save_tmrt', 'save_svf', 'save_kup', 'save_kdown', 'save_lup', 'save_ldown', 'save_shadow',
        'wall_method', 'utc_offset_hours', 'site_altitude_m', 'output_core',
        'output_chunk_steps', 'async_output', 'output_queue_size'}
    if required - case.keys():
        raise ValueError('Missing case options: ' + ', '.join(sorted(required - case.keys())))
    if case.keys() - allowed:
        raise ValueError('Unknown case options: ' + ', '.join(sorted(case.keys() - allowed)))
    for key in PATH_KEYS & case.keys():
        value = case[key]
        if value is None and key not in required and key != 'output_path':
            continue
        entries = value if key == 'met_file' and isinstance(value, list) else [value]
        if not entries or any(not isinstance(entry, str) or not entry.strip() for entry in entries):
            raise ValueError(f'{key} must be a nonempty path' + (' or ordered path list' if key == 'met_file' else ''))
        resolved = [str((path.parent / Path(entry).expanduser()).resolve()) for entry in entries]
        case[key] = resolved if isinstance(value, list) else resolved[0]
    case.setdefault('async_output', True)
    case.setdefault('output_queue_size', 2)
    case.setdefault('output_chunk_steps', 24)
    check_options(case)
    return case, settings


def check_rasters(case):
    configure_runtime()
    from osgeo import gdal, osr
    gdal.UseExceptions()
    reference = None
    keys = ['building_dsm_path', 'tree_path', 'dem_path']
    if case.get('landcover_path') is not None:
        keys.append('landcover_path')
    if case.get('wall_method') == 'geometry':
        keys.append('building_height_path')
    for key in keys:
        path = Path(case[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset = gdal.Open(str(path))
        if dataset.RasterCount != 1:
            raise ValueError(f'{path.name} must have exactly one band')
        grid = (dataset.RasterYSize, dataset.RasterXSize,
                dataset.GetGeoTransform(), dataset.GetProjection())
        rows, cols, affine, projection = grid
        crs = osr.SpatialReference()
        if (not projection or crs.ImportFromWkt(projection) != 0 or not crs.IsProjected()
                or not math.isclose(crs.GetLinearUnits(), 1., rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'{path.name} must use a projected CRS in metres')
        if (not np.isfinite(affine).all() or affine[1] <= 0 or affine[5] >= 0
                or affine[2] != 0 or affine[4] != 0
                or not math.isclose(affine[1], -affine[5], rel_tol=1e-9)):
            raise ValueError(f'{path.name} must be north-up with square pixels')
        if reference is not None and grid != reference:
            raise ValueError(f'{path.name} is not aligned with Building_DSM')
        reference = grid
        values = dataset.GetRasterBand(1).ReadAsArray()
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f'{path.name} must be a finite two-dimensional raster')
        if key == 'tree_path' and np.any(values < 0):
            raise ValueError('Canopy heights must be nonnegative')
        dataset = values = None
    if case.get('wall_method') == 'geometry':
        from shapely.geometry import shape
        from rasterio.warp import transform_geom
        from src.walls import building_height
        collection = json.loads(Path(case['building_geometry_path']).read_text(encoding='utf-8'))
        if collection.get('type') != 'FeatureCollection':
            raise ValueError('Building geometry must be a GeoJSON FeatureCollection')
        for feature in collection['features']:
            geometry = shape(transform_geom('EPSG:4326', reference[3], feature['geometry']))
            if geometry.geom_type not in ('Polygon', 'MultiPolygon') or geometry.is_empty or not geometry.is_valid:
                raise ValueError('Building geometry must contain valid nonempty polygons')
            height = building_height(feature.get('properties'),
                default_height_m=case.get('default_building_height_m', 6.),
                level_height_m=case.get('level_height_m', 3.),
                minimum_height_m=case.get('minimum_building_height_m', 2.))
            if not math.isfinite(height):
                raise ValueError('building fallback heights must be finite')
    return reference[:2]


def check_preparation(case, settings, case_path):
    if settings.get('case_type', 'local') == 'open_data':
        if 'start_date' in settings:
            start, end = (date.fromisoformat(settings[key]) for key in ('start_date', 'end_date'))
            if end < start:
                raise ValueError('end_date must be on or after start_date')
        if case.get('utc_offset_hours', 0) != 0:
            raise ValueError('open_data ERA5 requires utc_offset_hours=0')
        paths = [Path(case[key]) for key in _MORPHOLOGY]
        if any(path.name != name or path.parent != paths[0].parent for path, name in zip(paths, _MORPHOLOGY.values())):
            raise ValueError('open_data morphology paths must be Building_DSM.tif, Trees.tif and DEM.tif in one directory')
        if not isinstance(case['met_file'], str) or Path(case['met_file']).suffix.lower() != '.txt':
            raise ValueError('open_data met_file must be one .txt path')
        met = Path(case['met_file'])
        if not met.is_file() and 'start_date' not in settings:
            raise ValueError('start_date and end_date are required to prepare missing meteorology')
        if not all(path.is_file() for path in paths):
            configure_runtime()
            from scripts.morphology import prepare_morphology
            folder = paths[0].parent
            folder.mkdir(parents=True, exist_ok=True)
            size = settings['case_size_m']
            with TemporaryDirectory(prefix='.morphology-', dir=folder) as staging:
                prepare_morphology(latitude=case['latitude'], longitude=case['longitude'],
                                   width=size, height=size, resolution=settings.get('resolution_m', 1),
                                   output=staging, cache=Path(case_path).resolve().parent / 'cache' / 'morphology')
                for path in paths:
                    if not path.is_file():
                        (Path(staging) / path.name).replace(path)
        if not met.is_file():
            from scripts.meteorology import download_request, download_era5, prepare_met
            latitude, longitude = case['latitude'], case['longitude']
            start, end = settings['start_date'], settings['end_date']
            request = download_request(latitude, longitude, start, end)
            files = [download_era5(met.parent / 'era5', request)]
            prepare_met(files, met, latitude=latitude, longitude=longitude,
                        start_date=start, end_date=end)
    if 'core_area_m' in settings:
        if case.get('output_core') is not None:
            raise ValueError('Use core_area_m/buffer_area_m or output_core, not both.')
        if case.get('wall_method', 'dsm') != 'dsm':
            raise ValueError("Tiling requires wall_method='dsm'.")
        names = dict(_MORPHOLOGY)
        if case.get('landcover_path'):
            names['landcover_path'] = 'Landcover.tif'
        folder = Path(case['building_dsm_path']).parent
        if any(Path(case[key]) != folder / name for key, name in names.items()):
            raise ValueError('Tiling requires the standard raster filenames in one folder.')
        configure_runtime()
        import rasterio
        with rasterio.open(case['building_dsm_path']) as src:
            counts = [settings[key] / src.res[0] for key in ('core_area_m', 'buffer_area_m')]
        if not all(math.isclose(n, round(n), abs_tol=1e-8, rel_tol=0) for n in counts):
            raise ValueError('Core and buffer metres must be whole numbers of source pixels.')


def check_tiling(input_dir, *, core=1800, halo=100, format='vrt'):
    for name, value, minimum in (('core', core, 1), ('halo', halo, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if format not in ('vrt', 'tif'):
        raise ValueError("format must be 'vrt' or 'tif'")
    root = Path(input_dir)
    case = {key: root / name for key, name in _MORPHOLOGY.items()}
    if (root / 'Landcover.tif').exists():
        case['landcover_path'] = root / 'Landcover.tif'
    check_rasters(case)


def check_inputs(case, *, dtype=np.float32, output_path=None, no_write=False):
    check_options(case, dtype=dtype, output_path=output_path, no_write=no_write)
    shape = check_rasters(case)
    check_output_window(case.get('output_core'), shape, case.get('output_valid_mask'))
    check_weather(case['met_file'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare missing open-data inputs and check SOLWEIG inputs without running the model.')
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--output-core', type=int, nargs=4)
    parser.add_argument('--output-chunk-steps', type=int)
    parser.add_argument('--output-queue-size', type=int)
    parser.add_argument('--sync-output', action='store_true')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)
    try:
        case, settings = check_case(args.case)
        for name in ('output_core', 'output_chunk_steps', 'output_queue_size'):
            if getattr(args, name) is not None:
                case[name] = getattr(args, name)
        if args.sync_output:
            case['async_output'] = False
        output = Path(case.pop('output_path', args.case.parent / 'output'))
        check_preparation(case, settings, args.case)
        check_inputs(case, dtype=args.dtype, output_path=args.output or output, no_write=args.no_write)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f'Input check: {error}', file=sys.stderr)
        return 2
    print('Inputs OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
