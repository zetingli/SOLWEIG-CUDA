from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from osgeo import gdal
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds, transform_geom
from rasterio.windows import Window
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.walls import building_height

Bounds = tuple[float, float, float, float]
CHM_BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
GHSL_TILE_BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/"
    "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100/V1-0/tiles"
)
GHSL_TILE_PREFIX = "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100_V1_0"
GHSL_CRS = "ESRI:54009"
FABDEM_BASE_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"
GHSL_XMIN, GHSL_YMAX, GHSL_TILE_SIZE = -18_041_000.0, 9_000_000.0, 1_000_000.0
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
OSM_QUERY_BUFFER_M = 500.0


def _bounds_key(bounds: Bounds) -> str:
    return hashlib.sha256(",".join(f"{v:.8f}" for v in bounds).encode("ascii")).hexdigest()[:16]


def _osm_query_bounds(bounds: Bounds) -> Bounds:
    west, south, east, north = bounds
    latitude_pad = OSM_QUERY_BUFFER_M / 111_320.0
    longitude_pad = latitude_pad / max(1e-6, math.cos(math.radians(max(abs(south), abs(north)))))
    return (max(-180.0, west - longitude_pad), max(-90.0, south - latitude_pad),
            min(180.0, east + longitude_pad), min(90.0, north + latitude_pad))


@contextmanager
def _atomic_path(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=f".part{target.suffix}", dir=target.parent)
    os.close(handle)
    part = Path(filename)
    try:
        yield part
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def _readable_raster(path: Path) -> None:
    with rasterio.open(path) as src:
        if not src.crs or min(src.width, src.height, src.count) < 1:
            raise ValueError(f"Invalid raster: {path}")
        for col, row in ((0, 0), (src.width // 2, src.height // 2), (src.width - 1, src.height - 1)):
            src.read(1, window=Window(col, row, 1, 1))


def _cached(path: Path, validate: Callable[[Path], Any], force: bool = False) -> bool:
    if force or not path.is_file():
        return False
    try:
        validate(path)
        return True
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
        return False


def _read_collection(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError(f"Invalid GeoJSON FeatureCollection: {path}")
    for feature in data["features"]:
        geometry = shape(feature.get("geometry") or {})
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Invalid polygon in {path}")
    return data


def _download_file(url: str, target: Path, validate: Callable[[Path], Any], force: bool = False) -> Path:
    if _cached(target, validate, force):
        return target
    with _atomic_path(target) as part:
        with urlopen(Request(url, headers={"User-Agent": "SOLWEIG-CUDA/1.0"}), timeout=300) as response, part.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        validate(part)
    return target


def _overpass_query(bounds: Bounds) -> dict[str, Any]:
    west, south, east, north = bounds
    bbox = f"({south},{west},{north},{east})"
    query = (f'[out:json][timeout:60][maxsize:134217728];(way["building"]["building"!="no"]{bbox};'
             f'relation["building"]["building"!="no"]["type"="multipolygon"]{bbox};);out body geom qt;')
    errors = []
    for url in OVERPASS_URLS:
        try:
            request = Request(url, data=urlencode({"data": query}).encode("utf-8"), headers={
                "User-Agent": "SOLWEIG-CUDA/1.0", "Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
                raise ValueError("Invalid Overpass response")
            if data.get("remark"):
                raise RuntimeError(f"Incomplete Overpass response: {data['remark']}")
            return data
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("OSM download failed. " + "; ".join(errors))


def _coords(element: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(p["lon"]), float(p["lat"])) for p in element["geometry"]]


def _stitch_rings(segments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    remaining, rings = [list(s) for s in segments], []
    while remaining:
        ring = remaining.pop(0)
        while len(ring) >= 2 and ring[0] != ring[-1]:
            for index, segment in enumerate(remaining):
                if ring[-1] == segment[0]:
                    ring += segment[1:]
                elif ring[-1] == segment[-1]:
                    ring += list(reversed(segment[:-1]))
                elif ring[0] == segment[-1]:
                    ring = segment[:-1] + ring
                elif ring[0] == segment[0]:
                    ring = list(reversed(segment[1:])) + ring
                else:
                    continue
                remaining.pop(index)
                break
            else:
                raise ValueError("OSM multipolygon contains unclosed member ways")
        if len(ring) < 4:
            raise ValueError("OSM building ring has fewer than three vertices")
        rings.append(ring)
    return rings


def _geometry(element: dict[str, Any]) -> dict[str, Any]:
    if element["type"] == "way":
        rings = _stitch_rings([_coords(element)])
        geom = Polygon(rings[0])
    else:
        segments: dict[str, list] = {"outer": [], "inner": []}
        for member in element.get("members") or []:
            role = member.get("role") or "outer"
            if member.get("type") != "way" or role not in segments:
                continue
            segments[role].append(_coords(member))
        outers = _stitch_rings(segments["outer"])
        inners = _stitch_rings(segments["inner"])
        polygons = [[ring] for ring in outers]
        for inner in inners:
            containers = [i for i, outer in enumerate(outers) if Polygon(outer).covers(Polygon(inner))]
            polygons[min(containers, key=lambda i: Polygon(outers[i]).area)].append(inner)
        geom = unary_union([Polygon(rings[0], rings[1:]) for rings in polygons])
    if geom.is_empty or not geom.is_valid:
        raise ValueError(f"Invalid OSM building {element['type']}/{element.get('id')}")
    return mapping(geom)


def _to_geojson(data: dict[str, Any]) -> dict[str, Any]:
    elements = { (e.get("type"), e.get("id")): e for e in data["elements"]
                if (e.get("type") == "way" or
                    (e.get("type") == "relation" and e.get("tags", {}).get("type") == "multipolygon"
                     and any(m.get("type") == "way" and m.get("role", "") in ("", "outer")
                             for m in e.get("members", []))))
                and e.get("tags", {}).get("building") not in (None, "no") }
    relations = [e for e in elements.values() if e["type"] == "relation"]
    member_ids = {m["ref"] for e in relations for m in e.get("members", [])
                  if m.get("type") == "way" and m.get("role", "") in ("", "outer", "inner")}
    features = []
    for element in elements.values():
        if element["type"] == "way" and element["id"] in member_ids:
            continue
        features.append({"type": "Feature", "geometry": _geometry(element),
                         "properties": element.get("tags") or {}})
    return {"type": "FeatureCollection", "features": features}


def _download_osm(bounds: Bounds, output: Path, force: bool = False) -> Path:
    def validate_cache(path: Path) -> None:
        query_bounds = _read_collection(path).get("query_bounds_wgs84")
        if (not isinstance(query_bounds, (list, tuple)) or len(query_bounds) != 4
                or query_bounds[0] > bounds[0] or query_bounds[1] > bounds[1]
                or query_bounds[2] < bounds[2] or query_bounds[3] < bounds[3]):
            raise ValueError("OSM cache does not record coverage of the expanded query bounds")
    if _cached(output, validate_cache, force):
        return output
    try:
        data = _overpass_query(bounds)
    except (OSError, ValueError, RuntimeError):
        west, south, east, north = bounds
        midx, midy = (west + east) / 2, (south + north) / 2
        responses = [_overpass_query(b) for b in ((west, south, midx, midy), (midx, south, east, midy),
                                                (west, midy, midx, north), (midx, midy, east, north))]
        data = {"elements": [e for response in responses for e in response["elements"]]}
    collection = _to_geojson(data)
    collection["query_bounds_wgs84"] = bounds
    with _atomic_path(output) as part:
        part.write_text(json.dumps(collection), encoding="utf-8")
    return output


def _ghsl_ids(bounds: Bounds) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = bounds
    cols = range(max(1, math.floor((minx - GHSL_XMIN) / GHSL_TILE_SIZE) + 1),
                 min(37, math.ceil((maxx - GHSL_XMIN) / GHSL_TILE_SIZE)) + 1)
    rows = range(max(1, math.floor((GHSL_YMAX - maxy) / GHSL_TILE_SIZE) + 1),
                 min(18, math.ceil((GHSL_YMAX - miny) / GHSL_TILE_SIZE)) + 1)
    return [(row, col) for row in rows for col in cols]


def _validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff"))]
        if len(names) != 1:
            raise ValueError(f"Expected one GeoTIFF in GHSL archive: {path}")
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt GHSL archive: {path}")


def _download_ghsl(bounds: Bounds, cache: Path, force: bool = False) -> tuple[Path, ...]:
    projected = transform_bounds("EPSG:4326", GHSL_CRS, *bounds, densify_pts=41)
    paths = []
    for row, col in _ghsl_ids(projected):
        stem = f"{GHSL_TILE_PREFIX}_R{row}_C{col}"
        target = cache / "ghsl" / "extracted" / f"{stem}.tif"
        if not _cached(target, _readable_raster, force):
            archive_path = cache / "ghsl" / f"{stem}.zip"
            _download_file(f"{GHSL_TILE_BASE_URL}/{stem}.zip", archive_path, _validate_zip, force)
            with zipfile.ZipFile(archive_path) as archive, _atomic_path(target) as part:
                member = next(n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff")))
                with archive.open(member) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                _readable_raster(part)
        paths.append(target)
    if not paths:
        raise RuntimeError("No GHSL height tiles cover the requested bounds")
    return tuple(paths)


def _chm_tiles(index: Path, bounds: Bounds) -> list[str]:
    area = box(*bounds)
    tiles = set()
    for feature in _read_collection(index)["features"]:
        if shape(feature["geometry"]).intersection(area).area <= 0:
            continue
        tile = str(feature.get("properties", {}).get("tile", ""))
        if not re.fullmatch(r"[0-3]+", tile):
            raise ValueError("Meta/WRI tile index contains an invalid quadkey")
        tiles.add(tile)
    return sorted(tiles)


def _crop_raster(source: str | Path, bounds: Bounds, output: Path) -> Path:
    with ExitStack() as stack:
        stack.callback(gdal.VSICurlClearCache)
        stack.enter_context(rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                         CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3",
                                         GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="120"))
        with rasterio.open(source) as src:
            projected = transform_bounds("EPSG:4326", src.crs, *bounds, densify_pts=41)
            transform = src.transform
            left, bottom, right, top = projected
            col0, col1 = (left - transform.c) / transform.a, (right - transform.c) / transform.a
            row0, row1 = (top - transform.f) / transform.e, (bottom - transform.f) / transform.e
            x0, y0 = max(0, math.floor(col0) - 2), max(0, math.floor(row0) - 2)
            x1, y1 = min(src.width, math.ceil(col1) + 2), min(src.height, math.ceil(row1) + 2)
            window = Window(x0, y0, x1 - x0, y1 - y0)
            profile = dict(driver="GTiff", count=1, dtype="float32", crs=src.crs,
                           transform=src.window_transform(window), width=int(window.width), height=int(window.height),
                           nodata=-9999.0, tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER")
            with _atomic_path(output) as part:
                with rasterio.open(part, "w", **profile) as dst:
                    for _, block in dst.block_windows(1):
                        read_window = Window(x0 + block.col_off, y0 + block.row_off, block.width, block.height)
                        if src.nodata == 0:
                            values = src.read(1, window=read_window, out_dtype="float32")
                        else:
                            values = src.read(1, window=read_window, masked=True, out_dtype="float32").filled(-9999.0)
                        dst.write(values, 1, window=block)
                _readable_raster(part)
    return output


def _download_canopy(bounds: Bounds, cache: Path, force: bool = False) -> tuple[Path, ...]:
    index = _download_file(f"{CHM_BASE_URL}/tiles.geojson", cache / "chm_tiles.geojson", _read_collection, force)
    tiles = _chm_tiles(index, bounds)
    if not tiles:
        raise RuntimeError("No Meta/WRI v1 canopy-height tile covers the requested area")
    paths = []
    for tile in tiles:
        output = cache / "chm" / "crops" / f"{tile}_{_bounds_key(bounds)}.tif"
        if not _cached(output, _readable_raster, force):
            local = cache / "chm" / f"{tile}.tif"
            source = local if _cached(local, _readable_raster, force) else f"{CHM_BASE_URL}/chm/{tile}.tif"
            _crop_raster(source, bounds, output)
        paths.append(output)
    return tuple(paths)


def _download_dem(bounds: Bounds, output: Path, cache: Path, force: bool = False) -> Path:
    if _cached(output, _readable_raster, force):
        return output
    index = _download_file(f"{FABDEM_BASE_URL}/FABDEM_v1-2_tiles.geojson",
                           cache / "tiles.geojson", _read_collection, force)
    pad = 2.0 / 3600
    padded = (max(-180, bounds[0] - pad), max(-90, bounds[1] - pad),
              min(180, bounds[2] + pad), min(90, bounds[3] + pad))
    inner = (math.nextafter(padded[0], padded[2]), math.nextafter(padded[1], padded[3]),
             math.nextafter(padded[2], padded[0]), math.nextafter(padded[3], padded[1]))
    area = box(*inner)
    with ExitStack() as stack:
        stack.callback(gdal.VSICurlClearCache)
        stack.enter_context(rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                         CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".zip", GDAL_HTTP_TIMEOUT="120"))
        readers = []
        for feature in _read_collection(index)["features"]:
            if shape(feature["geometry"]).intersection(area).area <= 0:
                continue
            properties = feature["properties"]
            name = properties["file_name"][0] + properties["file_name"][2:]
            archive = (properties["zipfile_name"].replace("S-", "S").replace("N-", "N")
                       .replace("W000", "E000").replace("E180", "W180"))
            source = cache / name
            if not _cached(source, _readable_raster, force):
                source = f"/vsizip/{{/vsicurl/{FABDEM_BASE_URL}/{archive}}}/{name}"
            readers.append(stack.enter_context(rasterio.open(source)))
        with _atomic_path(output) as part:
            merge(readers, bounds=inner, dst_path=part, dst_kwds={"compress": "deflate", "predictor": 3})
            _readable_raster(part)
    return output


def acquire_sources(bounds_wgs84: Bounds, cache_dir: str | Path, *, force: bool = False) -> dict[str, Any]:
    bounds = bounds_wgs84
    cache = Path(cache_dir).expanduser().resolve()
    key = _bounds_key(bounds)
    dem_path = cache / "fabdem" / f"FABDEM_V1-2_{key}.tif"
    osm_bounds = _osm_query_bounds(bounds)
    osm = _download_osm(osm_bounds, cache / "osm" / f"buildings_{key}.geojson", force)
    ghsl = _download_ghsl(bounds, cache, force)
    canopy = _download_canopy(bounds, cache, force)
    dem = _download_dem(bounds, dem_path, cache / "fabdem" / "cache", force)
    return {"osm": osm, "ghsl": ghsl, "canopy": canopy, "dem": dem}


def plan_grid(*, latitude, longitude, width, height, resolution=1.0, crs=None):
    zone = min(60, int(math.floor((longitude + 180) / 6)) + 1)
    target = CRS.from_user_input(crs or f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}")
    x, y = transform("EPSG:4326", target, [longitude], [latitude])
    cols, rows = math.ceil(width / resolution), math.ceil(height / resolution)
    west, north = x[0] - cols * resolution / 2, y[0] + rows * resolution / 2
    affine = Affine(resolution, 0, west, 0, -resolution, north)
    bounds = array_bounds(rows, cols, affine)
    geographic = transform_bounds(target, "EPSG:4326", *bounds, densify_pts=21)
    return {
        "width": cols, "height": rows, "crs": target.to_string(),
        "transform": list(affine)[:6], "bounds": list(bounds),
        "bounds_wgs84": list(geographic),
    }


def _buildings(path, crs, settings):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    geometries, heights = [], []
    for feature in data["features"]:
        geometry = feature.get("geometry")
        projected = shape(transform_geom("EPSG:4326", crs, geometry))
        geometries.append(projected)
        heights.append(building_height(feature.get("properties"), **settings))
    return geometries, heights


def _warps(stack, paths, profile, resampling, *, sparse=False, fallback_nodata=None):
    readers = []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths:
        source = stack.enter_context(rasterio.open(path))
        nodata = source.nodata if source.nodata is not None else fallback_nodata
        if sparse and nodata == 0:
            nodata = np.nan
        args = dict(crs=profile["crs"], transform=profile["transform"],
                    width=profile["width"], height=profile["height"],
                    dtype="float32", nodata=np.nan, resampling=resampling)
        if nodata is not None:
            args["src_nodata"] = nodata
        readers.append(stack.enter_context(WarpedVRT(source, **args)))
    return readers


def _read(readers, window):
    result = np.full((int(window.height), int(window.width)), np.nan, dtype=np.float32)
    for source in readers:
        values = source.read(1, window=window, masked=True).filled(np.nan)
        result = np.fmax(result, values)
    return result


def prepare_morphology(*, latitude, longitude, width, height, output, resolution=1.0,
                       cache=None, crs=None, force=False, default_height=6.0,
                       level_height=3.0, minimum_height=2.0):
    grid = plan_grid(latitude=latitude, longitude=longitude, width=width, height=height,
                     resolution=resolution, crs=crs)
    settings = dict(default_height_m=default_height, level_height_m=level_height,
                    minimum_height_m=minimum_height)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any((output / name).exists() for name in ("Building_DSM.tif", "DEM.tif", "Trees.tif"))):
        raise ValueError("Choose a new or empty output directory.")
    cache = Path(cache).resolve() if cache else output.parent / ".morphology_cache"
    sources = acquire_sources(tuple(grid["bounds_wgs84"]), cache, force=force)
    geometries, heights = _buildings(sources["osm"], grid["crs"], settings)
    tree = STRtree(geometries) if geometries else None
    profile = dict(driver="GTiff", width=grid["width"], height=grid["height"],
                   crs=grid["crs"], transform=Affine(*grid["transform"]), count=1,
                   dtype="float32", compress="DEFLATE", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    output.mkdir(parents=True, exist_ok=True)
    outputs = {"building_dsm": "Building_DSM.tif", "dem": "DEM.tif", "trees": "Trees.tif"}
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=output) as staging:
        staging = Path(staging)
        with ExitStack() as stack:
            dem = _warps(stack, sources["dem"], profile, Resampling.bilinear, fallback_nodata=-9999)
            canopy = _warps(stack, sources["canopy"], profile, Resampling.bilinear, sparse=True)
            ghsl = _warps(stack, sources["ghsl"], profile, Resampling.nearest,
                          sparse=True, fallback_nodata=255)
            writers = [stack.enter_context(rasterio.open(staging / name, "w", **profile))
                       for name in outputs.values()]
            for _, window in writers[0].block_windows(1):
                terrain, trees = _read(dem, window), _read(canopy, window)
                local_transform = writers[0].window_transform(window)
                local_shape = terrain.shape
                fallback = np.zeros(local_shape, np.float32)
                if tree is not None:
                    local_bounds = array_bounds(*local_shape, local_transform)
                    candidates = sorted(tree.query(box(*local_bounds)), key=lambda i: heights[i])
                    if len(candidates):
                        fallback = rasterize([(geometries[i], heights[i]) for i in candidates],
                                             out_shape=local_shape, transform=local_transform,
                                             fill=0, dtype="float32", all_touched=False)
                footprint = fallback > 0
                sampled = _read(ghsl, window)
                use_ghsl = footprint & np.isfinite(sampled) & (sampled >= minimum_height)
                building = np.where(use_ghsl, sampled, fallback)
                dsm = terrain + building
                for writer, values in zip(writers, (dsm, terrain, trees)):
                    writer.write(values, 1, window=window)
        for name in outputs.values():
            (staging / name).replace(output / name)
    return {key: output / name for key, name in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description='Prepare morphology for a WGS84 bounding box.')
    parser.add_argument("--latitude", type=float, required=True, help="Centre latitude, WGS84 degrees.")
    parser.add_argument("--longitude", type=float, required=True, help="Centre longitude, WGS84 degrees.")
    parser.add_argument("--width", type=float, required=True, help="East-west extent in metres.")
    parser.add_argument("--height", type=float, required=True, help="North-south extent in metres.")
    parser.add_argument("--resolution", type=float, default=1, help="Output cell size in metres (default: 1).")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output folder.")
    parser.add_argument("--cache", type=Path, help="Reusable downloads; default: .morphology_cache beside output.")
    parser.add_argument("--crs", help="Projected metre CRS; default: local UTM zone.")
    parser.add_argument("--default-height", type=float, default=6, help="Building fallback height in metres.")
    parser.add_argument("--level-height", type=float, default=3, help="Metres per OSM building level.")
    parser.add_argument("--minimum-height", type=float, default=2, help="Minimum building height in metres.")
    parser.add_argument("--force", action="store_true", help="Refresh source downloads for this request.")
    args = vars(parser.parse_args())
    prepare_morphology(**args)
    print(f"Inputs ready: {args['output'].resolve()}")


if __name__ == "__main__":
    main()
