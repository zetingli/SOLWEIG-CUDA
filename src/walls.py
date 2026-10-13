from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import transform_geom
from scipy.ndimage import rotate
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape
from shapely.geometry.polygon import orient

from .runtime import cupy, get_kernel


def _filter_size(pixel_size_m: float) -> int:
    size = int(np.floor((1.0 / pixel_size_m + 1e-10) * 9))
    if size <= 2:
        size = 3
    elif size != 9 and size % 2 == 0:
        size += 1
    return size


@lru_cache(maxsize=8)
def _templates(size: int):
    radius = size // 2
    line = np.zeros((size, size))
    sides = np.zeros((size, size))
    line[:, radius] = 1
    sides[radius, :radius] = 1
    sides[radius, radius + 1:] = 2
    templates = []
    for angle in range(180):
        rotated_line = np.round(rotate(line, angle, order=1, reshape=False, mode="nearest"))
        rotated_sides = np.round(rotate(sides, angle, order=0, reshape=False, mode="nearest"))
        direction = 270 - angle
        if angle in (150, 30):
            rotated_sides[:, size - 1] = 0
        if direction == 225:
            rotated_line[0, 0] = rotated_line[size - 1, size - 1] = 1
        if direction == 135:
            rotated_line[0, size - 1] = rotated_line[size - 1, 0] = 1
        templates.append(tuple(
            np.argwhere(mask).astype(np.int64) - radius
            for mask in (rotated_line != 0, rotated_sides == 1, rotated_sides == 2)
        ))
    return tuple(templates)


def _height_array(surface: np.ndarray, threshold: float):
    cp = cupy()
    source = cp.asarray(surface, dtype=cp.float64, order="C")
    output = cp.empty(source.shape, dtype=cp.float64)
    if source.size:
        height_kernel = get_kernel("geometry.cu", "wall_heights", cp.cuda.Device().id)
        height_kernel(
            ((source.size + 255) // 256,), (256,),
            (source, output, np.int32(source.shape[0]), np.int32(source.shape[1]), np.float64(threshold)),
        )
    return output


def _selected_sums(surface: np.ndarray, positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    sums = np.zeros(positions.size, dtype=np.float64)
    if offsets.size:
        batch = max(1, 131072 // offsets.size)
        for start in range(0, positions.size, batch):
            stop = min(start + batch, positions.size)
            values = surface[positions[start:stop, None] + offsets[None, :]]
            sums[start:stop] = values.sum(axis=1, dtype=np.float64)
    return sums


def _aspect_array(wall_height: np.ndarray, surface: np.ndarray, pixel_size: float, wall_device):
    wall_mask = wall_height > 0
    aspect = np.zeros(surface.shape, dtype=np.float64)
    if not np.any(wall_mask):
        return aspect

    rows, cols = surface.shape
    size = _filter_size(pixel_size)
    radius = size // 2
    if rows > 2 * radius + 2 and cols > 2 * radius + 2:
        cp = cupy()
        templates = _templates(size)
        lines = [entry[0][:, 0] * cols + entry[0][:, 1] for entry in templates]
        starts = np.concatenate(([0], np.cumsum([line.size for line in lines]))).astype(np.int32)
        offsets = np.concatenate(lines).astype(np.int64)
        positions = np.flatnonzero(wall_mask).astype(np.int64, copy=False)
        angles_device = cp.empty(positions.size, dtype=cp.int16)
        wall_occupancy = (wall_device > 0).astype(cp.uint8)
        direction_kernel = get_kernel("geometry.cu", "wall_directions", cp.cuda.Device().id)
        direction_kernel(
            ((positions.size + 127) // 128,), (128,),
            (wall_occupancy, cp.asarray(positions), np.int64(positions.size),
             cp.asarray(offsets), cp.asarray(starts), np.int32(rows), np.int32(cols),
             np.int32(radius), angles_device),
        )
        angles = cp.asnumpy(angles_device)
        source_flat = np.ravel(surface, order="C")
        aspect_flat = aspect.ravel()
        for angle in np.unique(angles):
            if angle < 0:
                continue
            chosen = positions[angles == angle]
            _, side1, side2 = templates[int(angle)]
            first = _selected_sums(source_flat, chosen, side1[:, 0] * cols + side1[:, 1])
            second = _selected_sums(source_flat, chosen, side2[:, 0] * cols + side2[:, 1])
            direction = np.full(chosen.size, 270.0 - int(angle))
            direction[first > second] -= 180.0
            direction[direction < 0] += 360.0
            aspect_flat[chosen] = direction

    step = 1.0 / (1.0 / pixel_size)
    fy, fx = np.gradient(surface, step, step)
    gradient_aspect = -np.arctan2(fx, fy)
    gradient_aspect[gradient_aspect < 0] += 2 * np.pi
    unresolved = wall_mask & (aspect == 0)
    aspect += unresolved * (gradient_aspect / (math.pi / 180.0))
    return aspect


def calculate_dsm_walls(
    dsm: np.ndarray,
    pixel_size_m: float,
    minimum_wall_height_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    surface = np.asarray(dsm, dtype=np.float64)
    pixel_size = float(pixel_size_m)
    threshold = float(minimum_wall_height_m)
    wall_device = _height_array(surface, threshold)
    walls = cupy().asnumpy(wall_device)
    return walls, _aspect_array(walls, surface, pixel_size, wall_device)


_NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")
_HEIGHT_KEYS = ("height", "building:height", "est_height", "building:part:height")
_LEVEL_KEYS = ("building:levels", "levels", "building_levels")


def _geometry_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _numbers(value: Any) -> list[float]:
    text = str(value).strip().lower()
    if not text or text in {"yes", "no", "none", "null", "nan"}:
        return []
    return [
        number
        for token in _NUMBER_RE.findall(text)
        if (number := _geometry_positive(token.replace(",", "."))) is not None
    ]


def _length_metres(value: Any) -> float | None:
    direct = _geometry_positive(value)
    if direct is not None:
        return direct
    text = str(value).strip().lower()
    if ";" in text:
        values = [
            parsed
            for part in text.split(";")
            if (parsed := _length_metres(part)) is not None
        ]
        return max(values) if values else None
    values = _numbers(text)
    if not values:
        return None
    result = max(values) if "-" in text or " to " in text else values[0]
    return result * 0.3048 if "ft" in text or "feet" in text or "'" in text else result


def building_height(
    properties: Mapping[str, Any] | None,
    *,
    default_height_m: float = 6.0,
    level_height_m: float = 3.0,
    minimum_height_m: float = 2.0,
) -> float:

    default = float(default_height_m)
    level = float(level_height_m)
    minimum = float(minimum_height_m)
    properties = properties or {}
    for key in _HEIGHT_KEYS:
        height = _length_metres(properties.get(key)) if key in properties else None
        if height is not None:
            return max(minimum, height)
    for key in _LEVEL_KEYS:
        levels = max(_numbers(properties.get(key)) or [0]) if key in properties else 0
        if levels > 0:
            return max(minimum, levels * level)
    return max(minimum, default)


def load_buildings(
    geojson_path: str | Path,
    target_crs: str,
    *,
    default_height_m: float = 6.0,
    level_height_m: float = 3.0,
    minimum_height_m: float = 2.0,
) -> list[tuple[Any, float]]:

    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    buildings: list[tuple[Any, float]] = []
    for feature in data.get("features") or []:
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        projected = shapely_shape(transform_geom("EPSG:4326", target_crs, geometry))
        if projected.is_empty:
            continue
        height = building_height(
            feature.get("properties") or {},
            default_height_m=default_height_m,
            level_height_m=level_height_m,
            minimum_height_m=minimum_height_m,
        )
        buildings.append((projected, height))
    return buildings


def _polygon_parts(geometry: Any) -> Iterator[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def wall_segments(
    geometry: Any,
    height: float,
) -> tuple[list[tuple[LineString, float]], list[tuple[LineString, float]]]:

    heights: list[tuple[LineString, float]] = []
    aspects: list[tuple[LineString, float]] = []
    for polygon in _polygon_parts(geometry):
        polygon = orient(polygon, sign=1.0)
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = list(ring.coords)
            for first, second in zip(coordinates[:-1], coordinates[1:]):
                dx = float(second[0] - first[0])
                dy = float(second[1] - first[1])
                if dx == 0 and dy == 0:
                    continue
                line = LineString((first, second))
                aspect = (math.degrees(math.atan2(dy, -dx)) + 360.0) % 360.0
                heights.append((line, float(height)))
                aspects.append((line, float(aspect)))
    return heights, aspects


def calculate_geometry_walls(
    buildings: Iterable[tuple[Any, float]],
    ghsl_height: np.ndarray,
    out_shape: tuple[int, int],
    transform: Any,
    minimum_wall_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:

    shape = tuple(out_shape)
    ghsl = np.ma.filled(np.ma.asarray(ghsl_height), 0)
    minimum = float(minimum_wall_height_m)

    tile_bounds = shapely_box(*array_bounds(shape[0], shape[1], transform))
    height_shapes: list[tuple[LineString, float]] = []
    aspect_shapes: list[tuple[LineString, float]] = []
    for geometry, fallback_height in buildings:
        if geometry.is_empty or not geometry.intersects(tile_bounds):
            continue
        fallback = float(fallback_height)
        segment_heights, segment_aspects = wall_segments(
            geometry,
            max(minimum, fallback),
        )
        height_shapes.extend(segment_heights)
        aspect_shapes.extend(segment_aspects)

    if height_shapes:
        wall_mask = rasterize(
            [(geometry, 1) for geometry, _ in height_shapes],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        fallback = rasterize(
            height_shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="float32",
            all_touched=True,
        )
        aspect = rasterize(
            aspect_shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="float32",
            all_touched=True,
        )
    else:
        wall_mask = np.zeros(shape, dtype=bool)
        fallback = np.zeros(shape, dtype=np.float32)
        aspect = np.zeros(shape, dtype=np.float32)

    ghsl = np.nan_to_num(ghsl, nan=0.0, posinf=0.0, neginf=0.0)
    ghsl_support = wall_mask & (ghsl >= minimum)
    walls = np.where(wall_mask, np.where(ghsl_support, ghsl, fallback), 0)
    walls = walls.astype(np.float32)
    aspect = np.where(wall_mask, aspect, 0).astype(np.float32)
    np.remainder(aspect, np.float32(360.0), out=aspect)
    return walls, aspect


__all__ = [
    "building_height", "calculate_dsm_walls", "calculate_geometry_walls",
    "load_buildings", "wall_segments",
]
