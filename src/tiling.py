import argparse
from pathlib import Path

import rasterio
from osgeo import gdal


INPUTS = {
    "building_dsm_path": "Building_DSM.tif",
    "dem_path": "DEM.tif",
    "tree_path": "Trees.tif",
    "landcover_path": "Landcover.tif",
}


def create_tiles(input_dir, output_dir, *, core=1800, halo=100, format="vrt"):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    sources = {}
    for key, filename in INPUTS.items():
        path = input_dir / filename
        if key == "landcover_path" and not path.exists():
            continue
        sources[key] = path

    with rasterio.open(sources["building_dsm_path"]) as reference:
        width, height = reference.width, reference.height

    tiles = []
    for tile_row, row in enumerate(range(0, height, core)):
        for tile_col, col in enumerate(range(0, width, core)):
            core_width, core_height = min(core, width - col), min(core, height - row)
            left, top = max(0, col - halo), max(0, row - halo)
            right, bottom = min(width, col + core_width + halo), min(height, row + core_height + halo)
            tile_id = f"r{tile_row:03d}_c{tile_col:03d}"
            tiles.append({
                "tile_id": tile_id,
                "core_window": [col, row, core_width, core_height],
                "input_window": [left, top, right - left, bottom - top],
                "output_core": [row - top, col - left, core_height, core_width],
                "inputs": {key: str(output_dir / tile_id / f"{path.stem}.{format}")
                           for key, path in sources.items()},
            })
    gdal.UseExceptions()
    output_dir.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        (output_dir / tile["tile_id"]).mkdir(exist_ok=True)
        for key, path in tile["inputs"].items():
            options = {"format": "VRT" if format == "vrt" else "GTiff",
                       "srcWin": tile["input_window"]}
            if format == "tif":
                options["creationOptions"] = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
            result = gdal.Translate("" if format == "vrt" else str(path), str(sources[key]), **options)
            if format == "vrt":
                content = result.GetMetadata("xml:VRT")[0]
                target = Path(path)
                if not target.is_file() or target.read_text(encoding="utf-8") != content:
                    target.write_text(content, encoding="utf-8")
            else:
                result.FlushCache()
            result = None
    return tiles


def main(argv=None):
    parser = argparse.ArgumentParser(description='Split aligned rasters into tiles with halos.')
    parser.add_argument("input_dir", help="Folder containing prepared morphology rasters")
    parser.add_argument("--output", required=True, help="Directory for raster tiles")
    parser.add_argument("--core", type=int, default=1800, help="Core side length in pixels (default: 1800)")
    parser.add_argument("--halo", type=int, default=100, help="Halo in pixels (default: 100)")
    parser.add_argument("--format", choices=("vrt", "tif"), default="vrt")
    args = parser.parse_args(argv)
    tiles = create_tiles(args.input_dir, args.output, core=args.core,
                         halo=args.halo, format=args.format)
    print(f"Created {len(tiles)} tiles in {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
