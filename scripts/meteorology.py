from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


DATASET = "reanalysis-era5-single-levels-timeseries"
ALIASES = {
    "t2m": ("t2m", "2m_temperature", "2t"),
    "d2m": ("d2m", "2m_dewpoint_temperature", "2d"),
    "sp": ("sp", "surface_pressure"),
    "u10": ("u10", "10m_u_component_of_wind", "10u"),
    "v10": ("v10", "10m_v_component_of_wind", "10v"),
    "ssrd": ("ssrd", "surface_solar_radiation_downwards"),
}
HEADER = "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd"


def _grid_cell(latitude, longitude):
    return float(np.round(latitude * 4) / 4), float((np.round(longitude * 4) / 4 + 180) % 360 - 180)


def download_request(latitude, longitude, start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    lat, lon = _grid_cell(latitude, longitude)
    return {
        "variable": [aliases[1] for aliases in ALIASES.values()],
        "location": {"latitude": lat, "longitude": lon},
        "date": [f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"],
        "data_format": "netcdf",
    }


def download_era5(folder, request):
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"era5_{key}.nc"
    if target.is_file():
        return target
    with tempfile.TemporaryDirectory(prefix=".era5-", dir=folder) as directory:
        partial = Path(directory) / target.name
        cdsapi.Client().retrieve(DATASET, request, str(partial))
        partial.replace(target)
    return target


def _files(inputs):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        found = sorted(p for p in path.iterdir() if p.suffix.lower() in (".nc", ".nc4", ".zip")) if path.is_dir() else [path]
        for item in found:
            if item not in paths:
                paths.append(item)
    return paths


@contextmanager
def _netcdf_files(path):
    if not zipfile.is_zipfile(path):
        yield [path]
        return
    with tempfile.TemporaryDirectory(prefix="solweig_era5_") as directory, zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in (".nc", ".nc4")]
        files = []
        for index, member in enumerate(members):
            target = Path(directory) / f"{index}.nc"
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            files.append(target)
        yield files


def _point(dataset, expected):
    lat_name = next(k for k in ("latitude", "lat") if k in dataset.coords)
    lon_name = next(k for k in ("longitude", "lon") if k in dataset.coords)
    lat, lon = dataset[lat_name], dataset[lon_name]
    if lat.ndim == lon.ndim == 0:
        return dataset
    if lat.dims == lon.dims:
        return dataset.squeeze(dim=lat.dims)
    return dataset.isel({
        lat.dims[0]: int(np.argmin(np.abs(lat.values - expected[0]))),
        lon.dims[0]: int(np.argmin(np.abs((lon.values - expected[1] + 180) % 360 - 180))),
    })


def _series(variable):
    if "expver" in variable.dims:
        versions = [int(str(v)) for v in variable.expver.values]
        order = sorted(range(len(versions)), key=lambda i: versions[i] != 1)
        selected = variable.isel(expver=order[0], drop=True)
        for index in order[1:]:
            selected = selected.combine_first(variable.isel(expver=index, drop=True))
        variable = selected
    key = next(k for k in ("valid_time", "time") if k in variable.coords and variable[k].ndim == 1)
    coordinate = variable[key]
    dim = coordinate.dims[0]
    variable = variable.squeeze(dim=[other for other in variable.dims if other != dim], drop=True)
    times = pd.DatetimeIndex(coordinate.values)
    values = np.asarray(variable.values, dtype=np.float64).reshape(-1)
    return pd.Series(values, index=times).sort_index()


def prepare_met(files, output, *, latitude, longitude, start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    times = pd.date_range(start, end + pd.Timedelta(days=1), freq="h", inclusive="left")
    cell = _grid_cell(latitude, longitude)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output name; {output} already exists")
    fragments = {name: [] for name in ALIASES}
    for path in _files(files):
        with _netcdf_files(path) as parts:
            for netcdf in parts:
                with xr.open_dataset(netcdf, engine="netcdf4") as dataset:
                    selected = _point(dataset, cell)
                    for name, aliases in ALIASES.items():
                        actual = next((k for k in aliases if k in selected.data_vars), None)
                        if actual is None:
                            continue
                        variable = selected[actual]
                        fragments[name].append(_series(variable))
    fields = {}
    for name, pieces in fragments.items():
        joined = pd.concat(pieces).sort_index()
        if joined.index.has_duplicates:
            joined = joined.groupby(level=0).first()
        values = joined.reindex(times).to_numpy(dtype=np.float64)
        fields[name] = values
    ta, td = fields["t2m"] - 273.15, fields["d2m"] - 273.15
    values = np.full((len(times), 24), -999.0, dtype=np.float64)
    values[:, :4] = np.column_stack((times.year, times.dayofyear, times.hour, times.minute))
    values[:, 9] = np.hypot(fields["u10"], fields["v10"])
    values[:, 10] = np.clip(100 * np.exp(17.67 * td / (td + 243.5) - 17.67 * ta / (ta + 243.5)), 0, 100)
    values[:, 11] = ta
    values[:, 12] = fields["sp"] / 1000.0
    values[:, 14] = np.maximum(fields["ssrd"], 0) / 3600.0
    values[:, 23] = np.degrees(np.arctan2(-fields["u10"], -fields["v10"])) % 360
    values[values[:, 9] == 0, 23] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".met-", dir=output.parent) as directory:
        partial = Path(directory) / output.name
        with partial.open("w", encoding="utf-8", newline="\n") as destination:
            np.savetxt(destination, values, fmt=["%d"] * 4 + ["%.6f"] * 20, header=HEADER, comments="")
        partial.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description='Prepare SOLWEIG meteorology from ERA5.')
    parser.add_argument("files", nargs="*", type=Path, help="ERA5 NetCDF/ZIP files or a folder (split variables/months supported)")
    parser.add_argument("--download", action="store_true", help="download a CDS point time series instead of reading supplied files")
    parser.add_argument("--latitude", required=True, type=float)
    parser.add_argument("--longitude", required=True, type=float)
    parser.add_argument("--start", required=True, help="first UTC date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last UTC date, inclusive")
    parser.add_argument("--output", type=Path, default=Path("met.txt"))
    args = parser.parse_args()
    files = args.files
    if args.download:
        request = download_request(args.latitude, args.longitude, args.start, args.end)
        files = [download_era5(args.output.resolve().parent / "era5", request)]
    output = prepare_met(files, args.output, latitude=args.latitude, longitude=args.longitude,
                         start_date=args.start, end_date=args.end)
    print(f"Created {output}")
    print("Use utc_offset_hours=0 in run_timeseries.")


if __name__ == "__main__":
    main()
