from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from pathlib import Path

import numpy as np

from .runtime import (configure_runtime, cupy, DeviceBatch, OutputTransfer,
                      compute_stream_scope)

configure_runtime()
from osgeo import gdal
from rasterio.transform import Affine

from .umep_solweig import Solweig_2015a_metdata_noload, Tgmaps_v1
from .model import (Solweig_2022a_calc, prepare_model,
                    _MODEL_SCALARS, _MODEL_INTEGERS)
from .svf import svf_calculator, compute_diffsh
from .utci import utci_calculator
from .walls import calculate_dsm_walls, calculate_geometry_walls, load_buildings
from .output import (
    BoundedOutput,
    load_raster, _read_raster,
    _write_static_job, _write_device_static, _write_device_chunk,
)


landcover_classes_path = str(Path(__file__).resolve().parent / "landcoverclasses_2016a.txt")
LANDCOVER_GRASS_CODE = 5
LANDCOVER_BARE_SOIL_CODE = 6


# Wall and ground emissivity and albedo
albedo_b = 0.2
albedo_g = 0.15
ewall = 0.9
eground = 0.95
absK = 0.7
absL = 0.95

# Standing position
Fside = 0.22
Fup = 0.06
Fcyl = 0.28

cyl = True
elvis = 0
usevegdem = 1
onlyglobal = 1

firstdayleaf = 97
lastdayleaf = 300
conifer_bool = False


@dataclass(frozen=True)
class DayForcing:

    date: str
    values: np.ndarray
    timestamps: tuple[datetime, ...]


@dataclass(frozen=True)
class Meteorology:

    values: np.ndarray
    timestamps: tuple[datetime, ...]
    days: tuple[DayForcing, ...]
    timestep_minutes: float


@dataclass(frozen=True)
class _RasterTemplate:
    transform: tuple
    projection: str


@dataclass
class _SpatialContext:
    inputs: dict
    prepared: object
    template: object
    valid_mask: object
    static_files: dict
    utc: float


def run_timeseries(building_dsm_path, tree_path, dem_path, met_file, output_path,
        longitude, latitude, *, landcover_path=None, utc_offset_hours=None,
        site_altitude_m=3., wall_method='dsm', building_geometry_path=None,
        building_height_path=None, minimum_wall_height_m=2., dsm_wall_threshold_m=3.,
        default_building_height_m=6., level_height_m=3., minimum_building_height_m=2.,
        save_tmrt=True, save_svf=False, save_kup=False, save_kdown=False,
        save_lup=False, save_ldown=False, save_shadow=False, dtype=np.float32,
        no_write=False, output_core=None, output_chunk_steps=24,
        async_output=True, output_queue_size=2, output_valid_mask=None, progress=False):
    forcing = load_meteorology(met_file)
    dtype = np.dtype(dtype)
    output = Path(output_path).resolve()
    cp = cupy()
    if not no_write:
        (output / 'static').mkdir(parents=True, exist_ok=True)
    if gdal.GetConfigOption('GDAL_NUM_THREADS') is None:
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    context = None
    completed = False
    transfer = OutputTransfer()
    worker = BoundedOutput(enabled=bool(async_output and not no_write), max_inflight=output_queue_size)
    try:
        with compute_stream_scope():
            if progress:
                print(f"[{datetime.now():%H:%M:%S}] Spatial preparation started", flush=True)
            context = _prepare_spatial(building_dsm_path, tree_path, dem_path, output / 'static',
                longitude, latitude, dtype=dtype, no_write=no_write, save_svf=save_svf,
                output_worker=worker, output_transfer=transfer, wall_method=wall_method,
                building_geometry_path=building_geometry_path, building_height_path=building_height_path,
                minimum_wall_height_m=minimum_wall_height_m, dsm_wall_threshold_m=dsm_wall_threshold_m,
                default_building_height_m=default_building_height_m, level_height_m=level_height_m,
                minimum_building_height_m=minimum_building_height_m, landcover_path=landcover_path,
                utc_offset_hours=utc_offset_hours, site_altitude_m=site_altitude_m, progress=progress)
            crop, template = _output_window(output_core, context)
            if output_valid_mask is not None:
                output_valid_mask = np.asarray(output_valid_mask).copy()
                output_valid_mask.setflags(write=False)
            variables = ['UTCI'] + [name for name, enabled in (
                ('TMRT', save_tmrt), ('Kup', save_kup), ('Kdown', save_kdown), ('Lup', save_lup),
                ('Ldown', save_ldown), ('Shadow', save_shadow)) if enabled]
            files = {'static': context.static_files, 'days': {}}
            if progress:
                print(f"[{datetime.now():%H:%M:%S}] Time-series simulation started", flush=True)
            for day in forcing.days:
                worker.check()
                folder = output / day.date
                if not no_write:
                    folder.mkdir()
                files['days'][day.date] = _run_day(context, day, forcing.timestep_minutes, folder,
                    dtype=dtype, variables=variables, crop=crop, template=template,
                    chunk_steps=output_chunk_steps, no_write=no_write,
                    output_worker=worker, output_transfer=transfer,
                    output_valid_mask=output_valid_mask)
                if progress:
                    print(f"[{datetime.now():%H:%M:%S}] Daily calculation completed: {day.date}", flush=True)
            worker.finish()
            completed = True
            return files
    except BaseException:
        completed = False
        worker.abort()
        raise
    finally:
        context = None
        if completed:
            cp.get_default_memory_pool().free_all_blocks()


def _read_values(source):
    if isinstance(source, np.ndarray):
        values = np.array(source, dtype=np.float64, order="C", copy=True)
        return values.reshape(1, -1) if values.ndim == 1 else values
    if isinstance(source, (str, os.PathLike)):
        paths = [Path(source)]
    else:
        paths = [Path(path) for path in source]
    resolved = [path.expanduser().resolve() for path in paths]
    arrays = []
    for path in resolved:
        values = np.loadtxt(path, skiprows=1, ndmin=2, dtype=np.float64, encoding="utf-8-sig")
        arrays.append(values)
    return np.ascontiguousarray(arrays[0] if len(arrays) == 1 else np.concatenate(arrays), dtype=np.float64)


def load_meteorology(source) -> Meteorology:
    if isinstance(source, Meteorology):
        return source
    values = _read_values(source)
    fields = values[:, :4]
    stamps = []
    for year, doy, hour, minute in fields.astype(np.int64):
        year, doy, hour, minute = map(int, (year, doy, hour, minute))
        stamps.append(datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute))
    timestamps = tuple(stamps)
    timestep = (stamps[1] - stamps[0]).total_seconds() / 60 if len(stamps) > 1 else 0.0
    values.setflags(write=False)
    groups = []
    start = 0
    for stop in range(1, len(stamps) + 1):
        if stop == len(stamps) or stamps[stop].date() != stamps[start].date():
            day_values = values[start:stop]
            groups.append(DayForcing(stamps[start].date().isoformat(), day_values, timestamps[start:stop]))
            start = stop
    return Meteorology(values, timestamps, tuple(groups), timestep)


def _prepare_spatial(building_dsm_path, tree_path, dem_path, output_path,
        longitude, latitude, *, dtype, no_write, save_svf, output_worker, output_transfer,
        wall_method='dsm', building_geometry_path=None, building_height_path=None,
        minimum_wall_height_m=2., dsm_wall_threshold_m=3.,
        default_building_height_m=6., level_height_m=3., minimum_building_height_m=2.,
        landcover_path=None, utc_offset_hours=None, site_altitude_m=3., progress=False):
    cp = cupy()
    a, dataset = load_raster(building_dsm_path)
    temp1, _ = load_raster(tree_path)
    temp2, _ = load_raster(dem_path)

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    output_template = _RasterTemplate(tuple(geotransform), projection)

    method = str(wall_method).strip().lower()
    if method == "dsm":
        generated_walls, generated_aspect = calculate_dsm_walls(
            a.get(),
            pixel_size_m=abs(geotransform[1]),
            minimum_wall_height_m=float(dsm_wall_threshold_m),
        )
    else:
        height_values = _read_raster(building_height_path)
        buildings = load_buildings(
            building_geometry_path,
            projection,
            default_height_m=default_building_height_m,
            level_height_m=level_height_m,
            minimum_height_m=minimum_building_height_m,
        )
        generated_walls, generated_aspect = calculate_geometry_walls(
            buildings,
            height_values,
            (dataset.RasterYSize, dataset.RasterXSize),
            Affine.from_gdal(*geotransform),
            minimum_wall_height_m,
        )
    generated_walls = generated_walls.astype(np.float32).astype(np.float64)
    generated_aspect = generated_aspect.astype(np.float32).astype(np.float64)

    files = {}
    if not no_write:
        files['walls'] = str(Path(output_path) / 'walls.tif')
        files['aspect'] = str(Path(output_path) / 'aspect.tif')
        output_worker.reserve()
        output_worker.submit(_write_static_job, [
            (files['walls'], generated_walls, 'wall_height', 'm',
             {'wall_method': method, 'wall_backend': 'CUDA' if method == 'dsm' else 'geometry'}),
            (files['aspect'], generated_aspect, 'wall_aspect', 'degree_clockwise_from_north',
             {'wall_method': method})], output_template)

    walls = cp.array(generated_walls, dtype=cp.float64, copy=True)
    dirwalls = cp.array(generated_aspect, dtype=cp.float64, copy=True)

    dsm_raise = max(0.0, -float(a.min().item()))
    dem_raise = max(0.0, -float(temp2.min().item()))
    if dsm_raise:
        a = a + dsm_raise
    if dem_raise:
        temp2 = temp2 + dem_raise

    landcover = 0
    lcgrid = False
    lcgrid_np = None
    lc_class = None
    
    if landcover_path is not None:
        landcover = 1
        raw_lcgrid, _ = load_raster(landcover_path)
        lcgrid_np = _normalise_landcover_grid(raw_lcgrid.get())
        lcgrid = cp.asarray(lcgrid_np).astype(cp.float32, copy=False)
        lc_class = _read_landcover_classes()
        del raw_lcgrid
    
    rows, cols = a.shape
    scale = 1 / geotransform[1]
    lon = float(longitude)
    lat = float(latitude)
    location = {
        "longitude": lon,
        "latitude": lat,
        "altitude": float(site_altitude_m),
    }
    utc = lon / 15.0 if utc_offset_hours is None else float(utc_offset_hours)
    temp1[temp1 < 0.] = 0.
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    bush = cp.logical_not(vegdsm2 * vegdsm) * vegdsm
    amaxvalue = cp.maximum(a.max() - a.min(), temp1.max())
    buildings = a - temp2
    buildings[buildings < 2.] = 1.
    buildings[buildings >= 2.] = 0.
    valid_mask = (buildings == 1)
    Knight = cp.zeros((rows, cols), dtype=cp.float32)
    
    if landcover == 1:                                     
        (TgK_np, Tstart_np, alb_np, emis_np, TgK_wall_np, Tstart_wall_np, TmaxLST_np,
         TmaxLST_wall_np) = Tgmaps_v1(lcgrid_np, lc_class)
           
        TgK = cp.asarray(TgK_np).astype(cp.float32, copy=False)
        Tstart = cp.asarray(Tstart_np).astype(cp.float32, copy=False)
        alb_grid = cp.asarray(alb_np).astype(cp.float32, copy=False)
        emis_grid = cp.asarray(emis_np).astype(cp.float32, copy=False)
        TgK_wall = cp.array(float(np.asarray(TgK_wall_np).item()), dtype=cp.float32, copy=True)
        Tstart_wall = cp.array(float(np.asarray(Tstart_wall_np).item()), dtype=cp.float32, copy=True)
        TmaxLST = cp.asarray(TmaxLST_np).astype(cp.float32, copy=False)
        TmaxLST_wall = cp.array(float(np.asarray(TmaxLST_wall_np).item()), dtype=cp.float32, copy=True)
    else:
        TgK = Knight + 0.37
        Tstart = Knight - 3.41
        alb_grid = Knight + albedo_g
        emis_grid = Knight + eground
        TgK_wall = 0.37
        Tstart_wall = -3.41
        TmaxLST = Knight + 15.
        TmaxLST_wall = 15.
    lcgrid_model = lcgrid if landcover else Knight.copy()
    
    transVeg = 3. / 100.
    anisotropic_sky = 1
    patch_option = 2
    height = cp.array(1.1, dtype=cp.float32, copy=True)
    first = cp.round(height.copy())
    if first == 0.:
        first = cp.array(1., dtype=cp.float32, copy=True)
    second = cp.round(height * 20.)
    if progress:
        print(f"[{datetime.now():%H:%M:%S}] Sky-view factor calculation started", flush=True)
    svf_values = svf_calculator(
        patch_option,
        amaxvalue,
        a,
        vegdsm,
        vegdsm2,
        bush,
        scale,
    )
    svf_values = tuple(v.astype(cp.float32).astype(cp.float64) for v in svf_values[:15]) + svf_values[15:]
    (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg,
        svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg,
        vegshmat, vbshvegshmat, shmat, svftotal,
    ) = svf_values
    svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
    svftotal = svfbuveg
    asvf = cp.acos(cp.sqrt(svf))
    diffsh = compute_diffsh(shmat, vegshmat, transVeg)
    tmp = svf + svfveg - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = cp.asin(cp.exp(cp.log(1.0 - tmp) / 2.0))
    static = dict(
        dsm=a, scale=scale, rows=rows, cols=cols,
        svf=svf, svfN=svfN, svfW=svfW, svfE=svfE, svfS=svfS,
        svfveg=svfveg, svfNveg=svfNveg, svfEveg=svfEveg,
        svfSveg=svfSveg, svfWveg=svfWveg, svfaveg=svfaveg,
        svfEaveg=svfEaveg, svfSaveg=svfSaveg, svfWaveg=svfWaveg, svfNaveg=svfNaveg,
        vegdem=vegdsm, vegdem2=vegdsm2, albedo_b=albedo_b, absK=absK, absL=absL,
        ewall=ewall, Fside=Fside, Fup=Fup, Fcyl=Fcyl, usevegdem=usevegdem,
        onlyglobal=onlyglobal, buildings=buildings, location=location,
        landcover=landcover, lc_grid=lcgrid_model, dirwalls=dirwalls, walls=walls,
        cyl=cyl, elvis=elvis, amaxvalue=amaxvalue, bush=bush, TgK=TgK, Tstart=Tstart,
        alb_grid=alb_grid, emis_grid=emis_grid, TgK_wall=TgK_wall, Tstart_wall=Tstart_wall,
        TmaxLST=TmaxLST, TmaxLST_wall=TmaxLST_wall, first=first, second=second,
        svfalfa=svfalfa, svfbuveg=svfbuveg, diffsh=diffsh, shmat=shmat,
        vegshmat=vegshmat, vbshvegshmat=vbshvegshmat, anisotropic_sky=anisotropic_sky,
        asvf=asvf, patch_option=patch_option)
    for name in set(_MODEL_SCALARS) & static.keys():
        value = static[name]
        static[name] = int(value) if name in _MODEL_INTEGERS else float(value)
    prepared = prepare_model(**static, dtype=dtype)
    if save_svf and not no_write:
        output_worker.reserve()
        batch = DeviceBatch(1)
        try:
            batch.append({'SVF': svftotal})
            batch.seal()
            files['SVF'] = str(Path(output_path) / 'SVF.tif')
            output_worker.submit(_write_device_static, batch, output_transfer,
                                 files['SVF'], output_template)
        except BaseException:
            batch.release()
            raise
    return _SpatialContext(static, prepared, output_template, valid_mask, files, utc)


def _output_window(core, context):
    if core is None:
        return (slice(None), slice(None)), context.template
    row, col, nr, nc = core
    g = context.template.transform
    transform = (g[0]+col*g[1]+row*g[2], g[1], g[2],
                 g[3]+col*g[4]+row*g[5], g[4], g[5])
    template = _RasterTemplate(transform, context.template.projection)
    return (slice(row, row+nr), slice(col, col+nc)), template


def _run_day(context, day, interval_minutes, output, *, dtype, variables,
             crop, template, chunk_steps, no_write, output_worker, output_transfer,
             output_valid_mask=None):
    cp = cupy()
    met = day.values
    _, altitude, azimuth, zen, jday, _, dectime, altmax = Solweig_2015a_metdata_noload(
        met, context.inputs['location'], context.utc)

    # %Creating vectors from meteorological input
    DOY = met[:, 1]
    hours = met[:, 2]
    minu = met[:, 3]
    Ta = met[:, 11]
    RH = met[:, 10]
    radG = met[:, 14]
    radD = met[:, 21]
    radI = met[:, 22]
    P = met[:, 12]
    Ws = met[:, 9]

    if conifer_bool:
        leaf = np.ones(len(met), dtype=bool)
    elif firstdayleaf > lastdayleaf:
        leaf = (DOY > firstdayleaf) | (DOY < lastdayleaf)
    else:
        leaf = (DOY > firstdayleaf) & (DOY < lastdayleaf)
    psi = np.where(leaf, 3./100., .5)

    # %Initialization of maps
    Tgmap1 = cp.zeros(context.prepared.shape, dtype=cp.float32)
    Tgmap1E = cp.zeros(context.prepared.shape, dtype=cp.float32)
    Tgmap1S = cp.zeros(context.prepared.shape, dtype=cp.float32)
    Tgmap1W = cp.zeros(context.prepared.shape, dtype=cp.float32)
    Tgmap1N = cp.zeros(context.prepared.shape, dtype=cp.float32)
    TgOut1 = cp.zeros(context.prepared.shape, dtype=cp.float32)

    # Initialisation of time related variables
    timestepdec = (dectime[1] - dectime[0]) if len(met) > 1 else interval_minutes / 1440.
    timeadd = 0.
    firstdaytime = 1.
    CI = 1.

    # Daily water body temperature
    Twater = float(np.mean(Ta)) if context.inputs['landcover'] else None
    output_variables = () if no_write else tuple(variables)
    files = {name: [] for name in output_variables}
    batch = None
    chunk_start, chunk_number = 0, 0
    try:
        for i in range(len(Ta)):
            output_worker.check()
            # Nocturnal cloudfraction from Offerle et al. 2003
            if dectime[i] - np.floor(dectime[i]) == 0:
                CI = 1.
            (Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow,
             firstdaytime, timestepdec, timeadd, Tgmap1, Tgmap1E, Tgmap1S,
             Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth,
             Lwest, Lnorth, KsideI, TgOut1, TgOut, radIout, radDout, Lside,
             Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside) = Solweig_2022a_calc(
                **context.inputs, i=i,
                altitude=float(altitude[0, i]), azimuth=float(azimuth[0, i]),
                zen=float(zen[0, i]), jday=float(jday[0, i]), psi=float(psi[i]),
                dectime=float(dectime[i]), altmax=float(altmax[0, i]),
                Ta=float(Ta[i]), RH=float(RH[i]), radG=float(radG[i]),
                radD=float(radD[i]), radI=float(radI[i]), P=float(P[i]), Twater=Twater,
                firstdaytime=firstdaytime, timeadd=timeadd, timestepdec=timestepdec,
                Tgmap1=Tgmap1, Tgmap1E=Tgmap1E, Tgmap1S=Tgmap1S,
                Tgmap1W=Tgmap1W, Tgmap1N=Tgmap1N, CI=CI, TgOut1=TgOut1,
                dtype=dtype, prepared=context.prepared)

            utci_raw = utci_calculator(np.float32(Ta[i]), np.float32(RH[i]),
                                       Tmrt, np.float32(Ws[i]))
            utci = cp.full(utci_raw.shape, float('nan'), dtype=cp.float32)
            utci[context.valid_mask] = utci_raw[context.valid_mask]
            if output_variables and batch is None:
                output_worker.reserve()
                batch = DeviceBatch(min(chunk_steps, len(met)-i))
            if batch is not None:
                available = {'UTCI': utci, 'TMRT': Tmrt, 'Kdown': Kdown, 'Kup': Kup,
                             'Ldown': Ldown, 'Lup': Lup, 'Shadow': shadow}
                batch.append({name: available[name] for name in output_variables}, crop)
                if batch.count == batch.capacity:
                    batch.seal()
                    chunk_number += 1
                    suffix = f'_{chunk_number:04d}' if len(met) > chunk_steps else ''
                    paths = {}
                    for variable in output_variables:
                        path = str(output / f'{variable}{suffix}.tif')
                        files[variable].append(path)
                        paths[variable] = path
                    output_worker.submit(_write_device_chunk, batch, output_transfer, paths,
                        template, datetime.fromisoformat(day.date),
                        hours[chunk_start:i+1].copy(), minu[chunk_start:i+1].copy(), context.utc,
                        output_valid_mask)
                    batch = None
                    chunk_start = i+1
    finally:
        if batch is not None:
            batch.release()
    return files


def _read_landcover_classes():
    with open(landcover_classes_path, encoding="utf-8") as stream:
        rows = [line.split() for line in stream.readlines()[1:] if line.strip()]
    classes = np.asarray([[float(value) for value in row[1:7]] for row in rows], dtype=float)
    return classes


def _normalise_landcover_grid(values):
    source = np.asarray(values, dtype=float)
    rounded = np.rint(source)
    invalid = (~np.isfinite(source)) | (~np.isclose(source, rounded)) | (rounded < 1) | (rounded > 7)
    rounded[invalid] = LANDCOVER_BARE_SOIL_CODE
    vegetation = (rounded == 3) | (rounded == 4)
    rounded[vegetation] = LANDCOVER_GRASS_CODE
    result = rounded.astype(np.float32, copy=False)
    return np.ascontiguousarray(result)
