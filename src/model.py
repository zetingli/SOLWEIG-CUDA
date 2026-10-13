from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from functools import lru_cache
import math

import numpy as np

from .runtime import cupy, get_kernel, elementwise_kernel
from .umep_solweig import daylen, clearnessindex_2013b, diffusefraction, create_patches


SBC = 5.67051e-8


def Solweig_2022a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
                       amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1,
                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat,
                       anisotropic_sky, asvf, patch_option, *, dtype=np.float32, prepared=None):
    # This is the CUDA implementation of the SOLWEIG 2022a core function
    # Following the names and structure of the original SOLWEIG 2022a model
    # This function is later called from the main CUDA execution routine in the series.py file.
    # Entry point is now run.py (including tiling) -> series.py -> model.py -> this function
    # Note: This function is optimized for GPU execution using CuPy and direct CUDA kernels.
    #
    # Input variables:
    # dsm = digital surface model
    # scale = height to pixel size (2m pixel gives scale = 0.5)
    # svf,svfN,svfW,svfE,svfS = SVFs for building and ground
    # svfveg,svfNveg,svfEveg,svfSveg,svfWveg = Veg SVFs blocking sky
    # svfaveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg = Veg SVFs blocking buildings
    # vegdem = Vegetation canopy DSM
    # vegdem2 = Vegetation trunk zone DSM
    # albedo_b = buildings
    # absK = human absorption coefficient for shortwave radiation
    # absL = human absorption coefficient for longwave radiation
    # ewall = Emissivity of building walls
    # Fside = The angular factors between a person and the surrounding surfaces
    # Fup = The angular factors between a person and the surrounding surfaces
    # altitude = Sun altitude (degree)
    # azimuth = Sun azimuth (degree)
    # zen = Sun zenith angle (radians)
    # jday = day of year
    # usevegdem = use vegetation scheme
    # onlyglobal = calculate dir and diff from global
    # buildings = Boolena grid to identify building pixels
    # location = geographic location
    # psi = 1 - Transmissivity of shortwave through vegetation
    # landcover = use landcover scheme !!!NEW IN 2015a!!!
    # lc_grid = grid with landcoverclasses
    # dectime = decimal time
    # altmax = maximum sun altitude
    # dirwalls = aspect of walls
    # walls = one pixel row outside building footprints
    # cyl = consider man as cylinder instead of cude

    cp = cupy()
    if prepared is None:
        arguments = locals().copy()
        arguments.pop('prepared')
        arguments.pop('cp')
        prepared = prepare_model(**arguments)
    f = prepared.fields
    (svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg,
     svfSaveg, svfWaveg, svfNaveg, lc_grid, TgK, Tstart, alb_grid, emis_grid, TmaxLST, svfbuveg,
     asvf, svfalfa) = (f[n] for n in FIELD_NAMES)
    Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, TgOut1 = (
        _field(v, prepared.shape, prepared.dtype, scalar=True)
        for v in (Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, TgOut1))
    altitude, azimuth, zen, jday, Ta, RH, radG, radD, radI, P, CI, altmax, dectime = (
        np.float64(float(v)) for v in
        (altitude, azimuth, zen, jday, Ta, RH, radG, radD, radI, P, CI, altmax, dectime))
    opts = dict(dtype=prepared.dtype)
    D = prepared.dtype.type

    # # # Core program start # # #
    # Instrument offset in degrees
    t = 0.

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location['latitude'])

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = 1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5)) + elvis

    if altitude > 0:  # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, _, _ = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky:
            patchchoice = 1
            zenDeg = zen * (180 / np.pi)
            # Relative luminance
            lv, _, _ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)
            dRad = _diffuse(prepared, lv, radD)
        else:
            dRad = D(radD) * svfbuveg
            lv = None

        # Shadow  images
        cast = cast_shadow(prepared.shadow, azimuth, altitude)
        if usevegdem:
            vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = cast
            shadow = (sh - (1 - vegsh) * (1 - psi)).astype(prepared.dtype)
        else:
            sh, wallsh, wallsun, _, facesun = cast
            shadow = sh.astype(prepared.dtype)

        # # # Surface temperature parameterisation during daytime # # # #
        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, radD0 = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - zen / np.pi * 180) + 0.3454
        CI_Tg = radG / radI0 + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1

        radG0 = radI0 * np.sin(altitude * (np.pi / 180)) + radD0
        CI_TgG = radG / radG0 + (1 - corr)
        if (CI_TgG > 1) or (CI_TgG == np.inf):
            CI_TgG = 1

        Tg, Tgwall = surface_temperature(TgK, Tstart, TgK_wall, Tstart_wall, TmaxLST,
            TmaxLST_wall, altmax, dectime, SNUP, CI_TgG, landcover, **opts)

        # # # # Ground View Factors # # # #
        gr = gvf(prepared.gvf, wallsun, shadow, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid,
            SBC, albedo_b, Twater=Twater, lc_grid=lc_grid, landcover=landcover, **opts)
        (gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE,
         gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW,
         gvfLupN, gvfalbN, gvfalbnoshN, _, _) = gr.outputs
        Tg = gr.surface_temperature

        # # # # Lup, daytime # # # #
        # Surface temperature wave delay - new as from 2014a
        Lup, _, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1, **opts)
        LupE, _, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E, **opts)
        LupS, _, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S, **opts)
        LupW, _, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W, **opts)
        LupN, _, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N, **opts)

        # # For Tg output in POIs
        TgTemp = Tg * shadow + D(Ta)
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime,
            timeadd, timestepdec, TgOut1, **opts)  # timeadd only here v2021a

        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, rows, cols, **opts)  # Fraction shadow on building walls based on sun alt and svf
        F_sh = cp.where(cp.isnan(F_sh), 0.5, F_sh)

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = (D(radI) * shadow * D(np.sin(altitude * (np.pi / 180))) + dRad
                 + D(albedo_b) * (1 - svfbuveg) * (D(radG) * (1 - F_sh) + D(radD) * F_sh))

        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude,
            svfbuveg, albedo_b, F_sh, gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN,
            gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN, **opts)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI,
            radD, radG, shadow, svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
            azimuth, altitude, psi, t, albedo_b, F_sh, KupE, KupS, KupW, KupN, cyl, lv,
            anisotropic_sky, None, rows, cols, asvf, None, None, None,
            prepared=prepared.shortwave, **opts)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #
        Tgwall = 0.

        # Nocturnal K fluxes set to 0
        Knight = cp.zeros(prepared.shape, dtype=prepared.dtype)
        Kdown = Kwest = Kup = Keast = Ksouth = Knorth = KsideI = KsideD = F_sh = Tg = shadow = Knight
        CI_Tg = CI_TgG = CI
        dRad = Kside = Knight

        # # # # Lup # # # #
        Lup = D(SBC) * emis_grid * ((Knight + D(Ta) + Tg + D(273.15)) ** 4)
        if landcover:
            Lup = cp.where(lc_grid == 3, D(SBC * 0.98 * (float(Twater) + 273.15) ** 4), Lup)  # nocturnal Water temp

        LupE = LupS = LupW = LupN = Lup

        # # For Tg output in POIs
        TgOut = D(Ta) + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    # Anisotropic sky longwave radiation
    if anisotropic_sky:
        L_patches = lv.copy() if altitude > 0 else prepared.sky.copy()

        if CI < 0.95:
            esky = CI * esky + (1 - CI) * 1.

        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ = longwave(
            prepared.longwave if altitude > 0 else prepared.night_longwave,
            esky, Ta, Tgwall, ewall, Lup, altitude, azimuth, asvf, **opts)
    else:
        def isotropic(emissivity):
            return ((svf + svfveg - 1) * D(emissivity) * D(SBC) * D((Ta + 273.15) ** 4)
                + (2 - svfveg - svfaveg) * D(ewall) * D(SBC) * D((Ta + 273.15) ** 4)
                + (svfaveg - svf) * D(ewall) * D(SBC) * ((D(Ta + 273.15) + Tgwall) ** 4)
                + (2 - svf - svfveg) * D(1 - ewall) * D(emissivity) * D(SBC) * D((Ta + 273.15) ** 4))

        Ldown = isotropic(esky)
        Lside = cp.zeros(prepared.shape, dtype=prepared.dtype)
        L_patches = None

        if CI < 0.95:  # non - clear conditions
            c = 1 - CI
            Ldown = Ldown * D(1 - c) + D(c) * isotropic(1.)

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg,
        svfSveg, svfWveg, svfNveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth,
        altitude, Ta, Tgwall, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW, LupN,
        anisotropic_sky, **opts)

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky:
        Least, Lsouth, Lwest, Lnorth = Least + Least_, Lsouth + Lsouth_, Lwest + Lwest_, Lnorth + Lnorth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    Tmrt, _ = assemble_tmrt(Kdown, Kup, Knorth, Keast, Ksouth, Kwest, Ldown, Lup, Lnorth,
        Least, Lsouth, Lwest, KsideI, Kside, Lside, absK, absL, Fside, Fup, Fcyl, cyl,
        anisotropic_sky, **opts)

    # Add longwave to cardinal directions for output in POI
    if cyl == 1 and anisotropic_sky:
        Least, Lsouth, Lwest, Lnorth = Least + Least_, Lsouth + Lsouth_, Lwest + Lwest_, Lnorth + Lnorth_

    return (Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec,
            timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least,
            Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD,
            Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside)


@dataclass(frozen=True)
class PreparedGeometry:
    sky_patches: np.ndarray
    steradian: np.ndarray
    directional_cosines: np.ndarray
    visibility: object
    shape: tuple[int, int]
    device_id: int

    @property
    def patch_count(self):
        return len(self.sky_patches)


def prepare_geometry(sky_patches, shmat, vegshmat, vbshvegshmat):
    cp = cupy()
    sky = np.array(cp.asnumpy(sky_patches), copy=True)
    arrays = [cp.ascontiguousarray(cp.asarray(a))
              for a in (shmat, vegshmat, vbshvegshmat)]
    shape = arrays[0].shape
    pixels = shape[0] * shape[1]
    packed = cp.empty((len(sky), pixels), dtype=cp.uint8)
    elementwise_kernel(
        "raw T sh, raw U veg, raw V vb, int64 pixels, int32 patches", "uint8 packed",
        "geometry.cu", "solweig_cuda_pack_visibility", options=("-DSOLWEIG_PACK_VISIBILITY",),
    )(*arrays, np.int64(pixels), np.int32(len(sky)), packed)
    alt, azi = sky[:, 0], sky[:, 1]
    bands, counts = np.unique(alt, return_counts=True)
    omega = np.empty(len(sky), dtype=np.float64)
    rad = np.pi / 180
    for k in range(len(sky)):
        count = int(counts[bands == alt[k]][0])
        width = np.float64(360 / count * rad)
        if count > 1:
            omega[k] = width * (
                np.sin((alt[k] + alt[0]) * rad) - np.sin((alt[k] - alt[0]) * rad))
        else:
            omega[k] = width * (
                np.sin(alt[k] * rad) - np.sin((alt[k - 1] + alt[0]) * rad))
    directions = np.column_stack([
        np.where((azi > 360) | (azi < 180), np.cos((90 - azi) * rad), 0),
        np.where((azi > 180) & (azi < 360), np.cos((270 - azi) * rad), 0),
        np.where((azi > 270) | (azi < 90), np.cos((0 - azi) * rad), 0),
        np.where((azi > 90) & (azi < 270), np.cos((180 - azi) * rad), 0),
    ])
    for a in (sky, omega, directions):
        a.setflags(write=False)
    return PreparedGeometry(sky, omega, directions, packed, shape[:2], cp.cuda.Device().id)


@lru_cache(maxsize=256)
def _lw_sunshade_threshold(altitude):
    altitude = float(altitude)
    if not 0.0 < altitude < 89.0:
        return math.nan, math.nan
    with localcontext() as context:
        context.prec = 80
        factor = Decimal.from_float(180.0 / np.pi)
        guard = Decimal(2) ** -40
        center = Decimal.from_float(altitude)
        result = []
        for degrees, direction in ((center - guard, -math.inf),
                                   (center + guard, math.inf)):
            x = degrees / factor
            xx = -(x * x)
            sine = sine_term = x
            cosine = cosine_term = Decimal(1)
            for k in range(1, 40):
                sine_term *= xx / ((2 * k) * (2 * k + 1))
                cosine_term *= xx / ((2 * k - 1) * (2 * k))
                sine += sine_term
                cosine += cosine_term
            result.append(math.nextafter(float(sine / cosine), direction))
        return tuple(result)


def _lw_sunshade_thresholds(altitudes):
    return np.asarray([_lw_sunshade_threshold(float(alt)) for alt in altitudes],
                      dtype=np.float64)


def _lw_coefficients(geometry, esky, Ta, solar_altitude, solar_azimuth):
    sky = geometry.sky_patches
    alt, azi = sky[:, 0], sky[:, 1]
    rad = np.pi / 180
    emissivity = 1 - (1 - esky) * np.exp(0.308 * (1.7 - 1 / np.cos((90 - alt) * rad)))
    flux = emissivity * SBC * ((Ta + 273.15) ** 4) / np.pi
    down = flux * geometry.steradian * np.sin(alt * rad)
    side = flux * geometry.steradian * np.cos(alt * rad)
    down, side = down.astype(sky.dtype), side.astype(sky.dtype)
    difference = np.abs(solar_azimuth - azi)
    yi = np.minimum(2 * np.cos(difference * rad) * np.tan(solar_altitude * rad), 0)
    eligible = (difference > 90) & (difference < 270) & (solar_altitude > 0)
    sky_cardinal = side[:, None] * geometry.directional_cosines
    return np.ascontiguousarray(np.column_stack((
        down, side, geometry.steradian * np.sin(alt * rad),
        geometry.steradian * np.cos(alt * rad), geometry.directional_cosines,
        yi, alt, eligible, sky_cardinal, _lw_sunshade_thresholds(alt))), dtype=np.float64)


def _field(value, shape, dtype, *, scalar=False):
    cp = cupy()
    a = cp.ascontiguousarray(cp.asarray(value), dtype=dtype)
    if scalar and a.size == 1:
        a = cp.broadcast_to(a.reshape(()), shape).copy()
    return a


def longwave(geometry, esky, Ta, Tgwall, ewall, Lup, solar_altitude,
             solar_azimuth, asvf, *, dtype=np.float64):
    cp = cupy()
    dtype = np.dtype(dtype)
    esky, Ta, ewall, solar_altitude, solar_azimuth = (
        float(value) for value in (esky, Ta, ewall, solar_altitude, solar_azimuth))
    fields = (
        _field(Lup, geometry.shape, dtype),
        _field(Tgwall, geometry.shape, dtype, scalar=True),
        _field(asvf, geometry.shape, np.float64))
    coeff = cp.asarray(_lw_coefficients(geometry, esky, Ta, solar_altitude, solar_azimuth))
    patches = geometry.patch_count
    radiation = cp.empty((patches, 12), dtype=dtype)
    conditions = cp.empty(patches, dtype=cp.uint8)
    get_kernel(
        "longwave.cu", "longwave_coefficients", geometry.device_id, precision=dtype.name)(
        ((patches + 127) // 128,), (128,),
        (coeff, np.int32(patches), radiation, conditions))
    pixels = int(np.prod(geometry.shape))
    output = cp.empty((6, *geometry.shape), dtype=dtype)
    kernel = get_kernel(
        "longwave.cu", "longwave", geometry.device_id, precision=dtype.name)
    kernel(((pixels + 127) // 128,), (128,), (
        geometry.visibility, coeff, radiation, conditions, *fields,
        dtype.type(Ta), dtype.type(ewall), np.int64(pixels), np.int32(patches), output))
    return tuple(output[k] for k in range(6))


@dataclass(frozen=True)
class PreparedShortwave:
    geometry: PreparedGeometry
    diffuse_visibility: object


def prepare_shortwave(geometry, diffsh, *, dtype=np.float64):
    cp = cupy()
    dtype = np.dtype(dtype)
    source = _field(diffsh, (*geometry.shape, geometry.patch_count), None)
    pixels = int(np.prod(geometry.shape))
    diffuse = cp.empty((geometry.patch_count,pixels),dtype=dtype)
    elementwise_kernel(
        'raw U source, int64 pixels, int32 patches', 'T target',
        'geometry.cu', 'solweig_cuda_diffuse_transpose_cast', options=('-DSOLWEIG_TRANSPOSE_DIFFUSE',),
    )(source, np.int64(pixels), np.int32(geometry.patch_count), diffuse)
    return PreparedShortwave(geometry, diffuse)


def _sw_patch_coefficients(geometry, lv, radD, altitude, azimuth, t):
    sky = np.asarray(lv)
    alt, azi, luminance = sky[:, 0], sky[:, 1], sky[:, 2]
    rad = np.pi / 180
    omega = geometry.steradian
    rad_total = np.zeros(1, dtype=np.float64)
    for k in range(len(sky)):
        rad_total += luminance[k] * omega[k] * np.sin(alt[k] * rad)
    lum_chi = (luminance * radD) / rad_total
    cos_alt = np.cos(alt * rad)
    direction_cos = np.column_stack([
        np.cos((90 - azi + t) * rad), np.cos((180 - azi + t) * rad),
        np.cos((270 - azi + t) * rad), np.cos((0 - azi + t) * rad)])
    diffuse_flags = np.column_stack([
        (azi > 360) | (azi <= 180), (azi > 90) & (azi <= 270),
        (azi > 180) & (azi <= 360), (azi > 270) | (azi <= 90)])
    reflection_flags = np.column_stack([
        (azi > 360) | (azi < 180), (azi > 90) & (azi < 270),
        (azi > 180) & (azi < 360), (azi > 270) | (azi < 90)])
    difference = np.abs(azimuth - azi)
    yi = np.minimum((2 * np.cos(difference * rad)) * np.tan(altitude * rad), 0)
    box_sun_branch = (difference > 90) & (difference < 270)
    box_incidence = cos_alt[:, None] * direction_cos
    return np.ascontiguousarray(np.column_stack((
        omega, cos_alt, lum_chi, direction_cos, diffuse_flags,
        reflection_flags, yi, alt, box_sun_branch, box_incidence,
        _lw_sunshade_thresholds(alt))), dtype=np.float64)


def Kside_veg_v2022a(
    radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
    svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo, F_sh,
    KupE, KupS, KupW, KupN, cyl, lv, anisotropic_diffuse, diffsh, rows, cols,
    asvf, shmat, vegshmat, vbshvegshmat, *, dtype=np.float64, prepared=None,
):
    dtype = np.dtype(dtype)
    shape = (int(rows), int(cols))
    radI, radD, radG, azimuth, altitude, psi, t, albedo, cyl, anisotropic_diffuse = (
        float(v) for v in
        (radI, radD, radG, azimuth, altitude, psi, t, albedo, cyl, anisotropic_diffuse))
    cyl, anisotropic_diffuse = int(cyl), int(anisotropic_diffuse)
    cp = cupy()
    device = cp.cuda.Device().id
    if anisotropic_diffuse:
        if prepared is None:
            geometry = prepare_geometry(lv, shmat, vegshmat, vbshvegshmat)
            prepared = prepare_shortwave(geometry, diffsh, dtype=dtype)
        geometry = prepared.geometry
        coefficients = cp.asarray(_sw_patch_coefficients(
            geometry, cp.asnumpy(lv), radD, altitude, azimuth, t))
        visibility, diffuse = geometry.visibility, prepared.diffuse_visibility
        patches = geometry.patch_count
        horizon = _field(asvf, shape, np.float64, scalar=True)
    else:
        coefficients = cp.empty(0, dtype=cp.float64)
        visibility = cp.empty(0, dtype=cp.uint8)
        diffuse = cp.empty(0, dtype=dtype)
        horizon = cp.empty(0, dtype=cp.float64)
        patches = 0

    normalized = [
        _field(value, shape, dtype, scalar=True)
        for value in (shadow, F_sh, KupE, KupS, KupW, KupN,
                      svfE, svfS, svfW, svfN, svfEveg, svfSveg, svfWveg, svfNveg)]
    fields = cp.stack([normalized[i] for i in (0, 2, 3, 4, 5)]
                      if anisotropic_diffuse else normalized)
    radiation = cp.empty((patches, 11), dtype=dtype)
    branch_flags = cp.empty(patches, dtype=cp.uint16)
    if patches:
        get_kernel(
            "shortwave.cu", "shortwave_coefficients", device, precision=dtype.name)(
            ((patches + 127) // 128,), (128,), (
            coefficients, np.int32(patches), radiation, branch_flags))
    rad = np.pi / 180
    direct_masks = (
        azimuth > 360 - t or azimuth <= 180 - t,
        azimuth > 90 - t and azimuth <= 270 - t,
        azimuth > 180 - t and azimuth <= 360 - t,
        azimuth <= 90 - t or azimuth > 270 - t)
    direct = [np.sin((azimuth - rotation + t) * rad) if active else 0.0
              for rotation, active in zip((0, 90, 180, 270), direct_masks)]
    parameters = cp.asarray(np.array(
        [radI, radD, radG, np.cos(altitude * rad), psi, albedo, *direct],
        dtype=np.float64))
    pixels = shape[0] * shape[1]
    output = cp.empty((7, *shape), dtype=dtype)
    get_kernel(
        "shortwave.cu", "shortwave", device, precision=dtype.name)(
        ((pixels + 127) // 128,), (128,), (
        visibility, diffuse, fields, horizon, coefficients, parameters, radiation, branch_flags,
        np.int64(pixels), np.int32(patches), np.int32(cyl),
        np.int32(anisotropic_diffuse), output))
    return tuple(output[k] for k in range(7))


_GVF_FAST_BLOCK_SIZE = 128


@dataclass(frozen=True)
class GvfGeometry:
    walls: object
    buildings: object
    aspect: object
    offsets: object
    shade_bounds: object
    shape: tuple[int, int]
    first_steps: int
    second_steps: int
    scale: float
    device_id: int
    compact_masks: bool = False


@dataclass(frozen=True)
class GvfResult:
    outputs: tuple
    surface_temperature: object


def prepare_gvf_geometry(walls, buildings, dirwalls, scale, first, second):
    cp = cupy()
    scale, first, second = (float(v) for v in (scale, first, second))
    first_steps = max(1, int(np.round(first * scale)))
    second_steps = int(np.round(second * scale))
    walls = cp.ascontiguousarray(cp.asarray(walls), dtype=np.float64)
    shape = walls.shape
    buildings = _field(buildings, shape, np.float64)
    direction = _field(dirwalls, shape, np.float64)
    offsets = np.empty((18, second_steps, 2), dtype=np.int32)
    bounds = np.empty((18, 3), dtype=np.float64)
    for j, degrees in enumerate(np.arange(5, 359, 20)):
        azimuth = degrees * (np.pi / 180)
        sin_a, cos_a, tan_a = np.sin(azimuth), np.cos(azimuth), np.tan(azimuth)
        ss, sc = np.sign(sin_a), np.sign(cos_a)
        y_dominant = (np.pi / 4 <= azimuth < 3 * np.pi / 4) or (5 * np.pi / 4 <= azimuth < 7 * np.pi / 4)
        for n in range(second_steps):
            if y_dominant:
                dy, dx = ss * n, -sc * abs(np.round(n / tan_a))
            else:
                dy, dx = ss * abs(round(n * tan_a)), -sc * n
            offsets[j, n] = int(dx), int(dy)
        low, high = azimuth - np.pi / 2, azimuth + np.pi / 2
        if low >= 0 and high < 2 * np.pi:
            bounds[j] = low, high, 0
        elif low < 0 and high <= 2 * np.pi:
            bounds[j] = low + 2 * np.pi, high, 1
        else:
            bounds[j] = low, high - 2 * np.pi, 1
    aspect = direction * np.pi / 180
    compact_masks = bool(cp.all(((buildings == 0) | (buildings == 1)) & ~cp.signbit(buildings)))
    return GvfGeometry(walls.copy(), buildings.copy(), aspect,
                       cp.asarray(offsets), cp.asarray(bounds), shape,
                       first_steps, second_steps, scale, cp.cuda.Device().id, compact_masks)


def gvf(geometry, wallsun, shadow, Tg, Tgwall, Ta, emis_grid, ewall,
        alb_grid, SBC, albedo_b, Twater=None, lc_grid=None, landcover=0, *,
        dtype=np.float64):
    cp = cupy()
    dtype = np.dtype(dtype)
    Ta, ewall, SBC, albedo_b, landcover = (
        float(v) for v in (Ta, ewall, SBC, albedo_b, landcover))
    shape = geometry.shape
    wallsun = _field(wallsun, shape, np.float64)
    shadow = _field(shadow, shape, dtype)
    tg = _field(Tg, shape, dtype)
    tgwall = _field(Tgwall, shape, dtype, scalar=True)
    emis = _field(emis_grid, shape, dtype)
    alb = _field(alb_grid, shape, dtype)
    if landcover == 1:
        Twater = float(Twater)
        lc = _field(lc_grid, shape, np.float64)
        updated_tg = cp.where(lc == 3, dtype.type(Twater - Ta), tg)
    else:
        updated_tg = tg.copy()

    fields = cp.empty((13, *shape), dtype=dtype)
    fields[0] = geometry.buildings
    fields[1] = shadow
    fields[2] = (wallsun / geometry.walls * geometry.buildings) == 1
    ambient = SBC * emis * (Ta + 273.15) ** 4
    fields[3] = SBC * emis * (tg * shadow + Ta + 273.15) ** 4 - ambient
    fields[4] = SBC * emis * (updated_tg * shadow + Ta + 273.15) ** 4 - ambient
    fields[5] = alb * shadow
    fields[6] = alb
    fields[7] = SBC * ewall * (tgwall + Ta + 273.15) ** 4 - SBC * ewall * (Ta + 273.15) ** 4
    fields[8] = fields[4] * (fields[0] * -1 + 1)
    fields[9] = alb * (fields[0] * -1 + 1) * shadow
    fields[10] = alb * (fields[0] * -1 + 1)
    fields[11] = ambient
    fields[12] = geometry.walls > 0
    pixels = int(np.prod(shape))
    output = cp.empty((17, *shape), dtype=dtype)
    arguments = (fields, geometry.aspect, geometry.offsets, geometry.shade_bounds,
                 np.int32(shape[0]), np.int32(shape[1]), np.int32(geometry.first_steps),
                 np.int32(geometry.second_steps), dtype.type(albedo_b), output)
    if geometry.compact_masks:
        neighbors = cp.empty((pixels, 4), dtype=dtype)
        masks = cp.empty(pixels, dtype=cp.uint8)
        pack = get_kernel("gvf.cu", "gvf_pack", geometry.device_id, precision=dtype.name)
        scan = get_kernel("gvf.cu", "gvf_compact", geometry.device_id, precision=dtype.name)
        pack(((pixels + 255) // 256,), (256,), (fields, np.int64(pixels), neighbors, masks))
        scan(((pixels + _GVF_FAST_BLOCK_SIZE - 1) // _GVF_FAST_BLOCK_SIZE,), (_GVF_FAST_BLOCK_SIZE,),
             (*arguments, neighbors, masks))
    else:
        get_kernel(
            "gvf.cu", "gvf", geometry.device_id, precision=dtype.name)(
            ((pixels + 127) // 128,), (128,), arguments)
    return GvfResult(tuple(output[k] for k in range(17)), updated_tg)


@dataclass(frozen=True)
class PreparedShadow:
    fields: object
    shape: tuple
    building_max: float
    vegetation_max: float
    scale: float
    usevegdem: bool
    device_id: int
    scan_fields: object
    maximum_magnitude: float


def prepare_shadow(a, walls, aspect, scale, *, vegdem=None, vegdem2=None,
                   bush=None, amaxvalue=None):
    cp = cupy()
    shape = tuple(a.shape)
    scale = float(scale)
    vegetation = vegdem is not None
    args = (a, walls, aspect, vegdem if vegetation else 0,
            vegdem2 if vegetation else 0, bush if vegetation else 0)
    fields = cp.stack([_field(v, shape, np.float64, scalar=True) for v in args])
    scan_fields = cp.stack((fields[0], fields[3], fields[4])).astype(cp.float32)
    maximum_magnitude = float(cp.max(cp.abs(fields[(0, 3, 4), :, :])).item())
    return PreparedShadow(fields, shape, float(cp.max(fields[0]).item()),
                          float(amaxvalue) if vegetation else 0.,
                          scale, vegetation, cp.cuda.Device().id,
                          scan_fields, maximum_magnitude)


def _shadow_ray(geometry, azimuth, altitude):
    azi = azimuth * (np.pi / 180.) if geometry.usevegdem else math.radians(azimuth)
    alt = altitude * (np.pi / 180.) if geometry.usevegdem else math.radians(altitude)
    s, c, t = np.sin(azi), np.cos(azi), np.tan(azi)
    ymajor = np.pi/4 <= azi < 3*np.pi/4 or 5*np.pi/4 <= azi < 7*np.pi/4
    ds = abs(1 / (s if ymajor else c))
    tanalt = np.tan(alt) / geometry.scale
    first = 0 if geometry.usevegdem else 1
    maximum = geometry.vegetation_max if geometry.usevegdem else geometry.building_max
    index = np.arange(first, max(geometry.shape) + 2, dtype=np.float64)
    if ymajor:
        dy = np.sign(s) * index
        dx = -np.sign(c) * np.abs(np.round(index / t))
    else:
        dy = np.sign(s) * np.abs(np.round(index * t))
        dx = -np.sign(c) * index
    dz = (ds * index) * tanalt
    invalid = ((~(maximum >= dz)) | (np.abs(dx) >= geometry.shape[0])
               | (np.abs(dy) >= geometry.shape[1]))
    count = 0 if maximum < 0 else int(np.flatnonzero(invalid)[0]) + 1
    offsets = np.column_stack((dx[:count], dy[:count])).astype(np.int32)
    previous = np.concatenate(([0.], dz[:max(count - 1, 0)]))[:count]
    heights = np.column_stack((dz[:count], previous))
    low, high = azi-np.pi/2, azi+np.pi/2
    middle = low >= 0 and high < 2*np.pi
    if not middle:
        if low < 0:
            low += 2*np.pi
        else:
            high -= 2*np.pi
    return (np.asarray(offsets, dtype=np.int32).reshape(-1, 2),
            np.asarray(heights, dtype=np.float64).reshape(-1, 2), low, high, middle)


def cast_shadow(geometry, azimuth, altitude):
    cp = cupy()
    azimuth, altitude = float(azimuth), float(altitude)
    offsets, heights, low, high, middle = _shadow_ray(geometry, azimuth, altitude)
    epsilon = (8 * np.finfo(np.float32).eps
               * (geometry.maximum_magnitude + np.abs(heights[:, 0]))
               + 8 * np.finfo(np.float32).tiny)
    with np.errstate(over='ignore'):
        fast_heights = np.column_stack((heights, epsilon)).astype(np.float32)
        fast_heights[:, 2] = np.nextafter(fast_heights[:, 2], np.float32(np.inf))
    pixels = np.prod(geometry.shape)
    out = cp.empty((8, *geometry.shape), dtype=cp.float64)
    get_kernel("geometry.cu", "shadow_pixels", geometry.device_id)(((int(pixels)+255)//256,), (256,),
        (geometry.fields, geometry.scan_fields, cp.asarray(offsets),
         cp.asarray(heights), cp.asarray(fast_heights),
         np.int32(len(offsets)), np.int32(geometry.shape[0]), np.int32(geometry.shape[1]),
         np.float64(low), np.float64(high), np.int32(middle), np.int32(geometry.usevegdem), out))
    if geometry.usevegdem:
        return tuple(out[k] for k in range(8))
    return tuple(out[k] for k in (1, 3, 4, 6, 7))


def _flux_run(name, fields, params, output_count, dtype, shape=None, field_dtype=None):
    cp = cupy()
    dtype = np.dtype(dtype)
    field_dtype = field_dtype or dtype
    arrays = []
    matrix_scalar = False
    for value in fields:
        array = cp.asarray(value)
        if array.ndim == 2 and array.size > 1 and shape is None:
            shape = array.shape
        if array.ndim == 2:
            matrix_scalar = True
        arrays.append(cp.ascontiguousarray(array, dtype=field_dtype).reshape(-1))
    shape = ((1, 1) if matrix_scalar else ()) if shape is None else tuple(shape)
    shape = tuple(int(x) for x in shape)
    count = int(np.prod(shape)) if shape else 1
    pointers = cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))
    sizes = cp.asarray(np.array([a.size for a in arrays], dtype=np.int64))
    result = cp.empty((output_count, *shape), dtype=dtype)
    function = get_kernel(
        "flux.cu", name, cp.cuda.Device().id, precision=dtype.name)
    function(((count + 255) // 256,), (256,), (pointers, sizes, result, np.int64(count), *params))
    return tuple(result[index] for index in range(output_count))


def cylindric_wedge(zen, svfalfa, rows, cols, *, dtype=np.float64):
    rows, cols = float(rows), float(cols)
    return _flux_run("wedge", (svfalfa,), (np.float64(float(zen)),), 1,
                dtype, (rows, cols), np.dtype("float64"))[0]


def TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1, *, dtype=np.float64):
    dtype = np.dtype(dtype)
    first = int(firstdaytime)
    timeadd, timestepdec = float(timeadd), float(timestepdec)
    update = timeadd >= 59 / 1440
    if update:
        weight = math.exp(-33.27 * timeadd)
        next_timeadd = timestepdec if timestepdec > 59 / 1440 else 0.0
    else:
        next_timeadd = timeadd + timestepdec
        weight = math.exp(-33.27 * next_timeadd)
    delayed, next_state = _flux_run("delay", (gvfLup, Tgmap1),
                               (np.int32(first), np.int32(update), dtype.type(weight)),
                               2, dtype)
    return delayed, next_timeadd, next_state


def Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh,
                  gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN,
                  gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN,
                  *, dtype=np.float64):
    dtype = np.dtype(dtype)
    values = (svfbuveg, F_sh, gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN,
              gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)
    params = tuple(dtype.type(float(v)) for v in (radI, radD, radG))
    params += (np.float64(float(altitude)), dtype.type(float(albedo_b)))
    return _flux_run("kup", values, params, 5, dtype)


def Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                      svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tw,
                      SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW, LupN,
                      anisotropic_longwave, *, dtype=np.float64):
    dtype = np.dtype(dtype)
    values = (svfE, svfS, svfW, svfN, svfEveg, svfSveg, svfWveg, svfNveg,
              svfEaveg, svfSaveg, svfWaveg, svfNaveg, Tw, Ldown, F_sh, LupE, LupS, LupW, LupN)
    params = (np.float64(float(azimuth)), np.float64(float(altitude)))
    params += tuple(dtype.type(float(v)) for v in (Ta, SBC, ewall, esky))
    params += (np.float64(float(t)), dtype.type(float(CI)), np.int32(int(anisotropic_longwave)))
    return _flux_run("lside", values, params, 4, dtype)


def surface_temperature(TgK, Tstart, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall,
                        altmax, dectime, SNUP, CI_TgG, landcover=0, *, shape=None,
                        dtype=np.float64):
    dtype = np.dtype(dtype)
    dectime = float(dectime)
    params = (np.float64(float(altmax)), np.float64(dectime - math.floor(dectime)),
              np.float64(float(SNUP)), dtype.type(float(CI_TgG)),
              np.int32(int(landcover)))
    fields = (TgK, Tstart, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall)
    return _flux_run("surface", fields, params, 2, dtype, shape)


def assemble_tmrt(Kdown, Kup, Knorth, Keast, Ksouth, Kwest, Ldown, Lup, Lnorth,
                  Least, Lsouth, Lwest, KsideI, Kside, Lside, absK, absL, Fside,
                  Fup, Fcyl, cyl, anisotropic_sky, *, SBC=5.67051e-8,
                  dtype=np.float64):
    dtype = np.dtype(dtype)
    values = (Kdown, Kup, Knorth, Keast, Ksouth, Kwest, Ldown, Lup, Lnorth,
              Least, Lsouth, Lwest, KsideI, Kside, Lside)
    params = tuple(dtype.type(float(v)) for v in (absK, absL, Fside, Fup, Fcyl))
    params += (np.int32(int(cyl)), np.int32(int(anisotropic_sky)), dtype.type(float(SBC)))
    return _flux_run("assemble", values, params, 2, dtype)


def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice, patch_option):

    m_a1 = np.array([1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500])
    m_a2 = np.array([-0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289])
    m_a3 = np.array([-0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260])
    m_a4 = np.array([-1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590])
    m_b1 = np.array([-0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250])
    m_b2 = np.array([0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156])
    m_b3 = np.array([1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781])
    m_b4 = np.array([-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025])
    m_c1 = np.array([2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625])
    m_c2 = np.array([0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000])
    m_c3 = np.array([1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148])
    m_c4 = np.array([1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750])
    m_d1 = np.array([1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312])
    m_d2 = np.array([0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050])
    m_d3 = np.array([0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500])
    m_d4 = np.array([0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234])
    m_e1 = np.array([0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000])
    m_e2 = np.array([-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426])
    m_e3 = np.array([-0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564])
    m_e4 = np.array([0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636])

    acoeff = np.transpose(np.atleast_2d([m_a1, m_a2, m_a3, m_a4]))
    bcoeff = np.transpose(np.atleast_2d([m_b1, m_b2, m_b3, m_b4]))
    ccoeff = np.transpose(np.atleast_2d([m_c1, m_c2, m_c3, m_c4]))
    dcoeff = np.transpose(np.atleast_2d([m_d1, m_d2, m_d3, m_d4]))
    ecoeff = np.transpose(np.atleast_2d([m_e1, m_e2, m_e3, m_e4]))

    deg2rad = np.pi/180
    rad2deg = 180/np.pi
    altitude = 90-zen
    zen = zen * deg2rad
    azimuth = azimuth * deg2rad
    altitude = altitude * deg2rad
    Idh = radD
    # Ibh = radI/sin(altitude)
    Ibn = radI

    # Skyclearness
    PerezClearness = ((Idh+Ibn)/(Idh+1.041*np.power(zen, 3)))/(1+1.041*np.power(zen, 3))
    # Extra terrestrial radiation
    day_angle = jday*2*np.pi/365
    #I0=1367*(1+0.033*np.cos((2*np.pi*jday)/365))
    I0 = 1367*(1.00011+0.034221*np.cos(day_angle) + 0.00128*np.sin(day_angle)+0.000719 *
               np.cos(2*day_angle)+0.000077*np.sin(2*day_angle))    # New from robinsson

    # Optical air mass
    # m=1/altitude; old
    if altitude >= 10*deg2rad:
        AirMass = 1/np.sin(altitude)
    elif altitude < 0:   # below equation becomes complex
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*complex(altitude)/np.pi+6.07995, -1.6364)
    else:
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*altitude/np.pi+6.07995, -1.6364)

    # Skybrightness
    # if altitude*rad2deg+6.07995>=0
    PerezBrightness = (AirMass*radD)/I0
    if Idh <= 10:
        # m_a=0;m_b=0;m_c=0;m_d=0;m_e=0;
        PerezBrightness = 0
    #if altitude < 0:
        #print("Airmass")
        #print(AirMass)
        #print(PerezBrightness)
    # sky clearness bins
    if PerezClearness < 1.065:
        intClearness = 0
    if PerezClearness > 1.065 and PerezClearness < 1.230:
        intClearness = 1
    if PerezClearness > 1.230 and PerezClearness < 1.500:
        intClearness = 2
    if PerezClearness > 1.500 and PerezClearness < 1.950:
        intClearness = 3
    if PerezClearness > 1.950 and PerezClearness < 2.800:
        intClearness = 4
    if PerezClearness > 2.800 and PerezClearness < 4.500:
        intClearness = 5
    if PerezClearness > 4.500 and PerezClearness < 6.200:
        intClearness = 6
    if PerezClearness > 6.200:
        intClearness = 7

    m_a = acoeff[intClearness,  0] + acoeff[intClearness,  1] * zen + PerezBrightness * (acoeff[intClearness,  2] + acoeff[intClearness,  3] * zen)
    m_b = bcoeff[intClearness,  0] + bcoeff[intClearness,  1] * zen + PerezBrightness * (bcoeff[intClearness,  2] + bcoeff[intClearness,  3] * zen)
    m_e = ecoeff[intClearness,  0] + ecoeff[intClearness,  1] * zen + PerezBrightness * (ecoeff[intClearness,  2] + ecoeff[intClearness,  3] * zen)

    if intClearness > 0:
        m_c = ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen + PerezBrightness * (ccoeff[intClearness, 2] + ccoeff[intClearness, 3] * zen)
        m_d = dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen + PerezBrightness * (dcoeff[intClearness, 2] + dcoeff[intClearness, 3] * zen)
    else:
        # different equations for c & d in clearness bin no. 1,  from Robinsson
        m_c = np.exp(np.power(PerezBrightness * (ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen), ccoeff[intClearness, 2]))-1
        m_d = -np.exp(PerezBrightness * (dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen)) + dcoeff[intClearness, 2] + \
            PerezBrightness * dcoeff[intClearness, 3] * PerezBrightness

    # print 'a = ', m_a
    # print 'b = ', m_b
    # print 'e = ', m_e
    # print 'c = ', m_c
    # print 'd = ', m_d

    if patchchoice == 2:
        skyvaultalt = np.atleast_2d([])
        skyvaultazi = np.atleast_2d([])
        # Creating skyvault at one degree intervals
        skyvaultalt = np.ones([90, 361])*90
        skyvaultazi = np.empty((90, 361))
        for j in range(90):
            skyvaultalt[j, :] = 91-j
            skyvaultazi[j, :] = range(361)

    elif patchchoice == 1:
        # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
        skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

    skyvaultzen = (90 - skyvaultalt) * deg2rad
    skyvaultalt = skyvaultalt * deg2rad
    skyvaultazi = skyvaultazi * deg2rad

    # Angular distance from the sun from Robinsson
    cosSkySunAngle = np.sin(skyvaultalt) * np.sin(altitude) + \
                     np.cos(altitude) * np.cos(skyvaultalt) * np.cos(np.abs(skyvaultazi-azimuth))

    cosSkySunAngle = np.clip(cosSkySunAngle, -1.0, 1.0)

    # Main equation
    lv = (1 + m_a * np.exp(m_b / np.cos(skyvaultzen))) * ((1 + m_c * np.exp(m_d * np.arccos(cosSkySunAngle)) +
                                                           m_e * cosSkySunAngle * cosSkySunAngle))

    # Normalisation
    lv = lv / np.sum(lv)

    # plotting
    # axesm('stereo','Origin',[90 180],'MapLatLimit',[0 90],'Aspect','transverse')
    # framem off; gridm on; mlabel off; plabel off;axis on;
    # setm(gca,'MLabelParallel',-20)
    # geoshow(skyvaultalt*rad2deg,skyvaultazi*rad2deg,lv,'DisplayType','texture');
    # colorbar
    # set(gcf,'Color',[1 1 1])
    # pause(1)

    if patchchoice == 1:
        #x = np.atleast_2d([])
        #lv = np.transpose(np.append(np.append(np.append(x, skyvaultalt*rad2deg), skyvaultazi*rad2deg), lv))
        x = np.transpose(np.atleast_2d(skyvaultalt*rad2deg))
        y = np.transpose(np.atleast_2d(skyvaultazi*rad2deg))
        z = np.transpose(np.atleast_2d(lv))
        lv = np.append(np.append(x, y, axis=1), z, axis=1)
    return lv, PerezClearness, PerezBrightness


OUTPUT_NAMES = ('Tmrt','Kdown','Kup','Ldown','Lup','Tg','ea','esky','I0','CI',
    'shadow','firstdaytime','timestepdec','timeadd','Tgmap1','Tgmap1E','Tgmap1S',
    'Tgmap1W','Tgmap1N','Keast','Ksouth','Kwest','Knorth','Least','Lsouth','Lwest',
    'Lnorth','KsideI','TgOut1','TgOut','radI','radD','Lside','L_patches','CI_Tg',
    'CI_TgG','KsideD','dRad','Kside')
STATE_NAMES = ('CI','firstdaytime','timeadd','Tgmap1','Tgmap1E','Tgmap1S',
               'Tgmap1W','Tgmap1N','TgOut1')
FIELD_NAMES = ('svf','svfN','svfW','svfE','svfS','svfveg','svfNveg','svfEveg',
    'svfSveg','svfWveg','svfaveg','svfEaveg','svfSaveg','svfWaveg','svfNaveg',
    'lc_grid','TgK','Tstart','alb_grid','emis_grid','TmaxLST','svfbuveg','asvf','svfalfa')


@dataclass(frozen=True)
class PreparedModel:
    fields: dict
    shadow: object
    gvf: object
    longwave: object
    night_longwave: object
    shortwave: object
    sky: object
    dtype: np.dtype
    shape: tuple
    device_id: int


def prepare_model(*, dtype=np.float32, **inputs):
    cp = cupy()
    dtype = np.dtype(dtype)
    shape = (int(inputs['rows']), int(inputs['cols']))
    vegetation, anisotropic = int(inputs['usevegdem']), int(inputs['anisotropic_sky'])
    fields = {n: _field(inputs[n], shape,
        np.float64 if n in ('asvf','svfalfa','lc_grid') else dtype,
        scalar=True) for n in FIELD_NAMES}
    shadow = prepare_shadow(inputs['dsm'], inputs['walls'],
        _field(inputs['dirwalls'], shape, np.float64)*np.pi/180.,
        inputs['scale'], vegdem=inputs['vegdem'] if vegetation else None,
        vegdem2=inputs['vegdem2'] if vegetation else None,
        bush=inputs['bush'] if vegetation else None,
        amaxvalue=inputs['amaxvalue'] if vegetation else None)
    ground = prepare_gvf_geometry(inputs['walls'], inputs['buildings'], inputs['dirwalls'],
        inputs['scale'], inputs['first'], inputs['second'])
    sky = lw = night_lw = sw = None
    if anisotropic:
        alt, azi, *_ = create_patches(int(inputs['patch_option']))
        sky = np.column_stack((alt, azi, np.zeros_like(alt)))
        day_sky = sky.copy()
        day_sky[:,:2] = day_sky[:,:2]*(np.pi/180.)*(180./np.pi)
        lw = prepare_geometry(day_sky, inputs['shmat'], inputs['vegshmat'],
                              inputs['vbshvegshmat'])
        small = np.ones((1,1,len(sky)),dtype=np.uint8)
        metadata = prepare_geometry(sky,small,small,small)
        night_lw = replace(metadata,visibility=lw.visibility,shape=shape)
        sw = prepare_shortwave(lw, inputs['diffsh'], dtype=dtype)
    return PreparedModel(fields, shadow, ground, lw, night_lw, sw, sky, dtype, shape,
        cp.cuda.Device().id)


def _diffuse(prepared, lv, radD):
    cp = cupy()
    out = cp.empty(prepared.shape, dtype=prepared.dtype)
    n = out.size
    get_kernel('shortwave.cu', 'diffuse', prepared.device_id, precision=prepared.dtype.name)(
        ((n+255)//256,), (256,),
        (prepared.shortwave.diffuse_visibility, cp.asarray(lv[:,2].copy()),
         np.int64(n), np.int32(len(lv)), np.float64(radD), out))
    return out




_MODEL_SCALARS = (
    "i", "scale", "rows", "cols", "albedo_b", "absK", "absL", "ewall",
    "Fside", "Fup", "Fcyl", "altitude", "azimuth", "zen", "jday", "usevegdem",
    "onlyglobal", "psi", "landcover", "dectime", "altmax", "cyl", "elvis",
    "Ta", "RH", "radG", "radD", "radI", "P", "amaxvalue", "Twater",
    "TgK_wall", "Tstart_wall", "TmaxLST_wall", "first", "second",
    "firstdaytime", "timeadd", "timestepdec", "CI", "anisotropic_sky", "patch_option",
)
_MODEL_INTEGERS = {"i", "rows", "cols", "usevegdem", "onlyglobal", "landcover",
                   "cyl", "anisotropic_sky", "patch_option"}
