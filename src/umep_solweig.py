"""Core SOLWEIG 2022a model algorithms and utilities.

This file contains functions that are directly accessed from the SOLWEIG (SOlar 
and LongWave EnvironmentalIrradiance Geometry) model. 

The original entry point `Solweig_2022a_calc` is retained here, 
but the used entry point in the CUDA implementation is revised and put into model.py

All functions retain exactly the same names and structure as in the original SOLWEIG 
to ease cross-referencing.

Source: Original SOLWEIG 2022a/UMEP 4.1.1

Original SOLWEIG package notice: Copyright (C) 2016 by Fredrik Lindberg, UoG.
Upstream licence: GNU GPL version 2 or any later version.
Consolidated and adapted for SOLWEIG-CUDA by Zeting Li, October 2026.
See NOTICE.md and LICENSE in the repository root for source and licence details.

References
----------
Lindberg, F., Holmer, B. and Thorsson, S. (2008).
SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes
and mean radiant temperature in complex urban settings.
International Journal of Biometeorology, 52, 697–713.
doi:10.1007/s00484-008-0162-7

Lindberg, F. et al. (2018).
Urban Multi-scale Environmental Predictor (UMEP):
An integrated tool for city-based climate services.
Environmental Modelling & Software, 99, 70–87.
doi:10.1016/j.envsoft.2017.09.020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from copy import deepcopy
import math
from math import radians
import datetime
import calendar

# from .Lside_veg_v2015a import Lside_veg_v2015a
# from .Kside_veg_v2019a import Kside_veg_v2019a

# Anisotropic longwave

def Solweig_2022a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
                       amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, 
                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option):

#def Solweig_2021a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
#                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
#                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
#                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
#                       amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
#                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, 
#                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani):

    # This is the core function of the SOLWEIG model
    # 2016-Aug-28
    # Fredrik Lindberg, fredrikl@gvc.gu.se
    # Goteborg Urban Climate Group
    # Gothenburg University
    #
    # Input variables:
    # dsm = digital surface model
    # scale = height to pixel size (2m pixel gives scale = 0.5)
    # header = ESRI Ascii Grid header
    # sizey,sizex = no. of pixels in x and y
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
    # height = height of measurements point
    # psi = 1 - Transmissivity of shortwave through vegetation
    # output = output settings
    # fileformat = fileformat of output grids
    # landcover = use landcover scheme !!!NEW IN 2015a!!!
    # sensorheight = Sensorheight of wind sensor
    # lc_grid = grid with landcoverclasses
    # lc_class = table with landcover properties
    # dectime = decimal time
    # altmax = maximum sun altitude
    # dirwalls = aspect of walls
    # walls = one pixel row outside building footprints
    # cyl = consider man as cylinder instead of cude

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
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    if altitude > 0: # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
            if (CI > 1) or (CI == np.inf):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = zen*(180/np.pi)
            # Relative luminance
            lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)   
            # Total relative luminance from sky, i.e. from each patch, into each cell
            aniLum = np.zeros((rows, cols))
            for idx in range(lv.shape[0]):
                aniLum += diffsh[:,:,idx] * lv[idx,2]     

            dRad = aniLum * radD   # Total diffuse radiation from sky into each cell
        else:
            dRad = radD * svfbuveg
            patchchoice = 1
            lv = None
            # zenDeg = zen*(180/np.pi)
            # lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)   # Relative luminance

        # Shadow  images
        if usevegdem == 1:
            vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm, vegdem, vegdem2,
                                        azimuth, altitude, scale, amaxvalue, bush, walls, dirwalls * np.pi / 180.)
            shadow = sh - (1 - vegsh) * (1 - psi)
        else:
            sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13(dsm, azimuth, altitude, scale,
                                                                                   walls, dirwalls * np.pi / 180.)
            shadow = sh

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        # Tgamp = (TgK * altmax - Tstart) + Tstart # Old
        Tgamp = TgK * altmax + Tstart # Fixed 2021
        # Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall) # Old
        Tgampwall = TgK_wall * altmax + Tstart_wall
        Tg = Tgamp * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2) # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2) # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radG / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1

        deg2rad = np.pi/180
        radG0 = radI0 * (np.sin(altitude * deg2rad)) + _
        CI_TgG = (radG / radG0) + (1 - corr)
        if (CI_TgG > 1) or (CI_TgG == np.inf):
            CI_TgG = 1
        
        # Tg = Tg * CI_Tg  # new estimation
        # Tgwall = Tgwall * CI_Tg
        Tg = Tg * CI_TgG  # new estimation
        Tgwall = Tgwall * CI_TgG
        if landcover == 1:
            Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        # # # # Ground View Factors # # # #
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW,\
        gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm = gvf_2018a(wallsun, walls, buildings, scale, shadow, first,
                second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols,
                                                                 Twater, lc_grid, landcover)

        # # # # Lup, daytime # # # #
        # Surface temperature wave delay - new as from 2014a
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E)
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S)
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W)
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)
        
        # # For Tg output in POIs
        TgTemp = Tg * shadow + Ta
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime, timeadd, timestepdec, TgOut1) #timeadd only here v2021a

        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, rows, cols)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[np.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svfbuveg) * \
                            (radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))

        #Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + radD * svfbuveg + albedo_b * (1 - svfbuveg) * \
                            #(radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))

        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb,
                    gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        #Keast, Ksouth, Kwest, Knorth, KsideI, KsideD = Kside_veg_v2019a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
        #            svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo_b, F_sh, KupE, KupS, KupW,
        #            KupN, cyl, lv, ani, diffsh, rows, cols)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
                    svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo_b, F_sh, KupE, KupS, KupW,
                    KupN, cyl, lv, anisotropic_sky, diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = 0
        # CI_Tg = -999  # F_sh = []

        # Nocturnal K fluxes set to 0
        Knight = np.zeros((rows, cols))
        Kdown = np.zeros((rows, cols))
        Kwest = np.zeros((rows, cols))
        Kup = np.zeros((rows, cols))
        Keast = np.zeros((rows, cols))
        Ksouth = np.zeros((rows, cols))
        Knorth = np.zeros((rows, cols))
        KsideI = np.zeros((rows, cols))
        KsideD = np.zeros((rows, cols))
        F_sh = np.zeros((rows, cols))
        Tg = np.zeros((rows, cols))
        shadow = np.zeros((rows, cols))
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)

        dRad = np.zeros((rows,cols))

        Kside = np.zeros((rows,cols))

        # # # # Lup # # # #
        Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)
        if landcover == 1:
            Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp

        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        TgOut = Ta + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    # Anisotropic sky longwave radiation
    if anisotropic_sky == 1:
        if 'lv' not in locals():
            # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

            patch_emissivities = np.zeros(skyvaultalt.shape[0])

            x = np.transpose(np.atleast_2d(skyvaultalt))
            y = np.transpose(np.atleast_2d(skyvaultazi))
            z = np.transpose(np.atleast_2d(patch_emissivities))

            L_patches = np.append(np.append(x, y, axis=1), z, axis=1)

        else:
            L_patches = deepcopy(lv)

        if altitude < 0:
            CI = deepcopy(CI)

        if CI < 0.95:
            esky_c = CI * esky + (1 - CI) * 1.
            esky = esky_c

        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ \
                  = Lcyl_v2022a(esky, L_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, 
                                altitude, azimuth, rows, cols, asvf)

    else:
        Ldown = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) * ewall * SBC * \
                    ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
                    (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)
        # Ldown = Ldown - 25 # Shown by Jonsson et al.(2006) and Duarte et al.(2006)

        Lside = np.zeros((rows, cols))
        L_patches = None
 
        if CI < 0.95:  # non - clear conditions
            c = 1 - CI
            Ldown = Ldown * (1 - c) + c * ((svf + svfveg - 1) * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) *
                    ewall * SBC * ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) +
                    (2 - svf - svfveg) * (1 - ewall) * SBC * ((Ta + 273.15) ** 4))  # NOT REALLY TESTED!!! BUT MORE CORRECT?

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tgwall, SBC, ewall, Ldown,
                                                      esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_sky)

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # Human body considered as a cylinder with isotropic all-sky diffuse
    if cyl == 1 and anisotropic_sky == 0: 
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Human body considered as a cylinder with Perez et al. (1993) (anisotropic sky diffuse) 
    # and Martin and Berdahl (1984) (anisotropic sky longwave)
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)
    # Knorth = nan Ksouth = nan Kwest = nan Keast = nan
    else: # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)

    Tmrt = np.sqrt(np.sqrt((Sstr / (absL * SBC)))) - 273.2

    # Add longwave to cardinal directions for output in POI
    if (cyl == 1) and (anisotropic_sky == 1):
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, \
           timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
           Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, \
               Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside

    # return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, \
    #        timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, \
    #        Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, \
    #            Ldown, Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside





def cylindric_wedge(zen, svfalfa, rows, cols):

    # Fraction of sunlit walls based on sun altitude and svf wieghted building angles
    # input: 
    # sun zenith angle "beta"
    # svf related angle "alfa"

    beta=zen
    alfa=svfalfa
    
    # measure the size of the image
    # sizex=size(svfalfa,2)
    # sizey=size(svfalfa,1)
    
    xa=1-2./(np.tan(alfa)*np.tan(beta))
    ha=2./(np.tan(alfa)*np.tan(beta))
    ba=(1./np.tan(alfa))
    hkil=2.*ba*ha
    
    qa=np.zeros((rows, cols))
    # qa(length(svfalfa),length(svfalfa))=0;
    qa[xa<0]=np.tan(beta)/2
    
    Za=np.zeros((rows, cols))
    # Za(length(svfalfa),length(svfalfa))=0;
    Za[xa<0]=((ba[xa<0]**2)-((qa[xa<0]**2)/4))**0.5
    
    phi=np.zeros((rows, cols))
    #phi(length(svfalfa),length(svfalfa))=0;
    phi[xa<0]=np.arctan(Za[xa<0]/qa[xa<0])
    
    A=np.zeros((rows, cols))
    # A(length(svfalfa),length(svfalfa))=0;
    A[xa<0]=(np.sin(phi[xa<0])-phi[xa<0]*np.cos(phi[xa<0]))/(1-np.cos(phi[xa<0]))
    
    ukil=np.zeros((rows, cols))
    # ukil(length(svfalfa),length(svfalfa))=0
    ukil[xa<0]=2*ba[xa<0]*xa[xa<0]*A[xa<0]
    
    Ssurf=hkil+ukil
    
    F_sh=(2*np.pi*ba-Ssurf)/(2*np.pi*ba)#Xa
    
    return F_sh



def daylen(DOY, XLAT):
    # Calculation of declination of sun (Eqn. 16). Amplitude= +/-23.45
    # deg. Minimum = DOY 355 (DEC 21), maximum = DOY 172.5 (JUN 21/22).
    # Sun angles.  SOC limited for latitudes above polar circles.
    # Calculate daylength, sunrise and sunset (Eqn. 17)

    RAD=np.pi/180.0

    DEC = -23.45 * np.cos(2.0*np.pi*(DOY+10.0)/365.0)

    SOC = np.tan(RAD*DEC) * np.tan(RAD*XLAT)
    SOC = min(max(SOC,-1.0),1.0)
    # SOC=alt

    DAYL = 12.0 + 24.0*np.arcsin(SOC)/np.pi
    SNUP = 12.0 - DAYL/2.0
    SNDN = 12.0 + DAYL/2.0

    return DAYL, DEC, SNDN, SNUP



''' Model 1 is based on Unsworth & Monteith, 1975 '''
def model1(sky_patches, esky, Ta):

    # Stefan-Boltzmann's Constant
    SBC = 5.67051e-8

    # Degrees to radians
    deg2rad = np.pi / 180               

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    
    # Unique zeniths for the patches
    skyzen = 90-skyalt                  
    
    # Cosine of the zenith angles
    cosskyzen = np.cos(skyzen * deg2rad)

    # Estimate emissivities at different altitudes/zenith angles
    
    # Constants?
    a_c = 0.67
    b_c = 0.094

    # Natural log of the reduced depth of precipitable water
    ln_u_prec = esky/b_c-a_c/b_c-0.5
    
    # Reduced depth of precipitable water    
    u_prec = np.exp(ln_u_prec)          

    # Optical water depth
    owp = u_prec/cosskyzen              

    # Natural log of optical water depth
    log_owp = np.log(owp)               

    # Emissivity of each zenith angle, i.e. the zenith angle of each band of patches
    esky_band = a_c+b_c*log_owp            
    
    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:,0]
    
    # Empty array with length based on number of patches                
    patch_emissivity = np.zeros((p_alt.shape[0]))

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity/np.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band

''' Model 2 is based on Martin & Berdhal, 1984
    Ez = 1 - (1 - Es)*e**(b2*(1.7 - (1/np.cos(z)))) '''
def model2(sky_patches, esky, Ta):

    # Degrees to radians
    deg2rad = np.pi / 180               

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    
    # Unique zeniths for the patches
    skyzen = 90-skyalt                  
    
    # Constant (Ångström, 1915), proposed by Nahon et al., 2019
    b_c = 0.308
    # b_c = 0.1

    # Estimate emissivites at different altitudes/zenith angles
    esky_band = 1 - (1 - esky) * np.exp(b_c * (1.7 - (1 / np.cos(skyzen * deg2rad))))
    
    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:,0]
    
    # Empty array with length based on number of patches                
    patch_emissivity = np.zeros((p_alt.shape[0]))

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity/np.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band

''' Model 3 is based on Bliss, 1961.
    Ez = 1 - (1 - Es)**(1/(b1*np.cos(z))) '''
def model3(sky_patches, esky, Ta):

    # Degrees to radians
    deg2rad = np.pi / 180               

    # Unique altitudes in lv, i.e. unique altitude for the patches
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    
    # Unique zeniths for the patches
    skyzen = 90-skyalt                  
    
    # Constant, can (should?) be changed. Model gave unsatisfactory results in Nahon et al., 2019
    b_c = 1.8

    # Estimate emissivites at different altitudes/zenith angles
    esky_band = 1 - (1 - esky)**(1/(b_c * np.cos(skyzen * deg2rad)))

    # Estimating longwave radiation for each patch to a horizontal or vertical surface
    
    # Altitudes of the Robinson & Stone patches
    p_alt = sky_patches[:,0]
    
    # Empty array with length based on number of patches                
    patch_emissivity = np.zeros((p_alt.shape[0]))

    # Fill with emissivities
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity

    # Normalize
    patch_emissivity_normalized = patch_emissivity/np.sum(patch_emissivity)

    return patch_emissivity_normalized, esky_band

# import matplotlib.pyplot as plt


def gvf_2018a(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall,
              alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    azimuthA = np.arange(5, 359, 20)  # Search directions for Ground View Factors (GVF)

    #### Ground View Factors ####
    gvfLup = np.zeros((rows, cols))
    gvfalb = np.zeros((rows, cols))
    gvfalbnosh = np.zeros((rows, cols))
    gvfLupE = np.zeros((rows, cols))
    gvfLupS = np.zeros((rows, cols))
    gvfLupW = np.zeros((rows, cols))
    gvfLupN = np.zeros((rows, cols))
    gvfalbE = np.zeros((rows, cols))
    gvfalbS = np.zeros((rows, cols))
    gvfalbW = np.zeros((rows, cols))
    gvfalbN = np.zeros((rows, cols))
    gvfalbnoshE = np.zeros((rows, cols))
    gvfalbnoshS = np.zeros((rows, cols))
    gvfalbnoshW = np.zeros((rows, cols))
    gvfalbnoshN = np.zeros((rows, cols))
    gvfSum = np.zeros((rows, cols))

    #  sunwall=wallinsun_2015a(buildings,azimuth(i),shadow,psi(i),dirwalls,walls);
    sunwall = (wallsun / walls * buildings) == 1  # new as from 2015a

    for j in np.arange(0, azimuthA.__len__()):
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = sunonsurface_2018a(azimuthA[j], scale, buildings, shadow, sunwall,
                                                                    first,
                                                                    second, dirwalls * np.pi / 180, walls, Tg, Tgwall,
                                                                    Ta,
                                                                    emis_grid, ewall, alb_grid, SBC, albedo_b, Twater,
                                                                    lc_grid, landcover)

        gvfLup = gvfLup + gvfLupi
        gvfalb = gvfalb + gvfalbi
        gvfalbnosh = gvfalbnosh + gvfalbnoshi
        gvfSum = gvfSum + gvf2

        if (azimuthA[j] >= 0) and (azimuthA[j] < 180):
            gvfLupE = gvfLupE + gvfLupi
            gvfalbE = gvfalbE + gvfalbi
            gvfalbnoshE = gvfalbnoshE + gvfalbnoshi

        if (azimuthA[j] >= 90) and (azimuthA[j] < 270):
            gvfLupS = gvfLupS + gvfLupi
            gvfalbS = gvfalbS + gvfalbi
            gvfalbnoshS = gvfalbnoshS + gvfalbnoshi

        if (azimuthA[j] >= 180) and (azimuthA[j] < 360):
            gvfLupW = gvfLupW + gvfLupi
            gvfalbW = gvfalbW + gvfalbi
            gvfalbnoshW = gvfalbnoshW + gvfalbnoshi

        if (azimuthA[j] >= 270) or (azimuthA[j] < 90):
            gvfLupN = gvfLupN + gvfLupi
            gvfalbN = gvfalbN + gvfalbi
            gvfalbnoshN = gvfalbnoshN + gvfalbnoshi

    gvfLup = gvfLup / azimuthA.__len__() + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfalb = gvfalb / azimuthA.__len__()
    gvfalbnosh = gvfalbnosh / azimuthA.__len__()

    gvfLupE = gvfLupE / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupS = gvfLupS / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupW = gvfLupW / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4
    gvfLupN = gvfLupN / (azimuthA.__len__() / 2) + SBC * emis_grid * (Ta + 273.15) ** 4

    gvfalbE = gvfalbE / (azimuthA.__len__() / 2)
    gvfalbS = gvfalbS / (azimuthA.__len__() / 2)
    gvfalbW = gvfalbW / (azimuthA.__len__() / 2)
    gvfalbN = gvfalbN / (azimuthA.__len__() / 2)

    gvfalbnoshE = gvfalbnoshE / (azimuthA.__len__() / 2)
    gvfalbnoshS = gvfalbnoshS / (azimuthA.__len__() / 2)
    gvfalbnoshW = gvfalbnoshW / (azimuthA.__len__() / 2)
    gvfalbnoshN = gvfalbnoshN / (azimuthA.__len__() / 2)

    gvfNorm = gvfSum / (azimuthA.__len__())
    gvfNorm[buildings == 0] = 1
    
    return gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm


def Kside_veg_v2022a(radI,radD,radG,
                    shadow,svfS,svfW,svfN,svfE,svfEveg,svfSveg,svfWveg,svfNveg,
                    azimuth,altitude,psi,t,albedo,F_sh,
                    KupE,KupS,KupW,KupN,
                    cyl,lv,anisotropic_diffuse,diffsh,rows,cols,asvf,
                    shmat, vegshmat, vbshvegshmat):

    # New reflection equation 2012-05-25
    vikttot=4.4897
    aziE=azimuth+t
    aziS=azimuth-90+t
    aziW=azimuth-180+t
    aziN=azimuth-270+t
    deg2rad=np.pi/180
    KsideD = np.zeros((rows, cols))
    Kref_sun = np.zeros((rows, cols))
    Kref_sh = np.zeros((rows, cols))
    Kref_veg = np.zeros((rows, cols))
    Kside = np.zeros((rows, cols))

    Kref_veg_n = np.zeros((rows, cols))
    Kref_veg_s = np.zeros((rows, cols))
    Kref_veg_e = np.zeros((rows, cols))
    Kref_veg_w = np.zeros((rows, cols))

    Kref_sh_n = np.zeros((rows, cols))
    Kref_sh_s = np.zeros((rows, cols))
    Kref_sh_e = np.zeros((rows, cols))
    Kref_sh_w = np.zeros((rows, cols))

    Kref_sun_n = np.zeros((rows, cols))
    Kref_sun_s = np.zeros((rows, cols))
    Kref_sun_e = np.zeros((rows, cols))
    Kref_sun_w = np.zeros((rows, cols))

    KeastRef = np.zeros((rows, cols)); KwestRef = np.zeros((rows, cols)); KnorthRef = np.zeros((rows, cols)); KsouthRef = np.zeros((rows, cols))
    diffRadE = np.zeros((rows, cols)); diffRadS = np.zeros((rows, cols)); diffRadW = np.zeros((rows, cols)); diffRadN = np.zeros((rows, cols))

    ### Direct radiation ###
    if cyl == 1: ### Kside with cylinder ###
        KsideI=shadow*radI*np.cos(altitude*deg2rad)
        KeastI=0;KsouthI=0;KwestI=0;KnorthI=0
    else: ### Kside with weights ###
        if azimuth > (360-t)  or  azimuth <= (180-t):
            KeastI=radI*shadow*np.cos(altitude*deg2rad)*np.sin(aziE*deg2rad)
        else:
            KeastI=0
        if azimuth > (90-t)  and  azimuth <= (270-t):
            KsouthI=radI*shadow*np.cos(altitude*deg2rad)*np.sin(aziS*deg2rad)
        else:
            KsouthI=0
        if azimuth > (180-t)  and  azimuth <= (360-t):
            KwestI=radI*shadow*np.cos(altitude*deg2rad)*np.sin(aziW*deg2rad)
        else:
            KwestI=0
        if azimuth <= (90-t)  or  azimuth > (270-t):
            KnorthI=radI*shadow*np.cos(altitude*deg2rad)*np.sin(aziN*deg2rad)
        else:
            KnorthI=0

        KsideI=shadow*0

    ### Diffuse and reflected radiation ###
    [viktveg,viktwall]=Kvikt_veg(svfE,svfEveg,vikttot)
    svfviktbuvegE=(viktwall+(viktveg)*(1-psi))

    [viktveg,viktwall]=Kvikt_veg(svfS,svfSveg,vikttot)
    svfviktbuvegS=(viktwall+(viktveg)*(1-psi))

    [viktveg,viktwall]=Kvikt_veg(svfW,svfWveg,vikttot)
    svfviktbuvegW=(viktwall+(viktveg)*(1-psi))

    [viktveg,viktwall]=Kvikt_veg(svfN,svfNveg,vikttot)
    svfviktbuvegN=(viktwall+(viktveg)*(1-psi))

    ### Anisotropic Diffuse Radiation after Perez et al. 1993 ###
    if anisotropic_diffuse == 1:

        anisotropic_sky = True

        patch_altitude = lv[:, 0]
        patch_azimuth = lv[:, 1]
        if anisotropic_sky:
            patch_luminance = lv[:, 2]
        else:
            patch_luminance = np.zeros((patch_altitude.shape[0]))
            patch_luminance[:] = 1.0/patch_luminance.shape[0]

        # Unique altitudes for patches
        skyalt, skyalt_c = np.unique(patch_altitude, return_counts=True)

        radTot = np.zeros(1)

        # Calculation of steradian for each patch
        steradian = np.zeros((patch_altitude.shape[0]))
        for i in range(patch_altitude.shape[0]):
            # If there are more than one patch in a band
            if skyalt_c[skyalt == patch_altitude[i]] > 1:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (np.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad) \
                - np.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad))
            # If there is only one patch in band, i.e. 90 degrees
            else:
                steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (np.sin((patch_altitude[i]) * deg2rad) \
                    - np.sin((patch_altitude[i-1] + patch_altitude[0]) * deg2rad))

            radTot += (patch_luminance[i] * steradian[i] * np.sin(patch_altitude[i] * deg2rad)) # Radiance fraction normalization

        lumChi = (patch_luminance * radD) / radTot # Radiance fraction normalization

        if cyl == 1:
            for idx in range(patch_azimuth.shape[0]):
                # Angle of incidence, np.cos(0) because cylinder - always perpendicular
                anglIncC = np.cos(patch_altitude[idx] * deg2rad) * np.cos(0) # * np.sin(np.pi / 2) \
                    # + np.sin(patch_altitude[idx] * deg2rad) * np.cos(np.pi / 2)
                # Diffuse vertical radiation
                KsideD += diffsh[:, :, idx] * lumChi[idx] * anglIncC * steradian[idx]

                # Shortwave reflected on sunlit surfaces
                # sunlit_surface = ((albedo * radG) / np.pi)
                sunlit_surface = ((albedo * (radI * np.cos(altitude * deg2rad)) + (radD * 0.5)) / np.pi)
                # Shortwave reflected on shaded surfaces and vegetation
                shaded_surface = ((albedo * radD * 0.5) / np.pi)
                
                # Shortwave radiation reflected on vegetation - based on diffuse shortwave radiation
                temp_vegsh = ((vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
                Kref_veg += shaded_surface * temp_vegsh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)

                # Shortwave radiation reflected on buildings (shaded and sunlit) - based on global and diffuse shortwave radiation
                temp_vbsh = (1 - shmat[:,:,idx]) * vbshvegshmat[:,:,idx]
                temp_sh = (temp_vbsh == 1) # & (vbshvegshmat[:,:,idx] == 1)

                sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                Kref_sun += sunlit_surface * sunlit_patches * temp_sh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)
                Kref_sh += shaded_surface * shaded_patches * temp_sh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)

            Kside = KsideI + KsideD + Kref_sun + Kref_sh + Kref_veg

            Keast = (KupE * 0.5)
            Kwest = (KupW * 0.5)
            Knorth = (KupN * 0.5)
            Ksouth = (KupS * 0.5)

            # Keast  = (albedo * (svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
            # Ksouth = (albedo * (svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
            # Kwest  = (albedo * (svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
            # Knorth = (albedo * (svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
        else: # Box
            diffRadE = np.zeros((rows, cols)); diffRadS = np.zeros((rows, cols)); diffRadW = np.zeros((rows, cols)); diffRadN = np.zeros((rows, cols))

            for idx in range(patch_azimuth.shape[0]):
                if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] <= 180):
                    anglIncE = np.cos(patch_altitude[idx] * deg2rad) * np.cos((90 - patch_azimuth[idx] + t) * deg2rad) # * np.sin(np.pi / 2) \
                        # + np.sin(patch_altitude[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadE += diffsh[:, :, idx] * lumChi[idx] * anglIncE * steradian[idx] #* 0.5

                if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] <= 270):
                    anglIncS = np.cos(patch_altitude[idx] * deg2rad) * np.cos((180 - patch_azimuth[idx] + t) * deg2rad) # * np.sin(np.pi / 2) \
                        # + np.sin(patch_altitude[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadS += diffsh[:, :, idx] * lumChi[idx] * anglIncS * steradian[idx] #* 0.5

                if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] <= 360):
                    anglIncW = np.cos(patch_altitude[idx] * deg2rad) * np.cos((270 - patch_azimuth[idx] + t) * deg2rad) # * np.sin(np.pi / 2) \
                        # + np.sin(patch_altitude[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadW += diffsh[:, :, idx] * lumChi[idx] * anglIncW * steradian[idx] #* 0.5

                if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] <= 90):
                    anglIncN = np.cos(patch_altitude[idx] * deg2rad) * np.cos((0 - patch_azimuth[idx] + t) * deg2rad) # * np.sin(np.pi / 2) \
                        # + np.sin(patch_altitude[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadN += diffsh[:, :, idx] * lumChi[idx] * anglIncN * steradian[idx] #* 0.5
            
                # Shortwave reflected on sunlit surfaces
                # sunlit_surface = ((albedo * radG) / np.pi)
                sunlit_surface = ((albedo * (radI * np.cos(altitude * deg2rad)) + (radD * 0.5)) / np.pi)
                # Shortwave reflected on shaded surfaces and vegetation
                shaded_surface = ((albedo * radD * 0.5) / np.pi)
                
                # Shortwave radiation reflected on vegetation - based on diffuse shortwave radiation
                temp_vegsh = ((vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
                Kref_veg += shaded_surface * temp_vegsh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)

                if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                    Kref_veg_e += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                    Kref_veg_s += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                    Kref_veg_w += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                    Kref_veg_n += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((0 - patch_azimuth[idx] + t) * deg2rad)

                # Shortwave radiation reflected on buildings (shaded and sunlit) - based on global and diffuse shortwave radiation
                temp_vbsh = (1 - shmat[:,:,idx]) * vbshvegshmat[:,:,idx]
                temp_sh = (temp_vbsh == 1) # & (vbshvegshmat[:,:,idx] == 1)
                azimuth_difference = np.abs(azimuth - patch_azimuth[idx])

                if ((azimuth_difference > 90) and (azimuth_difference < 270)):
                    sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                    Kref_sun += sunlit_surface * sunlit_patches * temp_sh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)
                    Kref_sh += shaded_surface * shaded_patches * temp_sh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)

                    if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                        Kref_sun_e += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_e += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                        Kref_sun_s += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_s += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                        Kref_sun_w += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_w += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                        Kref_sun_n += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx] + t) * deg2rad)
                        Kref_sh_n += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx] + t) * deg2rad)
                else:
                    Kref_sh += shaded_surface * temp_sh * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad)

                    if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                        Kref_sh_e += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                        Kref_sh_s += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                        Kref_sh_w += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx] + t) * deg2rad)
                    if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                        Kref_sh_n += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx] + t) * deg2rad)

            Keast = KeastI + diffRadE + Kref_sun_e + Kref_sh_e + Kref_veg_e + KupE * 0.5
            Kwest = KwestI + diffRadW + Kref_sun_w + Kref_sh_w + Kref_veg_w + KupW * 0.5
            Knorth = KnorthI + diffRadN + Kref_sun_n + Kref_sh_n + Kref_veg_n + KupN * 0.5
            Ksouth = KsouthI + diffRadS + Kref_sun_s + Kref_sh_s + Kref_veg_s + KupS * 0.5

    else:
        KeastDG = (radD * (1 - svfviktbuvegE) + albedo * (
        svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
        Keast = KeastI + KeastDG

        KsouthDG = (radD * (1 - svfviktbuvegS) + albedo * (
        svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
        Ksouth = KsouthI + KsouthDG

        KwestDG = (radD * (1 - svfviktbuvegW) + albedo * (
        svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
        Kwest = KwestI + KwestDG

        KnorthDG = (radD * (1 - svfviktbuvegN) + albedo * (
        svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
        Knorth = KnorthI + KnorthDG

    return Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside
    # return Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kref_sh, Kref_sun, Kref_veg, Kside


def Kup_veg_2015a(radI,radD,radG,altitude,svfbuveg,albedo_b,F_sh,gvfalb,gvfalbE,gvfalbS,gvfalbW,gvfalbN,gvfalbnosh,gvfalbnoshE,gvfalbnoshS,gvfalbnoshW,gvfalbnoshN):

    Kup=(gvfalb*radI*np.sin(altitude*(np.pi/180.)))+(radD*svfbuveg+albedo_b*(1-svfbuveg)*(radG*(1-F_sh)+radD*F_sh))*gvfalbnosh

    KupE=(gvfalbE*radI*np.sin(altitude*(np.pi/180.)))+(radD*svfbuveg+albedo_b*(1-svfbuveg)*(radG*(1-F_sh)+radD*F_sh))*gvfalbnoshE

    KupS=(gvfalbS*radI*np.sin(altitude*(np.pi/180.)))+(radD*svfbuveg+albedo_b*(1-svfbuveg)*(radG*(1-F_sh)+radD*F_sh))*gvfalbnoshS

    KupW=(gvfalbW*radI*np.sin(altitude*(np.pi/180.)))+(radD*svfbuveg+albedo_b*(1-svfbuveg)*(radG*(1-F_sh)+radD*F_sh))*gvfalbnoshW

    KupN=(gvfalbN*radI*np.sin(altitude*(np.pi/180.)))+(radD*svfbuveg+albedo_b*(1-svfbuveg)*(radG*(1-F_sh)+radD*F_sh))*gvfalbnoshN

    return Kup, KupE, KupS, KupW, KupN


def Kvikt_veg(svf,svfveg,vikttot):

    # Least
    viktwall=(vikttot-(63.227*svf**6-161.51*svf**5+156.91*svf**4-70.424*svf**3+16.773*svf**2-0.4863*svf))/vikttot
    
    svfvegbu=(svfveg+svf-1)  # Vegetation plus buildings
    viktveg=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=viktveg-viktwall
    
    return viktveg,viktwall


''' This function combines the method to divide the sky vault into patches (Tregenza (1987) and Robinson & Stone (2004)) 
    and the approach by Unsworth & Monteith or Martin & Berdahl (1984) or Bliss (1961) to calculate emissivities of the 
    different parts of the sky vault. '''

def Lcyl_v2022a(esky, sky_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, solar_altitude, solar_azimuth, rows, cols, asvf):

    # Stefan-Boltzmann's Constant
    SBC = 5.67051e-8

    # Sky longwave radiation from emissivity based on Prata (1996)
    Ldown_prata = (esky * SBC * ((Ta + 273.15) ** 4))

    # Degrees to radians
    deg2rad = np.pi / 180

    # Unique altitudes for patches
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    # skyzen = 90-skyalt                  # Unique zeniths for the patches

    # Altitudes of the Robinson & Stone patches
    patch_altitude = sky_patches[:, 0]
    # Azimuths of the Robinson & Stone patches, used for box
    patch_azimuth = sky_patches[:, 1]
    
    emis_m = 2

    # Unsworth & Monteith (1975)
    if emis_m == 1:
        patch_emissivity_normalized, esky_band = model1(sky_patches, esky, Ta)
    # Martin & Berdahl (1984)
    elif emis_m == 2:
        patch_emissivity_normalized, esky_band = model2(sky_patches, esky, Ta)
    # Bliss (1961)
    elif emis_m == 3:
        patch_emissivity_normalized, esky_band = model3(sky_patches, esky, Ta)

    # Calculation of steradian for each patch
    steradian = np.zeros((patch_altitude.shape[0]))
    for i in range(patch_altitude.shape[0]):
        # If there are more than one patch in a band
        if skyalt_c[skyalt == patch_altitude[i]] > 1:
            steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (np.sin((patch_altitude[i] + patch_altitude[0]) * deg2rad) \
            - np.sin((patch_altitude[i] - patch_altitude[0]) * deg2rad))
        # If there is only one patch in band, i.e. 90 degrees
        else:
            steradian[i] = ((360 / skyalt_c[skyalt == patch_altitude[i]]) * deg2rad) * (np.sin((patch_altitude[i]) * deg2rad) \
                - np.sin((patch_altitude[i-1] + patch_altitude[0]) * deg2rad))

    # True = anisotropic sky, False = isotropic sky
    anisotropic_sky = True
    # anisotropic_sky = False

    # Longwave based on spectral flux density (divide by pi)
    Ldown = np.zeros((patch_altitude.shape[0]))
    Lside = np.zeros((patch_altitude.shape[0]))
    Lnormal = np.zeros((patch_altitude.shape[0]))
    for altitude in skyalt:
        # Anisotropic sky
        if anisotropic_sky:
            temp_emissivity = esky_band[skyalt == altitude]
        # Isotropic sky but with patches (need to switch anisotropic_sky to False)
        else:
            temp_emissivity = esky
        # Estimate longwave radiation on a horizontal surface (Ldown), vertical surface (Lside) and perpendicular (Lnormal)
        Ldown[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / np.pi) * steradian[patch_altitude == altitude] * np.sin(altitude * deg2rad)
        Lside[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / np.pi) * steradian[patch_altitude == altitude] * np.cos(altitude * deg2rad)
        Lnormal[patch_altitude == altitude] = ((temp_emissivity * SBC * ((Ta + 273.15) ** 4)) / np.pi) * steradian[patch_altitude == altitude]

    Lsky_normal = deepcopy(sky_patches)
    Lsky_down = deepcopy(sky_patches)
    Lsky_side = deepcopy(sky_patches)

    Lsky_normal[:,2] = Lnormal
    Lsky_down[:,2] = Ldown
    Lsky_side[:,2] = Lside

    # Estimate longwave radiation in each patch based on patch characteristics, i.e. sky, vegetation or building (shaded or sunlit)
    Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, \
            Least_, Lwest_, Lnorth_, Lsouth_ = define_patch_characteristics(solar_altitude, solar_azimuth, 
                                 patch_altitude, patch_azimuth, steradian,
                                 asvf,
                                 shmat, vegshmat, vbshvegshmat,
                                 Lsky_down, Lsky_side, Lsky_normal, Lup,
                                 Ta, Tgwall, ewall,
                                 rows, cols)

    return Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_
    # return Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Lsky_normal, Lsky_down, Lsky_side, Least_, Lwest_, Lnorth_, Lsouth_


def Lside_veg_v2022a(svfS,svfW,svfN,svfE,svfEveg,svfSveg,svfWveg,svfNveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg,azimuth,altitude,Ta,Tw,SBC,ewall,Ldown,esky,t,F_sh,CI,LupE,LupS,LupW,LupN,anisotropic_longwave):

    # This m-file is the current one that estimates L from the four cardinal points 20100414
    
    #Building height angle from svf
    svfalfaE=np.arcsin(np.exp((np.log(1-svfE))/2))
    svfalfaS=np.arcsin(np.exp((np.log(1-svfS))/2))
    svfalfaW=np.arcsin(np.exp((np.log(1-svfW))/2))
    svfalfaN=np.arcsin(np.exp((np.log(1-svfN))/2))
    
    vikttot=4.4897
    aziW=azimuth+t
    aziN=azimuth-90+t
    aziE=azimuth-180+t
    aziS=azimuth-270+t
    
    F_sh = 2*F_sh-1  #(cylindric_wedge scaled 0-1)
    
    c=1-CI
    Lsky_allsky = esky*SBC*((Ta+273.15)**4)*(1-c)+c*SBC*((Ta+273.15)**4)
    
    ## Least
    [viktveg, viktwall, viktsky, viktrefl] = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)
    
    if altitude > 0:  # daytime
        alfaB=np.arctan(svfalfaE)
        betaB=np.arctan(np.tan((svfalfaE)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaE)*(1+F_sh)) #TODO This should be considered in future versions
        if (azimuth > (180-t))  and  (azimuth <= (360-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziE*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    
    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupE*0.5
        Least=Lground
    else:
        Lsky=((svfE+svfEveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupE*0.5
        Lrefl=(Ldown+LupE)*(viktrefl)*(1-ewall)*0.5
        Least=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lsouth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfS,svfSveg,svfSaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaS)
        betaB=np.arctan(np.tan((svfalfaS)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaS)*(1+F_sh))
        if (azimuth <= (90-t))  or  (azimuth > (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziS*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupS*0.5
        Lsouth=Lground
    else:
        Lsky=((svfS+svfSveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupS*0.5
        Lrefl=(Ldown+LupS)*(viktrefl)*(1-ewall)*0.5
        Lsouth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lwest
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfW,svfWveg,svfWaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaW)
        betaB=np.arctan(np.tan((svfalfaW)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaW)*(1+F_sh))
        if (azimuth > (360-t))  or  (azimuth <= (180-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziW*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupW*0.5
        Lwest=Lground
    else:
        Lsky=((svfW+svfWveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupW*0.5
        Lrefl=(Ldown+LupW)*(viktrefl)*(1-ewall)*0.5
        Lwest=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lnorth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfN,svfNveg,svfNaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaN)
        betaB=np.arctan(np.tan((svfalfaN)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaN)*(1+F_sh))
        if (azimuth > (90-t))  and  (azimuth <= (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziN*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    # Longwave from ground (see Lcyl_v2022a for remaining fluxes)
    if anisotropic_longwave == 1:
        Lground=LupN*0.5
        Lnorth=Lground
    else:
        Lsky=((svfN+svfNveg-1)*Lsky_allsky)*viktsky*0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupN*0.5
        Lrefl=(Ldown+LupN)*(viktrefl)*(1-ewall)*0.5
        Lnorth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    return Least, Lsouth, Lwest, Lnorth

def Lvikt_veg(svf,svfveg,svfaveg,vikttot):

    # Least
    viktonlywall=(vikttot-(63.227*svf**6-161.51*svf**5+156.91*svf**4-70.424*svf**3+16.773*svf**2-0.4863*svf))/vikttot

    viktaveg=(vikttot-(63.227*svfaveg**6-161.51*svfaveg**5+156.91*svfaveg**4-70.424*svfaveg**3+16.773*svfaveg**2-0.4863*svfaveg))/vikttot

    viktwall=viktonlywall-viktaveg

    svfvegbu=(svfveg+svf-1)  # Vegetation plus buildings
    viktsky=(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu)/vikttot
    viktrefl=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=viktveg-viktwall

    return viktveg,viktwall,viktsky,viktrefl


''' This function defines if a patch seen from a pixel is sky, building or vegetation. 
    It also calculates if a building patch is sunlit or shaded. From this it estimates 
    corresponding longwave radiation originating from each surface.'''

def define_patch_characteristics(solar_altitude, solar_azimuth,
                             patch_altitude, patch_azimuth, steradian,
                             asvf,
                             shmat, vegshmat, vbshvegshmat,
                             Lsky_down, Lsky_side, Lsky, Lup,
                             Ta, Tgwall, ewall,
                             rows, cols):
    
    # Stefan-Boltzmann's Constant
    SBC = 5.67051e-8

    # Degrees to radians
    deg2rad = np.pi / 180

    # Define variables
    Ldown = np.zeros((rows,cols))
    Ldown_sky = np.zeros((rows,cols))
    Ldown_veg = np.zeros((rows,cols))
    Ldown_sun = np.zeros((rows,cols))
    Ldown_sh = np.zeros((rows,cols))
    Ldown_ref = np.zeros((rows,cols))

    Lside = np.zeros((rows,cols))
    Lside_sky = np.zeros((rows,cols))
    Lside_veg = np.zeros((rows,cols))
    Lside_sun = np.zeros((rows,cols))
    Lside_sh = np.zeros((rows,cols))
    Lside_ref = np.zeros((rows,cols))

    Least = np.zeros((rows,cols))
    Lwest = np.zeros((rows,cols))
    Lnorth = np.zeros((rows,cols))
    Lsouth = np.zeros((rows,cols))

    # Define patch characteristics (sky, vegetation or building, and sunlit or shaded if building)
    for idx in range(patch_altitude.shape[0]):
        # Calculations for patches on sky, shmat = 1 = sky is visible
        temp_sky = ((shmat[:,:,idx] == 1) & (vegshmat[:,:,idx] == 1))
        
        # Longwave radiation from sky to vertical surface
        Ldown_sky += temp_sky * Lsky_down[idx,2]
        
        # Longwave radiation from sky to horizontal surface
        Lside_sky += temp_sky * Lsky_side[idx,2]

        # Calculations for patches that are vegetation, vegshmat = 0 = shade from vegetation
        temp_vegsh = ((vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
        # Longwave radiation from vegetation surface (considered vertical)
        vegetation_surface = ((ewall * SBC * ((Ta + 273.15) ** 4)) / np.pi)

        # Longwave radiation reaching a vertical surface
        Lside_veg += vegetation_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh
        
        # Longwave radiation reaching a horizontal surface
        Ldown_veg += vegetation_surface * steradian[idx] * np.sin(patch_altitude[idx] * deg2rad) * temp_vegsh
        
        # Portion into cardinal directions to be used for standing box or POI output
        if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
            Least += temp_sky * Lsky_side[idx,2] * np.cos((90 - patch_azimuth[idx]) * deg2rad)
            Least += vegetation_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((90 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
            Lsouth += temp_sky * Lsky_side[idx,2] * np.cos((180 - patch_azimuth[idx]) * deg2rad)
            Lsouth += vegetation_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((180 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
            Lwest += temp_sky * Lsky_side[idx,2] * np.cos((270 - patch_azimuth[idx]) * deg2rad)
            Lwest += vegetation_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((270 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
            Lnorth += temp_sky * Lsky_side[idx,2] * np.cos((0 - patch_azimuth[idx]) * deg2rad)
            Lnorth += vegetation_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_vegsh * np.cos((0 - patch_azimuth[idx]) * deg2rad)

        # Calculations for patches that are buildings, shmat = 0 = shade from buildings
        temp_vbsh = (1 - shmat[:,:,idx]) * vbshvegshmat[:,:,idx]
        temp_sh = (temp_vbsh == 1)
        azimuth_difference = np.abs(solar_azimuth - patch_azimuth[idx])

        # Longwave radiation from sunlit surfaces
        sunlit_surface = ((ewall * SBC * ((Ta + Tgwall + 273.15) ** 4)) / np.pi)
        # Longwave radiation from shaded surfaces
        shaded_surface = ((ewall * SBC * ((Ta + 273.15) ** 4)) / np.pi)
        if ((azimuth_difference > 90) and (azimuth_difference < 270) and (solar_altitude > 0)):
            # Calculate which patches defined as buildings that are sunlit or shaded
            sunlit_patches, shaded_patches = shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
            
            # Calculate longwave radiation from sunlit walls to vertical surface
            Lside_sun += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh
            # Calculate longwave radiation from shaded walls to vertical surface
            Lside_sh += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh
            
            # Calculate longwave radiation from sunlit walls to horizontal surface
            Ldown_sun += sunlit_surface * sunlit_patches * steradian[idx] * np.sin(patch_altitude[idx] * deg2rad) * temp_sh
            # Calculate longwave radiation from shaded walls to horizontal surface
            Ldown_sh += shaded_surface * shaded_patches * steradian[idx] * np.sin(patch_altitude[idx] * deg2rad) * temp_sh
            
            # Portion into cardinal directions to be used for standing box or POI output
            if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                Least += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx]) * deg2rad)
                Least += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                Lsouth += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx]) * deg2rad)
                Lsouth += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                Lwest += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx]) * deg2rad)
                Lwest += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                Lnorth += sunlit_surface * sunlit_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx]) * deg2rad)
                Lnorth += shaded_surface * shaded_patches * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx]) * deg2rad)

        else:
            # Calculate longwave radiation from shaded walls reaching a vertical surface
            Lside_sh += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh
            
            # Calculate longwave radiation from shaded walls reaching a horizontal surface
            Ldown_sh += shaded_surface * steradian[idx] * np.sin(patch_altitude[idx] * deg2rad) * temp_sh

            # Portion into cardinal directions to be used for standing box or POI output
            if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
                Least += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
                Lsouth += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
                Lwest += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx]) * deg2rad)
            if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
                Lnorth += shaded_surface * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx]) * deg2rad)

    # Calculate reflected longwave in each patch
    reflected_on_surfaces = (((Ldown_sky+Lup)*(1-ewall)*0.5) / np.pi)
    for idx in range(patch_altitude.shape[0]):
        temp_sh = ((shmat[:,:,idx] == 0) | (vegshmat[:,:,idx] == 0) | (vbshvegshmat[:,:,idx] == 0))
        
        # Reflected longwave radiation reaching vertical surfaces
        Lside_ref += reflected_on_surfaces * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh
        
        # Reflected longwave radiation reaching horizontal surfaces
        Ldown_ref += reflected_on_surfaces * steradian[idx] * np.sin(patch_altitude[idx] * deg2rad) * temp_sh
        
        # Portion into cardinal directions to be used for standing box or POI output
        if (patch_azimuth[idx] > 360) or (patch_azimuth[idx] < 180):
            Least += reflected_on_surfaces * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((90 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 90) and (patch_azimuth[idx] < 270):
            Lsouth += reflected_on_surfaces * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((180 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 180) and (patch_azimuth[idx] < 360):
            Lwest += reflected_on_surfaces * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((270 - patch_azimuth[idx]) * deg2rad)
        if (patch_azimuth[idx] > 270) or (patch_azimuth[idx] < 90):
            Lnorth += reflected_on_surfaces * steradian[idx] * np.cos(patch_altitude[idx] * deg2rad) * temp_sh * np.cos((0 - patch_azimuth[idx]) * deg2rad)

    # Sum of all Lside components (sky, vegetation, sunlit and shaded buildings, reflected)
    Lside = Lside_sky + Lside_veg + Lside_sh + Lside_sun + Lside_ref

    # Sum of all Lside components (sky, vegetation, sunlit and shaded buildings, reflected)
    Ldown = Ldown_sky + Ldown_veg + Ldown_sh + Ldown_sun + Ldown_ref

    return Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Least, Lwest, Lnorth, Lsouth


''' This function calculates whether a point is sunlit or shaded
    based on a sky view factor (in a cylinder), solar altitude, solar azimuth '''

def shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, asvf):

    # Patch azimuth in relation to sun azimuth
    patch_to_sun_azi = np.abs(solar_azimuth - patch_azimuth)

    # Degrees to radians
    deg2rad = np.pi/180
    
    # Radians to degrees
    rad2deg = 180/np.pi

    # 
    xi = np.cos(patch_to_sun_azi * deg2rad)

    # 
    yi = 2 * xi * np.tan(solar_altitude * deg2rad)

    hsvf = np.tan(asvf)

    if yi > 0:
        yi_ = 0
    else:
        yi_ = yi

    # 
    tan_delta = hsvf + yi_

    # Degrees where below is in shade and above is sunlit
    sunlit_degrees = np.arctan(tan_delta) * rad2deg

    # Boolean for pixels where patch is sunlit
    sunlit_patches = sunlit_degrees < patch_altitude
    # Boolean for pixels where patch is shaded
    shaded_patches = sunlit_degrees > patch_altitude

    return sunlit_patches, shaded_patches


def sunonsurface_2018a(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta,
                       emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):
    # This version of sunonsurfce goes with SOLWEIG 2015a. It also simulates
    # Lup and albedo based on landcover information and shadow patterns.
    # Fredrik Lindberg, fredrikl@gvc.gu.se

    sizex = np.shape(walls)[0]
    sizey = np.shape(walls)[1]

    # sizex=size(buildings,1);sizey=size(buildings,2);
    wallbol = (walls > 0) * 1
    sunwall[sunwall > 0] = 1  # test 20160910

    # conversion into radians
    azimuth = azimuthA * (np.pi / 180)

    # loop parameters
    index = 0
    f = buildings
    Lup = SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4 - SBC * emis_grid * (Ta + 273.15) ** 4  # +Ta
    if landcover == 1:
        Tg[lc_grid == 3] = Twater - Ta  # Setting water temperature

    Lwall = SBC * ewall * (Tgwall + Ta + 273.15) ** 4 - SBC * ewall * (Ta + 273.15) ** 4  # +Ta
    albshadow = alb_grid * shadow
    alb = alb_grid
    # sh(sh<=0.1)=0;
    # sh=sh-(1-vegsh)*(1-psi);
    # shadow=sh-(1-vegsh)*(1-psi);
    # dx=0;
    # dy=0;
    # ds=0; ##ok<NASGU>

    tempsh = np.zeros((sizex, sizey))
    tempbu = np.zeros((sizex, sizey))
    tempbub = np.zeros((sizex, sizey))
    tempbubwall = np.zeros((sizex, sizey))
    tempwallsun = np.zeros((sizex, sizey))
    weightsumsh = np.zeros((sizex, sizey))
    weightsumwall = np.zeros((sizex, sizey))
    first = np.round(first * scale)
    if first < 1:
        first = 1
    second = np.round(second * scale)
    # tempTgsh=tempsh;
    weightsumLupsh = np.zeros((sizex, sizey))
    weightsumLwall = np.zeros((sizex, sizey))
    weightsumalbsh = np.zeros((sizex, sizey))
    weightsumalbwall = np.zeros((sizex, sizey))
    weightsumalbnosh = np.zeros((sizex, sizey))
    weightsumalbwallnosh = np.zeros((sizex, sizey))
    tempLupsh = np.zeros((sizex, sizey))
    tempalbsh = np.zeros((sizex, sizey))
    tempalbnosh = np.zeros((sizex, sizey))

    # other loop parameters
    pibyfour = np.pi / 4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)

    ## The Shadow casting algoritm
    for n in np.arange(0, second):
        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or (
                fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            # ds = dssin
        else:
            dy = signsinazimuth * abs(round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            # ds = dscos

        absdx = np.abs(dx)
        absdy = np.abs(dy)

        xc1 = ((dx + absdx) / 2)
        xc2 = (sizex + (dx - absdx) / 2)
        yc1 = ((dy + absdy) / 2)
        yc2 = (sizey + (dy - absdy) / 2)

        xp1 = -((dx - absdx) / 2)
        xp2 = (sizex - (dx + absdx) / 2)
        yp1 = -((dy - absdy) / 2)
        yp2 = (sizey - (dy + absdy) / 2)

        tempbu[int(xp1):int(xp2), int(yp1):int(yp2)] = buildings[int(xc1):int(xc2),
                                                       int(yc1):int(yc2)]  # moving building
        tempsh[int(xp1):int(xp2), int(yp1):int(yp2)] = shadow[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving shadow
        tempLupsh[int(xp1):int(xp2), int(yp1):int(yp2)] = Lup[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving Lup/shadow
        tempalbsh[int(xp1):int(xp2), int(yp1):int(yp2)] = albshadow[int(xc1):int(xc2),
                                                          int(yc1):int(yc2)]  # moving Albedo/shadow
        tempalbnosh[int(xp1):int(xp2), int(yp1):int(yp2)] = alb[int(xc1):int(xc2), int(yc1):int(yc2)]  # moving Albedo
        f = np.min([f, tempbu], axis=0)  # utsmetning av buildings

        shadow2 = tempsh * f
        weightsumsh = weightsumsh + shadow2

        Lupsh = tempLupsh * f
        weightsumLupsh = weightsumLupsh + Lupsh

        albsh = tempalbsh * f
        weightsumalbsh = weightsumalbsh + albsh

        albnosh = tempalbnosh * f
        weightsumalbnosh = weightsumalbnosh + albnosh

        tempwallsun[int(xp1):int(xp2), int(yp1):int(yp2)] = sunwall[int(xc1):int(xc2),
                                                            int(yc1):int(yc2)]  # moving buildingwall insun image
        tempb = tempwallsun * f
        tempbwall = f * -1 + 1
        tempbub = ((tempb + tempbub) > 0) * 1
        tempbubwall = ((tempbwall + tempbubwall) > 0) * 1
        weightsumLwall = weightsumLwall + tempbub * Lwall
        weightsumalbwall = weightsumalbwall + tempbub * albedo_b
        weightsumwall = weightsumwall + tempbub
        weightsumalbwallnosh = weightsumalbwallnosh + tempbubwall * albedo_b

        ind = 1
        if (n + 1) <= first:
            weightsumwall_first = weightsumwall / ind
            weightsumsh_first = weightsumsh / ind
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = (weightsumLwall) / ind  # *Lwall
            weightsumLupsh_first = weightsumLupsh / ind

            weightsumalbwall_first = weightsumalbwall / ind  # *albedo_b
            weightsumalbsh_first = weightsumalbsh / ind
            weightsumalbwallnosh_first = weightsumalbwallnosh / ind  # *albedo_b
            weightsumalbnosh_first = weightsumalbnosh / ind
            wallinfluence_first = weightsumalbwallnosh_first > 0
            #         gvf1=(weightsumwall+weightsumsh)/first;
            #         gvf1(gvf1>1)=1;
            ind += 1
        index += 1

    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0
    # gvf2(gvf2>1)=1;

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth - np.pi / 2
    azihigh = azimuth + np.pi / 2
    if azilow >= 0 and azihigh < 2 * np.pi:  # 90 to 270  (SHADOW)
        facesh = (np.logical_or(aspect < azilow, aspect >= azihigh).astype(float) - wallbol + 1)
    elif azilow < 0 and azihigh <= 2 * np.pi:  # 0 to 90
        azilow = azilow + 2 * np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)    # check for the -1
    elif azilow > 0 and azihigh >= 2 * np.pi:  # 270 to 360
        azihigh = azihigh - 2 * np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1  # (SHADOW)

    # removing walls in self shadoing
    keep = (weightsumwall == second) - facesh
    keep[keep == -1] = 0

    # gvf from shadow only
    gvf1 = ((weightsumwall_first + weightsumsh_first) / (first + 1)) * wallsuninfluence_first + \
           (weightsumsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumwall[keep == 1] = 0
    gvf2 = ((weightsumwall + weightsumsh) / (second + 1)) * wallsuninfluence_second + \
           (weightsumsh) / (second) * (wallsuninfluence_second * -1 + 1)

    gvf2[gvf2 > 1.] = 1.

    # gvf from shadow and Lup
    gvfLup1 = ((weightsumLwall_first + weightsumLupsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumLupsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumLwall[keep == 1] = 0
    gvfLup2 = ((weightsumLwall + weightsumLupsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumLupsh) / (second) * (wallsuninfluence_second * -1 + 1)

    # gvf from shadow and albedo
    gvfalb1 = ((weightsumalbwall_first + weightsumalbsh_first) / (first + 1)) * wallsuninfluence_first + \
              (weightsumalbsh_first) / (first) * (wallsuninfluence_first * -1 + 1)
    weightsumalbwall[keep == 1] = 0
    gvfalb2 = ((weightsumalbwall + weightsumalbsh) / (second + 1)) * wallsuninfluence_second + \
              (weightsumalbsh) / (second) * (wallsuninfluence_second * -1 + 1)

    # gvf from albedo only
    gvfalbnosh1 = ((weightsumalbwallnosh_first + weightsumalbnosh_first) / (first + 1)) * wallinfluence_first + \
                  (weightsumalbnosh_first) / (first) * (wallinfluence_first * -1 + 1)  #
    gvfalbnosh2 = ((weightsumalbwallnosh + weightsumalbnosh) / (second)) * wallinfluence_second + \
                  (weightsumalbnosh) / (second) * (wallinfluence_second * -1 + 1)

    # Weighting
    gvf = (gvf1 * 0.5 + gvf2 * 0.4) / 0.9
    gvfLup = (gvfLup1 * 0.5 + gvfLup2 * 0.4) / 0.9
    gvfLup = gvfLup + ((SBC * emis_grid * (Tg * shadow + Ta + 273.15) ** 4) - SBC * emis_grid * (Ta + 273.15) ** 4) * (
                buildings * -1 + 1)  # +Ta
    gvfalb = (gvfalb1 * 0.5 + gvfalb2 * 0.4) / 0.9
    gvfalb = gvfalb + alb_grid * (buildings * -1 + 1) * shadow
    gvfalbnosh = (gvfalbnosh1 * 0.5 + gvfalbnosh2 * 0.4) / 0.9
    gvfalbnosh = gvfalbnosh * buildings + alb_grid * (buildings * -1 + 1)

    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2


def Tgmaps_v1(lc_grid, lc_class):

    #Tgmaps_v1 Populates grids with cooeficients for Tg wave
    #   Detailed explanation goes here

    id = np.unique(lc_grid)
    TgK = np.copy(lc_grid)
    Tstart = np.copy(lc_grid)
    alb_grid = np.copy(lc_grid)
    emis_grid = np.copy(lc_grid)
    TmaxLST = np.copy(lc_grid)

    for i in np.arange(0, id.__len__()):
        row = (lc_class[:, 0] == id[i])
        Tstart[Tstart == id[i]] = lc_class[row, 4]
        alb_grid[alb_grid == id[i]] = lc_class[row, 1]
        emis_grid[emis_grid == id[i]] = lc_class[row, 2]
        TmaxLST[TmaxLST == id[i]] = lc_class[row, 5]
        TgK[TgK == id[i]] = lc_class[row, 3]

    wall_pos = np.where(lc_class[:, 0] == 99)
    TgK_wall = lc_class[wall_pos, 3]
    Tstart_wall = lc_class[wall_pos, 4]
    TmaxLST_wall = lc_class[wall_pos, 5]

    return TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall




def TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1):

    Tgmap0 = gvfLup  # current timestep
    if firstdaytime == 1:  # "first in morning"
        Tgmap1 = Tgmap0

    if timeadd >= (59/1440):  # more or equal to 59 min
        weight1 = np.exp(-33.27 * timeadd)  # surface temperature delay function - 1 step
        Tgmap1 = Tgmap0 * (1 - weight1) + Tgmap1 * weight1
        Lup = Tgmap1
        if timestepdec > (59/1440):
            timeadd = timestepdec
        else:
            timeadd = 0
    else:
        timeadd = timeadd + timestepdec
        weight1 = np.exp(-33.27 * timeadd)  # surface temperature delay function - 1 step
        Lup = (Tgmap0 * (1 - weight1) + Tgmap1 * weight1)

    return Lup, timeadd, Tgmap1

def clearnessindex_2013b(zen, jday, Ta, RH, radG, location, P):

    """ Clearness Index at the Earth's surface calculated from Crawford and Duchon 1999

    :param zen: zenith angle in radians
    :param jday: day of year
    :param Ta: air temperature
    :param RH: relative humidity
    :param radG: global shortwave radiation
    :param location: distionary including lat, lon and alt
    :param P: pressure
    :return:
    """

    if P == -999.0:
        p = 1013.  # Pressure in millibars
    else:
        p = P*10.  # Convert from hPa to millibars

    Itoa = 1370.0  # Effective solar constant
    D = sun_distance(jday)  # irradiance differences due to Sun-Earth distances
    m = 35. * np.cos(zen) * ((1224. * (np.cos(zen)**2) + 1) ** (-1/2.))     # optical air mass at p=1013
    Trpg = 1.021-0.084*(m*(0.000949*p+0.051))**0.5  # Transmission coefficient for Rayliegh scattering and permanent gases

    # empirical constant depending on latitude
    if location['latitude'] < 10.:
        G = [3.37, 2.85, 2.80, 2.64]
    elif location['latitude'] >= 10. and location['latitude'] < 20.:
        G = [2.99, 3.02, 2.70, 2.93]
    elif location['latitude'] >= 20. and location['latitude'] <30.:
        G = [3.60, 3.00, 2.98, 2.93]
    elif location['latitude'] >= 30. and location['latitude'] <40.:
        G = [3.04, 3.11, 2.92, 2.94]
    elif location['latitude'] >= 40. and location['latitude'] <50.:
        G = [2.70, 2.95, 2.77, 2.71]
    elif location['latitude'] >= 50. and location['latitude'] <60.:
        G = [2.52, 3.07, 2.67, 2.93]
    elif location['latitude'] >= 60. and location['latitude'] <70.:
        G = [1.76, 2.69, 2.61, 2.61]
    elif location['latitude'] >= 70. and location['latitude'] <80.:
        G = [1.60, 1.67, 2.24, 2.63]
    elif location['latitude'] >= 80. and location['latitude'] <90.:
        G = [1.11, 1.44, 1.94, 2.02]

    if jday > 335 or jday <= 60:
        G = G[0]
    elif jday > 60 and jday <= 152:
        G = G[1]
    elif jday > 152 and jday <= 244:
        G = G[2]
    elif jday > 244 and jday <= 335:
        G = G[3]

    # dewpoint calculation
    a2 = 17.27
    b2 = 237.7
    Td = (b2*(((a2*Ta)/(b2+Ta))+np.log(RH)))/(a2-(((a2*Ta)/(b2+Ta))+np.log(RH)))
    Td = (Td*1.8)+32  # Dewpoint (F)
    u = np.exp(0.1133-np.log(G+1)+0.0393*Td)  # Precipitable water
    Tw = 1-0.077*((u*m)**0.3)  # Transmission coefficient for water vapor
    Tar = 0.935**m  # Transmission coefficient for aerosols

    I0=Itoa*np.cos(zen)*Trpg*Tw*D*Tar
    if abs(zen)>np.pi/2:
        I0 = 0
    # b=I0==abs(zen)>np.pi/2
    # I0(b==1)=0
    # clear b;
    if not(np.isreal(I0)):
        I0 = 0

    corr=0.1473*np.log(90-(zen/np.pi*180))+0.3454  # 20070329

    CIuncorr = radG / I0
    CI = CIuncorr + (1-corr)
    I0et = Itoa*np.cos(zen)*D  # extra terrestial solar radiation
    Kt = radG / I0et
    if math.isnan(CI):
        CI = float('Inf')

    return I0, CI, Kt, I0et, CIuncorr


def create_patches(patch_option):

    deg2rad = np.pi/180

    # patch_option = 1 = 145 patches (Robinson & Stone, 2004)
    # patch_option = 2 = 153 patches (Wallenberg et al., 2022)
    # patch_option = 3 = 306 patches -> test 
    # patch_option = 4 = 612 patches -> test

    skyvaultalt = np.atleast_2d([])
    skyvaultazi = np.atleast_2d([])
            
    # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
    # Patch option 1, 145 patches, Original Robinson & Stone (2004) after Tregenza (1987)/Tregenza & Sharples (1993)
    if patch_option == 1:
        annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
        skyvaultaltint = np.array([6, 18, 30, 42, 54, 66, 78, 90]) # Robinson & Stone (2004)
        azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0]) # Fredrik/Nils
        patches_in_band = np.array([30, 30, 24, 24, 18, 12, 6, 1]) # Robinson & Stone (2004)
    # Patch option 1, 153 patches, Wallenberg et al. (2022)
    elif patch_option == 2:
        annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
        skyvaultaltint = np.array([6, 18, 30, 42, 54, 66, 78, 90]) # Robinson & Stone (2004)
        azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0]) # Fredrik/Nils
        patches_in_band = np.array([31, 30, 28, 24, 19, 13, 7, 1]) # Nils
    # Patch option 2, 306 patches, test
    elif patch_option == 3:
        annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
        skyvaultaltint = np.array([6, 18, 30, 42, 54, 66, 78, 90]) # Robinson & Stone (2004)
        azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0]) # Fredrik/Nils
        patches_in_band = np.array([31*2, 30*2, 28*2, 24*2, 19*2, 13*2, 7*2, 1]) # Nils
    # Patch option 3, 612 patches, test
    elif patch_option == 4:
        annulino = np.array([0, 4.5, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90]) # Nils
        skyvaultaltint = np.array([3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90]) # Nils
        patches_in_band = np.array([31*2, 31*2, 30*2, 30*2, 28*2, 28*2, 24*2, 24*2, 19*2, 19*2, 13*2, 13*2, 7*2, 7*2, 1]) # Nils
        azistart = np.array([0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0]) # Nils

    skyvaultaziint = np.array([360/patches for patches in patches_in_band])

    for j in range(0, skyvaultaltint.shape[0]):
        for k in range(0, patches_in_band[j]):
            skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])
            skyvaultazi = np.append(skyvaultazi, k*skyvaultaziint[j] + azistart[j])

    # skyvaultzen = (90 - skyvaultalt) * deg2rad
    # skyvaultalt = skyvaultalt * deg2rad
    # skyvaultazi = skyvaultazi * deg2rad

    return skyvaultalt, skyvaultazi, annulino, skyvaultaltint, patches_in_band, skyvaultaziint, azistart


def diffusefraction(radG,altitude,Kt,Ta,RH):
    """
    This function estimates diffuse and directbeam radiation according to
    Reindl et al (1990), Solar Energy 45:1

    :param radG:
    :param altitude:
    :param Kt:
    :param Ta:
    :param RH:
    :return:
    """

    alfa = altitude*(np.pi/180)

    if Ta <= -999.00 or RH <= -999.00 or np.isnan(Ta) or np.isnan(RH):
        if Kt <= 0.3:
            radD = radG*(1.020-0.248*Kt)
        elif Kt > 0.3 and Kt < 0.78:
            radD = radG*(1.45-1.67*Kt)
        else:
            radD = radG*0.147
    else:
        RH = RH/100
        if Kt <= 0.3:
            radD = radG*(1 - 0.232 * Kt + 0.0239 * np.sin(alfa) - 0.000682 * Ta + 0.0195 * RH)
        elif Kt > 0.3 and Kt < 0.78:
            radD = radG*(1.329- 1.716 * Kt + 0.267 * np.sin(alfa) - 0.00357 * Ta + 0.106 * RH)
        else:
            radD = radG*(0.426 * Kt - 0.256 * np.sin(alfa) + 0.00349 * Ta + 0.0734 * RH)

    radI = (radG - radD)/(np.sin(alfa))

    # Corrections for low sun altitudes (20130307)
    if radI < 0:
        radI = 0

    if altitude < 1 and radI > radG:
        radI=radG

    if radD > radG:
        radD = radG

    return radI, radD



def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice, patch_option):
    """
    This function calculates distribution of luminance on the skyvault based on
    Perez luminince distribution model.
    
    Created by:
    Fredrik Lindberg 20120527, fredrikl@gvc.gu.se
    Gothenburg University, Sweden
    Urban Climte Group
    
    Input parameters:
     - zen:     Zenith angle of the Sun (in degrees)
     - azimuth: Azimuth angle of the Sun (in degrees)
     - radD:    Horizontal diffuse radiation (W m-2)
     - radI:    Direct radiation perpendicular to the Sun beam (W m-2)
     - jday:    Day of year
    
    Output parameters:
     - lv:   Relative luminance map (same dimensions as theta. gamma)
    

    acoeff=[1.353 -0.258 -0.269 -1.437
           -1.222 -0.773 1.415 1.102
           -1.100 -0.252 0.895 0.016
           -0.585 -0.665 -0.267 0.712
           -0.600 -0.347 -2.500 2.323
           -1.016 -0.367 1.008 1.405
           -1.000 0.021 0.503 -0.512
           -1.050 0.029 0.426 0.359];
    
    bcoeff=[-0.767 0.001 1.273 -0.123
            -0.205 0.037 -3.913 0.916
             0.278 -0.181 -4.500 1.177
             0.723 -0.622 -5.681 2.630
             0.294 0.049 -5.681 1.842
             0.288 -0.533 -3.850 3.375
            -0.300 0.192 0.702 -1.632
            -0.325 0.116 0.778 0.003];
    
    ccoeff=[2.800 0.600 1.238 1.000
            6.975 0.177 6.448 -0.124
            24.22 -13.08 -37.70 34.84
            33.34 -18.30 -62.25 52.08
            21.00 -4.766 -21.59 7.249
            14.00 -0.999 -7.14 7.547
            19.00 -5.000 1.243 -1.91
            31.06 -14.50 -46.11 55.37];
    
    dcoeff=[1.874 0.630 0.974 0.281
           -1.580 -0.508 -1.781 0.108
           -5.00 1.522 3.923 -2.62
           -3.50 0.002 1.148 0.106
           -3.50 -0.155 1.406 0.399
           -3.40 -0.108 -1.075 1.57
           -4.00 0.025 0.384 0.266
           -7.23 0.405 13.35 0.623];
    
    ecoeff=[0.035 -0.125 -0.572 0.994
            0.262 0.067 -0.219 -0.428
           -0.016 0.160 0.420 -0.556
            0.466 -0.33 -0.088 -0.033
            0.003 0.077 -0.066 -0.129
           -0.067 0.402 0.302 -0.484
            1.047 -0.379 -2.452 1.466
            1.500 -0.643 1.856 0.564];

    :param zen:
    :param azimuth:
    :param radD:
    :param radI:
    :param jday:
    :param patchchoice:
    :return:
    """

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

# -*- coding: utf-8 -*-
# from scipy.ndimage.filters import median_filter


def shadowingfunction_wallheight_13(a, azimuth, altitude, scale, walls, aspect):
    """
    This m.file calculates shadows on a DSM and shadow height on building
    walls.
    
    INPUTS:
    a = DSM
    azimuth and altitude = sun position
    scale= scale of DSM (1 meter pixels=1, 2 meter pixels=0.5)
    walls= pixel row 'outside' buildings. will be calculated if empty
    aspect = normal aspect of buildings walls
    
    OUTPUT:
    sh=ground and roof shadow
    wallsh = height of wall that is in shadow
    wallsun = hieght of wall that is in sun
    
    Fredrik Lindberg 2012-03-19
    fredrikl@gvc.gu.se
    
     Utdate 2013-03-13 - bugfix for walls alinged with sun azimuths

    :param a:
    :param azimuth:
    :param altitude:
    :param scale:
    :param walls:
    :param aspect:
    :return:
    """

    if not walls.size:
        """
        walls = ordfilt2(a,4,[0 1 0; 1 0 1; 0 1 0])
        walls = walls-a
        walls[walls < 3]=0
        sizex = np.shape(a)[0]    #might be wrong
        sizey = np.shape(a)[1]
        dirwalls = filter1Goodwin_as_aspect_v3(walls,sizex,sizey,scale,a);
        aspect = dirwalls*np.pi/180
        """

    # conversion
    # degrees = np.pi/180
    azimuth = radians(azimuth)
    altitude = radians(altitude)

    # measure the size of the image
    sizex = np.shape(a)[0]
    sizey = np.shape(a)[1]

    # initialise parameters
    f = np.copy(a)
    dx = 0
    dy = 0
    dz = 0
    temp = np.zeros((sizex, sizey))
    wallbol = (walls > 0).astype(float)
    
    # other loop parameters
    amaxvalue = np.max(a)
    pibyfour = np.pi/4
    threetimespibyfour = 3 * pibyfour
    fivetimespibyfour = 5 * pibyfour
    seventimespibyfour = 7 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1/sinazimuth)
    dscos = np.abs(1/cosazimuth)
    tanaltitudebyscale = np.tan(altitude)/scale

    index = 1

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):

        if (pibyfour <= azimuth and azimuth < threetimespibyfour) or \
                (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            ds = dscos

        # note: dx and dy represent absolute values while ds is an incremental value
        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0

        absdx = np.abs(dx)
        absdy = np.abs(dy)

        xc1 = int((dx+absdx)/2)
        xc2 = int(sizex+(dx-absdx)/2)
        yc1 = int((dy+absdy)/2)
        yc2 = int(sizey+(dy-absdy)/2)

        xp1 = int(-((dx-absdx)/2))
        xp2 = int(sizex-(dx+absdx)/2)
        yp1 = int(-((dy-absdy)/2))
        yp2 = int(sizey-(dy+absdy)/2)

        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
        f = np.fmax(f, temp) #Moving building shadow

        index = index + 1

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth-np.pi/2
    azihigh = azimuth+np.pi/2

    if azilow >= 0 and azihigh < 2*np.pi:    # 90 to 270  (SHADOW)
        facesh = (np.logical_or(aspect < azilow, aspect >= azihigh).astype(float)-wallbol+1)
    elif azilow < 0 and azihigh <= 2*np.pi:    # 0 to 90
        azilow = azilow + 2*np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1    # (SHADOW)    # check for the -1
    elif azilow > 0 and azihigh >= 2*np.pi:    # 270 to 360
        azihigh = azihigh-2*np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh)*-1 + 1    # (SHADOW)

    sh = np.copy(f-a)    # shadow volume
    facesun = np.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = np.copy(walls-sh)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0    # Removing walls in "self"-shadow
    wallsh = np.copy(walls-wallsun)

    sh = np.logical_not(np.logical_not(sh)).astype(float)
    sh = sh * -1 + 1

    return sh, wallsh, wallsun, facesh, facesun


# import matplotlib.pylab as plt
def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
    """
    This function calculates shadows on a DSM and shadow height on building
    walls including both buildings and vegetion units.
    New functionallity to deal with pergolas, August 2021
    
    INPUTS:
    a = DSM
    vegdem = Vegetation canopy DSM (magl)
    vegdem2 = Trunkzone DSM (magl)
    azimuth and altitude = sun position
    scale= scale of DSM (1 meter pixels=1, 2 meter pixels=0.5)
    walls= pixel row 'outside' buildings. will be calculated if empty
    aspect=normal aspect of walls
    
    OUTPUT:
    sh=ground and roof shadow

    wallsh = height of wall that is in shadow
    wallsun = hieght of wall that is in sun

    original Matlab code:
    Fredrik Lindberg 2013-08-14
    fredrikl@gvc.gu.se

    :param a:
    :param vegdem:
    :param vegdem2:
    :param azimuth:
    :param altitude:
    :param scale:
    :param amaxvalue:
    :param bush:
    :param walls:
    :param aspect:
    :return:
    """

    # conversion
    degrees = np.pi/180.
    azimuth *= degrees
    altitude *= degrees
    
    # measure the size of the image
    sizex = np.shape(a)[0]
    sizey = np.shape(a)[1]
    
    # initialise parameters
    dx = 0
    dy = 0
    dz = 0
    temp = np.zeros((sizex, sizey))
    tempvegdem = np.zeros((sizex, sizey))
    tempvegdem2 = np.zeros((sizex, sizey))
    templastfabovea = np.zeros((sizex, sizey))
    templastgabovea = np.zeros((sizex, sizey))
    bushplant = bush > 1
    sh = np.zeros((sizex, sizey)) #shadows from buildings
    vbshvegsh = np.copy(sh) #vegetation blocking buildings
    vegsh = np.add(np.zeros((sizex, sizey)), bushplant, dtype=float) #vegetation shadow
    f = np.copy(a)
    shvoveg = np.copy(vegdem) # for vegetation shadowvolume
    # g = np.copy(sh)
    wallbol = (walls > 0).astype(float)

    # other loop parameters
    pibyfour = np.pi/4
    threetimespibyfour = 3*pibyfour
    fivetimespibyfour = 5*pibyfour
    seventimespibyfour = 7*pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1/sinazimuth)
    dscos = np.abs(1/cosazimuth)
    tanaltitudebyscale = np.tan(altitude)/scale

    index = 0

    # new case with pergola (thin vertical layer of vegetation), August 2021
    dzprev = 0

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if ((pibyfour <= azimuth) and (azimuth < threetimespibyfour)) or ((fivetimespibyfour <= azimuth) and (azimuth < seventimespibyfour)):
            dy = signsinazimuth * index
            dx = -1 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1 * signcosazimuth * index
            ds = dscos

        # note: dx and dy represent absolute values while ds is an incremental value
        dz = (ds * index) * tanaltitudebyscale
        tempvegdem[0:sizex, 0:sizey] = 0
        tempvegdem2[0:sizex, 0:sizey] = 0
        temp[0:sizex, 0:sizey] = 0
        templastfabovea[0:sizex, 0:sizey] = 0.
        templastgabovea[0:sizex, 0:sizey] = 0.
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx+absdx)/2)
        xc2 = int(sizex+(dx-absdx)/2)
        yc1 = int((dy+absdy)/2)
        yc2 = int(sizey+(dy-absdy)/2)
        xp1 = -int((dx-absdx)/2)
        xp2 = int(sizex-(dx+absdx)/2)
        yp1 = -int((dy-absdy)/2)
        yp2 = int(sizey-(dy+absdy)/2)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2]-dz

        f = np.fmax(f, temp) #Moving building shadow
        shvoveg = np.fmax(shvoveg, tempvegdem) # moving vegetation shadow volume
        sh[f > a] = 1
        sh[f <= a] = 0   
        fabovea = (tempvegdem > a).astype(int)   #vegdem above DEM
        gabovea = (tempvegdem2 > a).astype(int)   #vegdem2 above DEM
        
        #new pergola condition
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2]-dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2]-dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = np.add(np.add(np.add(fabovea, gabovea, dtype=float),lastfabovea, dtype=float),lastgabovea, dtype=float)
        vegsh2[vegsh2 == 4] = 0.
        # vegsh2[vegsh2 == 1] = 0. # This one is the ultimate question...
        vegsh2[vegsh2 > 0] = 1.

        # vegsh2 = fabovea - gabovea #old without pergolas
        # vegsh = np.max([vegsh, vegsh2], axis=0) #old without pergolas

        vegsh = np.fmax(vegsh, vegsh2)
        vegsh[vegsh*sh > 0] = 0    
        vbshvegsh = np.copy(vegsh) + vbshvegsh # removing shadows 'behind' buildings

        # # vegsh at high sun altitudes # Not needed when pergolas are included
        # if index == 0:
        #     firstvegdem = np.copy(tempvegdem) - np.copy(temp)
        #     firstvegdem[firstvegdem <= 0] = 1000
        #     vegsh[firstvegdem < dz] = 1
        #     vegsh *= (vegdem2 > a)
        #     vbshvegsh = np.zeros((sizex, sizey))

        # # Bush shadow on bush plant # Not needed when pergolas are included
        # if np.max(bush) > 0 and np.max(fabovea*bush) > 0:
        #     tempbush = np.zeros((sizex, sizey))
        #     tempbush[int(xp1):int(xp2), int(yp1):int(yp2)] = bush[int(xc1):int(xc2), int(yc1):int(yc2)] - dz
        #     g = np.max([g, tempbush], axis=0)
        #     g = bushplant * g
    
        index += 1

    # Removing walls in shadow due to selfshadowing
    azilow = azimuth - np.pi/2
    azihigh = azimuth + np.pi/2
    if azilow >= 0 and azihigh < 2*np.pi:    # 90 to 270  (SHADOW)
        facesh = np.logical_or(aspect < azilow, aspect >= azihigh).astype(float) - wallbol + 1    # TODO check
    elif azilow < 0 and azihigh <= 2*np.pi:    # 0 to 90
        azilow = azilow + 2*np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh) * -1 + 1    # (SHADOW)
    elif azilow > 0 and azihigh >= 2*np.pi:    # 270 to 360
        azihigh -= 2 * np.pi
        facesh = np.logical_or(aspect > azilow, aspect <= azihigh)*-1 + 1    # (SHADOW)

    sh = 1-sh
    vbshvegsh[vbshvegsh > 0] = 1
    vbshvegsh = vbshvegsh-vegsh
    
    # if np.max(bush) > 0: # Not needed when pergolas are included
    #     g = g-bush
    #     g[g > 0] = 1
    #     g[g < 0] = 0
    #     vegsh = vegsh-bushplant+g
    #     vegsh[vegsh < 0] = 0

    vegsh[vegsh > 0] = 1
    shvoveg = (shvoveg-a) * vegsh    #Vegetation shadow volume
    vegsh = 1-vegsh
    vbshvegsh = 1-vbshvegsh
    
    # wall shadows
    shvo = f - a   # building shadow volume
    facesun = np.logical_and(facesh + (walls > 0).astype(float) == 1, walls > 0).astype(float)
    wallsun = np.copy(walls-shvo)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0    # Removing walls in "self"-shadow
    wallsh = np.copy(walls-wallsun)

    wallshve = shvoveg * wallbol
    wallshve = wallshve - wallsh
    wallshve[wallshve < 0] = 0
    id = np.where(wallshve > walls)
    wallshve[id] = walls[id]
    wallsun = wallsun-wallshve    # problem with wallshve only
    id = np.where(wallsun < 0)
    wallshve[id] = 0
    wallsun[id] = 0
    
    return vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun


# from importdata import importdata
#import sun_position as sp

def Solweig_2015a_metdata_noload(inputdata, location, UTC):
    """
    This function is used to process the input meteorological file.
    It also calculates Sun position based on the time specified in the met-file

    :param inputdata:
    :param location:
    :param UTC:
    :return:
    """

    met = inputdata
    data_len = len(met[:, 0])
    dectime = met[:, 1]+met[:, 2] / 24 + met[:, 3] / (60*24.)
    dectimemin = met[:, 3] / (60*24.)
    if data_len == 1:
        halftimestepdec = 0
    else:
        halftimestepdec = (dectime[1] - dectime[0]) / 2.
    time = dict()
    time['sec'] = 0
    time['UTC'] = UTC
    sunmaximum = 0.
    leafon1 = 97  #TODO this should change
    leafoff1 = 300  #TODO this should change

    # initialize matrices
    altitude = np.empty(shape=(1, data_len))
    azimuth = np.empty(shape=(1, data_len))
    zen = np.empty(shape=(1, data_len))
    jday = np.empty(shape=(1, data_len))
    YYYY = np.empty(shape=(1, data_len))
    leafon = np.empty(shape=(1, data_len))
    altmax = np.empty(shape=(1, data_len))

    sunmax = dict()

    for i, row in enumerate(met[:, 0]):
        if met[i, 1] == 221:
            test = 4
        YMD = datetime.datetime(int(met[i, 0]), 1, 1) + datetime.timedelta(int(met[i, 1]) - 1)
        # Finding maximum altitude in 15 min intervals (20141027)
        if (i == 0) or (np.mod(dectime[i], np.floor(dectime[i])) == 0):
            fifteen = 0.
            sunmaximum = -90.
            sunmax['zenith'] = 90.
            while sunmaximum <= 90. - sunmax['zenith']:
                sunmaximum = 90. - sunmax['zenith']
                fifteen = fifteen + 15. / 1440.
                HM = datetime.timedelta(days=(60*10)/1440.0 + fifteen)
                YMDHM = YMD + HM
                time['year'] = YMDHM.year
                time['month'] = YMDHM.month
                time['day'] = YMDHM.day
                time['hour'] = YMDHM.hour
                time['min'] = YMDHM.minute
                sunmax = sun_position(time,location)
                sunmax = {k: np.asarray(v).item() for k, v in sunmax.items()}
        altmax[0, i] = sunmaximum

        half = datetime.timedelta(days=halftimestepdec)
        H = datetime.timedelta(hours=met[i, 2])
        M = datetime.timedelta(minutes=met[i, 3])
        YMDHM = YMD + H + M - half
        time['year'] = YMDHM.year
        time['month'] = YMDHM.month
        time['day'] = YMDHM.day
        time['hour'] = YMDHM.hour
        time['min'] = YMDHM.minute
        sun = sun_position(time, location)
        sun = {k: np.asarray(v).item() for k, v in sun.items()}
        if 89.0 < sun['zenith'] <= 90.0:
            sun['zenith'] = 89.0
        altitude[0, i] = 90. - sun['zenith']
        azimuth[0, i] = sun['azimuth']
        zen[0, i] = sun['zenith'] * (np.pi/180.)

        # day of year and check for leap year
        if calendar.isleap(time['year']):
            dayspermonth = np.atleast_2d([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        else:
            dayspermonth = np.atleast_2d([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        # jday[0, i] = np.sum(dayspermonth[0, 0:time['month']-1]) + time['day'] # bug when a new day 20191015
        YYYY[0, i] = met[i, 0]
        doy = YMD.timetuple().tm_yday
        jday[0, i] = doy
        if (doy > leafon1) | (doy < leafoff1):
            leafon[0, i] = 1
        else:
            leafon[0, i] = 0

    return YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax






__author__ = 'xlinfr'


def sun_distance(jday):
    """

    #% Calculatesrelative earth sun distance
    #% with day of year as input.
    #% Partridge and Platt, 1975
    """
    b = 2.*np.pi*jday/365.
    D = np.sqrt((1.00011+np.dot(0.034221, np.cos(b))+np.dot(0.001280, np.sin(b))+np.dot(0.000719,
                                        np.cos((2.*b)))+np.dot(0.000077, np.sin((2.*b)))))
    return D


# -*- coding: utf-8 -*-


def sun_position(time, location):
    """
    % sun = sun_position(time, location)
    %
    % This function compute the sun position (zenith and azimuth angle at the observer
    % location) as a function of the observer local time and position.
    %
    % It is an implementation of the algorithm presented by Reda et Andreas in:
    %   Reda, I., Andreas, A. (2003) Solar position algorithm for solar
    %   radiation application. National Renewable Energy Laboratory (NREL)
    %   Technical report NREL/TP-560-34302.
    % This document is avalaible at www.osti.gov/bridge
    %
    % This algorithm is based on numerical approximation of the exact equations.
    % The authors of the original paper state that this algorithm should be
    % precise at +/- 0.0003 degrees. I have compared it to NOAA solar table
    % (http://www.srrb.noaa.gov/highlights/sunrise/azel.html) and to USNO solar
    % table (http://aa.usno.navy.mil/data/docs/AltAz.html) and found very good
    % correspondance (up to the precision of those tables), except for large
    % zenith angle, where the refraction by the atmosphere is significant
    % (difference of about 1 degree). Note that in this code the correction
    % for refraction in the atmosphere as been implemented for a temperature
    % of 10C (283 kelvins) and a pressure of 1010 mbar. See the subfunction
    % �sun_topocentric_zenith_angle_calculation� for a possible modification
    % to explicitely model the effect of temperature and pressure as describe
    % in Reda & Andreas (2003).
    %
    % Input parameters:
    %   time: a structure that specify the time when the sun position is
    %   calculated.
    %       time.year: year. Valid for [-2000, 6000]
    %       time.month: month [1-12]
    %       time.day: calendar day [1-31]
    %       time.hour: local hour [0-23]
    %       time.min: minute [0-59]
    %       time.sec: second [0-59]
    %       time.UTC: offset hour from UTC. Local time = Greenwich time + time.UTC
    %   This input can also be passed using the Matlab time format ('dd-mmm-yyyy HH:MM:SS').
    %   In that case, the time has to be specified as UTC time (time.UTC = 0)
    %
    %   location: a structure that specify the location of the observer
    %       location.latitude: latitude (in degrees, north of equator is
    %       positive)
    %       location.longitude: longitude (in degrees, positive for east of
    %       Greenwich)
    %       location.altitude: altitude above mean sea level (in meters)
    %
    % Output parameters
    %   sun: a structure with the calculated sun position
    %       sun.zenith = zenith angle in degrees (angle from the vertical)
    %       sun.azimuth = azimuth angle in degrees, eastward from the north.
    % Only the sun zenith and azimuth angles are returned as output, but a lot
    % of other parameters are calculated that could also extracted as output of
    % this function.
    %
    % Exemple of use
    %
    % location.longitude = -105.1786;
    % location.latitude = 39.742476;
    % location.altitude = 1830.14;
    % time.year = 2005;
    % time.month = 10;
    % time.day = 17;
    % time.hour = 6;
    % time.min = 30;
    % time.sec = 30;
    % time.UTC = -7;
    % %
    % location.longitude = 11.94;
    % location.latitude = 57.70;
    % location.altitude = 3.0;
    % time.UTC = 1;
    % sun = sun_position(time, location);
    %
    % sun =
    %
    %      zenith: 50.1080438859849
    %      azimuth: 194.341174010338
    %
    % History
    %   09/03/2004  Original creation by Vincent Roy (vincent.roy@drdc-rddc.gc.ca)
    %   10/03/2004  Fixed a bug in julian_calculation subfunction (was
    %               incorrect for year 1582 only), Vincent Roy
    %   18/03/2004  Correction to the header (help display) only. No changes to
    %               the code. (changed the �elevation� field in �location� structure
    %               information to �altitude�), Vincent Roy
    %   13/04/2004  Following a suggestion from Jody Klymak (jklymak@ucsd.edu),
    %               allowed the 'time' input to be passed as a Matlab time string.
    %   22/08/2005  Following a bug report from Bruce Bowler
    %               (bbowler@bigelow.org), modified the julian_calculation function. Bug
    %               was 'MATLAB has allowed structure assignment  to a non-empty non-structure
    %               to overwrite the previous value.  This behavior will continue in this release,
    %               but will be an error in a  future version of MATLAB.  For advice on how to
    %               write code that  will both avoid this warning and work in future versions of
    %               MATLAB,  see R14SP2 Release Notes'. Script should now be
    %               compliant with futher release of Matlab...
    """

    # 1. Calculate the Julian Day, and Century. Julian Ephemeris day, century
    # and millenium are calculated using a mean delta_t of 33.184 seconds.
    julian = julian_calculation(time)
    #print(julian)

    # 2. Calculate the Earth heliocentric longitude, latitude, and radius
    # vector (L, B, and R)
    earth_heliocentric_position = earth_heliocentric_position_calculation(julian)

    # 3. Calculate the geocentric longitude and latitude
    sun_geocentric_position = sun_geocentric_position_calculation(earth_heliocentric_position)

    # 4. Calculate the nutation in longitude and obliquity (in degrees).
    nutation = nutation_calculation(julian)

    # 5. Calculate the true obliquity of the ecliptic (in degrees).
    true_obliquity = true_obliquity_calculation(julian, nutation)

    # 6. Calculate the aberration correction (in degrees)
    aberration_correction = abberation_correction_calculation(earth_heliocentric_position)

    # 7. Calculate the apparent sun longitude in degrees)
    apparent_sun_longitude = apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction)

    # 8. Calculate the apparent sideral time at Greenwich (in degrees)
    apparent_stime_at_greenwich = apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity)

    # 9. Calculate the sun rigth ascension (in degrees)
    sun_rigth_ascension = sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position)

    # 10. Calculate the geocentric sun declination (in degrees). Positive or
    # negative if the sun is north or south of the celestial equator.
    sun_geocentric_declination = sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity,
                                                                        sun_geocentric_position)

    # 11. Calculate the observer local hour angle (in degrees, westward from south).
    observer_local_hour = observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension)

    # 12. Calculate the topocentric sun position (rigth ascension, declination and
    # rigth ascension parallax in degrees)
    topocentric_sun_position = topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                                                  observer_local_hour, sun_rigth_ascension,
                                                                  sun_geocentric_declination)

    # 13. Calculate the topocentric local hour angle (in degrees)
    topocentric_local_hour = topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position)

    # 14. Calculate the topocentric zenith and azimuth angle (in degrees)
    sun = sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour)

    return sun


def julian_calculation(t_input):
    """
    % This function compute the julian day and julian century from the local
    % time and timezone information. Ephemeris are calculated with a delta_t=0
    % seconds.

    % If time input is a Matlab time string, extract the information from
    % this string and create the structure as defined in the main header of
    % this script.
    """
    if not isinstance(t_input, dict):
        # tt = datetime.datetime.strptime(t_input, "%Y-%m-%d %H:%M:%S.%f")    # if t_input is a string of this format
        # t_input should be a datetime object
        time = dict()
        time['UTC'] = 0
        time['year'] = t_input.year
        time['month'] = t_input.month
        time['day'] = t_input.day
        time['hour'] = t_input.hour
        time['min'] = t_input.minute
        time['sec'] = t_input.second
    else:
        time = t_input

    if time['month'] == 1 or time['month'] == 2:
        Y = time['year'] - 1
        M = time['month'] + 12
    else:
        Y = time['year']
        M = time['month']

    ut_time = ((time['hour'] - time['UTC'])/24) + (time['min']/(60*24)) + (time['sec']/(60*60*24))   # time of day in UT time.
    D = time['day'] + ut_time   # Day of month in decimal time, ex. 2sd day of month at 12:30:30UT, D=2.521180556

    # In 1582, the gregorian calendar was adopted
    if time['year'] == 1582:
        if time['month'] == 10:
            if time['day'] <= 4:   # The Julian calendar ended on October 4, 1582
                B = (0)
            elif time['day'] >= 15:   # The Gregorian calendar started on October 15, 1582
                A = np.floor(Y/100)
                B = 2 - A + np.floor(A/4)
            else:
                print('This date never existed!. Date automatically set to October 4, 1582')
                time['month'] = 10
                time['day'] = 4
                B = 0
        elif time['month'] < 10:   # Julian calendar
            B = 0
        else: # Gregorian calendar
            A = np.floor(Y/100)
            B = 2 - A + np.floor(A/4)
    elif time['year'] < 1582:   # Julian calendar
        B = 0
    else:
        A = np.floor(Y/100)    # Gregorian calendar
        B = 2 - A + np.floor(A/4)

    julian = dict()
    julian['day'] = D + B + np.floor(365.25*(Y+4716)) + np.floor(30.6001*(M+1)) - 1524.5

    delta_t = 0   # 33.184;
    julian['ephemeris_day'] = (julian['day']) + (delta_t/86400)
    julian['century'] = (julian['day'] - 2451545) / 36525
    julian['ephemeris_century'] = (julian['ephemeris_day'] - 2451545) / 36525
    julian['ephemeris_millenium'] = (julian['ephemeris_century']) / 10

    return julian


def earth_heliocentric_position_calculation(julian):
    """
    % This function compute the earth position relative to the sun, using
    % tabulated values. 
    
    % Tabulated values for the longitude calculation
    % L terms  from the original code.
    """
    # Tabulated values for the longitude calculation
    # L terms  from the original code. 
    L0_terms = np.array([[175347046.0, 0, 0],
                        [3341656.0, 4.6692568, 6283.07585],
                        [34894.0, 4.6261, 12566.1517],
                        [3497.0, 2.7441, 5753.3849],
                        [3418.0, 2.8289, 3.5231],
                        [3136.0, 3.6277, 77713.7715],
                        [2676.0, 4.4181, 7860.4194],
                        [2343.0, 6.1352, 3930.2097],
                        [1324.0, 0.7425, 11506.7698],
                        [1273.0, 2.0371, 529.691],
                        [1199.0, 1.1096, 1577.3435],
                        [990, 5.233, 5884.927],
                        [902, 2.045, 26.298],
                        [857, 3.508, 398.149],
                        [780, 1.179, 5223.694],
                        [753, 2.533, 5507.553],
                        [505, 4.583, 18849.228],
                        [492, 4.205, 775.523],
                        [357, 2.92, 0.067],
                        [317, 5.849, 11790.629],
                        [284, 1.899, 796.298],
                        [271, 0.315, 10977.079],
                        [243, 0.345, 5486.778],
                        [206, 4.806, 2544.314],
                        [205, 1.869, 5573.143],
                        [202, 2.4458, 6069.777],
                        [156, 0.833, 213.299],
                        [132, 3.411, 2942.463],
                        [126, 1.083, 20.775],
                        [115, 0.645, 0.98],
                        [103, 0.636, 4694.003],
                        [102, 0.976, 15720.839],
                        [102, 4.267, 7.114],
                        [99, 6.21, 2146.17],
                        [98, 0.68, 155.42],
                        [86, 5.98, 161000.69],
                        [85, 1.3, 6275.96],
                        [85, 3.67, 71430.7],
                        [80, 1.81, 17260.15],
                        [79, 3.04, 12036.46],
                        [71, 1.76, 5088.63],
                        [74, 3.5, 3154.69],
                        [74, 4.68, 801.82],
                        [70, 0.83, 9437.76],
                        [62, 3.98, 8827.39],
                        [61, 1.82, 7084.9],
                        [57, 2.78, 6286.6],
                        [56, 4.39, 14143.5],
                        [56, 3.47, 6279.55],
                        [52, 0.19, 12139.55],
                        [52, 1.33, 1748.02],
                        [51, 0.28, 5856.48],
                        [49, 0.49, 1194.45],
                        [41, 5.37, 8429.24],
                        [41, 2.4, 19651.05],
                        [39, 6.17, 10447.39],
                        [37, 6.04, 10213.29],
                        [37, 2.57, 1059.38],
                        [36, 1.71, 2352.87],
                        [36, 1.78, 6812.77],
                        [33, 0.59, 17789.85],
                        [30, 0.44, 83996.85],
                        [30, 2.74, 1349.87],
                        [25, 3.16, 4690.48]])

    L1_terms = np.array([[628331966747.0, 0, 0],
                        [206059.0, 2.678235, 6283.07585],
                        [4303.0, 2.6351, 12566.1517],
                        [425.0, 1.59, 3.523],
                        [119.0, 5.796, 26.298],
                        [109.0, 2.966, 1577.344],
                        [93, 2.59, 18849.23],
                        [72, 1.14, 529.69],
                        [68, 1.87, 398.15],
                        [67, 4.41, 5507.55],
                        [59, 2.89, 5223.69],
                        [56, 2.17, 155.42],
                        [45, 0.4, 796.3],
                        [36, 0.47, 775.52],
                        [29, 2.65, 7.11],
                        [21, 5.34, 0.98],
                        [19, 1.85, 5486.78],
                        [19, 4.97, 213.3],
                        [17, 2.99, 6275.96],
                        [16, 0.03, 2544.31],
                        [16, 1.43, 2146.17],
                        [15, 1.21, 10977.08],
                        [12, 2.83, 1748.02],
                        [12, 3.26, 5088.63],
                        [12, 5.27, 1194.45],
                        [12, 2.08, 4694],
                        [11, 0.77, 553.57],
                        [10, 1.3, 3286.6],
                        [10, 4.24, 1349.87],
                        [9, 2.7, 242.73],
                        [9, 5.64, 951.72],
                        [8, 5.3, 2352.87],
                        [6, 2.65, 9437.76],
                        [6, 4.67, 4690.48]])

    L2_terms = np.array([[52919.0, 0, 0],
                        [8720.0, 1.0721, 6283.0758],
                        [309.0, 0.867, 12566.152],
                        [27, 0.05, 3.52],
                        [16, 5.19, 26.3],
                        [16, 3.68, 155.42],
                        [10, 0.76, 18849.23],
                        [9, 2.06, 77713.77],
                        [7, 0.83, 775.52],
                        [5, 4.66, 1577.34],
                        [4, 1.03, 7.11],
                        [4, 3.44, 5573.14],
                        [3, 5.14, 796.3],
                        [3, 6.05, 5507.55],
                        [3, 1.19, 242.73],
                        [3, 6.12, 529.69],
                        [3, 0.31, 398.15],
                        [3, 2.28, 553.57],
                        [2, 4.38, 5223.69],
                        [2, 3.75, 0.98]])

    L3_terms = np.array([[289.0, 5.844, 6283.076],
                        [35, 0, 0],
                        [17, 5.49, 12566.15],
                        [3, 5.2, 155.42],
                        [1, 4.72, 3.52],
                        [1, 5.3, 18849.23],
                        [1, 5.97, 242.73]])
    L4_terms = np.array([[114.0, 3.142, 0],
                        [8, 4.13, 6283.08],
                        [1, 3.84, 12566.15]])

    L5_terms = np.array([1, 3.14, 0])
    L5_terms = np.atleast_2d(L5_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = L0_terms[:, 0]
    B0 = L0_terms[:, 1]
    C0 = L0_terms[:, 2]

    A1 = L1_terms[:, 0]
    B1 = L1_terms[:, 1]
    C1 = L1_terms[:, 2]

    A2 = L2_terms[:, 0]
    B2 = L2_terms[:, 1]
    C2 = L2_terms[:, 2]

    A3 = L3_terms[:, 0]
    B3 = L3_terms[:, 1]
    C3 = L3_terms[:, 2]

    A4 = L4_terms[:, 0]
    B4 = L4_terms[:, 1]
    C4 = L4_terms[:, 2]

    A5 = L5_terms[:, 0]
    B5 = L5_terms[:, 1]
    C5 = L5_terms[:, 2]

    JME = julian['ephemeris_millenium']

    # Compute the Earth Heliochentric longitude from the tabulated values.
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = np.sum(A4 * np.cos(B4 + (C4 * JME)))
    L5 = A5 * np.cos(B5 + (C5 * JME))

    earth_heliocentric_position = dict()
    earth_heliocentric_position['longitude'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                                          (L3 * np.power(JME, 3)) +
                                                          (L4 * np.power(JME, 4)) +
                                                          (L5 * np.power(JME, 5))) / 1e8
    # Convert the longitude to degrees.
    earth_heliocentric_position['longitude'] = earth_heliocentric_position['longitude'] * 180/np.pi

    # Limit the range to [0,360]
    earth_heliocentric_position['longitude'] = set_to_range(earth_heliocentric_position['longitude'], 0, 360)

    # Tabulated values for the earth heliocentric latitude. 
    # B terms  from the original code. 
    B0_terms = np.array([[280.0, 3.199, 84334.662],
                        [102.0, 5.422, 5507.553],
                        [80, 3.88, 5223.69],
                        [44, 3.7, 2352.87],
                        [32, 4, 1577.34]])

    B1_terms = np.array([[9, 3.9, 5507.55],
                         [6, 1.73, 5223.69]])

    A0 = B0_terms[:, 0]
    B0 = B0_terms[:, 1]
    C0 = B0_terms[:, 2]
    
    A1 = B1_terms[:, 0]
    B1 = B1_terms[:, 1]
    C1 = B1_terms[:, 2]
    
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))

    earth_heliocentric_position['latitude'] = (L0 + (L1 * JME)) / 1e8

    # Convert the latitude to degrees. 
    earth_heliocentric_position['latitude'] = earth_heliocentric_position['latitude'] * 180/np.pi

    # Limit the range to [0,360];
    earth_heliocentric_position['latitude'] = set_to_range(earth_heliocentric_position['latitude'], 0, 360)

    # Tabulated values for radius vector. 
    # R terms from the original code
    R0_terms = np.array([[100013989.0, 0, 0],
                        [1670700.0, 3.0984635, 6283.07585],
                        [13956.0, 3.05525, 12566.1517],
                        [3084.0, 5.1985, 77713.7715],
                        [1628.0, 1.1739, 5753.3849],
                        [1576.0, 2.8469, 7860.4194],
                        [925.0, 5.453, 11506.77],
                        [542.0, 4.564, 3930.21],
                        [472.0, 3.661, 5884.927],
                        [346.0, 0.964, 5507.553],
                        [329.0, 5.9, 5223.694],
                        [307.0, 0.299, 5573.143],
                        [243.0, 4.273, 11790.629],
                        [212.0, 5.847, 1577.344],
                        [186.0, 5.022, 10977.079],
                        [175.0, 3.012, 18849.228],
                        [110.0, 5.055, 5486.778],
                        [98, 0.89, 6069.78],
                        [86, 5.69, 15720.84],
                        [86, 1.27, 161000.69],
                        [85, 0.27, 17260.15],
                        [63, 0.92, 529.69],
                        [57, 2.01, 83996.85],
                        [56, 5.24, 71430.7],
                        [49, 3.25, 2544.31],
                        [47, 2.58, 775.52],
                        [45, 5.54, 9437.76],
                        [43, 6.01, 6275.96],
                        [39, 5.36, 4694],
                        [38, 2.39, 8827.39],
                        [37, 0.83, 19651.05],
                        [37, 4.9, 12139.55],
                        [36, 1.67, 12036.46],
                        [35, 1.84, 2942.46],
                        [33, 0.24, 7084.9],
                        [32, 0.18, 5088.63],
                        [32, 1.78, 398.15],
                        [28, 1.21, 6286.6],
                        [28, 1.9, 6279.55],
                        [26, 4.59, 10447.39]])

    R1_terms = np.array([[103019.0, 1.10749, 6283.07585],
                        [1721.0, 1.0644, 12566.1517],
                        [702.0, 3.142, 0],
                        [32, 1.02, 18849.23],
                        [31, 2.84, 5507.55],
                        [25, 1.32, 5223.69],
                        [18, 1.42, 1577.34],
                        [10, 5.91, 10977.08],
                        [9, 1.42, 6275.96],
                        [9, 0.27, 5486.78]])

    R2_terms = np.array([[4359.0, 5.7846, 6283.0758],
                        [124.0, 5.579, 12566.152],
                        [12, 3.14, 0],
                        [9, 3.63, 77713.77],
                        [6, 1.87, 5573.14],
                        [3, 5.47, 18849]])

    R3_terms = np.array([[145.0, 4.273, 6283.076],
                        [7, 3.92, 12566.15]])
    
    R4_terms = [4, 2.56, 6283.08]
    R4_terms = np.atleast_2d(R4_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = R0_terms[:, 0]
    B0 = R0_terms[:, 1]
    C0 = R0_terms[:, 2]
    
    A1 = R1_terms[:, 0]
    B1 = R1_terms[:, 1]
    C1 = R1_terms[:, 2]
    
    A2 = R2_terms[:, 0]
    B2 = R2_terms[:, 1]
    C2 = R2_terms[:, 2]
    
    A3 = R3_terms[:, 0]
    B3 = R3_terms[:, 1]
    C3 = R3_terms[:, 2]
    
    A4 = R4_terms[:, 0]
    B4 = R4_terms[:, 1]
    C4 = R4_terms[:, 2]

    # Compute the Earth heliocentric radius vector
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = A4 * np.cos(B4 + (C4 * JME))

    # Units are in AU
    earth_heliocentric_position['radius'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                             (L3 * np.power(JME, 3)) +
                                             (L4 * np.power(JME, 4))) / 1e8

    return earth_heliocentric_position


def sun_geocentric_position_calculation(earth_heliocentric_position):
    """
    % This function compute the sun position relative to the earth.
    """
    sun_geocentric_position = dict()
    sun_geocentric_position['longitude'] = earth_heliocentric_position['longitude'] + 180
    # Limit the range to [0,360];
    sun_geocentric_position['longitude'] = set_to_range(sun_geocentric_position['longitude'], 0, 360)

    sun_geocentric_position['latitude'] = -earth_heliocentric_position['latitude']
    # Limit the range to [0,360]
    sun_geocentric_position['latitude'] = set_to_range(sun_geocentric_position['latitude'], 0, 360)
    return sun_geocentric_position


def nutation_calculation(julian):
    """
    % This function compute the nutation in longtitude and in obliquity, in
    % degrees.
    :param julian:
    :return: nutation
    """

    # All Xi are in degrees.
    JCE = julian['ephemeris_century']

    # 1. Mean elongation of the moon from the sun
    p = np.atleast_2d([(1/189474), -0.0019142, 445267.11148, 297.85036])

    # X0 = polyval(p, JCE);
    X0 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]   # This is faster than polyval...

    # 2. Mean anomaly of the sun (earth)
    p = np.atleast_2d([-(1/300000), -0.0001603, 35999.05034, 357.52772])

    # X1 = polyval(p, JCE)
    X1 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 3. Mean anomaly of the moon
    p = np.atleast_2d([(1/56250), 0.0086972, 477198.867398, 134.96298])
    
    # X2 = polyval(p, JCE);
    X2 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 4. Moon argument of latitude
    p = np.atleast_2d([(1/327270), -0.0036825, 483202.017538, 93.27191])

    # X3 = polyval(p, JCE)
    X3 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 5. Longitude of the ascending node of the moon's mean orbit on the
    # ecliptic, measured from the mean equinox of the date
    p = np.atleast_2d([(1/450000), 0.0020708, -1934.136261, 125.04452])

    # X4 = polyval(p, JCE);
    X4 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # Y tabulated terms from the original code
    Y_terms = np.array([[0, 0, 0, 0, 1],
                        [-2, 0, 0, 2, 2],
                        [0, 0, 0, 2, 2],
                        [0, 0, 0, 0, 2],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [-2, 1, 0, 2, 2],
                        [0, 0, 0, 2, 1],
                        [0, 0, 1, 2, 2],
                        [-2, -1, 0, 2, 2],
                        [-2, 0, 1, 0, 0],
                        [-2, 0, 0, 2, 1],
                        [0, 0, -1, 2, 2],
                        [2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [2, 0, -1, 2, 2],
                        [0, 0, -1, 0, 1],
                        [0, 0, 1, 2, 1],
                        [-2, 0, 2, 0, 0],
                        [0, 0, -2, 2, 1],
                        [2, 0, 0, 2, 2],
                        [0, 0, 2, 2, 2],
                        [0, 0, 2, 0, 0],
                        [-2, 0, 1, 2, 2],
                        [0, 0, 0, 2, 0],
                        [-2, 0, 0, 2, 0],
                        [0, 0, -1, 2, 1],
                        [0, 2, 0, 0, 0],
                        [2, 0, -1, 0, 1],
                        [-2, 2, 0, 2, 2],
                        [0, 1, 0, 0, 1],
                        [-2, 0, 1, 0, 1],
                        [0, -1, 0, 0, 1],
                        [0, 0, 2, -2, 0],
                        [2, 0, -1, 2, 1],
                        [2, 0, 1, 2, 2],
                        [0, 1, 0, 2, 2],
                        [-2, 1, 1, 0, 0],
                        [0, -1, 0, 2, 2],
                        [2, 0, 0, 2, 1],
                        [2, 0, 1, 0, 0],
                        [-2, 0, 2, 2, 2],
                        [-2, 0, 1, 2, 1],
                        [2, 0, -2, 0, 1],
                        [2, 0, 0, 0, 1],
                        [0, -1, 1, 0, 0],
                        [-2, -1, 0, 2, 1],
                        [-2, 0, 0, 0, 1],
                        [0, 0, 2, 2, 1],
                        [-2, 0, 2, 0, 1],
                        [-2, 1, 0, 2, 1],
                        [0, 0, 1, -2, 0],
                        [-1, 0, 1, 0, 0],
                        [-2, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, -2, 2, 2],
                        [-1, -1, 1, 0, 0],
                        [0, 1, 1, 0, 0],
                        [0, -1, 1, 2, 2],
                        [2, -1, -1, 2, 2],
                        [0, 0, 3, 2, 2],
                        [2, -1, 0, 2, 2]])

    nutation_terms = np.array([[-171996, -174.2, 92025, 8.9],
                                [-13187, -1.6, 5736, -3.1],
                                [-2274, -0.2, 977, -0.5],
                                [2062, 0.2, -895, 0.5],
                                [1426, -3.4, 54, -0.1],
                                [712, 0.1, -7, 0],
                                [-517, 1.2, 224, -0.6],
                                [-386, -0.4, 200, 0],
                                [-301, 0, 129, -0.1],
                                [217, -0.5, -95, 0.3],
                                [-158, 0, 0, 0],
                                [129, 0.1, -70, 0],
                                [123, 0, -53, 0],
                                [63, 0, 0, 0],
                                [63, 0.1, -33, 0],
                                [-59, 0, 26, 0],
                                [-58, -0.1, 32, 0],
                                [-51, 0, 27, 0],
                                [48, 0, 0, 0],
                                [46, 0, -24, 0],
                                [-38, 0, 16, 0],
                                [-31, 0, 13, 0],
                                [29, 0, 0, 0],
                                [29, 0, -12, 0],
                                [26, 0, 0, 0],
                                [-22, 0, 0, 0],
                                [21, 0, -10, 0],
                                [17, -0.1, 0, 0],
                                [16, 0, -8, 0],
                                [-16, 0.1, 7, 0],
                                [-15, 0, 9, 0],
                                [-13, 0, 7, 0],
                                [-12, 0, 6, 0],
                                [11, 0, 0, 0],
                                [-10, 0, 5, 0],
                                [-8, 0, 3, 0],
                                [7, 0, -3, 0],
                                [-7, 0, 0, 0],
                                [-7, 0, 3, 0],
                                [-7, 0, 3, 0],
                                [6, 0, 0, 0],
                                [6, 0, -3, 0],
                                [6, 0, -3, 0],
                                [-6, 0, 3, 0],
                                [-6, 0, 3, 0],
                                [5, 0, 0, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0]])

    # Using the tabulated values, compute the delta_longitude and
    # delta_obliquity.
    Xi = np.array([X0, X1, X2, X3, X4])    # a col mat in octave

    tabulated_argument = Y_terms.dot(np.transpose(Xi)) * (np.pi/180)

    delta_longitude = (nutation_terms[:, 0] + (nutation_terms[:, 1] * JCE)) * np.sin(tabulated_argument)
    delta_obliquity = (nutation_terms[:, 2] + (nutation_terms[:, 3] * JCE)) * np.cos(tabulated_argument)

    nutation = dict()    # init nutation dictionary
    # Nutation in longitude
    nutation['longitude'] = np.sum(delta_longitude) / 36000000

    # Nutation in obliquity
    nutation['obliquity'] = np.sum(delta_obliquity) / 36000000

    return nutation


def true_obliquity_calculation(julian, nutation):
    """
    This function compute the true obliquity of the ecliptic.

    :param julian:
    :param nutation:
    :return:
    """

    p = np.atleast_2d([2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448])

    # mean_obliquity = polyval(p, julian.ephemeris_millenium/10);
    U = julian['ephemeris_millenium'] / 10
    mean_obliquity = p[0, 0] * np.power(U, 10) + p[0, 1] * np.power(U, 9) + \
                     p[0, 2] * np.power(U, 8) + p[0, 3] * np.power(U, 7) + \
                     p[0, 4] * np.power(U, 6) + p[0, 5] * np.power(U, 5) + \
                     p[0, 6] * np.power(U, 4) + p[0, 7] * np.power(U, 3) + \
                     p[0, 8] * np.power(U, 2) + p[0, 9] * U + p[0, 10]

    true_obliquity = (mean_obliquity/3600) + nutation['obliquity']

    return true_obliquity


def abberation_correction_calculation(earth_heliocentric_position):
    """
    This function compute the aberration_correction, as a function of the
    earth-sun distance.

    :param earth_heliocentric_position:
    :return:
    """
    aberration_correction = -20.4898/(3600*earth_heliocentric_position['radius'])
    return aberration_correction


def apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction):
    """
    This function compute the sun apparent longitude

    :param sun_geocentric_position:
    :param nutation:
    :param aberration_correction:
    :return:
    """
    apparent_sun_longitude = sun_geocentric_position['longitude'] + nutation['longitude'] + aberration_correction
    return apparent_sun_longitude


def apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity):
    """
    This function compute the apparent sideral time at Greenwich.

    :param julian:
    :param nutation:
    :param true_obliquity:
    :return:
    """

    JD = julian['day']
    JC = julian['century']

    # Mean sideral time, in degrees
    mean_stime = 280.46061837 + (360.98564736629*(JD-2451545)) + \
                 (0.000387933*np.power(JC, 2)) - \
                 (np.power(JC, 3)/38710000)

    # Limit the range to [0-360];
    mean_stime = set_to_range(mean_stime, 0, 360)

    apparent_stime_at_greenwich = mean_stime + (nutation['longitude'] * np.cos(true_obliquity * np.pi/180))
    return apparent_stime_at_greenwich


def sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """
    This function compute the sun rigth ascension.
    :param apparent_sun_longitude:
    :param true_obliquity:
    :param sun_geocentric_position:
    :return:
    """

    argument_numerator = (np.sin(apparent_sun_longitude * np.pi/180) * np.cos(true_obliquity * np.pi/180)) - \
        (np.tan(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180))
    argument_denominator = np.cos(apparent_sun_longitude * np.pi/180);

    sun_rigth_ascension = np.arctan2(argument_numerator, argument_denominator) * 180/np.pi
    # Limit the range to [0,360];
    sun_rigth_ascension = set_to_range(sun_rigth_ascension, 0, 360)
    return sun_rigth_ascension


def sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """

    :param apparent_sun_longitude:
    :param true_obliquity:
    :param sun_geocentric_position:
    :return:
    """

    argument = (np.sin(sun_geocentric_position['latitude'] * np.pi/180) * np.cos(true_obliquity * np.pi/180)) + \
        (np.cos(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180) *
         np.sin(apparent_sun_longitude * np.pi/180))

    sun_geocentric_declination = np.arcsin(argument) * 180/np.pi
    return sun_geocentric_declination


def observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension):
    """
    This function computes observer local hour.

    :param apparent_stime_at_greenwich:
    :param location:
    :param sun_rigth_ascension:
    :return:
    """

    observer_local_hour = apparent_stime_at_greenwich + location['longitude'] - sun_rigth_ascension
    # Set the range to [0-360]
    observer_local_hour = set_to_range(observer_local_hour, 0, 360)
    return observer_local_hour


def topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                       observer_local_hour, sun_rigth_ascension, sun_geocentric_declination):
    """
    This function compute the sun position (rigth ascension and declination)
    with respect to the observer local position at the Earth surface.

    :param earth_heliocentric_position:
    :param location:
    :param observer_local_hour:
    :param sun_rigth_ascension:
    :param sun_geocentric_declination:
    :return:
    """

    # Equatorial horizontal parallax of the sun in degrees
    eq_horizontal_parallax = 8.794 / (3600 * earth_heliocentric_position['radius'])

    # Term u, used in the following calculations (in radians)
    u = np.arctan(0.99664719 * np.tan(location['latitude'] * np.pi/180))

    # Term x, used in the following calculations
    x = np.cos(u) + ((location['altitude']/6378140) * np.cos(location['latitude'] * np.pi/180))

    # Term y, used in the following calculations
    y = (0.99664719 * np.sin(u)) + ((location['altitude']/6378140) * np.sin(location['latitude'] * np.pi/180))

    # Parallax in the sun rigth ascension (in radians)
    nominator = -x * np.sin(eq_horizontal_parallax * np.pi/180) * np.sin(observer_local_hour * np.pi/180)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (x * np.sin(eq_horizontal_parallax * np.pi/180) *
                                                                    np.cos(observer_local_hour * np.pi/180))
    sun_rigth_ascension_parallax = np.arctan2(nominator, denominator)
    # Conversion to degrees.
    topocentric_sun_position = dict()
    topocentric_sun_position['rigth_ascension_parallax'] = sun_rigth_ascension_parallax * 180/np.pi

    # Topocentric sun rigth ascension (in degrees)
    topocentric_sun_position['rigth_ascension'] = sun_rigth_ascension + (sun_rigth_ascension_parallax * 180/np.pi)

    # Topocentric sun declination (in degrees)
    nominator = (np.sin(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180))) * \
                np.cos(sun_rigth_ascension_parallax)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180)) * \
                                                                   np.cos(observer_local_hour * np.pi/180)
    topocentric_sun_position['declination'] = np.arctan2(nominator, denominator) * 180/np.pi
    return topocentric_sun_position


def topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position):
    """
    This function compute the topocentric local jour angle in degrees

    :param observer_local_hour:
    :param topocentric_sun_position:
    :return:
    """

    topocentric_local_hour = observer_local_hour - topocentric_sun_position['rigth_ascension_parallax']
    return topocentric_local_hour


def sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour):
    """
    This function compute the sun zenith angle, taking into account the
    atmospheric refraction. A default temperature of 283K and a
    default pressure of 1010 mbar are used.

    :param location:
    :param topocentric_sun_position:
    :param topocentric_local_hour:
    :return:
    """

    # Topocentric elevation, without atmospheric refraction
    argument = (np.sin(location['latitude'] * np.pi/180) * np.sin(topocentric_sun_position['declination'] * np.pi/180)) + \
    (np.cos(location['latitude'] * np.pi/180) * np.cos(topocentric_sun_position['declination'] * np.pi/180) *
     np.cos(topocentric_local_hour * np.pi/180))
    true_elevation = np.arcsin(argument) * 180/np.pi

    # Atmospheric refraction correction (in degrees)
    argument = true_elevation + (10.3/(true_elevation + 5.11))
    refraction_corr = 1.02 / (60 * np.tan(argument * np.pi/180))

    # For exact pressure and temperature correction, use this,
    # with P the pressure in mbar amd T the temperature in Kelvins:
    # refraction_corr = (P/1010) * (283/T) * 1.02 / (60 * tan(argument * pi/180));

    # Apparent elevation
    apparent_elevation = true_elevation + refraction_corr

    sun = dict()
    sun['zenith'] = 90 - apparent_elevation

    # Topocentric azimuth angle. The +180 conversion is to pass from astronomer
    # notation (westward from south) to navigation notation (eastward from
    # north);
    nominator = np.sin(topocentric_local_hour * np.pi/180)
    denominator = (np.cos(topocentric_local_hour * np.pi/180) * np.sin(location['latitude'] * np.pi/180)) - \
    (np.tan(topocentric_sun_position['declination'] * np.pi/180) * np.cos(location['latitude'] * np.pi/180))
    sun['azimuth'] = (np.arctan2(nominator, denominator) * 180/np.pi) + 180

    # Set the range to [0-360]
    sun['azimuth'] = set_to_range(sun['azimuth'], 0, 360)
    return sun


def set_to_range(var, min_interval, max_interval):
    """
    Sets a variable in range min_interval and max_interval

    :param var:
    :param min_interval:
    :param max_interval:
    :return:
    """
    var = var - max_interval * np.floor(var/max_interval)

    if var < min_interval:
        var = var + max_interval
    return var

