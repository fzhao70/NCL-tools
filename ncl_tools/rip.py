"""
NCL RIP (Read/Interpolate/Plot) Functions

This module provides Python implementations of NCL's RIP functions for calculating
convective available potential energy (CAPE), convective inhibition (CIN),
lifted condensation level (LCL), and level of free convection (LFC).

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/rip.shtml
"""

import numpy as np
from typing import Union, Optional, Tuple


# Physical constants
GRAVITY = 9.81  # m/s²
R_DRY = 287.04  # J/(kg·K) - gas constant for dry air
R_VAPOR = 461.5  # J/(kg·K) - gas constant for water vapor
CP_DRY = 1005.7  # J/(kg·K) - specific heat at constant pressure for dry air
EPSILON = R_DRY / R_VAPOR  # ~0.622
L_VAPOR = 2.5e6  # J/kg - latent heat of vaporization
P0 = 100000.0  # Pa - reference pressure (1000 hPa)


def _calculate_theta(t, p):
    """
    Calculate potential temperature.

    Parameters
    ----------
    t : ndarray
        Temperature (K)
    p : ndarray
        Pressure (Pa)

    Returns
    -------
    ndarray
        Potential temperature (K)
    """
    return t * (P0 / p) ** (R_DRY / CP_DRY)


def _calculate_mixing_ratio_sat(t, p):
    """
    Calculate saturation mixing ratio.

    Parameters
    ----------
    t : ndarray
        Temperature (K)
    p : ndarray
        Pressure (Pa)

    Returns
    -------
    ndarray
        Saturation mixing ratio (kg/kg)
    """
    # Bolton's formula for saturation vapor pressure (hPa)
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))

    # Convert to Pa
    es_pa = es * 100.0

    # Mixing ratio
    ws = EPSILON * es_pa / (p - es_pa)

    return ws


def _calculate_theta_e(t, p, q):
    """
    Calculate equivalent potential temperature.

    Parameters
    ----------
    t : ndarray
        Temperature (K)
    p : ndarray
        Pressure (Pa)
    q : ndarray
        Mixing ratio (kg/kg)

    Returns
    -------
    ndarray
        Equivalent potential temperature (K)
    """
    # Calculate potential temperature
    theta = _calculate_theta(t, p)

    # Calculate theta_e (simplified Bolton formula)
    # theta_e = theta * exp((Lv * q) / (Cp * T))
    theta_e = theta * np.exp((L_VAPOR * q) / (CP_DRY * t))

    return theta_e


def _calculate_lcl(t_parcel, p_parcel, q_parcel):
    """
    Calculate lifted condensation level.

    Parameters
    ----------
    t_parcel : float
        Parcel temperature (K)
    p_parcel : float
        Parcel pressure (Pa)
    q_parcel : float
        Parcel mixing ratio (kg/kg)

    Returns
    -------
    tuple
        (p_lcl, t_lcl, z_lcl) - Pressure (Pa), temperature (K), and height (m) at LCL
    """
    # Calculate dewpoint temperature
    # Using approximation: Td = T - ((100 - RH) / 5)
    # First get RH from mixing ratio
    ws = _calculate_mixing_ratio_sat(t_parcel, p_parcel)
    rh = 100.0 * q_parcel / ws if ws > 0 else 0.0
    rh = np.clip(rh, 0, 100)

    t_c = t_parcel - 273.15
    td_c = t_c - ((100 - rh) / 5.0)
    td = td_c + 273.15

    # Bolton's formula for LCL
    # T_LCL = 1 / (1/(Td - 56) + ln(T/Td)/800) + 56
    if td > 0:
        t_lcl = 1.0 / (1.0 / (td - 56.0) + np.log(t_parcel / td) / 800.0) + 56.0
    else:
        t_lcl = td

    # LCL pressure using dry adiabatic process
    # p_lcl = p * (T_lcl / T)^(Cp/R)
    p_lcl = p_parcel * (t_lcl / t_parcel) ** (CP_DRY / R_DRY)

    return p_lcl, t_lcl


def _lift_parcel(t_parcel, p_parcel, q_parcel, p_levels):
    """
    Lift a parcel adiabatically through pressure levels.

    Parameters
    ----------
    t_parcel : float
        Initial parcel temperature (K)
    p_parcel : float
        Initial parcel pressure (Pa)
    q_parcel : float
        Parcel mixing ratio (kg/kg)
    p_levels : ndarray
        Pressure levels to lift through (Pa)

    Returns
    -------
    ndarray
        Parcel temperature at each pressure level (K)
    """
    p_lcl, t_lcl = _calculate_lcl(t_parcel, p_parcel, q_parcel)

    t_lifted = np.zeros_like(p_levels, dtype=np.float64)

    for i, p in enumerate(p_levels):
        if p >= p_parcel:
            # Below parcel level - use environmental temperature
            t_lifted[i] = t_parcel
        elif p >= p_lcl:
            # Dry adiabatic ascent to LCL
            t_lifted[i] = t_parcel * (p / p_parcel) ** (R_DRY / CP_DRY)
        else:
            # Moist adiabatic ascent above LCL
            # Simplified: use pseudo-adiabatic process
            # Start from LCL conditions
            t_lifted[i] = t_lcl * (p / p_lcl) ** (R_DRY / CP_DRY)

            # Adjust for latent heat release (simplified)
            ws = _calculate_mixing_ratio_sat(t_lifted[i], p)
            gamma_m = GRAVITY * (1 + (L_VAPOR * ws) / (R_DRY * t_lifted[i])) / \
                     (CP_DRY + (L_VAPOR ** 2 * ws * EPSILON) / (R_DRY * t_lifted[i] ** 2))

            # Correction factor
            correction = 1.0 - (gamma_m / (GRAVITY / CP_DRY))
            t_lifted[i] = t_lcl * (p / p_lcl) ** (R_DRY / (CP_DRY * correction))

    return t_lifted


def _calculate_cape_cin_column(p, t, q, z, z_sfc, p_sfc):
    """
    Calculate CAPE, CIN, LCL, and LFC for a single column.

    Parameters
    ----------
    p : ndarray
        Pressure levels (Pa), ordered top-to-bottom or bottom-to-top
    t : ndarray
        Temperature (K)
    q : ndarray
        Mixing ratio (kg/kg)
    z : ndarray
        Geopotential height (m)
    z_sfc : float
        Surface height (m)
    p_sfc : float
        Surface pressure (Pa)

    Returns
    -------
    tuple
        (cape, cin, lcl, lfc) in (J/kg, J/kg, m, m)
    """
    # Ensure top-to-bottom ordering
    if p[0] > p[-1]:
        p = p[::-1]
        t = t[::-1]
        q = q[::-1]
        z = z[::-1]

    # Find levels below 3000m AGL for parcel search
    z_agl = z - z_sfc
    below_3km = z_agl < 3000.0

    if not np.any(below_3km):
        return np.nan, np.nan, np.nan, np.nan

    # Calculate equivalent potential temperature
    theta_e = _calculate_theta_e(t, p, q)

    # Find parcel with maximum theta_e below 3000m (most unstable parcel)
    theta_e_low = np.where(below_3km, theta_e, -np.inf)
    max_idx = np.argmax(theta_e_low)

    if max_idx < 0 or max_idx >= len(p):
        return np.nan, np.nan, np.nan, np.nan

    # Define parcel as 500m deep layer average
    z_parcel = z[max_idx]
    parcel_top_idx = max_idx
    parcel_bot_idx = max_idx

    # Find indices for 500m layer
    for i in range(max_idx, len(z)):
        if z[i] - z_parcel < 500:
            parcel_bot_idx = i
        else:
            break

    # Average properties over 500m layer
    p_parcel = np.mean(p[max_idx:parcel_bot_idx+1])
    t_parcel = np.mean(t[max_idx:parcel_bot_idx+1])
    q_parcel = np.mean(q[max_idx:parcel_bot_idx+1])
    z_parcel_base = z[max_idx]

    # Lift parcel
    t_lifted = _lift_parcel(t_parcel, p_parcel, q_parcel, p)

    # Calculate virtual temperatures
    # Tv = T * (1 + 0.61*q)
    tv_env = t * (1.0 + 0.61 * q)
    tv_parcel = t_lifted * (1.0 + 0.61 * q_parcel)

    # Calculate buoyancy
    buoyancy = GRAVITY * (tv_parcel - tv_env) / tv_env

    # Find LCL and LFC
    p_lcl, t_lcl = _calculate_lcl(t_parcel, p_parcel, q_parcel)

    # Find LCL height by interpolation
    lcl_idx = np.searchsorted(p[::-1], p_lcl)
    lcl_idx = len(p) - 1 - lcl_idx
    if 0 < lcl_idx < len(z) - 1:
        # Interpolate
        frac = (np.log(p_lcl) - np.log(p[lcl_idx])) / \
               (np.log(p[lcl_idx+1]) - np.log(p[lcl_idx]))
        z_lcl = z[lcl_idx] + frac * (z[lcl_idx+1] - z[lcl_idx])
    else:
        z_lcl = z_sfc

    # Find LFC (first level above parcel where buoyancy becomes positive)
    lfc_idx = -1
    for i in range(max_idx, len(buoyancy)):
        if buoyancy[i] > 0:
            lfc_idx = i
            break

    if lfc_idx < 0:
        # No LFC found - no CAPE
        return 0.0, 0.0, z_lcl - z_sfc, np.nan

    z_lfc = z[lfc_idx] - z_sfc

    # Find EL (equilibrium level - where buoyancy becomes negative again)
    el_idx = len(buoyancy) - 1
    for i in range(lfc_idx + 1, len(buoyancy)):
        if buoyancy[i] <= 0:
            el_idx = i - 1
            break

    # Calculate CAPE (integrate positive buoyancy from LFC to EL)
    cape = 0.0
    for i in range(lfc_idx, el_idx):
        if i + 1 < len(z) and buoyancy[i] > 0:
            dz = z[i+1] - z[i]
            cape += buoyancy[i] * dz

    # Calculate CIN (integrate negative buoyancy from parcel to LFC)
    cin = 0.0
    for i in range(max_idx, lfc_idx):
        if i + 1 < len(z) and buoyancy[i] < 0:
            dz = z[i+1] - z[i]
            cin += buoyancy[i] * dz

    # CIN is reported as positive value
    cin = abs(cin)

    return cape, cin, z_lcl - z_sfc, z_lfc


def wrf_cape_2d(p, t, q, z, zsfc, psfc, opt):
    """
    Computes maximum CAPE, CIN, LCL, and LFC.

    Parameters
    ----------
    p : array_like
        Pressure (Pa), any dimension ordering
    t : array_like
        Temperature (K), same dimensions as p
    q : array_like
        Water vapor mixing ratio (kg/kg), same dimensions as p
    z : array_like
        Geopotential height (m), same dimensions as p
    zsfc : array_like or float
        Surface height (m)
    psfc : array_like or float
        Surface pressure (hPa)
    opt : bool or int
        False for pressure level data, True for terrain-following data

    Returns
    -------
    ndarray
        Array with dimensions [4, ...] where:
        [0] = CAPE (J/kg)
        [1] = CIN (J/kg)
        [2] = LCL (m AGL)
        [3] = LFC (m AGL)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/wrf_cape_2d.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    zsfc = np.asarray(zsfc, dtype=np.float64)
    psfc = np.asarray(psfc, dtype=np.float64) * 100.0  # Convert hPa to Pa

    # Determine dimensions
    if p.ndim == 4:
        # (time, lev, lat, lon)
        ntime, nlev, nlat, nlon = p.shape
        result = np.zeros((4, ntime, nlat, nlon), dtype=np.float64)

        for it in range(ntime):
            for ilat in range(nlat):
                for ilon in range(nlon):
                    cape, cin, lcl, lfc = _calculate_cape_cin_column(
                        p[it, :, ilat, ilon],
                        t[it, :, ilat, ilon],
                        q[it, :, ilat, ilon],
                        z[it, :, ilat, ilon],
                        zsfc[it, ilat, ilon] if zsfc.ndim == 3 else zsfc[ilat, ilon] if zsfc.ndim == 2 else zsfc,
                        psfc[it, ilat, ilon] if psfc.ndim == 3 else psfc[ilat, ilon] if psfc.ndim == 2 else psfc
                    )
                    result[0, it, ilat, ilon] = cape
                    result[1, it, ilat, ilon] = cin
                    result[2, it, ilat, ilon] = lcl
                    result[3, it, ilat, ilon] = lfc

    elif p.ndim == 3:
        # (lev, lat, lon)
        nlev, nlat, nlon = p.shape
        result = np.zeros((4, nlat, nlon), dtype=np.float64)

        for ilat in range(nlat):
            for ilon in range(nlon):
                cape, cin, lcl, lfc = _calculate_cape_cin_column(
                    p[:, ilat, ilon],
                    t[:, ilat, ilon],
                    q[:, ilat, ilon],
                    z[:, ilat, ilon],
                    zsfc[ilat, ilon] if zsfc.ndim == 2 else zsfc,
                    psfc[ilat, ilon] if psfc.ndim == 2 else psfc
                )
                result[0, ilat, ilon] = cape
                result[1, ilat, ilon] = cin
                result[2, ilat, ilon] = lcl
                result[3, ilat, ilon] = lfc

    elif p.ndim == 1:
        # Single column
        result = np.zeros(4, dtype=np.float64)
        cape, cin, lcl, lfc = _calculate_cape_cin_column(
            p, t, q, z,
            float(zsfc) if np.ndim(zsfc) == 0 else zsfc[0],
            float(psfc) if np.ndim(psfc) == 0 else psfc[0]
        )
        result[0] = cape
        result[1] = cin
        result[2] = lcl
        result[3] = lfc

    else:
        raise ValueError("Unsupported number of dimensions")

    return result


def wrf_cape_3d(p, t, q, z, zsfc, psfc, opt):
    """
    Computes CAPE and CIN for entire 3D domain.

    Parameters
    ----------
    p : array_like
        Pressure (Pa), any dimension ordering
    t : array_like
        Temperature (K), same dimensions as p
    q : array_like
        Water vapor mixing ratio (kg/kg), same dimensions as p
    z : array_like
        Geopotential height (m), same dimensions as p
    zsfc : array_like or float
        Surface height (m)
    psfc : array_like or float
        Surface pressure (hPa)
    opt : bool or int
        False for pressure level data, True for terrain-following data

    Returns
    -------
    ndarray
        Array with dimensions [2, ...] where:
        [0] = CAPE (J/kg)
        [1] = CIN (J/kg)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/wrf_cape_3d.shtml
    """
    # Use wrf_cape_2d and extract only CAPE and CIN
    full_result = wrf_cape_2d(p, t, q, z, zsfc, psfc, opt)

    # Return only CAPE and CIN (indices 0 and 1)
    return full_result[0:2]


def rip_cape_2d(p, t, q, z, zsfc, psfc, opt):
    """
    Computes CAPE, CIN, LCL, and LFC (deprecated - use wrf_cape_2d).

    Parameters
    ----------
    p : array_like
        Pressure (hPa), ordered top-to-bottom
    t : array_like
        Temperature (K)
    q : array_like
        Mixing ratio (kg/kg)
    z : array_like
        Geopotential height (m)
    zsfc : array_like or float
        Surface height (m)
    psfc : array_like or float
        Surface pressure (hPa)
    opt : bool or int
        False for pressure level data, True for terrain-following data

    Returns
    -------
    ndarray
        Array with dimensions [4, ...] where:
        [0] = CAPE (J/kg)
        [1] = CIN (J/kg)
        [2] = LCL (m)
        [3] = LFC (m)

    Notes
    -----
    This function is deprecated. Use wrf_cape_2d instead.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/rip_cape_2d.shtml
    """
    # Convert hPa to Pa for pressure
    p_pa = np.asarray(p, dtype=np.float64) * 100.0

    # Call wrf_cape_2d (it expects Pa and hPa for psfc)
    return wrf_cape_2d(p_pa, t, q, z, zsfc, psfc, opt)


def rip_cape_3d(p, t, q, z, zsfc, psfc, opt):
    """
    Computes CAPE and CIN (deprecated - use wrf_cape_3d).

    Parameters
    ----------
    p : array_like
        Pressure (hPa), ordered top-to-bottom
    t : array_like
        Temperature (K)
    q : array_like
        Mixing ratio (kg/kg)
    z : array_like
        Geopotential height (m)
    zsfc : array_like or float
        Surface height (m)
    psfc : array_like or float
        Surface pressure (hPa)
    opt : bool or int
        False for pressure level data, True for terrain-following data

    Returns
    -------
    ndarray
        Array with dimensions [2, ...] where:
        [0] = CAPE (J/kg)
        [1] = CIN (J/kg)

    Notes
    -----
    This function is deprecated. Use wrf_cape_3d instead.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/rip_cape_3d.shtml
    """
    # Convert hPa to Pa for pressure
    p_pa = np.asarray(p, dtype=np.float64) * 100.0

    # Call wrf_cape_3d
    return wrf_cape_3d(p_pa, t, q, z, zsfc, psfc, opt)


__all__ = [
    'wrf_cape_2d',
    'wrf_cape_3d',
    'rip_cape_2d',
    'rip_cape_3d'
]
