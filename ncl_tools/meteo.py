"""
Meteorology functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
meteorological calculation functions.

All functions use NumPy and SciPy for robust implementations.
"""

import numpy as np


# Constants
R_DRY = 287.04  # Gas constant for dry air (J/(kg*K))
R_VAPOR = 461.5  # Gas constant for water vapor (J/(kg*K))
CP = 1004.0  # Specific heat at constant pressure (J/(kg*K))
EPSILON = 0.622  # Ratio of molecular weights (Rd/Rv)
GAMMA = 0.286  # Rd/Cp
GRAVITY = 9.80665  # Acceleration due to gravity (m/s^2)
EARTH_OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)


def satvpr_water_bolton(temp):
    """
    Calculate saturation vapor pressure over water using Bolton's equation.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin

    Returns
    -------
    ndarray
        Saturation vapor pressure in Pascals

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Monthly Weather Review, 108, 1046-1053.
    """
    temp = np.asarray(temp, dtype=np.float64)

    # Convert to Celsius
    temp_c = temp - 273.15

    # Bolton's equation (returns hPa, convert to Pa)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

    return es * 100.0  # Convert hPa to Pa


def dewtemp_trh(temp, rh):
    """
    Calculate dew point temperature from temperature and relative humidity.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    rh : array_like
        Relative humidity in percent (0-100)

    Returns
    -------
    ndarray
        Dew point temperature in Kelvin

    References
    ----------
    Magnus formula approximation
    """
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    # Convert to Celsius
    temp_c = temp - 273.15

    # Calculate saturation vapor pressure
    es = satvpr_water_bolton(temp) / 100.0  # Convert to hPa

    # Actual vapor pressure
    e = es * rh / 100.0

    # Magnus formula inverted for dew point
    dewpoint_c = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

    return dewpoint_c + 273.15  # Convert back to Kelvin


def relhum_ttd(temp, dewpt):
    """
    Calculate relative humidity from temperature and dew point temperature.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    dewpt : array_like
        Dew point temperature in Kelvin

    Returns
    -------
    ndarray
        Relative humidity in percent (0-100)
    """
    temp = np.asarray(temp, dtype=np.float64)
    dewpt = np.asarray(dewpt, dtype=np.float64)

    # Calculate saturation vapor pressures
    es_temp = satvpr_water_bolton(temp)
    es_dewpt = satvpr_water_bolton(dewpt)

    # Relative humidity
    rh = 100.0 * es_dewpt / es_temp

    return rh


def mixhum_ptd(pres, dewpt):
    """
    Calculate mixing ratio given pressure and dew point temperature.

    Parameters
    ----------
    pres : array_like
        Pressure in Pascals
    dewpt : array_like
        Dew point temperature in Kelvin

    Returns
    -------
    ndarray
        Mixing ratio in kg/kg

    Notes
    -----
    Mixing ratio is the mass of water vapor per mass of dry air.
    """
    pres = np.asarray(pres, dtype=np.float64)
    dewpt = np.asarray(dewpt, dtype=np.float64)

    # Saturation vapor pressure at dew point
    es = satvpr_water_bolton(dewpt)

    # Mixing ratio formula: w = epsilon * e / (p - e)
    mixr = EPSILON * es / (pres - es)

    return mixr


def mixhum_ptrh(pres, temp, rh):
    """
    Calculate mixing ratio given pressure, temperature, and relative humidity.

    Parameters
    ----------
    pres : array_like
        Pressure in Pascals
    temp : array_like
        Temperature in Kelvin
    rh : array_like
        Relative humidity in percent (0-100)

    Returns
    -------
    ndarray
        Mixing ratio in kg/kg
    """
    pres = np.asarray(pres, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    # Saturation vapor pressure
    es = satvpr_water_bolton(temp)

    # Actual vapor pressure
    e = es * rh / 100.0

    # Mixing ratio
    mixr = EPSILON * e / (pres - e)

    return mixr


def relhum(temp, mixr, pres):
    """
    Calculate relative humidity from temperature, mixing ratio, and pressure.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    mixr : array_like
        Mixing ratio in kg/kg
    pres : array_like
        Pressure in Pascals

    Returns
    -------
    ndarray
        Relative humidity in percent (0-100)
    """
    temp = np.asarray(temp, dtype=np.float64)
    mixr = np.asarray(mixr, dtype=np.float64)
    pres = np.asarray(pres, dtype=np.float64)

    # Actual vapor pressure from mixing ratio: e = (w * p) / (epsilon + w)
    e = mixr * pres / (EPSILON + mixr)

    # Saturation vapor pressure
    es = satvpr_water_bolton(temp)

    # Relative humidity
    rh = 100.0 * e / es

    return rh


def temp_virtual(temp, mixr):
    """
    Calculate virtual temperature.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    mixr : array_like
        Mixing ratio in kg/kg

    Returns
    -------
    ndarray
        Virtual temperature in Kelvin

    Notes
    -----
    Virtual temperature is the temperature dry air would need to have
    to have the same density as moist air at the same pressure.
    """
    temp = np.asarray(temp, dtype=np.float64)
    mixr = np.asarray(mixr, dtype=np.float64)

    # Tv = T * (1 + w/epsilon) / (1 + w)
    tv = temp * (1.0 + mixr / EPSILON) / (1.0 + mixr)

    return tv


def pot_temp(temp, pres, pref=100000.0):
    """
    Calculate potential temperature.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    pres : array_like
        Pressure in Pascals
    pref : float, optional
        Reference pressure in Pascals (default: 100000 Pa = 1000 hPa)

    Returns
    -------
    ndarray
        Potential temperature in Kelvin

    Notes
    -----
    Potential temperature is the temperature a parcel would have if
    brought adiabatically to a reference pressure (usually 1000 hPa).
    """
    temp = np.asarray(temp, dtype=np.float64)
    pres = np.asarray(pres, dtype=np.float64)

    # Theta = T * (p0/p)^(R/Cp)
    theta = temp * np.power(pref / pres, GAMMA)

    return theta


def dpres_plevel(plev):
    """
    Calculate pressure layer thickness of constant pressure level coordinate system.

    Parameters
    ----------
    plev : array_like
        Pressure levels in Pascals (1D array)

    Returns
    -------
    ndarray
        Pressure layer thickness in Pascals

    Notes
    -----
    Returns an array the same size as plev containing the pressure
    differences between adjacent levels. The returned array has the
    form: dp(k) = p(k) - p(k-1) for k > 0, and dp(0) = 0.
    """
    plev = np.asarray(plev, dtype=np.float64)

    dp = np.zeros_like(plev)

    if len(plev) > 1:
        # dp(k) = abs(p(k+1) - p(k))
        dp[:-1] = np.abs(np.diff(plev))
        dp[-1] = dp[-2]  # Extrapolate last level

    return dp


def pres_sigma(psfc, sigma, ptop=0.0):
    """
    Calculate pressure at specified sigma levels.

    Parameters
    ----------
    psfc : array_like
        Surface pressure in Pascals
    sigma : array_like
        Sigma levels (0 to 1)
    ptop : float, optional
        Pressure at top of model in Pascals (default: 0)

    Returns
    -------
    ndarray
        Pressure at sigma levels in Pascals

    Notes
    -----
    Sigma coordinates: sigma = (p - ptop) / (psfc - ptop)
    Therefore: p = ptop + sigma * (psfc - ptop)
    """
    psfc = np.asarray(psfc, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # p = ptop + sigma * (psfc - ptop)
    pres = ptop + sigma * (psfc - ptop)

    return pres


def omega_to_w(omega, pres, temp):
    """
    Convert omega vertical velocity (Pa/s) to w (m/s).

    Parameters
    ----------
    omega : array_like
        Omega vertical velocity in Pa/s
    pres : array_like
        Pressure in Pascals
    temp : array_like
        Temperature in Kelvin

    Returns
    -------
    ndarray
        Vertical velocity in m/s

    Notes
    -----
    w = -omega * (R * T) / (p * g)
    """
    omega = np.asarray(omega, dtype=np.float64)
    pres = np.asarray(pres, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # w = -omega * R * T / (p * g)
    w = -omega * R_DRY * temp / (pres * GRAVITY)

    return w


def w_to_omega(w, pres, temp):
    """
    Convert vertical velocity (m/s) to omega (Pa/s).

    Parameters
    ----------
    w : array_like
        Vertical velocity in m/s
    pres : array_like
        Pressure in Pascals
    temp : array_like
        Temperature in Kelvin

    Returns
    -------
    ndarray
        Omega vertical velocity in Pa/s

    Notes
    -----
    omega = -w * (p * g) / (R * T)
    """
    w = np.asarray(w, dtype=np.float64)
    pres = np.asarray(pres, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # omega = -w * p * g / (R * T)
    omega = -w * pres * GRAVITY / (R_DRY * temp)

    return omega


def wind_speed(u, v):
    """
    Calculate wind speed from zonal and meridional components.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s
    v : array_like
        Meridional wind component in m/s

    Returns
    -------
    ndarray
        Wind speed in m/s
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    return np.sqrt(u**2 + v**2)


def wind_direction(u, v):
    """
    Calculate meteorological wind direction from zonal and meridional components.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s
    v : array_like
        Meridional wind component in m/s

    Returns
    -------
    ndarray
        Wind direction in degrees (0-360), where 0/360 is north

    Notes
    -----
    Meteorological convention: direction wind is coming FROM.
    0° = North, 90° = East, 180° = South, 270° = West
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Calculate direction (math convention: counterclockwise from east)
    dir_math = np.arctan2(v, u) * 180.0 / np.pi

    # Convert to meteorological convention (clockwise from north)
    # Direction wind is coming FROM
    dir_met = 270.0 - dir_math

    # Normalize to 0-360
    dir_met = np.where(dir_met < 0, dir_met + 360.0, dir_met)
    dir_met = np.where(dir_met >= 360, dir_met - 360.0, dir_met)

    return dir_met


def wind_component(wspd, wdir):
    """
    Calculate zonal and meridional wind components from speed and direction.

    Parameters
    ----------
    wspd : array_like
        Wind speed in m/s
    wdir : array_like
        Meteorological wind direction in degrees (0-360)

    Returns
    -------
    tuple
        (u, v) - zonal and meridional wind components in m/s

    Notes
    -----
    Input direction is meteorological convention (direction FROM which wind blows).
    """
    wspd = np.asarray(wspd, dtype=np.float64)
    wdir = np.asarray(wdir, dtype=np.float64)

    # Convert meteorological direction to mathematical angle
    angle = (270.0 - wdir) * np.pi / 180.0

    # Calculate components
    u = wspd * np.cos(angle)
    v = wspd * np.sin(angle)

    return (u, v)


def coriolis_param(lat):
    """
    Calculate Coriolis parameter.

    Parameters
    ----------
    lat : array_like
        Latitude in degrees

    Returns
    -------
    ndarray
        Coriolis parameter in 1/s

    Notes
    -----
    f = 2 * Omega * sin(latitude)
    where Omega is Earth's angular velocity
    """
    lat = np.asarray(lat, dtype=np.float64)

    # Convert to radians
    lat_rad = lat * np.pi / 180.0

    # f = 2 * Omega * sin(lat)
    f = 2.0 * EARTH_OMEGA * np.sin(lat_rad)

    return f


def hydro(pres, temp, q, z_bot, lev_dim=-1):
    """
    Calculate geopotential height using the hydrostatic equation.

    Parameters
    ----------
    pres : array_like
        Pressure in Pascals (must be monotonically increasing or decreasing)
    temp : array_like
        Temperature in Kelvin
    q : array_like
        Specific humidity in kg/kg
    z_bot : float or array_like
        Height at bottom level in meters
    lev_dim : int, optional
        Dimension index for vertical levels (default: -1)

    Returns
    -------
    ndarray
        Geopotential height in meters

    Notes
    -----
    Integrates the hydrostatic equation vertically.
    """
    pres = np.asarray(pres, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Calculate virtual temperature
    tv = temp * (1.0 + 0.61 * q)

    # Move level dimension to end if necessary
    if lev_dim != -1:
        pres = np.moveaxis(pres, lev_dim, -1)
        tv = np.moveaxis(tv, lev_dim, -1)

    nlev = pres.shape[-1]
    z = np.zeros_like(pres)

    # Set bottom level
    z[..., 0] = z_bot

    # Integrate upward (assuming pressure decreases with height)
    for k in range(1, nlev):
        # dz = -R * Tv * dln(p)
        dz = -R_DRY * (tv[..., k] + tv[..., k-1]) / 2.0 * \
             np.log(pres[..., k] / pres[..., k-1])
        z[..., k] = z[..., k-1] + dz

    # Move dimension back if necessary
    if lev_dim != -1:
        z = np.moveaxis(z, -1, lev_dim)

    return z


def prcwater_dp(q, dp, lev_dim=-1):
    """
    Calculate total column precipitable water.

    Parameters
    ----------
    q : array_like
        Specific humidity in kg/kg
    dp : array_like
        Pressure layer thickness in Pascals
    lev_dim : int, optional
        Dimension index for vertical levels (default: -1)

    Returns
    -------
    ndarray
        Precipitable water in kg/m^2

    Notes
    -----
    PW = (1/g) * integral(q * dp)
    """
    q = np.asarray(q, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)

    # Integrate: PW = (1/g) * sum(q * dp)
    pw = np.sum(q * dp, axis=lev_dim) / GRAVITY

    return pw


def static_stability(temp, pres, lev_dim=-1):
    """
    Calculate static stability parameter.

    Parameters
    ----------
    temp : array_like
        Temperature in Kelvin
    pres : array_like
        Pressure in Pascals
    lev_dim : int, optional
        Dimension index for vertical levels (default: -1)

    Returns
    -------
    ndarray
        Static stability parameter in K/Pa

    Notes
    -----
    Static stability = -T/theta * d(theta)/dp
    where theta is potential temperature
    """
    temp = np.asarray(temp, dtype=np.float64)
    pres = np.asarray(pres, dtype=np.float64)

    # Calculate potential temperature
    theta = pot_temp(temp, pres)

    # Calculate d(theta)/dp
    dtheta_dp = np.gradient(theta, axis=lev_dim) / np.gradient(pres, axis=lev_dim)

    # Static stability
    sigma = -temp / theta * dtheta_dp

    return sigma


def uv2dv_cfd(u, v, lat, lon):
    """
    Calculate divergence from u and v using centered finite differences.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s
    v : array_like
        Meridional wind component in m/s
    lat : array_like
        Latitude in degrees (1D or 2D)
    lon : array_like
        Longitude in degrees (1D or 2D)

    Returns
    -------
    ndarray
        Divergence in 1/s

    Notes
    -----
    Assumes rightmost two dimensions are (lat, lon) or (y, x).
    Uses centered finite differences on a sphere.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Earth radius
    R = 6.371e6  # meters

    # Convert to radians
    if lat.ndim == 1 and lon.ndim == 1:
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    else:
        lat_2d = lat
        lon_2d = lon

    lat_rad = lat_2d * np.pi / 180.0
    lon_rad = lon_2d * np.pi / 180.0

    # Calculate gradients
    # du/dx
    du_dlon = np.gradient(u, axis=-1) / np.gradient(lon_rad, axis=-1)

    # dv/dy
    dv_dlat = np.gradient(v, axis=-2) / np.gradient(lat_rad, axis=-2)

    # Divergence on sphere: div = (1/(R*cos(lat))) * du/dlon + (1/R) * d(v*cos(lat))/dlat
    cos_lat = np.cos(lat_rad)

    # Simplified divergence
    div = (1.0 / (R * cos_lat)) * du_dlon + (1.0 / R) * dv_dlat

    return div


def uv2vr_cfd(u, v, lat, lon):
    """
    Calculate relative vorticity from u and v using centered finite differences.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s
    v : array_like
        Meridional wind component in m/s
    lat : array_like
        Latitude in degrees (1D or 2D)
    lon : array_like
        Longitude in degrees (1D or 2D)

    Returns
    -------
    ndarray
        Relative vorticity in 1/s

    Notes
    -----
    Assumes rightmost two dimensions are (lat, lon) or (y, x).
    Uses centered finite differences on a sphere.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Earth radius
    R = 6.371e6  # meters

    # Convert to radians
    if lat.ndim == 1 and lon.ndim == 1:
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    else:
        lat_2d = lat
        lon_2d = lon

    lat_rad = lat_2d * np.pi / 180.0
    lon_rad = lon_2d * np.pi / 180.0

    # Calculate gradients
    # dv/dx
    dv_dlon = np.gradient(v, axis=-1) / np.gradient(lon_rad, axis=-1)

    # du/dy
    du_dlat = np.gradient(u, axis=-2) / np.gradient(lat_rad, axis=-2)

    # Vorticity on sphere: vor = (1/(R*cos(lat))) * dv/dlon - (1/R) * d(u*cos(lat))/dlat
    cos_lat = np.cos(lat_rad)

    # Simplified vorticity
    vor = (1.0 / (R * cos_lat)) * dv_dlon - (1.0 / R) * du_dlat

    return vor


def wetbulb_stull(temp, rh):
    """
    Calculate wet bulb temperature using Stull's method.

    Parameters
    ----------
    temp : array_like
        Temperature in Celsius
    rh : array_like
        Relative humidity in percent (0-100)

    Returns
    -------
    ndarray
        Wet bulb temperature in Celsius

    References
    ----------
    Stull, R., 2011: Wet-Bulb Temperature from Relative Humidity and Air Temperature.
    J. Appl. Meteor. Climatol., 50, 2267-2269.
    """
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    # Stull's formula (empirical fit)
    tw = temp * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + \
         0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh) - \
         np.arctan(rh - 1.676331) + \
         np.arctan(temp + rh) - 4.686035

    return tw
