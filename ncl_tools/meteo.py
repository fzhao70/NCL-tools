"""
Meteorology functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
meteorological calculation functions with NCL-compatible signatures.

All functions use NumPy and follow NCL parameter conventions exactly.

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/meteo.shtml
"""

import numpy as np


# Constants
R_DRY = 287.058  # Gas constant for dry air (J/(kg*K))
R_VAPOR = 461.5  # Gas constant for water vapor (J/(kg*K))
CP = 1004.0  # Specific heat at constant pressure (J/(kg*K))
EPSILON = 0.622  # Ratio of molecular weights (Rd/Rv)
GAMMA = R_DRY / CP  # 0.286
GRAVITY = 9.80665  # Acceleration due to gravity (m/s^2)
EARTH_OMEGA = 7.292e-5  # Earth's angular velocity (rad/s)


def satvpr_water_bolton(t, iounit):
    """
    Calculate saturation vapor pressure over water using Bolton's equation.

    Parameters
    ----------
    t : array_like
        Temperature (units specified by iounit[0])
    iounit : array_like
        Integer array of length 2:
        - iounit[0]: Input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: Output pressure units (0=hPa, 1=Pa, 2=kPa)

    Returns
    -------
    ndarray
        Saturation vapor pressure in units specified by iounit[1]

    References
    ----------
    Bolton, D., 1980: The computation of equivalent potential temperature.
    Monthly Weather Review, 108, 1046-1053.

    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/satvpr_water_bolton.shtml
    """
    t = np.asarray(t, dtype=np.float64)
    iounit = np.asarray(iounit, dtype=np.int32)

    # Convert to Celsius
    if iounit[0] == 0:  # Already Celsius
        temp_c = t
    elif iounit[0] == 1:  # Kelvin
        temp_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        temp_c = (t - 32.0) * 5.0 / 9.0
    else:
        raise ValueError("iounit[0] must be 0 (C), 1 (K), or 2 (F)")

    # Bolton's equation (returns hPa)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

    # Convert to requested output units
    if iounit[1] == 0:  # hPa
        return es
    elif iounit[1] == 1:  # Pa
        return es * 100.0
    elif iounit[1] == 2:  # kPa
        return es / 10.0
    else:
        raise ValueError("iounit[1] must be 0 (hPa), 1 (Pa), or 2 (kPa)")


def dewtemp_trh(tk, rh):
    """
    Calculate dew point temperature from temperature and relative humidity.

    Parameters
    ----------
    tk : array_like
        Temperature in Kelvin
    rh : array_like
        Relative humidity in percent (0-100)

    Returns
    -------
    ndarray
        Dew point temperature in Kelvin

    References
    ----------
    Equations from John Dutton's "Ceaseless Wind" (pp 273-274)

    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dewtemp_trh.shtml
    """
    tk = np.asarray(tk, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    # Convert to Celsius
    temp_c = tk - 273.15

    # Calculate saturation vapor pressure using Bolton's formula
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

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

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/relhum_ttd.shtml
    """
    temp = np.asarray(temp, dtype=np.float64)
    dewpt = np.asarray(dewpt, dtype=np.float64)

    # Convert to Celsius
    temp_c = temp - 273.15
    dewpt_c = dewpt - 273.15

    # Calculate saturation vapor pressures using Bolton's formula
    es_temp = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    es_dewpt = 6.112 * np.exp(17.67 * dewpt_c / (dewpt_c + 243.5))

    # Relative humidity
    rh = 100.0 * es_dewpt / es_temp

    return rh


def mixhum_ptd(p, tdk, iswit):
    """
    Calculate mixing ratio or specific humidity from pressure and dew point.

    Parameters
    ----------
    p : array_like
        Pressure in Pascals
    tdk : array_like
        Dew point temperature in Kelvin
    iswit : int
        Output control:
        - 1: mixing ratio in kg/kg
        - 2: specific humidity in kg/kg
        - -1: mixing ratio in g/kg
        - -2: specific humidity in g/kg

    Returns
    -------
    ndarray
        Mixing ratio or specific humidity in units determined by iswit

    Notes
    -----
    Mixing ratio is the mass of water vapor per mass of dry air: w = ε * e / (p - e)
    Specific humidity is the mass of water vapor per total mass: q = ε * e / p

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/mixhum_ptd.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    tdk = np.asarray(tdk, dtype=np.float64)

    # Convert dew point to Celsius
    tdk_c = tdk - 273.15

    # Saturation vapor pressure at dew point using Bolton's formula (returns hPa)
    es = 6.112 * np.exp(17.67 * tdk_c / (tdk_c + 243.5))
    es = es * 100.0  # Convert to Pa

    # Calculate based on iswit
    abs_iswit = abs(iswit)

    if abs_iswit == 1:  # Mixing ratio
        result = EPSILON * es / (p - es)
    elif abs_iswit == 2:  # Specific humidity
        result = EPSILON * es / p
    else:
        raise ValueError("iswit must be ±1 (mixing ratio) or ±2 (specific humidity)")

    # Convert to g/kg if negative
    if iswit < 0:
        result = result * 1000.0

    return result


def mixhum_ptrh(p, tk, rh, iswit):
    """
    Calculate mixing ratio or specific humidity from pressure, temperature, and RH.

    Parameters
    ----------
    p : array_like
        Pressure in hPa (mb)
    tk : array_like
        Temperature in Kelvin
    rh : array_like
        Relative humidity in percent (0-100)
    iswit : int
        Output control:
        - 1: mixing ratio in kg/kg
        - 2: specific humidity in kg/kg
        - -1: mixing ratio in g/kg
        - -2: specific humidity in g/kg

    Returns
    -------
    ndarray
        Mixing ratio or specific humidity in units determined by iswit

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/mixhum_ptrh.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    tk = np.asarray(tk, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    # Convert to Celsius
    temp_c = tk - 273.15

    # Saturation vapor pressure using Bolton's formula (returns hPa)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

    # Actual vapor pressure (hPa)
    e = es * rh / 100.0

    # Calculate based on iswit
    abs_iswit = abs(iswit)

    if abs_iswit == 1:  # Mixing ratio
        result = EPSILON * e / (p - e)
    elif abs_iswit == 2:  # Specific humidity
        result = EPSILON * e / p
    else:
        raise ValueError("iswit must be ±1 (mixing ratio) or ±2 (specific humidity)")

    # Convert to g/kg if negative
    if iswit < 0:
        result = result * 1000.0

    return result


def relhum(t, w, p):
    """
    Calculate relative humidity from temperature, mixing ratio, and pressure.

    Parameters
    ----------
    t : array_like
        Temperature in Kelvin
    w : array_like
        Mixing ratio in kg/kg
    p : array_like
        Pressure in Pascals

    Returns
    -------
    ndarray
        Relative humidity in percent (0-100)

    Notes
    -----
    NCL uses a table look-up procedure. This implementation uses formulas.
    Between 0C and -20C, NCL calculates with respect to mixed phase.
    Below -20C, NCL calculates with respect to ice.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/relhum.shtml
    """
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # Actual vapor pressure from mixing ratio: e = (w * p) / (epsilon + w)
    e = w * p / (EPSILON + w)

    # Convert to Celsius
    temp_c = t - 273.15

    # Saturation vapor pressure using Bolton's formula (returns hPa)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    es = es * 100.0  # Convert to Pa

    # Relative humidity
    rh = 100.0 * e / es

    return rh


def temp_virtual(t, w, iounit):
    """
    Calculate virtual temperature.

    Parameters
    ----------
    t : array_like
        Temperature (units specified by iounit[0])
    w : array_like
        Mixing ratio (units specified by iounit[1])
    iounit : array_like
        Integer array of length 3:
        - iounit[0]: Input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: Input mixing ratio units (0=kg/kg, 1=g/kg)
        - iounit[2]: Output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    ndarray
        Virtual temperature in units specified by iounit[2]

    Notes
    -----
    Virtual temperature is the temperature dry air would need to have
    to have the same density as moist air at the same pressure.
    Formula: Tv = T * (1 + w/epsilon) / (1 + w) ≈ T * (1 + 0.61*w)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/temp_virtual.shtml
    """
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    iounit = np.asarray(iounit, dtype=np.int32)

    # Convert temperature to Kelvin
    if iounit[0] == 0:  # Celsius
        temp_k = t + 273.15
    elif iounit[0] == 1:  # Kelvin
        temp_k = t
    elif iounit[0] == 2:  # Fahrenheit
        temp_k = (t - 32.0) * 5.0 / 9.0 + 273.15
    else:
        raise ValueError("iounit[0] must be 0 (C), 1 (K), or 2 (F)")

    # Convert mixing ratio to kg/kg
    if iounit[1] == 0:  # kg/kg
        mixr = w
    elif iounit[1] == 1:  # g/kg
        mixr = w / 1000.0
    else:
        raise ValueError("iounit[1] must be 0 (kg/kg) or 1 (g/kg)")

    # Calculate virtual temperature in Kelvin
    # Using the approximation: Tv ≈ T * (1 + 0.61*w)
    tv_k = temp_k * (1.0 + 0.61 * mixr)

    # Convert to output units
    if iounit[2] == 0:  # Celsius
        return tv_k - 273.15
    elif iounit[2] == 1:  # Kelvin
        return tv_k
    elif iounit[2] == 2:  # Fahrenheit
        return (tv_k - 273.15) * 9.0 / 5.0 + 32.0
    else:
        raise ValueError("iounit[2] must be 0 (C), 1 (K), or 2 (F)")


def pot_temp(p, t, dim, opt):
    """
    Calculate potential temperature.

    Parameters
    ----------
    p : array_like
        Pressure in Pascals
    t : array_like
        Temperature in Kelvin
    dim : int
        Dimension of t that corresponds to p. Use -1 if p and t have same shape.
    opt : bool
        Currently not used. Set to False.

    Returns
    -------
    ndarray
        Potential temperature in Kelvin

    Notes
    -----
    Potential temperature is the temperature a parcel would have if
    brought adiabatically to a reference pressure (1000 hPa).
    Formula: theta = T * (p0/p)^(R/Cp) where p0 = 100000 Pa

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/pot_temp.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Reference pressure (1000 hPa = 100000 Pa)
    p0 = 100000.0

    # Theta = T * (p0/p)^(R/Cp)
    theta = t * np.power(p0 / p, GAMMA)

    return theta


def dpres_plevel(plev, psfc, ptop, iopt):
    """
    Calculate pressure layer thickness of constant pressure level coordinate system.

    Parameters
    ----------
    plev : array_like
        One-dimensional array of pressure levels in Pascals (ascending or descending)
    psfc : array_like
        Surface pressure in Pascals (scalar or up to 3D with rightmost dims as lat,lon)
    ptop : float
        Pressure at top of column in Pascals (should be <= min(plev))
    iopt : int
        Currently not used. Set to 0.

    Returns
    -------
    ndarray
        Pressure layer thickness in Pascals

    Notes
    -----
    At each grid point, sum of all layer thicknesses equals (psfc - ptop).
    Values above ptop and below psfc are set to fill values.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dpres_plevel.shtml
    """
    plev = np.asarray(plev, dtype=np.float64)
    psfc = np.asarray(psfc, dtype=np.float64)
    ptop = float(ptop)

    nlev = len(plev)

    # Determine output shape
    if psfc.ndim == 0:  # scalar
        out_shape = (nlev,)
    elif psfc.ndim == 2:  # (lat, lon)
        out_shape = (nlev,) + psfc.shape
    elif psfc.ndim == 3:  # (time, lat, lon)
        out_shape = (nlev,) + psfc.shape[1:]
        # For 3D, need to handle time dimension
        ntime = psfc.shape[0]
        result = np.zeros((ntime, nlev) + psfc.shape[1:], dtype=np.float64)
        for t in range(ntime):
            result[t] = dpres_plevel(plev, psfc[t], ptop, iopt)
        return result
    else:
        raise ValueError("psfc must be scalar, 2D, or 3D")

    dp = np.zeros(out_shape, dtype=np.float64)

    # Determine if pressure is ascending or descending
    if plev[1] > plev[0]:  # ascending (bottom to top)
        ascending = True
    else:  # descending (top to bottom)
        ascending = False

    # For each level, calculate thickness
    if psfc.ndim == 0:  # scalar case
        for k in range(nlev):
            if ascending:
                p_bot = plev[k-1] if k > 0 else psfc
                p_top = plev[k]
            else:
                p_top = plev[k]
                p_bot = plev[k+1] if k < nlev-1 else psfc

            # Check if level is within valid range
            if p_top < ptop or p_bot > psfc:
                dp[k] = np.nan
            else:
                # Adjust boundaries
                p_bot = min(p_bot, psfc)
                p_top = max(p_top, ptop)
                dp[k] = p_bot - p_top
    else:  # 2D case (lat, lon)
        for k in range(nlev):
            if ascending:
                p_bot = plev[k-1] if k > 0 else psfc
                p_top = plev[k]
            else:
                p_top = plev[k]
                p_bot = plev[k+1] if k < nlev-1 else psfc

            # Vectorized boundary checks
            valid = (p_top >= ptop) & (p_bot <= psfc)

            if np.isscalar(p_bot):
                p_bot_adj = np.minimum(p_bot, psfc)
            else:
                p_bot_adj = p_bot.copy()

            if np.isscalar(p_top):
                p_top_adj = np.maximum(p_top, ptop)
            else:
                p_top_adj = p_top

            dp[k] = np.where(valid, p_bot_adj - p_top_adj, np.nan)

    return dp


def pres_sigma(sigma, ps):
    """
    Calculate pressure at specified sigma levels.

    Parameters
    ----------
    sigma : array_like
        One-dimensional array of sigma levels (0 to 1)
    ps : array_like
        Surface pressure (at least 2D with rightmost dims as lat,lon)

    Returns
    -------
    ndarray
        Pressure at sigma levels

    Notes
    -----
    Sigma coordinates: sigma = p / ps
    Therefore: p = sigma * ps

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/pres_sigma.shtml
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)

    # Reshape sigma for broadcasting
    # ps is (..., lat, lon), we need sigma to be (nsigma, 1, 1)
    sigma_bc = sigma.reshape(-1, *([1] * (ps.ndim)))

    # p = sigma * ps
    pres = sigma_bc * ps

    return pres


def omega_to_w(omega, p, t):
    """
    Convert omega vertical velocity (Pa/s) to w (m/s).

    Parameters
    ----------
    omega : array_like
        Omega vertical velocity in Pa/s
    p : array_like
        Pressure in Pascals
    t : array_like
        Temperature in Kelvin

    Returns
    -------
    ndarray
        Vertical velocity in m/s

    Notes
    -----
    Formula: w = -omega / (rho * g) where rho = p / (R * T)
    Therefore: w = -omega * R * T / (p * g)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/omega_to_w.shtml
    """
    omega = np.asarray(omega, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # w = -omega * R * T / (p * g)
    w = -omega * R_DRY * t / (p * GRAVITY)

    return w


def w_to_omega(w, p, t):
    """
    Convert vertical velocity (m/s) to omega (Pa/s).

    Parameters
    ----------
    w : array_like
        Vertical velocity in m/s
    p : array_like
        Pressure in Pascals
    t : array_like
        Temperature in Kelvin

    Returns
    -------
    ndarray
        Omega vertical velocity in Pa/s

    Notes
    -----
    Formula: omega = -w * rho * g where rho = p / (R * T)
    Therefore: omega = -w * p * g / (R * T)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/w_to_omega.shtml
    """
    w = np.asarray(w, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # omega = -w * p * g / (R * T)
    omega = -w * p * GRAVITY / (R_DRY * t)

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

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/wind_speed.shtml
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    return np.sqrt(u**2 + v**2)


def wind_direction(u, v, opt):
    """
    Calculate meteorological wind direction from zonal and meridional components.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s
    v : array_like
        Meridional wind component in m/s
    opt : int or float
        How to handle calm winds (u=v=0):
        - 0: return 0.0
        - 1: return missing value (NaN)
        - other: return that value

    Returns
    -------
    ndarray
        Wind direction in degrees (0-360), where 0/360 is north

    Notes
    -----
    Meteorological convention: direction wind is coming FROM.
    0° = North, 90° = East, 180° = South, 270° = West

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/wind_direction.shtml
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

    # Handle calm winds (u=v=0)
    calm = (u == 0) & (v == 0)
    if np.any(calm):
        if opt == 0:
            calm_value = 0.0
        elif opt == 1:
            calm_value = np.nan
        else:
            calm_value = float(opt)
        dir_met = np.where(calm, calm_value, dir_met)

    return dir_met


def wind_component(wspd, wdir, opt):
    """
    Calculate zonal and meridional wind components from speed and direction.

    Parameters
    ----------
    wspd : array_like
        Wind speed in m/s
    wdir : array_like
        Meteorological wind direction in degrees (0-360)
    opt : int
        Currently not used. Set to 0.

    Returns
    -------
    tuple
        (u, v) - zonal and meridional wind components in m/s

    Notes
    -----
    Input direction is meteorological convention (direction FROM which wind blows).

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/wind_component.shtml
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
    Formula: f = 2 * Omega * sin(latitude)
    where Omega = 7.292e-5 rad/s is Earth's angular velocity

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/coriolis_param.shtml
    """
    lat = np.asarray(lat, dtype=np.float64)

    # Convert to radians
    lat_rad = lat * np.pi / 180.0

    # f = 2 * Omega * sin(lat)
    f = 2.0 * EARTH_OMEGA * np.sin(lat_rad)

    return f


def hydro(p, tkv, zsfc):
    """
    Calculate geopotential height using the hydrostatic equation.

    Parameters
    ----------
    p : array_like
        Pressure in mb (hPa), with rightmost dimension as level.
        Order must be bottom-to-top. Must include surface pressure.
    tkv : array_like
        Virtual temperature in Kelvin, same size as p.
        Order must be bottom-to-top. Must include surface temperature.
    zsfc : array_like
        Surface geopotential height in gpm (geopotential meters).
        Size is same as p minus the rightmost (level) dimension.

    Returns
    -------
    ndarray
        Geopotential height in gpm (geopotential meters)

    Notes
    -----
    Integrates the hydrostatic equation vertically using average layer temperatures.
    Uses g = 9.80655 m/s^2 (WMO value at 45 degrees latitude).

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/hydro.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    tkv = np.asarray(tkv, dtype=np.float64)
    zsfc = np.asarray(zsfc, dtype=np.float64)

    # Convert pressure from mb to Pa
    p_pa = p * 100.0

    nlev = p.shape[-1]
    z = np.zeros_like(p)

    # Set surface level
    z[..., 0] = zsfc

    # Integrate upward using hypsometric equation
    # dz = (R * Tv_avg / g) * ln(p_lower / p_upper)
    for k in range(1, nlev):
        # Average virtual temperature between levels
        tv_avg = (tkv[..., k] + tkv[..., k-1]) / 2.0

        # Thickness of layer
        dz = (R_DRY * tv_avg / GRAVITY) * np.log(p_pa[..., k-1] / p_pa[..., k])

        z[..., k] = z[..., k-1] + dz

    return z


def prcwater_dp(q, dp):
    """
    Calculate total column precipitable water.

    Parameters
    ----------
    q : array_like
        Specific humidity in kg/kg (rightmost dimension must be level)
    dp : array_like
        Pressure layer thickness in Pa (same size as q or 1D with level dimension)

    Returns
    -------
    ndarray
        Precipitable water in kg/m^2

    Notes
    -----
    Formula: PW = (1/g) * integral(q * dp)
    Rightmost dimension is assumed to be the level dimension.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/prcwater_dp.shtml
    """
    q = np.asarray(q, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)

    # Integrate along rightmost dimension: PW = (1/g) * sum(q * dp)
    pw = np.sum(q * dp, axis=-1) / GRAVITY

    return pw


def static_stability(p, t, dim, sopt):
    """
    Calculate static stability parameter.

    Parameters
    ----------
    p : array_like
        Pressure in Pascals
    t : array_like
        Temperature in Kelvin
    dim : int
        Dimension of t that corresponds to p
    sopt : int
        Option flag:
        - 0: return static stability only
        - 1: return list with [static_stability, theta, dthdp]

    Returns
    -------
    ndarray or list
        If sopt=0: static stability in K/Pa
        If sopt=1: list containing [sigma, theta, dthetadp]

    Notes
    -----
    Static stability: sigma = -T/theta * d(theta)/dp
    where theta is potential temperature.
    Measures gravitational resistance of atmosphere to vertical displacements.

    References
    ----------
    Bluestein (1992): pg 197, eqn 4.3.8
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/static_stability.shtml
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # Calculate potential temperature
    theta = pot_temp(p, t, dim, False)

    # Calculate d(theta)/dp using gradient
    dtheta_dp = np.gradient(theta, axis=dim) / np.gradient(p, axis=dim)

    # Static stability: sigma = -(T/theta) * d(theta)/dp
    sigma = -(t / theta) * dtheta_dp

    if sopt == 0:
        return sigma
    elif sopt == 1:
        return [sigma, theta, dtheta_dp]
    else:
        raise ValueError("sopt must be 0 or 1")


def uv2dv_cfd(u, v, lat, lon, boundOpt):
    """
    Calculate divergence from u and v using centered finite differences.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s (rightmost 2 dims must be lat, lon)
    v : array_like
        Meridional wind component in m/s (rightmost 2 dims must be lat, lon)
    lat : array_like
        Latitude in degrees (1D)
    lon : array_like
        Longitude in degrees (1D)
    boundOpt : int
        Boundary condition option:
        - 0: boundary points set to missing
        - 1: assume cyclic in longitude
        - 2: use one-sided differences at boundaries
        - 3: cyclic in longitude + one-sided at lat boundaries

    Returns
    -------
    ndarray
        Divergence in 1/s

    Notes
    -----
    Uses centered finite differences on a sphere.
    For global grids, spherical harmonic functions are more accurate.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/uv2dv_cfd.shtml
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Earth radius
    R = 6.371e6  # meters

    # Create 2D lat/lon grids
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    lat_rad = lat_2d * np.pi / 180.0
    lon_rad = lon_2d * np.pi / 180.0
    cos_lat = np.cos(lat_rad)

    # Calculate gradients
    # du/dlon
    du_dlon = np.gradient(u, axis=-1) / np.gradient(lon_rad, axis=-1)

    # dv/dlat
    dv_dlat = np.gradient(v, axis=-2) / np.gradient(lat_rad, axis=-2)

    # Divergence on sphere: div = (1/(R*cos(lat))) * du/dlon + (1/R) * d(v*cos(lat))/dlat
    # Simplified: div = (1/(R*cos(lat))) * du/dlon + (1/R) * dv/dlat
    div = (1.0 / (R * cos_lat)) * du_dlon + (1.0 / R) * dv_dlat

    # Apply boundary conditions
    if boundOpt == 0:
        # Set boundaries to NaN
        div[..., 0, :] = np.nan
        div[..., -1, :] = np.nan
        div[..., :, 0] = np.nan
        div[..., :, -1] = np.nan
    elif boundOpt == 1:
        # Cyclic in longitude, boundaries in lat set to NaN
        div[..., 0, :] = np.nan
        div[..., -1, :] = np.nan
    elif boundOpt == 2:
        # Use one-sided differences (already handled by gradient)
        pass
    elif boundOpt == 3:
        # Cyclic in longitude, one-sided in latitude
        # Gradient already uses one-sided at boundaries
        pass
    else:
        raise ValueError("boundOpt must be 0, 1, 2, or 3")

    return div


def uv2vr_cfd(u, v, lat, lon, boundOpt):
    """
    Calculate relative vorticity from u and v using centered finite differences.

    Parameters
    ----------
    u : array_like
        Zonal wind component in m/s (rightmost 2 dims must be lat, lon)
    v : array_like
        Meridional wind component in m/s (rightmost 2 dims must be lat, lon)
    lat : array_like
        Latitude in degrees (1D)
    lon : array_like
        Longitude in degrees (1D)
    boundOpt : int
        Boundary condition option:
        - 0: boundary points set to missing
        - 1: assume cyclic in longitude
        - 2: use one-sided differences at boundaries
        - 3: cyclic in longitude + one-sided at lat boundaries

    Returns
    -------
    ndarray
        Relative vorticity in 1/s

    Notes
    -----
    Uses centered finite differences on a sphere.
    For global grids, spherical harmonic functions are more accurate.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/uv2vr_cfd.shtml
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Earth radius
    R = 6.371e6  # meters

    # Create 2D lat/lon grids
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    lat_rad = lat_2d * np.pi / 180.0
    lon_rad = lon_2d * np.pi / 180.0
    cos_lat = np.cos(lat_rad)

    # Calculate gradients
    # dv/dlon
    dv_dlon = np.gradient(v, axis=-1) / np.gradient(lon_rad, axis=-1)

    # du/dlat
    du_dlat = np.gradient(u, axis=-2) / np.gradient(lat_rad, axis=-2)

    # Vorticity on sphere: vor = (1/(R*cos(lat))) * dv/dlon - (1/R) * d(u*cos(lat))/dlat
    # Simplified: vor = (1/(R*cos(lat))) * dv/dlon - (1/R) * du/dlat
    vor = (1.0 / (R * cos_lat)) * dv_dlon - (1.0 / R) * du_dlat

    # Apply boundary conditions
    if boundOpt == 0:
        # Set boundaries to NaN
        vor[..., 0, :] = np.nan
        vor[..., -1, :] = np.nan
        vor[..., :, 0] = np.nan
        vor[..., :, -1] = np.nan
    elif boundOpt == 1:
        # Cyclic in longitude, boundaries in lat set to NaN
        vor[..., 0, :] = np.nan
        vor[..., -1, :] = np.nan
    elif boundOpt == 2:
        # Use one-sided differences (already handled by gradient)
        pass
    elif boundOpt == 3:
        # Cyclic in longitude, one-sided in latitude
        pass
    else:
        raise ValueError("boundOpt must be 0, 1, 2, or 3")

    return vor


def wetbulb_stull(t, rh, iounit, opt):
    """
    Calculate wet bulb temperature using Stull's method.

    Parameters
    ----------
    t : array_like
        Temperature (units specified by iounit[0])
    rh : array_like
        Relative humidity in percent (0-100)
    iounit : array_like
        Integer array of length 2:
        - iounit[0]: Input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: Output temperature units (0=°C, 1=K, 2=°F)
    opt : bool
        Currently not used. Set to False.

    Returns
    -------
    ndarray
        Wet bulb temperature in units specified by iounit[1]

    References
    ----------
    Stull, R., 2011: Wet-Bulb Temperature from Relative Humidity and Air Temperature.
    J. Appl. Meteor. Climatol., 50, 2267-2269.

    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/wetbulb_stull.shtml
    """
    t = np.asarray(t, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    iounit = np.asarray(iounit, dtype=np.int32)

    # Convert to Celsius
    if iounit[0] == 0:  # Already Celsius
        temp_c = t
    elif iounit[0] == 1:  # Kelvin
        temp_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        temp_c = (t - 32.0) * 5.0 / 9.0
    else:
        raise ValueError("iounit[0] must be 0 (C), 1 (K), or 2 (F)")

    # Stull's formula (empirical fit)
    tw = temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + \
         0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh) - \
         np.arctan(rh - 1.676331) + \
         np.arctan(temp_c + rh) - 4.686035

    # Convert to requested output units
    if iounit[1] == 0:  # Celsius
        return tw
    elif iounit[1] == 1:  # Kelvin
        return tw + 273.15
    elif iounit[1] == 2:  # Fahrenheit
        return tw * 9.0 / 5.0 + 32.0
    else:
        raise ValueError("iounit[1] must be 0 (C), 1 (K), or 2 (F)")
