"""
NCL Climatology Functions

This module provides Python implementations of NCL's climatology functions,
including monthly/daily climatologies, anomalies, seasonal means, and
annual cycle removal.

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/climo.shtml
"""

import numpy as np
from typing import Union, Optional


# ==============================================================================
# MONTHLY CLIMATOLOGY FUNCTIONS
# ==============================================================================

def clmMonTLL(x):
    """
    Calculates long term monthly means (monthly climatology) from monthly data.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with dimensions [12, lat, lon] containing monthly climatology

    Notes
    -----
    Computes monthly climatology by averaging all January values together,
    all February values together, etc. across multiple years.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [time, lat, lon]")

    ntime, nlat, nlon = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [nyears, 12, lat, lon] and take mean over years
    x_reshaped = x.reshape(nyears, 12, nlat, nlon)
    clm = np.mean(x_reshaped, axis=0)

    return clm


def clmMonTLLL(x):
    """
    Calculates long term monthly means (monthly climatology) from monthly data.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [time, lev, lat, lon]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with dimensions [12, lev, lat, lon] containing monthly climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonTLLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [time, lev, lat, lon]")

    ntime, nlev, nlat, nlon = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [nyears, 12, lev, lat, lon] and take mean over years
    x_reshaped = x.reshape(nyears, 12, nlev, nlat, nlon)
    clm = np.mean(x_reshaped, axis=0)

    return clm


def clmMonLLT(x):
    """
    Calculates long term monthly means (monthly climatology) from monthly data.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [lat, lon, time]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with dimensions [lat, lon, 12] containing monthly climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [lat, lon, time]")

    nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [lat, lon, nyears, 12] and take mean over years
    x_reshaped = x.reshape(nlat, nlon, nyears, 12)
    clm = np.mean(x_reshaped, axis=2)

    return clm


def clmMonLLLT(x):
    """
    Calculates long term monthly means (monthly climatology) from monthly data.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [lev, lat, lon, time]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with dimensions [lev, lat, lon, 12] containing monthly climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMonLLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [lev, lat, lon, time]")

    nlev, nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [lev, lat, lon, nyears, 12] and take mean over years
    x_reshaped = x.reshape(nlev, nlat, nlon, nyears, 12)
    clm = np.mean(x_reshaped, axis=3)

    return clm


# ==============================================================================
# DAILY CLIMATOLOGY FUNCTIONS
# ==============================================================================

def clmDayTLL(x, yyyyddd):
    """
    Calculates long term daily means (daily climatology) from daily data.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Daily data spanning multiple years
    yyyyddd : array_like
        1D integer array of size time with format yyyyddd
        (e.g., 1905001 = Jan 1, 1905; 1905032 = Feb 1, 1905)

    Returns
    -------
    ndarray
        Array with dimensions [366, lat, lon] (or 365/360 depending on calendar)
        containing daily climatology

    Notes
    -----
    Computes climatological daily means by averaging all values for each
    day-of-year across multiple years.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    yyyyddd = np.asarray(yyyyddd, dtype=np.int32)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [time, lat, lon]")

    ntime, nlat, nlon = x.shape

    if len(yyyyddd) != ntime:
        raise ValueError("yyyyddd length must match time dimension")

    # Extract day-of-year from yyyyddd
    doy = yyyyddd % 1000

    # Determine maximum day (366 for leap years, 365 for regular, 360 for 360_day)
    max_doy = doy.max()

    # Initialize output array
    clm = np.zeros((max_doy, nlat, nlon), dtype=np.float64)
    count = np.zeros(max_doy, dtype=np.int32)

    # Accumulate values for each day-of-year
    for t in range(ntime):
        d = doy[t] - 1  # Convert to 0-based indexing
        if 0 <= d < max_doy:
            clm[d] += x[t]
            count[d] += 1

    # Compute means
    for d in range(max_doy):
        if count[d] > 0:
            clm[d] /= count[d]
        else:
            clm[d] = np.nan

    return clm


def clmDayTLLL(x, yyyyddd):
    """
    Calculates long term daily means (daily climatology) from daily data.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [time, lev, lat, lon]
        Daily data spanning multiple years
    yyyyddd : array_like
        1D integer array of size time with format yyyyddd

    Returns
    -------
    ndarray
        Array with dimensions [366, lev, lat, lon] containing daily climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayTLLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    yyyyddd = np.asarray(yyyyddd, dtype=np.int32)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [time, lev, lat, lon]")

    ntime, nlev, nlat, nlon = x.shape

    if len(yyyyddd) != ntime:
        raise ValueError("yyyyddd length must match time dimension")

    # Extract day-of-year from yyyyddd
    doy = yyyyddd % 1000
    max_doy = doy.max()

    # Initialize output array
    clm = np.zeros((max_doy, nlev, nlat, nlon), dtype=np.float64)
    count = np.zeros(max_doy, dtype=np.int32)

    # Accumulate values for each day-of-year
    for t in range(ntime):
        d = doy[t] - 1
        if 0 <= d < max_doy:
            clm[d] += x[t]
            count[d] += 1

    # Compute means
    for d in range(max_doy):
        if count[d] > 0:
            clm[d] /= count[d]
        else:
            clm[d] = np.nan

    return clm


def clmDayHourTLL(x, yyyymmddhh, hours):
    """
    Calculates climatological day-hour means at specified hours for each day.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Hourly or sub-daily data
    yyyymmddhh : array_like
        1D integer array with format yyyymmddhh (e.g., 2010010100)
    hours : array_like
        1D array of integers specifying which hours to extract (0-23)

    Returns
    -------
    ndarray
        Array with dimensions [366*nhours, lat, lon] containing
        day-hour climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayHourTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    yyyymmddhh = np.asarray(yyyymmddhh, dtype=np.int64)
    hours = np.asarray(hours, dtype=np.int32)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [time, lat, lon]")

    ntime, nlat, nlon = x.shape
    nhours = len(hours)

    # Extract components
    hh = yyyymmddhh % 100
    dd = (yyyymmddhh // 100) % 100
    mm = (yyyymmddhh // 10000) % 100
    yyyy = yyyymmddhh // 1000000

    # Calculate day-of-year
    # Simplified calculation (approximate)
    doy = np.zeros(ntime, dtype=np.int32)
    for i in range(ntime):
        # Simple day-of-year calculation (ignoring leap years for simplicity)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        doy[i] = sum(days_in_month[:mm[i]-1]) + dd[i]

    # Initialize output
    clm = np.zeros((366 * nhours, nlat, nlon), dtype=np.float64)
    count = np.zeros(366 * nhours, dtype=np.int32)

    # Accumulate values
    for t in range(ntime):
        if hh[t] in hours:
            hour_idx = np.where(hours == hh[t])[0][0]
            idx = (doy[t] - 1) * nhours + hour_idx
            if 0 <= idx < 366 * nhours:
                clm[idx] += x[t]
                count[idx] += 1

    # Compute means
    for i in range(366 * nhours):
        if count[i] > 0:
            clm[i] /= count[i]
        else:
            clm[i] = np.nan

    return clm


def clmDayHourTLLL(x, yyyymmddhh, hours):
    """
    Calculates climatological day-hour means at specified hours for each day.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [time, lev, lat, lon]
    yyyymmddhh : array_like
        1D integer array with format yyyymmddhh
    hours : array_like
        1D array of integers specifying which hours to extract

    Returns
    -------
    ndarray
        Array with dimensions [366*nhours, lev, lat, lon]

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmDayHourTLLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    yyyymmddhh = np.asarray(yyyymmddhh, dtype=np.int64)
    hours = np.asarray(hours, dtype=np.int32)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [time, lev, lat, lon]")

    ntime, nlev, nlat, nlon = x.shape
    nhours = len(hours)

    # Extract components
    hh = yyyymmddhh % 100
    dd = (yyyymmddhh // 100) % 100
    mm = (yyyymmddhh // 10000) % 100

    # Calculate day-of-year
    doy = np.zeros(ntime, dtype=np.int32)
    for i in range(ntime):
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        doy[i] = sum(days_in_month[:mm[i]-1]) + dd[i]

    # Initialize output
    clm = np.zeros((366 * nhours, nlev, nlat, nlon), dtype=np.float64)
    count = np.zeros(366 * nhours, dtype=np.int32)

    # Accumulate values
    for t in range(ntime):
        if hh[t] in hours:
            hour_idx = np.where(hours == hh[t])[0][0]
            idx = (doy[t] - 1) * nhours + hour_idx
            if 0 <= idx < 366 * nhours:
                clm[idx] += x[t]
                count[idx] += 1

    # Compute means
    for i in range(366 * nhours):
        if count[i] > 0:
            clm[i] /= count[i]
        else:
            clm[i] = np.nan

    return clm


# ==============================================================================
# MONTHLY ANOMALY FUNCTIONS
# ==============================================================================

def calcMonAnomTLL(x, xAve):
    """
    Calculates monthly anomalies by subtracting long term mean from each point.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Time dimension must be a multiple of 12
    xAve : array_like
        3D array with dimensions [12, lat, lon]
        Monthly climatology (from clmMonTLL)

    Returns
    -------
    ndarray
        Array with same dimensions as x containing monthly anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    xAve = np.asarray(xAve, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input x must be 3D array [time, lat, lon]")

    if xAve.shape[0] != 12:
        raise ValueError("xAve first dimension must be 12 (months)")

    ntime, nlat, nlon = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    # Subtract climatology from each corresponding month
    anom = np.zeros_like(x)
    for t in range(ntime):
        month_idx = t % 12
        anom[t] = x[t] - xAve[month_idx]

    return anom


def calcMonAnomTLLL(x, xAve):
    """
    Calculates monthly anomalies by subtracting long term mean from each point.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [time, lev, lat, lon]
    xAve : array_like
        4D array with dimensions [12, lev, lat, lon]

    Returns
    -------
    ndarray
        Array with same dimensions as x containing monthly anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomTLLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    xAve = np.asarray(xAve, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input x must be 4D array [time, lev, lat, lon]")

    if xAve.shape[0] != 12:
        raise ValueError("xAve first dimension must be 12 (months)")

    ntime = x.shape[0]

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    anom = np.zeros_like(x)
    for t in range(ntime):
        month_idx = t % 12
        anom[t] = x[t] - xAve[month_idx]

    return anom


def calcMonAnomLLT(x, xAve):
    """
    Calculates monthly anomalies by subtracting long term mean from each point.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [lat, lon, time]
    xAve : array_like
        3D array with dimensions [lat, lon, 12]

    Returns
    -------
    ndarray
        Array with same dimensions as x containing monthly anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    xAve = np.asarray(xAve, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input x must be 3D array [lat, lon, time]")

    if xAve.shape[2] != 12:
        raise ValueError("xAve last dimension must be 12 (months)")

    nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    anom = np.zeros_like(x)
    for t in range(ntime):
        month_idx = t % 12
        anom[:, :, t] = x[:, :, t] - xAve[:, :, month_idx]

    return anom


def calcMonAnomLLLT(x, xAve):
    """
    Calculates monthly anomalies by subtracting long term mean from each point.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [lev, lat, lon, time]
    xAve : array_like
        4D array with dimensions [lev, lat, lon, 12]

    Returns
    -------
    ndarray
        Array with same dimensions as x containing monthly anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/calcMonAnomLLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    xAve = np.asarray(xAve, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input x must be 4D array [lev, lat, lon, time]")

    if xAve.shape[3] != 12:
        raise ValueError("xAve last dimension must be 12 (months)")

    nlev, nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    anom = np.zeros_like(x)
    for t in range(ntime):
        month_idx = t % 12
        anom[:, :, :, t] = x[:, :, :, t] - xAve[:, :, :, month_idx]

    return anom


def calcDayAnomTLL(x, yyyyddd, xClmDay):
    """
    Calculates daily anomalies from a daily data climatology.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
    yyyyddd : array_like
        1D integer array with format yyyyddd
    xClmDay : array_like
        3D array with dimensions [366, lat, lon]
        Daily climatology from clmDayTLL

    Returns
    -------
    ndarray
        Array with same dimensions as x containing daily anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/calcDayAnomTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    yyyyddd = np.asarray(yyyyddd, dtype=np.int32)
    xClmDay = np.asarray(xClmDay, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input x must be 3D array [time, lat, lon]")

    ntime = x.shape[0]

    if len(yyyyddd) != ntime:
        raise ValueError("yyyyddd length must match time dimension")

    # Extract day-of-year
    doy = yyyyddd % 1000

    # Compute anomalies
    anom = np.zeros_like(x)
    for t in range(ntime):
        d = doy[t] - 1  # Convert to 0-based indexing
        if 0 <= d < xClmDay.shape[0]:
            anom[t] = x[t] - xClmDay[d]
        else:
            anom[t] = np.nan

    return anom


# ==============================================================================
# SEASONAL AVERAGING FUNCTIONS
# ==============================================================================

def month_to_season(xMon, season):
    """
    Computes a user-specified three-month seasonal mean.

    Parameters
    ----------
    xMon : array_like
        Array with time as leftmost dimension
        Can be 1D (time), 3D (time,lat,lon), or 4D (time,lev,lat,lon)
        Time dimension must be divisible by 12
    season : str
        Three-character season code (e.g., "DJF", "JJA", "SON", "MAM")

    Returns
    -------
    ndarray
        Seasonal means with time dimension reduced by factor of 12

    Notes
    -----
    Valid seasons: DJF, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND, NDJ
    First (DJF) and last (NDJ) are two-month averages.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_season.shtml
    """
    xMon = np.asarray(xMon, dtype=np.float64)
    season = season.upper()

    # Season to month indices mapping
    seasons = {
        'DJF': [11, 0, 1],   # Dec, Jan, Feb
        'JFM': [0, 1, 2],    # Jan, Feb, Mar
        'FMA': [1, 2, 3],    # Feb, Mar, Apr
        'MAM': [2, 3, 4],    # Mar, Apr, May
        'AMJ': [3, 4, 5],    # Apr, May, Jun
        'MJJ': [4, 5, 6],    # May, Jun, Jul
        'JJA': [5, 6, 7],    # Jun, Jul, Aug
        'JAS': [6, 7, 8],    # Jul, Aug, Sep
        'ASO': [7, 8, 9],    # Aug, Sep, Oct
        'SON': [8, 9, 10],   # Sep, Oct, Nov
        'OND': [9, 10, 11],  # Oct, Nov, Dec
        'NDJ': [10, 11, 0]   # Nov, Dec, Jan
    }

    if season not in seasons:
        raise ValueError(f"Invalid season: {season}. Valid: {list(seasons.keys())}")

    month_indices = seasons[season]
    ntime = xMon.shape[0]

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be divisible by 12")

    nyears = ntime // 12

    # Reshape to separate years
    if xMon.ndim == 1:
        x_reshaped = xMon.reshape(nyears, 12)
        result = np.zeros(nyears, dtype=np.float64)
    elif xMon.ndim == 3:
        nlat, nlon = xMon.shape[1:]
        x_reshaped = xMon.reshape(nyears, 12, nlat, nlon)
        result = np.zeros((nyears, nlat, nlon), dtype=np.float64)
    elif xMon.ndim == 4:
        nlev, nlat, nlon = xMon.shape[1:]
        x_reshaped = xMon.reshape(nyears, 12, nlev, nlat, nlon)
        result = np.zeros((nyears, nlev, nlat, nlon), dtype=np.float64)
    else:
        raise ValueError("Input must be 1D, 3D, or 4D")

    # Compute seasonal means
    if season == 'DJF':
        # Special case: first year is JF only (2 months)
        if xMon.ndim == 1:
            result[0] = np.mean(x_reshaped[0, [0, 1]])
            for year in range(1, nyears):
                result[year] = np.mean(x_reshaped[year-1:year+1, [11, 0, 1]].reshape(-1))
        elif xMon.ndim == 3:
            result[0] = np.mean(x_reshaped[0, [0, 1]], axis=0)
            for year in range(1, nyears):
                result[year] = np.mean([x_reshaped[year-1, 11],
                                       x_reshaped[year, 0],
                                       x_reshaped[year, 1]], axis=0)
        else:  # 4D
            result[0] = np.mean(x_reshaped[0, [0, 1]], axis=0)
            for year in range(1, nyears):
                result[year] = np.mean([x_reshaped[year-1, 11],
                                       x_reshaped[year, 0],
                                       x_reshaped[year, 1]], axis=0)
    elif season == 'NDJ':
        # Special case: last year is ND only (2 months)
        if xMon.ndim == 1:
            for year in range(nyears - 1):
                result[year] = np.mean([x_reshaped[year, 10],
                                       x_reshaped[year, 11],
                                       x_reshaped[year+1, 0]])
            result[-1] = np.mean(x_reshaped[-1, [10, 11]])
        elif xMon.ndim == 3:
            for year in range(nyears - 1):
                result[year] = np.mean([x_reshaped[year, 10],
                                       x_reshaped[year, 11],
                                       x_reshaped[year+1, 0]], axis=0)
            result[-1] = np.mean(x_reshaped[-1, [10, 11]], axis=0)
        else:  # 4D
            for year in range(nyears - 1):
                result[year] = np.mean([x_reshaped[year, 10],
                                       x_reshaped[year, 11],
                                       x_reshaped[year+1, 0]], axis=0)
            result[-1] = np.mean(x_reshaped[-1, [10, 11]], axis=0)
    else:
        # Normal case: average the three months
        for year in range(nyears):
            if xMon.ndim == 1:
                result[year] = np.mean(x_reshaped[year, month_indices])
            else:
                result[year] = np.mean(x_reshaped[year, month_indices], axis=0)

    return result


def month_to_season12(xMon):
    """
    Computes all twelve three-month seasonal means.

    Parameters
    ----------
    xMon : array_like
        Array with time as leftmost dimension
        Time dimension must be divisible by 12

    Returns
    -------
    ndarray
        All seasonal means stacked

    Notes
    -----
    Computes: DJF, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND, NDJ

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_season12.shtml
    """
    seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

    results = []
    for season in seasons:
        results.append(month_to_season(xMon, season))

    return np.array(results)


def month_to_seasonN(xMon, seasons):
    """
    Computes a user-specified list of three-month seasonal means.

    Parameters
    ----------
    xMon : array_like
        Array with time as leftmost dimension
    seasons : list of str
        List of season codes (e.g., ["DJF", "JJA"])

    Returns
    -------
    list of ndarray
        List of seasonal means for each requested season

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_seasonN.shtml
    """
    results = []
    for season in seasons:
        results.append(month_to_season(xMon, season))

    return results


# ==============================================================================
# REMOVE ANNUAL CYCLE FUNCTIONS
# ==============================================================================

def rmAnnCycle1D(x):
    """
    Removes annual cycle from a one-dimensional time series.

    Parameters
    ----------
    x : array_like
        1D array of monthly values
        Length must be a multiple of 12

    Returns
    -------
    ndarray
        Time series with annual cycle removed (anomalies from monthly means)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/rmAnnCycle1D.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    ntime = len(x)

    if ntime % 12 != 0:
        raise ValueError("Length must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [nyears, 12]
    x_reshaped = x.reshape(nyears, 12)

    # Compute monthly climatology
    clm = np.mean(x_reshaped, axis=0)

    # Subtract climatology
    anom = np.zeros_like(x)
    for t in range(ntime):
        month_idx = t % 12
        anom[t] = x[t] - clm[month_idx]

    return anom


def rmMonAnnCycTLL(x):
    """
    Removes the annual cycle from monthly data.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with same dimensions containing anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/rmMonAnnCycTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    # Compute climatology
    clm = clmMonTLL(x)

    # Compute anomalies
    anom = calcMonAnomTLL(x, clm)

    return anom


def rmMonAnnCycLLT(x):
    """
    Removes the annual cycle from monthly data.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [lat, lon, time]

    Returns
    -------
    ndarray
        Array with same dimensions containing anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/rmMonAnnCycLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    clm = clmMonLLT(x)
    anom = calcMonAnomLLT(x, clm)

    return anom


def rmMonAnnCycLLLT(x):
    """
    Removes the annual cycle from monthly data.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [lev, lat, lon, time]

    Returns
    -------
    ndarray
        Array with same dimensions containing anomalies

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/rmMonAnnCycLLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    clm = clmMonLLLT(x)
    anom = calcMonAnomLLLT(x, clm)

    return anom


# ==============================================================================
# STANDARD DEVIATION FUNCTIONS
# ==============================================================================

def stdMonTLL(x):
    """
    Calculates standard deviations of monthly means.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [time, lat, lon]
        Time dimension must be a multiple of 12

    Returns
    -------
    ndarray
        Array with dimensions [12, lat, lon] containing monthly standard deviations

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/stdMonTLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [time, lat, lon]")

    ntime, nlat, nlon = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [nyears, 12, lat, lon]
    x_reshaped = x.reshape(nyears, 12, nlat, nlon)

    # Compute standard deviation over years for each month
    std = np.std(x_reshaped, axis=0, ddof=1)

    return std


def stdMonTLLL(x):
    """
    Calculates standard deviations of monthly means.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [time, lev, lat, lon]

    Returns
    -------
    ndarray
        Array with dimensions [12, lev, lat, lon]

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/stdMonTLLL.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [time, lev, lat, lon]")

    ntime, nlev, nlat, nlon = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [nyears, 12, lev, lat, lon]
    x_reshaped = x.reshape(nyears, 12, nlev, nlat, nlon)

    # Compute standard deviation over years
    std = np.std(x_reshaped, axis=0, ddof=1)

    return std


def stdMonLLT(x):
    """
    Calculates standard deviations of monthly means.

    Parameters
    ----------
    x : array_like
        3D array with dimensions [lat, lon, time]

    Returns
    -------
    ndarray
        Array with dimensions [lat, lon, 12]

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/stdMonLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError("Input must be 3D array [lat, lon, time]")

    nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [lat, lon, nyears, 12]
    x_reshaped = x.reshape(nlat, nlon, nyears, 12)

    # Compute standard deviation over years
    std = np.std(x_reshaped, axis=2, ddof=1)

    return std


def stdMonLLLT(x):
    """
    Calculates standard deviations of monthly means.

    Parameters
    ----------
    x : array_like
        4D array with dimensions [lev, lat, lon, time]

    Returns
    -------
    ndarray
        Array with dimensions [lev, lat, lon, 12]

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/stdMonLLLT.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 4:
        raise ValueError("Input must be 4D array [lev, lat, lon, time]")

    nlev, nlat, nlon, ntime = x.shape

    if ntime % 12 != 0:
        raise ValueError("Time dimension must be a multiple of 12")

    nyears = ntime // 12

    # Reshape to [lev, lat, lon, nyears, 12]
    x_reshaped = x.reshape(nlev, nlat, nlon, nyears, 12)

    # Compute standard deviation over years
    std = np.std(x_reshaped, axis=3, ddof=1)

    return std


# ==============================================================================
# SMOOTHING FUNCTIONS
# ==============================================================================

def smthClmDayTLL(clmDay, nHarm):
    """
    Calculates a smooth mean daily annual cycle using FFT.

    Parameters
    ----------
    clmDay : array_like
        3D array with dimensions [ntim, lat, lon]
        Output from clmDayTLL (typically 365 or 366 days)
    nHarm : int
        Number of harmonics to retain (typically 1-3)

    Returns
    -------
    ndarray
        Smoothed daily climatology with same dimensions as input

    Notes
    -----
    Uses FFT to retain only the specified number of harmonics,
    producing a smoothed annual cycle.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/smthClmDayTLL.shtml
    """
    clmDay = np.asarray(clmDay, dtype=np.float64)

    if clmDay.ndim != 3:
        raise ValueError("Input must be 3D array [time, lat, lon]")

    ntim, nlat, nlon = clmDay.shape

    # Apply FFT along time dimension
    result = np.zeros_like(clmDay)

    for ilat in range(nlat):
        for ilon in range(nlon):
            # Get time series for this grid point
            ts = clmDay[:, ilat, ilon]

            # Skip if all NaN
            if np.all(np.isnan(ts)):
                result[:, ilat, ilon] = np.nan
                continue

            # Perform FFT
            fft_vals = np.fft.rfft(ts)

            # Zero out high frequency harmonics
            if nHarm < len(fft_vals) - 1:
                fft_vals[nHarm+1:] = 0

            # Inverse FFT
            result[:, ilat, ilon] = np.fft.irfft(fft_vals, n=ntim)

    return result


def smthClmDayTLLL(clmDay, nHarm):
    """
    Calculates a smooth mean daily annual cycle using FFT.

    Parameters
    ----------
    clmDay : array_like
        4D array with dimensions [ntim, lev, lat, lon]
    nHarm : int
        Number of harmonics to retain

    Returns
    -------
    ndarray
        Smoothed daily climatology

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/smthClmDayTLLL.shtml
    """
    clmDay = np.asarray(clmDay, dtype=np.float64)

    if clmDay.ndim != 4:
        raise ValueError("Input must be 4D array [time, lev, lat, lon]")

    ntim, nlev, nlat, nlon = clmDay.shape

    result = np.zeros_like(clmDay)

    for ilev in range(nlev):
        for ilat in range(nlat):
            for ilon in range(nlon):
                ts = clmDay[:, ilev, ilat, ilon]

                if np.all(np.isnan(ts)):
                    result[:, ilev, ilat, ilon] = np.nan
                    continue

                fft_vals = np.fft.rfft(ts)

                if nHarm < len(fft_vals) - 1:
                    fft_vals[nHarm+1:] = 0

                result[:, ilev, ilat, ilon] = np.fft.irfft(fft_vals, n=ntim)

    return result


# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

def clmMon2clmDay(clmMon, nDay, option=0):
    """
    Create a daily climatology from a monthly climatology.

    Parameters
    ----------
    clmMon : array_like
        Monthly climatology with dimensions [..., 12]
    nDay : int
        Number of days in the year (365 or 366)
    option : int, optional
        Interpolation option (default: 0 = linear)

    Returns
    -------
    ndarray
        Daily climatology with dimensions [..., nDay]

    Notes
    -----
    Interpolates monthly climatology to daily values.
    Assumes monthly values represent the middle of each month.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/clmMon2clmDay.shtml
    """
    clmMon = np.asarray(clmMon, dtype=np.float64)

    if clmMon.shape[-1] != 12:
        raise ValueError("Last dimension must be 12 (months)")

    # Day-of-year for middle of each month (approximate)
    days_in_month = [31, 28 if nDay == 365 else 29, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]

    month_midpoints = []
    cumsum = 0
    for days in days_in_month:
        month_midpoints.append(cumsum + days / 2)
        cumsum += days

    month_midpoints = np.array(month_midpoints)

    # Target days
    target_days = np.arange(nDay) + 0.5

    # Prepare output shape
    out_shape = list(clmMon.shape[:-1]) + [nDay]
    result = np.zeros(out_shape, dtype=np.float64)

    # Flatten all dimensions except last
    orig_shape = clmMon.shape
    clmMon_flat = clmMon.reshape(-1, 12)
    result_flat = result.reshape(-1, nDay)

    # Interpolate each spatial point
    for i in range(clmMon_flat.shape[0]):
        # Extend monthly values to handle wrapping
        values_extended = np.concatenate([
            clmMon_flat[i, [-1]],
            clmMon_flat[i, :],
            clmMon_flat[i, [0]]
        ])

        days_extended = np.concatenate([
            [-month_midpoints[0]],
            month_midpoints,
            [nDay + month_midpoints[0]]
        ])

        # Linear interpolation
        result_flat[i, :] = np.interp(target_days, days_extended, values_extended)

    return result


__all__ = [
    # Monthly climatology
    'clmMonTLL',
    'clmMonTLLL',
    'clmMonLLT',
    'clmMonLLLT',
    # Daily climatology
    'clmDayTLL',
    'clmDayTLLL',
    'clmDayHourTLL',
    'clmDayHourTLLL',
    # Monthly anomalies
    'calcMonAnomTLL',
    'calcMonAnomTLLL',
    'calcMonAnomLLT',
    'calcMonAnomLLLT',
    # Daily anomalies
    'calcDayAnomTLL',
    # Seasonal means
    'month_to_season',
    'month_to_season12',
    'month_to_seasonN',
    # Remove annual cycle
    'rmAnnCycle1D',
    'rmMonAnnCycTLL',
    'rmMonAnnCycLLT',
    'rmMonAnnCycLLLT',
    # Standard deviations
    'stdMonTLL',
    'stdMonTLLL',
    'stdMonLLT',
    'stdMonLLLT',
    # Smoothing
    'smthClmDayTLL',
    'smthClmDayTLLL',
    # Conversion
    'clmMon2clmDay'
]
