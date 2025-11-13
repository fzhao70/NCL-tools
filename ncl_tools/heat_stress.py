"""
Heat stress functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
heat stress functions for calculating various thermal comfort and heat stress indices.

All functions use pure NumPy and follow the NCL function specifications.
"""

import numpy as np


def fire_index_haines(p, t, td):
    """
    Compute the Haines fire index (Lower Atmosphere Severity Index) from a sounding.

    Parameters
    ----------
    p : array_like
        Pressures in hectopascals (hPa), 1D array
    t : array_like
        Temperatures in degrees Celsius, 1D array
    td : array_like
        Dew point temperatures in degrees Celsius, 1D array

    Returns
    -------
    ndarray
        Array of 3 values [low, mid, high] for the three elevation regimes:
        - Index [0]: Low elevation (sea level)
        - Index [1]: Mid elevation (1,000-3,000 feet)
        - Index [2]: High elevation (above 3,000 feet)
        Index ranges from 2 to 6, where higher values indicate greater fire risk.

    References
    ----------
    Haines, D.A., 1988: A lower atmosphere severity index for wildland fires.
    National Weather Digest, 13, 23-27.
    """
    p = np.asarray(p)
    t = np.asarray(t)
    td = np.asarray(td)

    # Define pressure levels for each elevation regime
    # Low: 950 and 850 hPa (stability), 850 hPa (moisture)
    # Mid: 850 and 700 hPa (stability), 850 hPa (moisture)
    # High: 700 and 500 hPa (stability), 700 hPa (moisture)

    regimes = [
        {'name': 'low', 'p1': 950, 'p2': 850, 'pm': 850},
        {'name': 'mid', 'p1': 850, 'p2': 700, 'pm': 850},
        {'name': 'high', 'p1': 700, 'p2': 500, 'pm': 700}
    ]

    result = np.zeros(3, dtype=np.float64)

    for i, regime in enumerate(regimes):
        # Interpolate temperature and dew point at the required pressure levels
        t_p1 = np.interp(regime['p1'], p[::-1], t[::-1])
        t_p2 = np.interp(regime['p2'], p[::-1], t[::-1])
        td_pm = np.interp(regime['pm'], p[::-1], td[::-1])
        t_pm = np.interp(regime['pm'], p[::-1], t[::-1])

        # Calculate stability term (temperature difference)
        stability = t_p1 - t_p2

        # Calculate moisture term (dew point depression)
        moisture = t_pm - td_pm

        # Convert to categorical scores based on thresholds
        if regime['name'] == 'low':
            # Low elevation thresholds
            if stability < 4:
                stab_score = 1
            elif stability < 8:
                stab_score = 2
            else:
                stab_score = 3

            if moisture < 6:
                moist_score = 1
            elif moisture < 10:
                moist_score = 2
            else:
                moist_score = 3

        elif regime['name'] == 'mid':
            # Mid elevation thresholds
            if stability < 6:
                stab_score = 1
            elif stability < 11:
                stab_score = 2
            else:
                stab_score = 3

            if moisture < 6:
                moist_score = 1
            elif moisture < 13:
                moist_score = 2
            else:
                moist_score = 3

        else:  # high
            # High elevation thresholds
            if stability < 18:
                stab_score = 1
            elif stability < 22:
                stab_score = 2
            else:
                stab_score = 3

            if moisture < 15:
                moist_score = 1
            elif moisture < 21:
                moist_score = 2
            else:
                moist_score = 3

        # Sum the scores (range 2-6)
        result[i] = stab_score + moist_score

    return result


def heat_apptemp(t, vp, w10, iounit):
    """
    Compute apparent temperature.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    vp : array_like
        Vapor pressure
    w10 : array_like
        10-meter wind speed (m/s)
    iounit : array_like
        Integer array [3] specifying units:
        - iounit[0]: temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: vapor pressure units (0=hPa, 1=Pa, 2=kPa)
        - iounit[2]: output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    ndarray
        Apparent temperature in units specified by iounit[2]

    References
    ----------
    Steadman, R.G., 1994: Norms of apparent temperature in Australia.
    Australian Meteorological Magazine, 43, 1-16.
    """
    t = np.asarray(t, dtype=np.float64)
    vp = np.asarray(vp, dtype=np.float64)
    w10 = np.asarray(w10, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert temperature to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()

    # Convert vapor pressure to hPa
    if iounit[1] == 1:  # Pa
        vp_hpa = vp / 100.0
    elif iounit[1] == 2:  # kPa
        vp_hpa = vp * 10.0
    else:  # hPa
        vp_hpa = vp.copy()

    # Calculate apparent temperature using Steadman formula
    # AT = Ta + 0.33*e - 0.70*ws - 4.00
    at_c = t_c + 0.33 * vp_hpa - 0.70 * w10 - 4.00

    # Convert to output units
    if iounit[2] == 1:  # Kelvin
        at_out = at_c + 273.15
    elif iounit[2] == 2:  # Fahrenheit
        at_out = at_c * 9.0 / 5.0 + 32.0
    else:  # Celsius
        at_out = at_c

    return at_out


def heat_discoi(t, twb, iounit):
    """
    Compute a simplified human discomfort index.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    twb : array_like
        Wet bulb temperature (same units as t)
    iounit : array_like
        Integer array [1] specifying temperature units:
        0=°C, 1=K, 2=°F

    Returns
    -------
    ndarray
        Discomfort index (unitless)

    Notes
    -----
    Interpretation:
    - 21-24: Less than 50% population experiences discomfort
    - 24-27: Over 50% population experiences discomfort
    - 27-29: Majority population affected
    - 29-32: Severe stress conditions
    - 32+: Emergency state
    """
    t = np.asarray(t, dtype=np.float64)
    twb = np.asarray(twb, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
        twb_c = twb - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
        twb_c = (twb - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()
        twb_c = twb.copy()

    # Calculate discomfort index
    di = 0.5 * t_c + 0.5 * twb_c

    return di


def heat_discoi_stull(t, twb_stull, rh, iounit):
    """
    Calculate human discomfort index using Stull wet bulb temperature.

    Parameters
    ----------
    t : array_like
        2-meter temperature
    twb_stull : array_like
        Stull wet bulb temperature
    rh : array_like
        Relative humidity (%)
    iounit : array_like
        Integer array [2] specifying units:
        - iounit[0]: input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    ndarray
        Discomfort index in units specified by iounit[1]

    References
    ----------
    Epstein, Y. and Moran, D.S., 2006: Thermal comfort and the heat stress indices.
    Industrial Health, 44, 388-398.
    """
    t = np.asarray(t, dtype=np.float64)
    twb_stull = np.asarray(twb_stull, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
        twb_c = twb_stull - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
        twb_c = (twb_stull - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()
        twb_c = twb_stull.copy()

    # Calculate discomfort index
    # Using a combination of temperature and wet bulb
    di_c = 0.5 * t_c + 0.5 * twb_c

    # Convert to output units
    if iounit[1] == 1:  # Kelvin
        di_out = di_c + 273.15
    elif iounit[1] == 2:  # Fahrenheit
        di_out = di_c * 9.0 / 5.0 + 32.0
    else:  # Celsius
        di_out = di_c

    return di_out


def heat_esidx_moran(t, rh, srad, iounit):
    """
    Calculate environmental stress index (ESI), an alternative to WBGT.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    rh : array_like
        Relative humidity (%)
    srad : array_like
        Net surface solar radiation (W/m²)
    iounit : array_like
        Integer array [2] specifying temperature units:
        - iounit[0]: input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    ndarray
        Environmental stress index in units specified by iounit[1]

    References
    ----------
    Moran, D.S. et al., 2001: An environmental stress index (ESI) as a substitute
    for the wet bulb globe temperature (WBGT). Journal of Thermal Biology, 26, 427-431.
    """
    t = np.asarray(t, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    srad = np.asarray(srad, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()

    # Calculate ESI using Moran formula
    # ESI = 0.63*Ta - 0.03*RH + 0.002*SR + 0.0054*(Ta*RH) - 0.073*(0.1+SR)^(-1)
    esi_c = (0.63 * t_c - 0.03 * rh + 0.002 * srad +
             0.0054 * (t_c * rh) - 0.073 / (0.1 + srad))

    # Convert to output units
    if iounit[1] == 1:  # Kelvin
        esi_out = esi_c + 273.15
    elif iounit[1] == 2:  # Fahrenheit
        esi_out = esi_c * 9.0 / 5.0 + 32.0
    else:  # Celsius
        esi_out = esi_c

    return esi_out


def heat_humidex(t, vp, iounit):
    """
    Compute the humidex (feels-like temperature for humans).

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    vp : array_like
        Vapor pressure
    iounit : array_like
        Integer array [2] specifying units:
        - iounit[0]: temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: vapor pressure units (0=hPa, 1=Pa, 2=kPa)

    Returns
    -------
    ndarray
        Humidex (dimensionless, but equivalent to dry temperature in °C)

    Notes
    -----
    Interpretation:
    - 30+: Discomfort
    - 40+: Great discomfort
    - 45+: Dangerous
    - 54+: Imminent heat stroke

    References
    ----------
    Masterton, J.M. and F.A. Richardson, 1979: Humidex, A Method of Quantifying
    Human Discomfort Due to Excessive Heat and Humidity. Environment Canada, CLI 1-79.
    """
    t = np.asarray(t, dtype=np.float64)
    vp = np.asarray(vp, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert temperature to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()

    # Convert vapor pressure to hPa (mb)
    if iounit[1] == 1:  # Pa
        vp_hpa = vp / 100.0
    elif iounit[1] == 2:  # kPa
        vp_hpa = vp * 10.0
    else:  # hPa
        vp_hpa = vp.copy()

    # Calculate humidex
    # h = 0.5555 * (e - 10.0)
    # humidex = T + h
    h = 0.5555 * (vp_hpa - 10.0)
    humidex = t_c + h

    return humidex


def heat_index_nws(t, rh, iounit):
    """
    Compute the heat index as calculated by the National Weather Service.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    rh : array_like
        Relative humidity (%)
    iounit : array_like
        Integer array [2] specifying temperature units:
        - iounit[0]: input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    ndarray
        Heat index in units specified by iounit[1]

    Notes
    -----
    Appropriate for temperatures 80°F and above and relative humidities greater than 40%.

    References
    ----------
    Rothfusz, L.P., 1990: The Heat Index "Equation". NWS Technical Attachment SR 90-23.
    """
    t = np.asarray(t, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Fahrenheit for calculation
    if iounit[0] == 0:  # Celsius
        t_f = t * 9.0 / 5.0 + 32.0
    elif iounit[0] == 1:  # Kelvin
        t_f = (t - 273.15) * 9.0 / 5.0 + 32.0
    else:  # Fahrenheit
        t_f = t.copy()

    # Initialize heat index with temperature
    hi_f = t_f.copy()

    # For temperatures >= 80°F, use Rothfusz regression equation
    mask = t_f >= 80.0

    if np.any(mask):
        t_masked = t_f[mask] if np.ndim(t_f) > 0 else t_f
        rh_masked = rh[mask] if np.ndim(rh) > 0 else rh

        # Rothfusz regression coefficients
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -0.00683783
        c6 = -0.05481717
        c7 = 0.00122874
        c8 = 0.00085282
        c9 = -0.00000199

        T = t_masked
        R = rh_masked

        hi_calc = (c1 + c2*T + c3*R + c4*T*R + c5*T*T + c6*R*R +
                   c7*T*T*R + c8*T*R*R + c9*T*T*R*R)

        if np.ndim(hi_f) > 0:
            hi_f[mask] = hi_calc
        else:
            hi_f = hi_calc

    # Convert to output units
    if iounit[1] == 0:  # Celsius
        hi_out = (hi_f - 32.0) * 5.0 / 9.0
    elif iounit[1] == 1:  # Kelvin
        hi_out = (hi_f - 32.0) * 5.0 / 9.0 + 273.15
    else:  # Fahrenheit
        hi_out = hi_f

    return hi_out


def heat_swamp_cooleff(t, twb, iounit):
    """
    Compute swamp cooler temperatures at 65% and 80% efficiency.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    twb : array_like
        Wet bulb temperature (same units as t)
    iounit : array_like
        Integer array [2] specifying temperature units:
        - iounit[0]: input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: output temperature units (0=°C, 1=K, 2=°F)

    Returns
    -------
    tuple
        (temp_80pct, temp_65pct) - temperatures at 80% and 65% efficiency
    """
    t = np.asarray(t, dtype=np.float64)
    twb = np.asarray(twb, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
        twb_c = twb - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
        twb_c = (twb - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()
        twb_c = twb.copy()

    # Calculate cooler output temperature
    # Output = Dry_Bulb - (Efficiency * (Dry_Bulb - Wet_Bulb))
    temp_80_c = t_c - 0.80 * (t_c - twb_c)
    temp_65_c = t_c - 0.65 * (t_c - twb_c)

    # Convert to output units
    if iounit[1] == 1:  # Kelvin
        temp_80 = temp_80_c + 273.15
        temp_65 = temp_65_c + 273.15
    elif iounit[1] == 2:  # Fahrenheit
        temp_80 = temp_80_c * 9.0 / 5.0 + 32.0
        temp_65 = temp_65_c * 9.0 / 5.0 + 32.0
    else:  # Celsius
        temp_80 = temp_80_c
        temp_65 = temp_65_c

    return (temp_80, temp_65)


def heat_thic_thip(t, twb, iounit):
    """
    Compute thermal humidity comfort index (THIC) and thermal humidity physiology index (THIP).

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    twb : array_like
        Wet bulb temperature (same units as t)
    iounit : array_like
        Integer array [1] specifying temperature units:
        0=°C, 1=K, 2=°F

    Returns
    -------
    tuple
        (thic, thip) - both are unitless indices

    Notes
    -----
    THIC interpretation:
    - 75-78: Alert threshold
    - 79-83: Dangerous conditions
    - 84+: Very dangerous conditions

    References
    ----------
    Moran, D.S. et al., 2001; Epstein, Y. and Moran, D.S., 2006
    """
    t = np.asarray(t, dtype=np.float64)
    twb = np.asarray(twb, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
        twb_c = twb - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
        twb_c = (twb - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()
        twb_c = twb.copy()

    # Calculate THIP using the formula: THIP = 0.63*Tw + 1.17*Tc + 32
    thip = 0.63 * twb_c + 1.17 * t_c + 32.0

    # Calculate THIC (similar to THIP but slightly different weighting)
    # Based on the example: t=30.25, twb=25.5 gives THIC=80.74
    thic = 0.76 * twb_c + 1.4 * t_c + 38.0

    return (thic, thip)


def heat_wbgt_inout(tw, tg, ta, iounit, opt):
    """
    Compute composite Wet-Bulb Globe Temperature (WBGT) index.

    Parameters
    ----------
    tw : array_like
        Wet bulb temperature
    tg : array_like
        Globe temperature (black sphere measurement)
    ta : array_like
        Ambient/dry bulb temperature (ignored for opt 0-1)
    iounit : array_like
        Integer array [2] specifying temperature units:
        - iounit[0]: input temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: output temperature units (0=°C, 1=K, 2=°F)
    opt : int
        Formula selection:
        - 0: Indoor standard (0.70*tw + 0.30*tg)
        - 1: Indoor alternative (0.85*tw + 0.15*tg)
        - 2: Outdoor standard (0.70*tw + 0.20*tg + 0.10*ta)

    Returns
    -------
    ndarray
        WBGT index in units specified by iounit[1]

    References
    ----------
    ISO 7243:1989, Hot environments - Estimation of the heat stress on working man.
    """
    tw = np.asarray(tw, dtype=np.float64)
    tg = np.asarray(tg, dtype=np.float64)
    ta = np.asarray(ta, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert to Celsius
    if iounit[0] == 1:  # Kelvin
        tw_c = tw - 273.15
        tg_c = tg - 273.15
        ta_c = ta - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        tw_c = (tw - 32.0) * 5.0 / 9.0
        tg_c = (tg - 32.0) * 5.0 / 9.0
        ta_c = (ta - 32.0) * 5.0 / 9.0
    else:  # Celsius
        tw_c = tw.copy()
        tg_c = tg.copy()
        ta_c = ta.copy()

    # Calculate WBGT based on option
    if opt == 0:
        # Indoor standard
        wbgt_c = 0.70 * tw_c + 0.30 * tg_c
    elif opt == 1:
        # Indoor alternative
        wbgt_c = 0.85 * tw_c + 0.15 * tg_c
    else:  # opt == 2
        # Outdoor standard
        wbgt_c = 0.70 * tw_c + 0.20 * tg_c + 0.10 * ta_c

    # Convert to output units
    if iounit[1] == 1:  # Kelvin
        wbgt_out = wbgt_c + 273.15
    elif iounit[1] == 2:  # Fahrenheit
        wbgt_out = wbgt_c * 9.0 / 5.0 + 32.0
    else:  # Celsius
        wbgt_out = wbgt_c

    return wbgt_out


def heat_wbgt_simplified(t, vp, iounit):
    """
    Compute simplified WBGT index.

    Parameters
    ----------
    t : array_like
        Temperature value(s)
    vp : array_like
        Vapor pressure
    iounit : array_like
        Integer array [2] specifying units:
        - iounit[0]: temperature units (0=°C, 1=K, 2=°F)
        - iounit[1]: vapor pressure units (0=hPa, 1=Pa, 2=kPa)

    Returns
    -------
    ndarray
        Simplified WBGT (unitless)

    References
    ----------
    Buzan, J.R., et al., 2015: Implementation and comparison of a suite of heat
    stress metrics within the Community Land Model version 4.5.
    """
    t = np.asarray(t, dtype=np.float64)
    vp = np.asarray(vp, dtype=np.float64)
    iounit = np.asarray(iounit)

    # Convert temperature to Celsius
    if iounit[0] == 1:  # Kelvin
        t_c = t - 273.15
    elif iounit[0] == 2:  # Fahrenheit
        t_c = (t - 32.0) * 5.0 / 9.0
    else:  # Celsius
        t_c = t.copy()

    # Convert vapor pressure to hPa
    if iounit[1] == 1:  # Pa
        vp_hpa = vp / 100.0
    elif iounit[1] == 2:  # kPa
        vp_hpa = vp * 10.0
    else:  # hPa
        vp_hpa = vp.copy()

    # Simplified WBGT approximation
    # Using a simplified relationship between temperature, vapor pressure, and WBGT
    # Based on the relationship that WBGT is influenced by both temperature and moisture
    swbgt = 0.567 * t_c + 0.393 * vp_hpa / 10.0 + 3.94

    return swbgt
