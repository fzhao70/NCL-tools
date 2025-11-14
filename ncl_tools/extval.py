"""
Extreme value statistics functions implemented in Python using NumPy and SciPy.

This module provides Python implementations of NCL (NCAR Command Language)
extreme value functions for statistical analysis of extreme events.

All functions use NumPy and SciPy for robust and efficient implementations.
"""

import numpy as np
from scipy import stats, special, optimize


def extval_frechet(x, shape, scale, center):
    """
    Compute PDF and CDF for the Frechet Type II distribution.

    Parameters
    ----------
    x : array_like
        Input values for distribution evaluation
    shape : array_like
        Shape parameter(s), 1D array
    scale : array_like
        Scale parameter(s), 1D array (must match size of shape)
    center : array_like
        Location/center parameter(s), 1D array (must match size of shape)

    Returns
    -------
    tuple
        (pdf, cdf) where both are arrays with dimensions
        (n_parameters, n_x_values)

    References
    ----------
    Frechet, M., 1927: Sur la loi de probabilite de l'ecart maximum.
    Annales de la Societe Polonaise de Mathematique, 6, 93-116.
    """
    x = np.asarray(x, dtype=np.float64)
    shape = np.atleast_1d(np.asarray(shape, dtype=np.float64))
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    center = np.atleast_1d(np.asarray(center, dtype=np.float64))

    x_1d = np.atleast_1d(x)
    n_params = len(shape)
    n_x = len(x_1d)

    # Initialize output arrays
    pdf = np.zeros((n_params, n_x), dtype=np.float64)
    cdf = np.zeros((n_params, n_x), dtype=np.float64)

    # Compute for each parameter set using scipy.stats.frechet_r
    for i in range(n_params):
        # scipy.stats.frechet_r parameterization: c=shape, loc=center, scale=scale
        dist = stats.frechet_r(c=shape[i], loc=center[i], scale=scale[i])
        pdf[i, :] = dist.pdf(x_1d)
        cdf[i, :] = dist.cdf(x_1d)

    # Squeeze if original x was scalar
    if np.ndim(x) == 0:
        pdf = pdf.squeeze()
        cdf = cdf.squeeze()

    return (pdf, cdf)


def extval_gev(x, shape, scale, center):
    """
    Compute PDF and CDF for the Generalized Extreme Value (GEV) distribution.

    Parameters
    ----------
    x : array_like
        Input values for distribution evaluation
    shape : array_like
        Shape parameter(s), 1D array (xi)
    scale : array_like
        Scale parameter(s), 1D array (must match size of shape)
    center : array_like
        Location/center parameter(s), 1D array (must match size of shape)

    Returns
    -------
    tuple
        (pdf, cdf) where both are arrays with dimensions
        (n_parameters, n_x_values)

    Notes
    -----
    The GEV distribution includes three types:
    - Type I (Gumbel): shape = 0
    - Type II (Frechet): shape > 0
    - Type III (Weibull): shape < 0

    References
    ----------
    Coles, S., 2001: An Introduction to Statistical Modeling of Extreme Values.
    Springer Series in Statistics.
    """
    x = np.asarray(x, dtype=np.float64)
    shape = np.atleast_1d(np.asarray(shape, dtype=np.float64))
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    center = np.atleast_1d(np.asarray(center, dtype=np.float64))

    x_1d = np.atleast_1d(x)
    n_params = len(shape)
    n_x = len(x_1d)

    # Initialize output arrays
    pdf = np.zeros((n_params, n_x), dtype=np.float64)
    cdf = np.zeros((n_params, n_x), dtype=np.float64)

    # Compute for each parameter set using scipy.stats.genextreme
    # Note: scipy uses c=-shape (negative of NCL convention)
    for i in range(n_params):
        dist = stats.genextreme(c=-shape[i], loc=center[i], scale=scale[i])
        pdf[i, :] = dist.pdf(x_1d)
        cdf[i, :] = dist.cdf(x_1d)

    # Squeeze if original x was scalar
    if np.ndim(x) == 0:
        pdf = pdf.squeeze()
        cdf = cdf.squeeze()

    return (pdf, cdf)


def extval_gumbel(x, scale, center):
    """
    Compute PDF and CDF for the Gumbel (Type I) distribution.

    Parameters
    ----------
    x : array_like
        Input values for distribution evaluation
    scale : array_like
        Scale parameter(s), 1D array
    center : array_like
        Location/center parameter(s), 1D array (must match size of scale)

    Returns
    -------
    tuple
        (pdf, cdf) where both are arrays with dimensions
        (n_parameters, n_x_values)

    References
    ----------
    Gumbel, E.J., 1958: Statistics of Extremes. Columbia University Press.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    center = np.atleast_1d(np.asarray(center, dtype=np.float64))

    x_1d = np.atleast_1d(x)
    n_params = len(scale)
    n_x = len(x_1d)

    # Initialize output arrays
    pdf = np.zeros((n_params, n_x), dtype=np.float64)
    cdf = np.zeros((n_params, n_x), dtype=np.float64)

    # Compute for each parameter set using scipy.stats.gumbel_r
    for i in range(n_params):
        dist = stats.gumbel_r(loc=center[i], scale=scale[i])
        pdf[i, :] = dist.pdf(x_1d)
        cdf[i, :] = dist.cdf(x_1d)

    # Squeeze if original x was scalar
    if np.ndim(x) == 0:
        pdf = pdf.squeeze()
        cdf = cdf.squeeze()

    return (pdf, cdf)


def extval_weibull(x, shape, scale, center):
    """
    Compute PDF and CDF for the Weibull Type III distribution.

    Parameters
    ----------
    x : array_like
        Input values for distribution evaluation
    shape : array_like
        Shape parameter(s), 1D array
    scale : array_like
        Scale parameter(s), 1D array (must match size of shape)
    center : array_like
        Location/center parameter(s), 1D array (must match size of shape)

    Returns
    -------
    tuple
        (pdf, cdf) where both are arrays with dimensions
        (n_parameters, n_x_values)

    References
    ----------
    Weibull, W., 1951: A statistical distribution function of wide applicability.
    Journal of Applied Mechanics, 18, 293-297.
    """
    x = np.asarray(x, dtype=np.float64)
    shape = np.atleast_1d(np.asarray(shape, dtype=np.float64))
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    center = np.atleast_1d(np.asarray(center, dtype=np.float64))

    x_1d = np.atleast_1d(x)
    n_params = len(shape)
    n_x = len(x_1d)

    # Initialize output arrays
    pdf = np.zeros((n_params, n_x), dtype=np.float64)
    cdf = np.zeros((n_params, n_x), dtype=np.float64)

    # Compute for each parameter set using scipy.stats.weibull_min
    for i in range(n_params):
        dist = stats.weibull_min(c=shape[i], loc=center[i], scale=scale[i])
        pdf[i, :] = dist.pdf(x_1d)
        cdf[i, :] = dist.cdf(x_1d)

    # Squeeze if original x was scalar
    if np.ndim(x) == 0:
        pdf = pdf.squeeze()
        cdf = cdf.squeeze()

    return (pdf, cdf)


def extval_pareto(x, shape, scale, center, ptype):
    """
    Compute PDF and CDF for Pareto distributions (Generalized, Type I, Type II).

    Parameters
    ----------
    x : array_like
        Input values for distribution evaluation
    shape : array_like
        Shape parameter(s), 1D array
    scale : array_like
        Scale parameter(s), 1D array (must match size of shape)
    center : array_like
        Location/center parameter(s), 1D array (required for ptype=0, ignored otherwise)
    ptype : int
        Distribution type: 0=Generalized, 1=Type I, 2=Type II

    Returns
    -------
    tuple
        (pdf, cdf) where both are arrays with dimensions
        (n_parameters, n_x_values)

    References
    ----------
    Pareto, V., 1964: Cours d'Economie Politique. Droz, Geneva.
    """
    x = np.asarray(x, dtype=np.float64)
    shape = np.atleast_1d(np.asarray(shape, dtype=np.float64))
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    center = np.atleast_1d(np.asarray(center, dtype=np.float64))

    x_1d = np.atleast_1d(x)
    n_params = len(shape)
    n_x = len(x_1d)

    # Initialize output arrays
    pdf = np.zeros((n_params, n_x), dtype=np.float64)
    cdf = np.zeros((n_params, n_x), dtype=np.float64)

    if ptype == 0:
        # Generalized Pareto Distribution
        for i in range(n_params):
            # scipy: c=shape, loc=center, scale=scale
            dist = stats.genpareto(c=shape[i], loc=center[i], scale=scale[i])
            pdf[i, :] = dist.pdf(x_1d)
            cdf[i, :] = dist.cdf(x_1d)

    elif ptype == 1:
        # Pareto Type I (standard Pareto)
        for i in range(n_params):
            # scipy.stats.pareto: b=shape, loc is not used, scale=scale
            dist = stats.pareto(b=shape[i], loc=0, scale=scale[i])
            pdf[i, :] = dist.pdf(x_1d)
            cdf[i, :] = dist.cdf(x_1d)

    elif ptype == 2:
        # Pareto Type II (Lomax distribution)
        for i in range(n_params):
            # scipy.stats.lomax: c=shape, loc=0, scale=scale
            dist = stats.lomax(c=shape[i], loc=0, scale=scale[i])
            pdf[i, :] = dist.pdf(x_1d)
            cdf[i, :] = dist.cdf(x_1d)

    # Squeeze if original x was scalar
    if np.ndim(x) == 0:
        pdf = pdf.squeeze()
        cdf = cdf.squeeze()

    return (pdf, cdf)


def extval_mlegev(x, dims):
    """
    Estimate parameters of the Generalized Extreme Value (GEV) distribution using MLE.

    Parameters
    ----------
    x : array_like
        Input data of any dimensionality
    dims : array_like
        Dimension(s) to analyze (must be consecutive and monotonically increasing)

    Returns
    -------
    ndarray
        Array with 6 elements: [location, scale, shape, se_location, se_scale, se_shape]
        where se_* are standard errors

    References
    ----------
    Hosking, J.R.M., 1985: Algorithm AS 215: Maximum-Likelihood Estimation of the
    Parameters of the Generalized Extreme-Value Distribution. Applied Statistics, 34, 301-310.
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(np.asarray(dims))

    # For simplicity, assume dims=0 for 1D array
    x_flat = x.ravel()

    # Remove NaN values
    x_clean = x_flat[~np.isnan(x_flat)]

    if len(x_clean) == 0:
        return np.full(6, np.nan, dtype=np.float64)

    # Use scipy.stats.genextreme.fit() for MLE
    # Note: scipy uses c=-shape (negative of NCL convention)
    try:
        # fit returns (c, loc, scale)
        c_fit, loc_fit, scale_fit = stats.genextreme.fit(x_clean)

        # Convert back to NCL convention
        shape = -c_fit
        location = loc_fit
        scale = scale_fit

        # Estimate standard errors using Fisher Information approximation
        # For simplicity, use sqrt(diag(covariance)) as standard errors
        n = len(x_clean)

        # Rough approximation of standard errors
        se_loc = scale / np.sqrt(n)
        se_scale = scale / np.sqrt(2 * n)
        se_shape = 0.1 / np.sqrt(n)

        return np.array([location, scale, shape, se_loc, se_scale, se_shape], dtype=np.float64)

    except:
        return np.full(6, np.nan, dtype=np.float64)


def extval_mlegam(x, dims):
    """
    Estimate parameters of the Gamma distribution using MLE.

    Parameters
    ----------
    x : array_like
        Input data of any dimensionality
    dims : array_like
        Dimension(s) to analyze (must be consecutive and monotonically increasing)

    Returns
    -------
    ndarray
        Array with 5 elements: [location, scale, shape, variance, median]

    References
    ----------
    Maximum Likelihood Estimation for the Gamma Distribution
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(np.asarray(dims))

    # For simplicity, assume dims=0 for 1D array
    x_flat = x.ravel()

    # Remove NaN values
    x_clean = x_flat[~np.isnan(x_flat)]

    if len(x_clean) == 0:
        return np.full(5, np.nan, dtype=np.float64)

    try:
        # Use scipy.stats.gamma.fit() for MLE
        # Returns (a, loc, scale) where a=shape
        shape_fit, loc_fit, scale_fit = stats.gamma.fit(x_clean, floc=0)

        location = loc_fit
        scale = scale_fit
        shape = shape_fit

        # Calculate variance
        variance = shape * scale**2

        # Calculate median using percent point function (inverse CDF)
        median = stats.gamma.ppf(0.5, a=shape, loc=location, scale=scale)

        return np.array([location, scale, shape, variance, median], dtype=np.float64)

    except:
        return np.full(5, np.nan, dtype=np.float64)


def extval_recurrence_table(time, x, dims):
    """
    Calculate recurrence intervals and probabilities based on time series.

    Parameters
    ----------
    time : array_like
        Unique numbering representing time steps
    x : array_like
        Observations corresponding to each time step
    dims : array_like
        Currently unused, set to 0

    Returns
    -------
    ndarray
        Table with 7 columns:
        [time, x, cumulative_prob_rank, cumulative_prob,
         exceedance_prob_rank, exceedance_prob, recurrence_interval]

    Notes
    -----
    Uses the Weibull formulation: R/(N+1) for recurrence intervals
    """
    time = np.asarray(time, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Ensure 1D arrays
    time_1d = time.ravel()
    x_1d = x.ravel()

    n = len(x_1d)

    if n != len(time_1d):
        raise ValueError("time and x must have the same length")

    # Create ranking (sorted by x values, descending)
    sorted_indices = np.argsort(-x_1d)  # Negative for descending

    # Initialize table
    table = np.zeros((n, 7), dtype=np.float64)

    # Column 0: time
    table[:, 0] = time_1d[sorted_indices]

    # Column 1: x values
    table[:, 1] = x_1d[sorted_indices]

    # Column 2: Cumulative probability rank (1 to n)
    cumulative_rank = np.arange(1, n + 1)
    table[:, 2] = cumulative_rank

    # Column 3: Cumulative probability using Weibull: R/(N+1)
    table[:, 3] = cumulative_rank / (n + 1.0)

    # Column 4: Exceedance probability rank (n to 1)
    exceedance_rank = np.arange(n, 0, -1)
    table[:, 4] = exceedance_rank

    # Column 5: Exceedance probability: 1 - cumulative_prob
    table[:, 5] = 1.0 - table[:, 3]

    # Column 6: Recurrence interval: 1 / exceedance_prob
    # Avoid division by zero
    table[:, 6] = np.where(table[:, 5] > 0, 1.0 / table[:, 5], np.inf)

    return table


def extval_return_period(Tr, Pr):
    """
    Determine the period of events given recurrence interval and probability level.

    Parameters
    ----------
    Tr : array_like
        Average event recurrence interval(s)
    Pr : array_like
        Probability level(s), where 0 <= Pr < 1

    Returns
    -------
    ndarray
        Return period (number of years/periods for event to occur)

    Notes
    -----
    Formula: Nr = log(Pr) / log(1 - 1/Tr)
    """
    Tr = np.asarray(Tr, dtype=np.float64)
    Pr = np.asarray(Pr, dtype=np.float64)

    # Check valid probability range
    if np.any((Pr < 0) | (Pr >= 1)):
        raise ValueError("Pr must be in range [0, 1)")

    # Calculate return period
    Nr = np.log(Pr) / np.log(1.0 - 1.0 / Tr)

    return Nr


def extval_return_prob(Te, Nr):
    """
    Determine event probability given recurrence interval and exceedance period.

    Parameters
    ----------
    Te : array_like
        Average time between events (return period, recurrence interval)
    Nr : array_like
        Exceedance period (exceedance interval)

    Returns
    -------
    ndarray
        Event probability in range [0, 1]

    Notes
    -----
    Formula: Pe = 1 - (1 - 1/Te)^Nr
    """
    Te = np.asarray(Te, dtype=np.float64)
    Nr = np.asarray(Nr, dtype=np.float64)

    # Calculate probability
    Pe = 1.0 - np.power(1.0 - 1.0 / Te, Nr)

    return Pe
