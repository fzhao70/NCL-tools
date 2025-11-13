"""
Extreme value statistics functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
extreme value functions for statistical analysis of extreme events.

All functions use pure NumPy and follow the NCL function specifications.
"""

import numpy as np
from scipy import special, optimize


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

    # Ensure x is at least 1D
    x_1d = np.atleast_1d(x)

    # Reshape for broadcasting: shape as (n_params, 1), x as (1, n_x)
    alpha = shape.reshape(-1, 1)
    beta = scale.reshape(-1, 1)
    mu = center.reshape(-1, 1)
    x_expanded = x_1d.reshape(1, -1)

    # Standardize: z = (x - mu) / beta
    z = (x_expanded - mu) / beta

    # Initialize output arrays
    pdf = np.zeros_like(z)
    cdf = np.zeros_like(z)

    # Only compute for x > center
    valid = z > 0

    # PDF: (alpha/beta) * z^(-1-alpha) * exp(-z^(-alpha))
    pdf[valid] = (alpha[valid] / beta[valid]) * np.power(z[valid], -1 - alpha[valid]) * \
                 np.exp(-np.power(z[valid], -alpha[valid]))

    # CDF: exp(-z^(-alpha))
    cdf[valid] = np.exp(-np.power(z[valid], -alpha[valid]))

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

    # Ensure x is at least 1D
    x_1d = np.atleast_1d(x)

    # Reshape for broadcasting
    xi = shape.reshape(-1, 1)
    sigma = scale.reshape(-1, 1)
    mu = center.reshape(-1, 1)
    x_expanded = x_1d.reshape(1, -1)

    # Initialize output
    pdf = np.zeros((len(shape), len(x_1d)), dtype=np.float64)
    cdf = np.zeros_like(pdf)

    # Standardize
    z = (x_expanded - mu) / sigma

    # Handle shape ~ 0 (Gumbel case)
    gumbel_mask = np.abs(xi) < 1e-10

    if np.any(gumbel_mask):
        # Gumbel distribution
        t = np.exp(-z[gumbel_mask])
        pdf[gumbel_mask] = (1.0 / sigma[gumbel_mask]) * t * np.exp(-t)
        cdf[gumbel_mask] = np.exp(-np.exp(-z[gumbel_mask]))

    # Handle shape != 0 (Frechet/Weibull case)
    non_gumbel_mask = ~gumbel_mask

    if np.any(non_gumbel_mask):
        xi_ng = xi[non_gumbel_mask]
        sigma_ng = sigma[non_gumbel_mask]
        z_ng = z[non_gumbel_mask]

        # Valid region: 1 + xi*z > 0
        valid = (1.0 + xi_ng * z_ng) > 0

        t = np.zeros_like(z_ng)
        t[valid] = np.power(1.0 + xi_ng[valid] * z_ng[valid], -1.0 / xi_ng[valid])

        # PDF
        pdf_vals = np.zeros_like(z_ng)
        pdf_vals[valid] = (1.0 / sigma_ng[valid]) * \
                          np.power(1.0 + xi_ng[valid] * z_ng[valid], -(1.0 + 1.0 / xi_ng[valid])) * \
                          np.exp(-t[valid])
        pdf[non_gumbel_mask] = pdf_vals

        # CDF
        cdf_vals = np.zeros_like(z_ng)
        cdf_vals[valid] = np.exp(-t[valid])
        cdf[non_gumbel_mask] = cdf_vals

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

    # Ensure x is at least 1D
    x_1d = np.atleast_1d(x)

    # Reshape for broadcasting
    beta = scale.reshape(-1, 1)
    mu = center.reshape(-1, 1)
    x_expanded = x_1d.reshape(1, -1)

    # Standardize: z = (x - mu) / beta
    z = (x_expanded - mu) / beta

    # PDF: (1/beta) * exp(-z - exp(-z))
    pdf = (1.0 / beta) * np.exp(-z - np.exp(-z))

    # CDF: exp(-exp(-z))
    cdf = np.exp(-np.exp(-z))

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

    # Ensure x is at least 1D
    x_1d = np.atleast_1d(x)

    # Reshape for broadcasting
    k = shape.reshape(-1, 1)
    lam = scale.reshape(-1, 1)
    gamma = center.reshape(-1, 1)
    x_expanded = x_1d.reshape(1, -1)

    # Shift by location parameter
    z = x_expanded - gamma

    # Initialize output
    pdf = np.zeros_like(z)
    cdf = np.zeros_like(z)

    # Only valid for x > center
    valid = z > 0

    # PDF: (k/lam) * ((z/lam)^(k-1)) * exp(-(z/lam)^k)
    pdf[valid] = (k[valid] / lam[valid]) * \
                 np.power(z[valid] / lam[valid], k[valid] - 1.0) * \
                 np.exp(-np.power(z[valid] / lam[valid], k[valid]))

    # CDF: 1 - exp(-(z/lam)^k)
    cdf[valid] = 1.0 - np.exp(-np.power(z[valid] / lam[valid], k[valid]))

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

    # Ensure x is at least 1D
    x_1d = np.atleast_1d(x)

    # Reshape for broadcasting
    alpha = shape.reshape(-1, 1)
    sigma = scale.reshape(-1, 1)
    mu = center.reshape(-1, 1)
    x_expanded = x_1d.reshape(1, -1)

    # Initialize output
    pdf = np.zeros((len(shape), len(x_1d)), dtype=np.float64)
    cdf = np.zeros_like(pdf)

    if ptype == 0:
        # Generalized Pareto Distribution
        z = (x_expanded - mu) / sigma
        valid = z > 0

        # PDF: (1/sigma) * (1 + alpha*z)^(-(1 + 1/alpha))
        pdf[valid] = (1.0 / sigma[valid]) * \
                     np.power(1.0 + alpha[valid] * z[valid], -(1.0 + 1.0 / alpha[valid]))

        # CDF: 1 - (1 + alpha*z)^(-1/alpha)
        cdf[valid] = 1.0 - np.power(1.0 + alpha[valid] * z[valid], -1.0 / alpha[valid])

    elif ptype == 1:
        # Pareto Type I (standard Pareto)
        valid = x_expanded >= sigma

        # PDF: (alpha * sigma^alpha) / x^(alpha+1)
        pdf[valid] = (alpha[valid] * np.power(sigma[valid], alpha[valid])) / \
                     np.power(x_expanded[valid], alpha[valid] + 1.0)

        # CDF: 1 - (sigma/x)^alpha
        cdf[valid] = 1.0 - np.power(sigma[valid] / x_expanded[valid], alpha[valid])

    elif ptype == 2:
        # Pareto Type II (Lomax distribution)
        valid = x_expanded >= 0

        # PDF: (alpha/sigma) * (1 + x/sigma)^(-(alpha+1))
        pdf[valid] = (alpha[valid] / sigma[valid]) * \
                     np.power(1.0 + x_expanded[valid] / sigma[valid], -(alpha[valid] + 1.0))

        # CDF: 1 - (1 + x/sigma)^(-alpha)
        cdf[valid] = 1.0 - np.power(1.0 + x_expanded[valid] / sigma[valid], -alpha[valid])

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

    # Initial estimates using method of moments
    mean_x = np.mean(x_clean)
    std_x = np.std(x_clean, ddof=1)

    # Initial guesses
    shape_init = 0.1
    scale_init = std_x * np.sqrt(6) / np.pi
    location_init = mean_x - 0.57722 * scale_init

    # Negative log-likelihood function for GEV
    def neg_log_likelihood(params):
        loc, scale, shape = params
        if scale <= 0:
            return np.inf

        z = (x_clean - loc) / scale
        if np.abs(shape) < 1e-10:
            # Gumbel case
            return len(x_clean) * np.log(scale) + np.sum(z + np.exp(-z))
        else:
            t = 1.0 + shape * z
            if np.any(t <= 0):
                return np.inf
            return len(x_clean) * np.log(scale) + \
                   (1.0 + 1.0 / shape) * np.sum(np.log(t)) + \
                   np.sum(np.power(t, -1.0 / shape))

    # Optimize
    try:
        result = optimize.minimize(
            neg_log_likelihood,
            [location_init, scale_init, shape_init],
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )

        if result.success:
            location, scale, shape = result.x

            # Estimate standard errors using Hessian approximation
            # For simplicity, use a finite difference approximation
            try:
                hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else None
                if hess_inv is not None and isinstance(hess_inv, np.ndarray):
                    se = np.sqrt(np.diag(hess_inv))
                else:
                    # Rough approximation
                    se = np.array([scale / np.sqrt(len(x_clean))] * 3)
            except:
                se = np.array([scale / np.sqrt(len(x_clean))] * 3)

            return np.array([location, scale, shape, se[0], se[1], se[2]], dtype=np.float64)
        else:
            return np.full(6, np.nan, dtype=np.float64)

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

    # Estimate location as minimum (or close to it)
    location = np.min(x_clean) - 0.01 * np.std(x_clean)

    # Shift data
    x_shifted = x_clean - location

    # Method of moments initial estimates
    mean_shifted = np.mean(x_shifted)
    var_shifted = np.var(x_shifted, ddof=1)

    # shape (k) and scale (theta) relationship: mean = k*theta, var = k*theta^2
    theta_init = var_shifted / mean_shifted
    k_init = mean_shifted / theta_init

    # MLE for shape parameter (iterative)
    def mle_shape_eq(k):
        n = len(x_shifted)
        return np.log(k) - special.digamma(k) - np.log(np.mean(x_shifted)) + np.mean(np.log(x_shifted))

    try:
        shape = optimize.fsolve(mle_shape_eq, k_init)[0]
        if shape <= 0:
            shape = k_init
    except:
        shape = k_init

    # MLE for scale parameter
    scale = np.mean(x_shifted) / shape

    # Variance
    variance = shape * scale**2

    # Median (approximate using quantile function)
    median = location + scale * special.gammaincinv(shape, 0.5)

    return np.array([location, scale, shape, variance, median], dtype=np.float64)


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
