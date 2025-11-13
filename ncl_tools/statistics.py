"""
Statistics functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
statistical functions with NCL-compatible signatures.

All functions use NumPy/SciPy and follow NCL parameter conventions exactly.

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/statistics.shtml
"""

import numpy as np
from scipy import stats as sp_stats


# ==============================================================================
# DIMENSIONAL STATISTICS
# ==============================================================================

def dim_avg_n(x, dims):
    """
    Calculate average over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to average over (must be consecutive and monotonically increasing)

    Returns
    -------
    ndarray
        Average over specified dimensions (double if x is double, else float)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_avg_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    # Convert dims to tuple for numpy
    dims_tuple = tuple(dims)

    return np.mean(x, axis=dims_tuple)


def dim_stddev_n(x, dims):
    """
    Calculate sample standard deviation over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to calculate standard deviation over

    Returns
    -------
    ndarray
        Standard deviation (double if x is double, else float)

    Notes
    -----
    Uses sample standard deviation (divides by N-1, not N)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_stddev_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.std(x, axis=dims_tuple, ddof=1)


def dim_variance_n(x, dims):
    """
    Calculate unbiased sample variance over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to calculate variance over

    Returns
    -------
    ndarray
        Variance (double if x is double, else float)

    Notes
    -----
    Uses unbiased estimator (divides by N-1)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_variance_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.var(x, axis=dims_tuple, ddof=1)


def dim_max_n(x, dims):
    """
    Calculate maximum over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to find maximum over

    Returns
    -------
    ndarray
        Maximum values (same type as input)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_max_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.max(x, axis=dims_tuple)


def dim_min_n(x, dims):
    """
    Calculate minimum over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to find minimum over

    Returns
    -------
    ndarray
        Minimum values (same type as input)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_min_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.min(x, axis=dims_tuple)


def dim_median_n(x, dims):
    """
    Calculate median over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to calculate median over

    Returns
    -------
    ndarray
        Median values (double if x is double, else float)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_median_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.median(x, axis=dims_tuple)


def dim_sum_n(x, dims):
    """
    Calculate sum over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to sum over

    Returns
    -------
    ndarray
        Sum (double if x is double, else float)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_sum_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    return np.sum(x, axis=dims_tuple)


def dim_cumsum_n(x, dim, opt):
    """
    Calculate cumulative sum along specified dimension.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dim : int
        Dimension to calculate cumsum along
    opt : int
        Option: 0=forward cumsum, 1=reverse cumsum

    Returns
    -------
    ndarray
        Cumulative sum (same shape as input)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_cumsum_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)

    if opt == 0:
        # Forward cumsum
        result = np.cumsum(x, axis=dim)
    elif opt == 1:
        # Reverse cumsum
        result = np.flip(np.cumsum(np.flip(x, axis=dim), axis=dim), axis=dim)
    else:
        raise ValueError("opt must be 0 (forward) or 1 (reverse)")

    return result


def dim_rmvmean_n(x, dims):
    """
    Remove mean from specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to remove mean from

    Returns
    -------
    ndarray
        Array with mean removed (same shape as input)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_rmvmean_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    # Calculate mean and subtract
    mean = np.mean(x, axis=dims_tuple, keepdims=True)

    return x - mean


def dim_rmvmed_n(x, dims):
    """
    Remove median from specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    dims : int or array_like
        Dimension(s) to remove median from

    Returns
    -------
    ndarray
        Array with median removed (same shape as input)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_rmvmed_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    # Calculate median and subtract
    median = np.median(x, axis=dims_tuple, keepdims=True)

    return x - median


def dim_standardize_n(x, opt, dims):
    """
    Standardize (normalize) data over specified dimensions.

    Parameters
    ----------
    x : array_like
        Input array of any dimensionality
    opt : int
        Option for standard deviation calculation:
        - opt=1: use population std (divide by N)
        - opt=0 or other: use sample std (divide by N-1)
    dims : int or array_like
        Dimension(s) to standardize over

    Returns
    -------
    ndarray
        Standardized array (same shape as input)

    Notes
    -----
    This function removes the mean and divides by standard deviation.
    The opt parameter only controls which std calculation to use.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_standardize_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    # Remove mean
    mean = np.mean(x, axis=dims_tuple, keepdims=True)
    result = x - mean

    # Divide by standard deviation
    # opt=1: population std (ddof=0), otherwise sample std (ddof=1)
    ddof = 0 if opt == 1 else 1
    std = np.std(x, axis=dims_tuple, ddof=ddof, keepdims=True)
    result = result / std

    return result


def dim_rmsd_n(x, y, dims):
    """
    Calculate root-mean-square-difference between two arrays.

    Parameters
    ----------
    x : array_like
        First input array
    y : array_like
        Second input array
    dims : int or array_like
        Dimension(s) to calculate RMSD over

    Returns
    -------
    ndarray
        RMSD values

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dim_rmsd_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dims = np.atleast_1d(dims)

    dims_tuple = tuple(dims)

    # RMSD = sqrt(mean((x - y)^2))
    diff_sq = (x - y) ** 2
    rmsd = np.sqrt(np.mean(diff_sq, axis=dims_tuple))

    return rmsd


# ==============================================================================
# CORRELATION
# ==============================================================================

def escorc_n(x, y, dims_x, dims_y):
    """
    Calculate Pearson correlation coefficient at lag 0.

    Parameters
    ----------
    x : array_like
        First input array
    y : array_like
        Second input array
    dims_x : int
        Dimension of x to correlate over
    dims_y : int
        Dimension of y to correlate over

    Returns
    -------
    ndarray
        Correlation coefficients (double if either input is double, else float)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/escorc_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Move correlation dimensions to last position
    x = np.moveaxis(x, dims_x, -1)
    y = np.moveaxis(y, dims_y, -1)

    # Calculate correlation along last axis
    # Remove means
    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    # Calculate correlation coefficient
    numerator = np.sum(x_centered * y_centered, axis=-1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=-1) * np.sum(y_centered**2, axis=-1))

    cor = numerator / denominator

    return cor


def escovc(x, y):
    """
    Calculate sample cross-covariance at lag 0.

    Parameters
    ----------
    x : array_like
        First input array (rightmost dimension is typically time)
    y : array_like
        Second input array (rightmost dimension is typically time)

    Returns
    -------
    ndarray
        Covariance (double if either input is double, else float)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/escovc.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Remove means along last axis
    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    # Calculate covariance
    n = x.shape[-1]
    cov = np.sum(x_centered * y_centered, axis=-1) / (n - 1)

    return cov


def pattern_cor(x, y, w, opt):
    """
    Calculate pattern correlation.

    Parameters
    ----------
    x : array_like
        First pattern array
    y : array_like
        Second pattern array
    w : array_like or None
        Weights (None for unweighted)
    opt : int
        Option: 0=centered, 1=uncentered

    Returns
    -------
    float or ndarray
        Pattern correlation coefficient

    Notes
    -----
    Pattern correlation is commonly used to compare spatial patterns.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/pattern_cor.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if w is None:
        w = np.ones_like(x.flat)
    else:
        w = np.asarray(w, dtype=np.float64).flatten()

    x_flat = x.flatten()
    y_flat = y.flatten()

    if opt == 0:
        # Centered: remove weighted means
        x_mean = np.average(x_flat, weights=w)
        y_mean = np.average(y_flat, weights=w)
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
    else:
        # Uncentered
        x_centered = x_flat
        y_centered = y_flat

    # Calculate weighted correlation
    numerator = np.sum(w * x_centered * y_centered)
    denominator = np.sqrt(np.sum(w * x_centered**2) * np.sum(w * y_centered**2))

    cor = numerator / denominator

    return cor


# ==============================================================================
# REGRESSION
# ==============================================================================

def regline(x, y):
    """
    Calculate simple linear regression for 1D arrays.

    Parameters
    ----------
    x : array_like
        Independent variable (1D)
    y : array_like
        Dependent variable (1D)

    Returns
    -------
    tuple
        (slope, y_intercept) where slope is the regression coefficient

    Notes
    -----
    The return value (slope) will have attributes:
    - yintercept: y-intercept
    - tval: t-statistic
    - rstd: standard error
    - nptxy: number of points used
    - xave: mean of x
    - yave: mean of y

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/regline.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)

    slope = numerator / denominator
    y_intercept = y_mean - slope * x_mean

    # Calculate statistics
    n = len(x)
    y_pred = slope * x + y_intercept
    residuals = y - y_pred
    sse = np.sum(residuals**2)
    mse = sse / (n - 2)
    se_slope = np.sqrt(mse / np.sum((x - x_mean)**2))
    t_stat = slope / se_slope

    # Store as attributes (we'll return a class to hold these)
    class RegressionResult:
        def __init__(self, slope, yintercept, tval, rstd, nptxy, xave, yave):
            self.value = slope
            self.yintercept = yintercept
            self.tval = tval
            self.rstd = rstd
            self.nptxy = nptxy
            self.xave = xave
            self.yave = yave

    result = RegressionResult(slope, y_intercept, t_stat, se_slope, n, x_mean, y_mean)

    return result


def regCoef_n(x, y, dims_x, dims_y):
    """
    Calculate linear regression coefficients.

    Parameters
    ----------
    x : array_like
        Independent variable
    y : array_like
        Dependent variable
    dims_x : int
        Dimension of x to regress along
    dims_y : int
        Dimension of y to regress along

    Returns
    -------
    ndarray
        Regression coefficients (double if either input is double, else float)

    Notes
    -----
    The return value will have attributes:
    - yintercept: y-intercept values
    - tval: t-statistics
    - rstd: standard errors
    - nptxy: number of points used

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/regCoef_n.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Move regression dimensions to last position
    x = np.moveaxis(x, dims_x, -1)
    y = np.moveaxis(y, dims_y, -1)

    # Calculate means
    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)

    # Calculate regression coefficients
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered, axis=-1)
    denominator = np.sum(x_centered**2, axis=-1)

    slope = numerator / denominator
    y_intercept = np.squeeze(y_mean) - slope * np.squeeze(x_mean)

    # Calculate statistics
    n = x.shape[-1]
    y_pred = np.expand_dims(slope, -1) * x + np.expand_dims(y_intercept, -1)
    residuals = y - y_pred
    sse = np.sum(residuals**2, axis=-1)
    mse = sse / (n - 2)
    se_slope = np.sqrt(mse / np.sum(x_centered**2, axis=-1))
    t_stat = slope / se_slope

    # Create result object with attributes
    class RegressionResult(np.ndarray):
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            return obj

    result = RegressionResult(slope)
    result.yintercept = y_intercept
    result.tval = t_stat
    result.rstd = se_slope
    result.nptxy = n

    return result


# ==============================================================================
# TREND ANALYSIS
# ==============================================================================

def dtrend_n(y, return_info, dim):
    """
    Estimate and remove least squares linear trend.

    Parameters
    ----------
    y : array_like
        Input array
    return_info : bool
        If True, attach slope and y_intercept as attributes
    dim : int
        Dimension to detrend along

    Returns
    -------
    ndarray
        Detrended array (same shape as input, mean removed)

    Notes
    -----
    Does not handle missing values. Use dtrend_msg_n if missing values exist.
    Assumes equally spaced data along detrend dimension.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_n.shtml
    """
    y = np.asarray(y, dtype=np.float64)

    # Move detrend dimension to last position
    y = np.moveaxis(y, dim, -1)

    # Create time index
    n = y.shape[-1]
    t = np.arange(n, dtype=np.float64)

    # Calculate trend
    t_mean = np.mean(t)
    y_mean = np.mean(y, axis=-1, keepdims=True)

    t_centered = t - t_mean
    y_centered = y - y_mean

    # Regression coefficient (slope)
    numerator = np.sum(y_centered * t_centered, axis=-1, keepdims=True)
    denominator = np.sum(t_centered**2)

    slope = numerator / denominator
    y_intercept = np.squeeze(y_mean, axis=-1) - slope * t_mean

    # Calculate trend line
    trend = slope * t + np.expand_dims(y_intercept, -1)

    # Remove trend and mean
    detrended = y - trend - y_mean

    # Move dimension back
    detrended = np.moveaxis(detrended, -1, dim)

    if return_info:
        # Flatten slope and y_intercept if multi-dimensional
        slope = np.squeeze(slope)

        class DetrendResult(np.ndarray):
            def __new__(cls, input_array):
                obj = np.asarray(input_array).view(cls)
                return obj

        result = DetrendResult(detrended)
        result.slope = slope
        result.y_intercept = y_intercept
        return result
    else:
        return detrended


def dtrend_msg_n(x, y, remove_mean, return_info, dim):
    """
    Estimate and remove least squares linear trend (missing values allowed).

    Parameters
    ----------
    x : array_like
        One-dimensional coordinate array (e.g., time). If equally spaced,
        can be generated internally by passing None and using iota.
    y : array_like
        Input array (may contain missing values as NaN)
    remove_mean : bool
        If True: remove mean from detrended output
        If False: do not remove mean (return with mean intact)
    return_info : bool
        If True, attach slope and y_intercept as attributes
    dim : int
        Dimension of y to detrend along

    Returns
    -------
    ndarray
        Detrended array (same shape as input)

    Notes
    -----
    If x is None, assumes equally spaced coordinates.
    Missing values (NaN) are handled appropriately.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/dtrend_msg_n.shtml
    """
    y = np.asarray(y, dtype=np.float64)

    # Move detrend dimension to last position
    y = np.moveaxis(y, dim, -1)

    # Create or use provided coordinate
    n = y.shape[-1]
    if x is None:
        x = np.arange(n, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)

    # Handle missing values
    valid = ~np.isnan(y)

    # Calculate trend with missing values
    result = np.zeros_like(y)
    slopes = []
    intercepts = []

    # For each point in other dimensions
    for idx in np.ndindex(y.shape[:-1]):
        y_slice = y[idx]
        valid_slice = valid[idx]

        if np.sum(valid_slice) < 2:
            # Not enough valid points
            result[idx] = np.nan
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue

        x_valid = x[valid_slice]
        y_valid = y_slice[valid_slice]

        # Calculate trend
        x_mean = np.mean(x_valid)
        y_mean = np.mean(y_valid)

        x_centered = x_valid - x_mean
        y_centered = y_valid - y_mean

        slope = np.sum(x_centered * y_centered) / np.sum(x_centered**2)
        y_intercept = y_mean - slope * x_mean

        slopes.append(slope)
        intercepts.append(y_intercept)

        # Calculate and remove trend
        trend = slope * x + y_intercept
        detrended = y_slice - trend

        if remove_mean:
            # Also remove mean if requested
            detrended = detrended - np.nanmean(detrended)

        result[idx] = detrended

    # Move dimension back
    result = np.moveaxis(result, -1, dim)

    if return_info:
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        class DetrendResult(np.ndarray):
            def __new__(cls, input_array):
                obj = np.asarray(input_array).view(cls)
                return obj

        result_obj = DetrendResult(result)
        result_obj.slope = slopes
        result_obj.y_intercept = intercepts
        return result_obj
    else:
        return result


# ==============================================================================
# STATISTICAL TESTS
# ==============================================================================

def ttest(ave1, var1, s1, ave2, var2, s2, iflag, tval_opt):
    """
    Perform Student's t-test.

    Parameters
    ----------
    ave1 : float or array_like
        Mean of first sample
    var1 : float or array_like
        Variance of first sample
    s1 : int or array_like
        Number of statistically independent observations in first sample
    ave2 : float or array_like
        Mean of second sample
    var2 : float or array_like
        Variance of second sample
    s2 : int or array_like
        Number of statistically independent observations in second sample
    iflag : bool or int
        If False/0: assume equal variances (pooled variance)
        If True/1: assume unequal variances (Welch's t-test)
    tval_opt : bool or int
        If True/1: return both probability and t-value
        If False/0: return probability only

    Returns
    -------
    float or ndarray or tuple
        If tval_opt=False: probability only
        If tval_opt=True: (probability, t-value)

    Notes
    -----
    Tests null hypothesis that means are equal (H0: ave1 = ave2)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/ttest.shtml
    """
    ave1 = np.asarray(ave1, dtype=np.float64)
    ave2 = np.asarray(ave2, dtype=np.float64)
    var1 = np.asarray(var1, dtype=np.float64)
    var2 = np.asarray(var2, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.int64)
    s2 = np.asarray(s2, dtype=np.int64)

    if not iflag:
        # iflag=False: Equal variances (pooled variance)
        pooled_var = ((s1 - 1) * var1 + (s2 - 1) * var2) / (s1 + s2 - 2)
        se = np.sqrt(pooled_var * (1.0 / s1 + 1.0 / s2))
        df = s1 + s2 - 2
    else:
        # iflag=True: Unequal variances (Welch's t-test)
        se = np.sqrt(var1 / s1 + var2 / s2)
        # Welch-Satterthwaite degrees of freedom
        df = (var1 / s1 + var2 / s2)**2 / \
             ((var1 / s1)**2 / (s1 - 1) + (var2 / s2)**2 / (s2 - 1))
        df = df.astype(np.int64)

    # Calculate t-statistic
    t_val = (ave1 - ave2) / se

    # Calculate two-tailed probability
    prob = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_val), df))

    if tval_opt:
        # Return both probability and t-value
        return (prob, t_val)
    else:
        # Return probability only
        return prob


def ftest(var1, n1, var2, n2, opt):
    """
    Perform F-test for variances.

    Parameters
    ----------
    var1 : float or array_like
        Variance of first sample
    n1 : int or array_like
        Sample size of first sample
    var2 : float or array_like
        Variance of second sample
    n2 : int or array_like
        Sample size of second sample
    opt : int
        Option: 0=return probability, 1=return F-statistic

    Returns
    -------
    float or ndarray
        Either F-statistic or probability depending on opt

    Notes
    -----
    Tests null hypothesis that variances are equal

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/ftest.shtml
    """
    var1 = np.asarray(var1, dtype=np.float64)
    var2 = np.asarray(var2, dtype=np.float64)
    n1 = np.asarray(n1, dtype=np.int64)
    n2 = np.asarray(n2, dtype=np.int64)

    # Calculate F-statistic
    f_stat = var1 / var2

    df1 = n1 - 1
    df2 = n2 - 1

    if opt == 1:
        # Return F-statistic
        return f_stat
    else:
        # Return two-tailed probability
        # For two-tailed test, use 2 * min(lower tail, upper tail)
        prob_lower = sp_stats.f.cdf(f_stat, df1, df2)
        prob_upper = 1.0 - prob_lower
        prob = 2.0 * np.minimum(prob_lower, prob_upper)
        return prob


def student_t(t, df):
    """
    Calculate two-tailed probability of Student's t-distribution.

    Parameters
    ----------
    t : float or array_like
        T-statistic values
    df : int or array_like
        Degrees of freedom

    Returns
    -------
    float or ndarray
        Two-tailed probability

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/student_t.shtml
    """
    t = np.asarray(t, dtype=np.float64)
    df = np.asarray(df, dtype=np.int64)

    # Two-tailed probability
    prob = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t), df))

    return prob


# ==============================================================================
# OTHER STATISTICS
# ==============================================================================

def equiv_sample_size(x, siglvl, dims):
    """
    Estimate equivalent sample size accounting for autocorrelation.

    Parameters
    ----------
    x : array_like
        Input time series
    siglvl : float
        Significance level (e.g., 0.05 for 95% confidence)
    dims : int or array_like
        Dimension(s) representing time

    Returns
    -------
    float or ndarray
        Equivalent sample size

    Notes
    -----
    Uses lag-1 autocorrelation to estimate effective sample size.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/equiv_sample_size.shtml
    """
    x = np.asarray(x, dtype=np.float64)
    dims = np.atleast_1d(dims)

    # Move time dimension to last position
    if len(dims) == 1:
        x = np.moveaxis(x, dims[0], -1)

    n = x.shape[-1]

    # Calculate lag-1 autocorrelation
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_centered = x - x_mean

    # Lag-1 autocorrelation
    r1 = np.sum(x_centered[..., :-1] * x_centered[..., 1:], axis=-1) / \
         np.sum(x_centered**2, axis=-1)

    # Effective sample size
    # n_eff = n * (1 - r1) / (1 + r1)
    n_eff = n * (1.0 - r1) / (1.0 + r1)

    # Ensure n_eff is at least 1
    n_eff = np.maximum(n_eff, 1.0)

    return n_eff


def taylor_stats(x, y, opt):
    """
    Calculate Taylor diagram statistics.

    Parameters
    ----------
    x : array_like
        Reference pattern
    y : array_like
        Test pattern
    opt : int or array_like
        Options (currently unused, set to 0)

    Returns
    -------
    ndarray
        Array with [correlation, std_ratio, centered_rmsd]

    Notes
    -----
    Used for creating Taylor diagrams to compare model performance.

    References
    ----------
    Taylor, K. E., 2001: Summarizing multiple aspects of model performance
    in a single diagram. J. Geophys. Res., 106, 7183-7192.

    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/taylor_stats.shtml
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    # Calculate correlation
    cor = np.corrcoef(x, y)[0, 1]

    # Calculate standard deviations
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    std_ratio = std_y / std_x

    # Calculate centered RMSD
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    centered_rmsd = np.sqrt(np.mean((x_centered - y_centered)**2))

    # Normalize centered RMSD by reference std
    centered_rmsd_norm = centered_rmsd / std_x

    return np.array([cor, std_ratio, centered_rmsd_norm])


def bin_avg(x, y, bins):
    """
    Calculate bin averages.

    Parameters
    ----------
    x : array_like
        Input data to bin
    y : array_like
        Values to average within bins
    bins : array_like
        Bin edges

    Returns
    -------
    tuple
        (bin_centers, bin_averages, bin_counts)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/bin_avg.shtml
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    bins = np.asarray(bins, dtype=np.float64)

    # Calculate bin averages
    bin_sums = np.zeros(len(bins) - 1)
    bin_counts = np.zeros(len(bins) - 1, dtype=np.int64)

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(mask):
            bin_sums[i] = np.sum(y[mask])
            bin_counts[i] = np.sum(mask)

    # Calculate averages (avoid division by zero)
    bin_avgs = np.zeros(len(bins) - 1)
    nonzero = bin_counts > 0
    bin_avgs[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    bin_avgs[~nonzero] = np.nan

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    return (bin_centers, bin_avgs, bin_counts)


def bin_sum(x, y, bins):
    """
    Calculate bin sums.

    Parameters
    ----------
    x : array_like
        Input data to bin
    y : array_like
        Values to sum within bins
    bins : array_like
        Bin edges

    Returns
    -------
    tuple
        (bin_centers, bin_sums, bin_counts)

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/bin_sum.shtml
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    bins = np.asarray(bins, dtype=np.float64)

    # Calculate bin sums
    bin_sums = np.zeros(len(bins) - 1)
    bin_counts = np.zeros(len(bins) - 1, dtype=np.int64)

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(mask):
            bin_sums[i] = np.sum(y[mask])
            bin_counts[i] = np.sum(mask)

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    return (bin_centers, bin_sums, bin_counts)
