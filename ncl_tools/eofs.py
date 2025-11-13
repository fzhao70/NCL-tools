"""
NCL EOF (Empirical Orthogonal Function) Functions

This module provides Python implementations of NCL's EOF analysis functions,
including standard EOF computation, time series calculation, reconstruction,
and varimax rotation.

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/eofs.shtml
"""

import numpy as np
from typing import Union, Optional, Tuple


class EOFResult(np.ndarray):
    """
    Array subclass to hold EOF results with attributes.

    Attributes may include: eval, pcvar, matrix, method, eval_transpose
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


class TimeSeriesResult(np.ndarray):
    """
    Array subclass to hold time series results with attributes.

    Attributes may include: ts_mean
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return


def eofunc(data, neval, optEOF):
    """
    Computes Empirical Orthogonal Functions (EOFs, aka Principal Component Analysis).

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    neval : int
        Number of eigenvalues/eigenvectors to return
    optEOF : int, bool, or dict
        Options for EOF computation. Can be:
        - int/bool: 0/False for covariance (default), 1/True for correlation
        - dict with keys:
          - 'jopt': 0 for covariance, 1 for correlation
          - 'pcrit': percentage threshold for missing data (default 50)

    Returns
    -------
    EOFResult
        Array of normalized EOFs with shape (..., neval) where ... represents
        the spatial dimensions. Contains attributes:
        - eval: eigenvalues
        - pcvar: percent variance for each eigenvalue
        - matrix: "covariance" or "correlation"
        - method: "transpose" or "no transpose"
        - eval_transpose: eigenvalues of transposed matrix (if transposed)

    Notes
    -----
    - Missing values (NaN) are ignored during computation
    - Input often contains anomalies (mean removed) but not required
    - The function may transpose input for computational efficiency
    - EOF signs are arbitrary; interpret with corresponding time series

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc.shtml
    """
    data = np.asarray(data, dtype=np.float64)

    # Parse options
    if isinstance(optEOF, dict):
        jopt = optEOF.get('jopt', 0)
        pcrit = optEOF.get('pcrit', 50.0)
    elif isinstance(optEOF, (int, bool)):
        jopt = int(optEOF)
        pcrit = 50.0
    else:
        jopt = 0
        pcrit = 50.0

    # Get dimensions
    original_shape = data.shape
    nobs = original_shape[-1]

    # Reshape to 2D: [space, time]
    nspace = np.prod(original_shape[:-1])
    data_2d = data.reshape(nspace, nobs)

    # Remove rows with too many missing values
    missing_pct = np.isnan(data_2d).sum(axis=1) / nobs * 100
    valid_mask = missing_pct < (100 - pcrit)
    data_valid = data_2d[valid_mask]

    if data_valid.shape[0] == 0:
        raise ValueError("No valid data points after applying pcrit threshold")

    # Decide whether to transpose for computational efficiency
    # Transpose if nspace < nobs
    if data_valid.shape[0] < data_valid.shape[1]:
        transpose = True
        data_work = data_valid.T  # [time, space]
    else:
        transpose = False
        data_work = data_valid  # [space, time]

    # Handle missing values by removing mean (nanmean)
    data_mean = np.nanmean(data_work, axis=1, keepdims=True)
    data_anom = data_work - data_mean

    # Replace NaN with 0 for matrix operations
    data_anom = np.nan_to_num(data_anom, nan=0.0)

    # Compute covariance or correlation matrix
    if jopt == 0:
        # Covariance matrix
        if transpose:
            # C = X^T X / (n-1) where X is [time, space]
            cov_matrix = np.dot(data_anom.T, data_anom) / (data_anom.shape[0] - 1)
        else:
            # C = X X^T / (n-1) where X is [space, time]
            cov_matrix = np.dot(data_anom, data_anom.T) / (data_anom.shape[1] - 1)
        matrix_type = "covariance"
    else:
        # Correlation matrix: standardize first
        data_std = np.nanstd(data_work, axis=1, keepdims=True, ddof=1)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        data_standardized = data_anom / data_std
        data_standardized = np.nan_to_num(data_standardized, nan=0.0)

        if transpose:
            cov_matrix = np.dot(data_standardized.T, data_standardized) / (data_standardized.shape[0] - 1)
        else:
            cov_matrix = np.dot(data_standardized, data_standardized.T) / (data_standardized.shape[1] - 1)
        matrix_type = "correlation"

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top neval
    eigenvalues = eigenvalues[:neval]
    eigenvectors = eigenvectors[:, :neval]

    # Compute percent variance
    pcvar = eigenvalues / eigenvalues.sum() * 100.0

    # Prepare output EOFs
    if transpose:
        # EOFs are in reduced space, need to project back
        eofs = np.dot(data_anom, eigenvectors)  # [time, neval]
        # Normalize
        for i in range(neval):
            norm = np.linalg.norm(eofs[:, i])
            if norm > 0:
                eofs[:, i] /= norm

        # Reshape back: should be [space, neval] but we transposed
        # Actually, in transpose case, EOFs are in time space
        # Need to expand back to spatial dimensions
        eofs_full = np.zeros((nspace, neval))
        eofs_full[valid_mask, :] = np.dot(data_valid, eofs) / (data_valid.shape[1] - 1)

        # Normalize
        for i in range(neval):
            norm = np.linalg.norm(eofs_full[:, i])
            if norm > 0:
                eofs_full[:, i] /= norm

        method = "transpose"
        eval_transpose = eigenvalues.copy()
    else:
        # EOFs are already in spatial space
        eofs_full = np.zeros((nspace, neval))
        eofs_full[valid_mask, :] = eigenvectors
        method = "no transpose"
        eval_transpose = None

    # Reshape to original spatial dimensions plus neval
    output_shape = list(original_shape[:-1]) + [neval]
    eofs_reshaped = eofs_full.reshape(output_shape)

    # Create result with attributes
    result = EOFResult(
        eofs_reshaped,
        eval=eigenvalues,
        pcvar=pcvar,
        matrix=matrix_type,
        method=method
    )

    if eval_transpose is not None:
        result.eval_transpose = eval_transpose

    return result


def eofunc_n(data, neval, optEOF, time_dim):
    """
    Computes EOFs with specified time dimension.

    Parameters
    ----------
    data : array_like
        Multi-dimensional array
    neval : int
        Number of eigenvalues/eigenvectors to return
    optEOF : int, bool, or dict
        Options for EOF computation
    time_dim : int
        Dimension index representing time/observations

    Returns
    -------
    EOFResult
        Array of normalized EOFs with time dimension replaced by neval dimension

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_n.shtml
    """
    data = np.asarray(data, dtype=np.float64)

    # Move time dimension to last position
    data_moved = np.moveaxis(data, time_dim, -1)

    # Call eofunc
    result = eofunc(data_moved, neval, optEOF)

    # Move EOF dimension back to where time was
    result_moved = np.moveaxis(result, -1, time_dim)

    # Create new result with moved data but keep attributes
    final_result = EOFResult(
        result_moved,
        eval=result.eval,
        pcvar=result.pcvar,
        matrix=result.matrix,
        method=result.method
    )

    if hasattr(result, 'eval_transpose'):
        final_result.eval_transpose = result.eval_transpose

    return final_result


def eofunc_ts(data, evec, optETS):
    """
    Calculates time series amplitudes (principal components) for EOFs.

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    evec : array_like
        EOF patterns from eofunc
    optETS : int, bool, or dict
        Options for time series computation:
        - 0/False: use anomalies (default)
        - 1/True: use standardized data
        - dict with 'jopt' key

    Returns
    -------
    TimeSeriesResult
        2D array of shape [neval, time] containing time series amplitudes.
        Contains attribute ts_mean with the means that were subtracted.

    Notes
    -----
    - Output time series are orthogonal
    - Mean is subtracted from each time series

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_ts.shtml
    """
    data = np.asarray(data, dtype=np.float64)
    evec = np.asarray(evec, dtype=np.float64)

    # Parse options
    if isinstance(optETS, dict):
        jopt = optETS.get('jopt', 0)
    elif isinstance(optETS, (int, bool)):
        jopt = int(optETS)
    else:
        jopt = 0

    # Get dimensions
    original_shape = data.shape
    nobs = original_shape[-1]
    neval = evec.shape[-1]

    # Reshape data to 2D: [space, time]
    nspace = np.prod(original_shape[:-1])
    data_2d = data.reshape(nspace, nobs)

    # Reshape EOFs to 2D: [space, neval]
    evec_2d = evec.reshape(nspace, neval)

    # Remove mean
    data_mean = np.nanmean(data_2d, axis=1, keepdims=True)
    data_anom = data_2d - data_mean

    if jopt == 1:
        # Standardize
        data_std = np.nanstd(data_2d, axis=1, keepdims=True, ddof=1)
        data_std[data_std == 0] = 1.0
        data_anom = data_anom / data_std

    # Replace NaN with 0
    data_anom = np.nan_to_num(data_anom, nan=0.0)

    # Compute time series: PC = EOF^T * Data
    # evec_2d is [space, neval], data_anom is [space, time]
    # Result should be [neval, time]
    time_series = np.dot(evec_2d.T, data_anom)

    # Subtract mean from each time series
    ts_mean = np.mean(time_series, axis=1, keepdims=True)
    time_series = time_series - ts_mean

    # Create result with attribute
    result = TimeSeriesResult(
        time_series,
        ts_mean=ts_mean.squeeze()
    )

    return result


def eofunc_ts_n(data, evec, optETS, time_dim):
    """
    Calculates time series amplitudes with specified time dimension.

    Parameters
    ----------
    data : array_like
        Multi-dimensional array
    evec : array_like
        EOF patterns from eofunc_n
    optETS : int, bool, or dict
        Options for time series computation
    time_dim : int
        Dimension index representing time/observations

    Returns
    -------
    TimeSeriesResult
        2D array of shape [neval, time] containing time series amplitudes

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_ts_n.shtml
    """
    data = np.asarray(data, dtype=np.float64)
    evec = np.asarray(evec, dtype=np.float64)

    # Move time dimension to last position
    data_moved = np.moveaxis(data, time_dim, -1)

    # Move EOF dimension to last position in evec
    evec_moved = np.moveaxis(evec, time_dim, -1)

    # Call eofunc_ts
    result = eofunc_ts(data_moved, evec_moved, optETS)

    return result


def eof2data(evec, evec_ts):
    """
    Reconstructs data from EOFs and time series.

    Parameters
    ----------
    evec : array_like
        EOFs from eofunc (shape: [..., neval])
    evec_ts : array_like
        Time series from eofunc_ts (shape: [neval, time])

    Returns
    -------
    ndarray
        Reconstructed data with shape [..., time]

    Notes
    -----
    - To fully reconstruct original data, must add back ts_mean attribute
    - If using subset of EOFs, reconstructs that fraction of variance

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eof2data.shtml
    """
    evec = np.asarray(evec, dtype=np.float64)
    evec_ts = np.asarray(evec_ts, dtype=np.float64)

    # Get dimensions
    spatial_shape = evec.shape[:-1]
    neval = evec.shape[-1]
    ntime = evec_ts.shape[-1]

    # Reshape EOFs to 2D: [space, neval]
    nspace = np.prod(spatial_shape)
    evec_2d = evec.reshape(nspace, neval)

    # Reconstruct: Data = EOF * PC^T
    # evec_2d is [space, neval], evec_ts is [neval, time]
    # Result should be [space, time]
    reconstructed = np.dot(evec_2d, evec_ts)

    # Reshape back to original spatial dimensions plus time
    output_shape = list(spatial_shape) + [ntime]
    reconstructed = reconstructed.reshape(output_shape)

    return reconstructed


def eof2data_n(evec, evec_ts, time_dim):
    """
    Reconstructs data from EOFs and time series with specified dimension.

    Parameters
    ----------
    evec : array_like
        EOFs from eofunc_n
    evec_ts : array_like
        Time series from eofunc_ts_n (shape: [neval, time])
    time_dim : int
        Dimension index where time should be placed in output

    Returns
    -------
    ndarray
        Reconstructed data with time at specified dimension

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eof2data_n.shtml
    """
    evec = np.asarray(evec, dtype=np.float64)
    evec_ts = np.asarray(evec_ts, dtype=np.float64)

    # Move EOF dimension to last position
    evec_moved = np.moveaxis(evec, time_dim, -1)

    # Reconstruct
    reconstructed = eof2data(evec_moved, evec_ts)

    # Move time dimension to specified position
    result = np.moveaxis(reconstructed, -1, time_dim)

    return result


def eofunc_north(eval, N, prinfo):
    """
    Evaluates eigenvalue separation using North et al. (1982) methodology.

    Parameters
    ----------
    eval : array_like
        Eigenvalues to test (from EOF eval, pcvar, or eval_transpose attribute)
    N : int
        Maximum possible number of eigenvalues (typically number of time steps)
    prinfo : bool
        If True, print diagnostic information

    Returns
    -------
    ndarray
        Boolean array indicating if each eigenvalue is significantly separated
        from the next one

    Notes
    -----
    Uses equation 24 from North et al. (1982) to compute error estimates:
    delta_lambda = lambda * sqrt(2/N)

    An eigenvalue is considered significantly separated if:
    lambda[i] - lambda[i+1] > delta_lambda[i]

    References
    ----------
    North, G.R. et al (1982): "Sampling Errors in the Estimation of Empirical
    Orthogonal Functions," Monthly Weather Review, 110, 699-706.
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_north.shtml
    """
    eval = np.asarray(eval, dtype=np.float64)
    neval = len(eval)

    # Compute error bounds using North et al. equation 24
    # delta_lambda = lambda * sqrt(2/N)
    delta_lambda = eval * np.sqrt(2.0 / N)

    # Test separation: lambda[i] - lambda[i+1] > delta_lambda[i]
    separated = np.zeros(neval, dtype=bool)

    for i in range(neval - 1):
        separation = eval[i] - eval[i + 1]
        separated[i] = separation > delta_lambda[i]

    # Last eigenvalue is compared to zero (or not separated by definition)
    separated[-1] = False

    if prinfo:
        print("\nEigenvalue Separation Test (North et al. 1982)")
        print("=" * 60)
        print(f"{'Mode':<6} {'Eigenvalue':<15} {'Delta Lambda':<15} {'Separated':<10}")
        print("-" * 60)
        for i in range(neval):
            sep_str = "Yes" if separated[i] else "No"
            print(f"{i+1:<6} {eval[i]:<15.6f} {delta_lambda[i]:<15.6f} {sep_str:<10}")
        print("=" * 60)

    return separated


def eofunc_varimax(evec, optEVX):
    """
    Rotates EOFs using varimax rotation with Kaiser normalization.

    Parameters
    ----------
    evec : array_like
        EOFs to rotate (from eofunc)
    optEVX : int
        Scaling option:
        - 0: Return normalized eigenvectors
        - 1: Scale by sqrt(eigenvalues)
        - -1: Same as 1 but return scaled rotated eigenvectors

    Returns
    -------
    EOFResult
        Rotated EOFs with attributes:
        - pcvar_varimax: percent variance (may not be in descending order)
        - variance_varimax: variance values

    Notes
    -----
    - Uses Kaiser row normalization
    - Results are orthonormal (not just orthogonal)
    - Output may not be in descending variance order
    - Use eofunc_varimax_reorder to sort by variance

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_varimax.shtml
    """
    evec = np.asarray(evec, dtype=np.float64)

    # Get dimensions
    spatial_shape = evec.shape[:-1]
    neval = evec.shape[-1]
    nspace = np.prod(spatial_shape)

    # Reshape to 2D: [space, neval]
    evec_2d = evec.reshape(nspace, neval)

    # Get eigenvalues if available as attribute
    if hasattr(evec, 'eval'):
        eigenvalues = evec.eval
    else:
        # Estimate from variance of each EOF
        eigenvalues = np.var(evec_2d, axis=0)

    # Kaiser normalization: normalize rows
    row_norms = np.sqrt(np.sum(evec_2d**2, axis=1, keepdims=True))
    row_norms[row_norms == 0] = 1.0
    evec_normalized = evec_2d / row_norms

    # Perform varimax rotation
    # Use iterative algorithm
    max_iter = 100
    tol = 1e-6

    rotation_matrix = np.eye(neval)

    for iteration in range(max_iter):
        # Compute rotated loadings
        rotated = np.dot(evec_normalized, rotation_matrix)

        # Varimax criterion: maximize variance of squared loadings
        # Compute gradient
        d = rotated**3 - rotated * np.sum(rotated**2, axis=0, keepdims=True) / nspace

        # SVD to find optimal rotation update
        u, s, vt = np.linalg.svd(np.dot(evec_normalized.T, d))

        # Update rotation matrix
        new_rotation = np.dot(u, vt)

        # Check convergence
        if np.allclose(rotation_matrix, new_rotation, atol=tol):
            break

        rotation_matrix = new_rotation

    # Apply rotation
    rotated_evec = np.dot(evec_normalized, rotation_matrix)

    # Reverse Kaiser normalization
    rotated_evec = rotated_evec * row_norms

    # Optionally scale by eigenvalues
    if optEVX != 0:
        scaling = np.sqrt(eigenvalues)
        rotated_evec = rotated_evec * scaling[np.newaxis, :]

    # Compute variance explained by rotated EOFs
    variance = np.sum(rotated_evec**2, axis=0)
    pcvar = variance / variance.sum() * 100.0

    # Reshape back to original shape
    output_shape = list(spatial_shape) + [neval]
    rotated_evec = rotated_evec.reshape(output_shape)

    # Create result with attributes
    result = EOFResult(
        rotated_evec,
        pcvar_varimax=pcvar,
        variance_varimax=variance
    )

    return result


def eofunc_varimax_reorder(evec_rotated):
    """
    Reorders varimax-rotated EOFs by descending variance.

    Parameters
    ----------
    evec_rotated : EOFResult
        Rotated EOFs from eofunc_varimax with variance_varimax attribute

    Returns
    -------
    EOFResult
        Reordered EOFs with updated attributes

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/eofunc_varimax_reorder.shtml
    """
    evec = np.asarray(evec_rotated, dtype=np.float64)

    # Get variance
    if hasattr(evec_rotated, 'variance_varimax'):
        variance = evec_rotated.variance_varimax
    else:
        raise ValueError("Input must have variance_varimax attribute")

    # Sort by descending variance
    idx = np.argsort(variance)[::-1]

    # Reorder along last dimension
    evec_reordered = evec[..., idx]
    variance_reordered = variance[idx]

    # Recompute percent variance
    pcvar_reordered = variance_reordered / variance_reordered.sum() * 100.0

    # Create result with updated attributes
    result = EOFResult(
        evec_reordered,
        pcvar_varimax=pcvar_reordered,
        variance_varimax=variance_reordered
    )

    return result


def eofcov(data, neval):
    """
    Calculates EOFs via a covariance matrix (original NCL implementation).

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    neval : int
        Number of eigenvalues/eigenvectors to return

    Returns
    -------
    EOFResult
        Array of normalized EOFs with attributes:
        - trace: trace of covariance matrix
        - eval: eigenvalues
        - pcvar: percent variance for each eigenvalue
        - eof_function: 0 (indicating eofcov)

    Notes
    -----
    This is the original NCL function for calculating EOFs.
    It may be slower than eofunc for large matrices.
    Uses covariance matrix directly without adaptive algorithms.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofcov.shtml
    """
    # Call eofunc with covariance option
    result = eofunc(data, neval, {'jopt': 0})

    # Compute trace (sum of eigenvalues)
    trace = np.sum(result.eval)

    # Add trace and eof_function attributes
    result.trace = trace
    result.eof_function = 0

    return result


def eofcor(data, neval):
    """
    Calculates EOFs via a correlation matrix (original NCL implementation).

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    neval : int
        Number of eigenvalues/eigenvectors to return

    Returns
    -------
    EOFResult
        Array of normalized EOFs with attributes:
        - trace: trace of correlation matrix
        - eval: eigenvalues
        - pcvar: percent variance for each eigenvalue
        - eof_function: 1 (indicating eofcor)

    Notes
    -----
    This is the original NCL function for calculating EOFs via correlation.
    It may be slower than eofunc for large matrices.
    Uses correlation matrix directly without adaptive algorithms.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofcor.shtml
    """
    # Call eofunc with correlation option
    result = eofunc(data, neval, {'jopt': 1})

    # Compute trace (sum of eigenvalues)
    trace = np.sum(result.eval)

    # Add trace and eof_function attributes
    result.trace = trace
    result.eof_function = 1

    return result


def eofcov_ts(data, evec):
    """
    Calculates time series amplitudes for covariance-based EOFs.

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    evec : array_like
        EOF patterns from eofcov

    Returns
    -------
    TimeSeriesResult
        2D array of shape [neval, time] containing time series amplitudes

    Notes
    -----
    This function is deprecated in favor of eofunc_ts.
    Provided for NCL compatibility.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofcov_ts.shtml
    """
    return eofunc_ts(data, evec, 0)


def eofcor_ts(data, evec):
    """
    Calculates time series amplitudes for correlation-based EOFs.

    Parameters
    ----------
    data : array_like
        Multi-dimensional array where rightmost dimension is observations (time)
    evec : array_like
        EOF patterns from eofcor

    Returns
    -------
    TimeSeriesResult
        2D array of shape [neval, time] containing time series amplitudes

    Notes
    -----
    This function is deprecated in favor of eofunc_ts.
    Provided for NCL compatibility.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/eofcor_ts.shtml
    """
    return eofunc_ts(data, evec, 1)


__all__ = [
    'eofunc',
    'eofunc_n',
    'eofunc_ts',
    'eofunc_ts_n',
    'eofcov',
    'eofcov_ts',
    'eofcor',
    'eofcor_ts',
    'eof2data',
    'eof2data_n',
    'eofunc_north',
    'eofunc_varimax',
    'eofunc_varimax_reorder',
    'EOFResult',
    'TimeSeriesResult'
]
