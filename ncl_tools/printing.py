"""
NCL Printing Functions

This module provides Python implementations of NCL's printing and output functions,
including variable summaries, min/max display, formatted matrix output, and ASCII table.

References
----------
NCL Documentation: https://www.ncl.ucar.edu/Document/Functions/printing.shtml
"""

import numpy as np
import sys
import re
from typing import Union, Optional


def printVarSummary(data, var_name="data"):
    """
    Prints a summary of a variable's information.

    Parameters
    ----------
    data : array_like or any
        Variable to summarize
    var_name : str, optional
        Name to display for the variable (default: "data")

    Notes
    -----
    Displays variable name, type, dimensions, size, and attributes (if available).
    Does not print the actual data values.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/printVarSummary.shtml
    """
    # Handle different input types
    if isinstance(data, np.ndarray):
        dtype_name = str(data.dtype)
        shape = data.shape
        ndim = data.ndim
        size = data.size
        nbytes = data.nbytes

        print(f"\nVariable: {var_name}")
        print(f"Type: {dtype_name}")
        print(f"Total Size: {nbytes} bytes")
        print(f"            {size} values")

        if ndim > 0:
            print(f"Number of Dimensions: {ndim}")
            print(f"Dimensions and sizes:\t[{'] x ['.join(map(str, shape))}]")
        else:
            print("Number of Dimensions: 0 (scalar)")

        # Print attributes if they exist (for structured arrays or custom objects)
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            print(f"Number of Attributes: {len(data.dtype.names)}")
            for name in data.dtype.names:
                print(f"    {name} : (field)")
        elif hasattr(data, '__dict__') and len(data.__dict__) > 0:
            attrs = {k: v for k, v in data.__dict__.items() if not k.startswith('_')}
            if attrs:
                print(f"Number of Attributes: {len(attrs)}")
                for key, value in attrs.items():
                    if isinstance(value, (np.ndarray, list)) and len(str(value)) > 50:
                        print(f"    {key} : {type(value).__name__}[...]")
                    else:
                        print(f"    {key} : {value}")

    else:
        # Non-array data
        print(f"\nVariable: {var_name}")
        print(f"Type: {type(data).__name__}")
        if hasattr(data, '__len__'):
            try:
                print(f"Length: {len(data)}")
            except:
                pass
        print(f"Value: {data}")


def printMinMax(data, opt, var_name="data"):
    """
    Prints the minimum and maximum values of a variable.

    Parameters
    ----------
    data : array_like
        Numeric variable of any dimensionality
    opt : bool or int
        If True/1: add line feed before printing
        If False/0: no line feed before printing
    var_name : str, optional
        Variable name to display (default: "data")

    Notes
    -----
    Automatically displays variable name, units (if available as attribute),
    and min/max values.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Contributed/printMinMax.shtml
    """
    data = np.asarray(data)

    # Add line feed if requested
    if opt:
        print()

    # Get variable name or long_name attribute
    display_name = var_name
    if hasattr(data, 'long_name'):
        display_name = data.long_name
    elif hasattr(data, 'description'):
        display_name = data.description
    elif hasattr(data, 'standard_name'):
        display_name = data.standard_name

    # Get units if available
    units_str = ""
    if hasattr(data, 'units'):
        units_str = f" ({data.units})"

    # Calculate min and max, handling NaN values
    if np.issubdtype(data.dtype, np.floating):
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
    else:
        min_val = np.min(data)
        max_val = np.max(data)

    # Print formatted output
    print(f"{display_name}{units_str} : min={min_val}   max={max_val}")


def write_matrix(data, fmtf, option):
    """
    Outputs formatted 2D arrays to standard output or a file.

    Parameters
    ----------
    data : array_like
        2D array of numeric type (integer, float, or double)
    fmtf : str
        Fortran-style format string (e.g., "7f10.2", "5i6", "3e15.5")
    option : bool or dict
        If False: output to stdout
        If True or dict: use optional attributes:
            - 'fout': output filename (default: stdout)
            - 'title': title string to print above data
            - 'tspace': leading spaces before title (default: 0)
            - 'row': if True, print row numbers (default: False)

    Notes
    -----
    Supported format descriptors:
    - Integers: Iw or Iw.m (e.g., "I5", "I8.5")
    - Floats: Fw.d, Ew.d, Gw.d (e.g., "F10.2", "E15.5", "G12.4")

    Format string can include repetition counts (e.g., "7f7.2" for seven F7.2 fields)
    and multiple descriptors separated by commas (e.g., "2i5,3f10.2").

    Maximum row count with row numbers: 99,999

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/write_matrix.shtml
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("write_matrix requires a 2D array")

    nrows, ncols = data.shape

    # Parse options
    if isinstance(option, dict):
        fout = option.get('fout', None)
        title = option.get('title', None)
        tspace = option.get('tspace', 0)
        row = option.get('row', False)
    elif option:
        fout = None
        title = None
        tspace = 0
        row = False
    else:
        fout = None
        title = None
        tspace = 0
        row = False

    # Open output file or use stdout
    if fout:
        f = open(fout, 'w')
    else:
        f = sys.stdout

    try:
        # Print title if provided
        if title:
            f.write(' ' * tspace + title + '\n')

        # Parse format string
        formats = _parse_fortran_format(fmtf, ncols)

        # Print each row
        for i in range(nrows):
            line = ""

            # Add row number if requested
            if row:
                if nrows > 99999:
                    raise ValueError("Maximum row count with row numbers is 99,999")
                line += f"{i:5d} "

            # Format each column
            for j in range(ncols):
                val = data[i, j]
                if j < len(formats):
                    line += _format_value(val, formats[j])
                else:
                    # Use last format for remaining columns
                    line += _format_value(val, formats[-1])

            f.write(line.rstrip() + '\n')

    finally:
        if fout:
            f.close()


def _parse_fortran_format(fmtf, ncols):
    """
    Parse Fortran-style format string.

    Returns list of format tuples: (type, width, decimals)
    """
    formats = []
    fmtf = fmtf.upper().strip()

    # Split by commas
    parts = [p.strip() for p in fmtf.split(',')]

    for part in parts:
        # Match repetition and format descriptor
        # Examples: 7F10.2, I5, 3E15.5, F8.3
        match = re.match(r'(\d*)([IFEG])(\d+)(?:\.(\d+))?', part)
        if match:
            repeat = int(match.group(1)) if match.group(1) else 1
            fmt_type = match.group(2)
            width = int(match.group(3))
            decimals = int(match.group(4)) if match.group(4) else 0

            for _ in range(repeat):
                formats.append((fmt_type, width, decimals))

    # If not enough formats specified, repeat the last one
    if len(formats) < ncols:
        if formats:
            last_fmt = formats[-1]
            for _ in range(ncols - len(formats)):
                formats.append(last_fmt)
        else:
            # Default format
            formats = [('F', 12, 4)] * ncols

    return formats


def _format_value(val, fmt):
    """
    Format a single value according to format specification.

    Parameters
    ----------
    val : float
        Value to format
    fmt : tuple
        (type, width, decimals)

    Returns
    -------
    str
        Formatted string
    """
    fmt_type, width, decimals = fmt

    if np.isnan(val):
        return ' ' * (width - 3) + 'NaN'

    if fmt_type == 'I':
        # Integer format
        try:
            return f"{int(val):>{width}d}"
        except:
            return ' ' * (width - 3) + '***'

    elif fmt_type == 'F':
        # Fixed-point format
        try:
            fmt_str = f"{{:>{width}.{decimals}f}}"
            result = fmt_str.format(val)
            if len(result) > width:
                return '*' * width
            return result
        except:
            return '*' * width

    elif fmt_type == 'E':
        # Exponential format
        try:
            fmt_str = f"{{:>{width}.{decimals}e}}"
            result = fmt_str.format(val)
            if len(result) > width:
                return '*' * width
            return result
        except:
            return '*' * width

    elif fmt_type == 'G':
        # General format (uses E or F depending on magnitude)
        try:
            fmt_str = f"{{:>{width}.{decimals}g}}"
            result = fmt_str.format(val)
            if len(result) > width:
                return '*' * width
            return result
        except:
            return '*' * width

    else:
        # Default: fixed-point
        return f"{val:>{width}.{decimals}f}"


def show_ascii():
    """
    Prints the ASCII table to the screen.

    Displays all 128 ASCII characters (0-127) with their decimal values.
    Control characters are labeled with their standard abbreviations,
    and printable characters show their actual symbols.

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/show_ascii.shtml
    """
    # ASCII control character names
    control_chars = [
        "nul", "soh", "stx", "etx", "eot", "enq", "ack", "bel",
        "bs",  "ht",  "nl",  "vt",  "np",  "cr",  "so",  "si",
        "dle", "dc1", "dc2", "dc3", "dc4", "nak", "syn", "etb",
        "can", "em",  "sub", "esc", "fs",  "gs",  "rs",  "us"
    ]

    print("\nASCII Character Table")
    print("=" * 80)

    # Print in rows of 8 columns
    for row in range(16):
        line = ""
        for col in range(8):
            ascii_val = row * 8 + col
            if ascii_val >= 128:
                break

            if ascii_val < 32:
                # Control character
                char_repr = control_chars[ascii_val]
            elif ascii_val == 32:
                char_repr = "sp"  # space
            elif ascii_val == 127:
                char_repr = "del"  # delete
            else:
                # Printable character
                char_repr = chr(ascii_val)

            line += f"{ascii_val:3d}:{char_repr:>4s}  "

        print(line)

    print("=" * 80)


def printFileVarSummary(file_handle, var_name):
    """
    Prints a summary of a file variable's information.

    Parameters
    ----------
    file_handle : file object or dict
        File handle or dictionary representing file structure
    var_name : str
        Name of the variable to summarize

    Notes
    -----
    This function is a placeholder for NCL compatibility.
    In Python, you would typically use xarray or netCDF4 libraries
    to inspect file variables without loading them into memory.

    For netCDF files, use:
        import netCDF4
        ds = netCDF4.Dataset('file.nc')
        print(ds.variables['varname'])

    Or with xarray:
        import xarray as xr
        ds = xr.open_dataset('file.nc')
        print(ds['varname'])

    References
    ----------
    NCL: https://www.ncl.ucar.edu/Document/Functions/Built-in/printFileVarSummary.shtml
    """
    print("\nNote: printFileVarSummary is a placeholder.")
    print("For netCDF files, use netCDF4 or xarray libraries:")
    print("  import xarray as xr")
    print("  ds = xr.open_dataset('file.nc')")
    print(f"  print(ds['{var_name}'])")


__all__ = [
    'printVarSummary',
    'printMinMax',
    'write_matrix',
    'show_ascii',
    'printFileVarSummary'
]
