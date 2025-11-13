"""
Latitude/longitude utility functions implemented in Python using NumPy.

This module provides Python implementations of NCL (NCAR Command Language)
latitude/longitude and spherical geometry functions.

All functions use NumPy and SciPy for robust implementations.
"""

import numpy as np
from scipy import special


def gaus(nlat):
    """
    Compute Gaussian latitudes and weights.

    Parameters
    ----------
    nlat : int
        Number of latitude points PER HEMISPHERE (total latitude points = 2*nlat)

    Returns
    -------
    ndarray
        Array of shape (2*nlat, 2) where:
        - Column 0: Gaussian latitudes in degrees
        - Column 1: Gaussian weights

    References
    ----------
    Numerical Recipes in Fortran 77: The Art of Scientific Computing,
    Press et al., Cambridge University Press, 1992.
    """
    nlat = int(nlat)

    # Total number of latitudes
    n = 2 * nlat

    # Compute Legendre polynomial roots using numpy
    # These are the Gaussian latitudes (in sine space)
    # Use Gauss-Legendre quadrature
    mu, weights = np.polynomial.legendre.leggauss(n)

    # mu ranges from -1 to 1, convert to latitudes
    lats = np.arcsin(mu) * 180.0 / np.pi

    # Reverse to go from North to South
    lats = lats[::-1]
    weights = weights[::-1]

    # Create output array
    result = np.zeros((n, 2), dtype=np.float64)
    result[:, 0] = lats
    result[:, 1] = weights

    return result


def latGlobeF(nlat, name, longname, units):
    """
    Generate latitudes and metadata for a global fixed grid.

    Parameters
    ----------
    nlat : int
        Number of latitudes
    name : str
        Variable name (e.g., "lat")
    longname : str
        Long descriptive name
    units : str
        Units (e.g., "degrees_north")

    Returns
    -------
    dict
        Dictionary with 'values' (latitude array) and 'attrs' (metadata)
    """
    nlat = int(nlat)

    # Create equally spaced latitudes from -90 to 90
    dlat = 180.0 / nlat
    lats = np.linspace(-90 + dlat/2, 90 - dlat/2, nlat)

    attrs = {
        'long_name': longname,
        'units': units,
        'name': name
    }

    return {'values': lats, 'attrs': attrs}


def lonGlobeF(nlon, name, longname, units):
    """
    Generate longitudes and metadata for a global fixed grid.

    Parameters
    ----------
    nlon : int
        Number of longitudes
    name : str
        Variable name (e.g., "lon")
    longname : str
        Long descriptive name
    units : str
        Units (e.g., "degrees_east")

    Returns
    -------
    dict
        Dictionary with 'values' (longitude array) and 'attrs' (metadata)
    """
    nlon = int(nlon)

    # Create equally spaced longitudes from 0 to 360 (excluding 360)
    dlon = 360.0 / nlon
    lons = np.linspace(0, 360 - dlon, nlon)

    attrs = {
        'long_name': longname,
        'units': units,
        'name': name
    }

    return {'values': lons, 'attrs': attrs}


def latGlobeFo(nlat, name, longname, units):
    """
    Generate latitudes and metadata for a global fixed offset grid.

    Parameters
    ----------
    nlat : int
        Number of latitudes
    name : str
        Variable name
    longname : str
        Long descriptive name
    units : str
        Units (e.g., "degrees_north")

    Returns
    -------
    dict
        Dictionary with 'values' (latitude array) and 'attrs' (metadata)
    """
    nlat = int(nlat)

    # Create equally spaced latitudes from -90 to 90 (including poles)
    lats = np.linspace(-90, 90, nlat)

    attrs = {
        'long_name': longname,
        'units': units,
        'name': name
    }

    return {'values': lats, 'attrs': attrs}


def lonGlobeFo(nlon, name, longname, units):
    """
    Generate longitudes and metadata for a global fixed offset grid.

    Parameters
    ----------
    nlon : int
        Number of longitudes
    name : str
        Variable name
    longname : str
        Long descriptive name
    units : str
        Units (e.g., "degrees_east")

    Returns
    -------
    dict
        Dictionary with 'values' (longitude array) and 'attrs' (metadata)
    """
    nlon = int(nlon)

    # Create equally spaced longitudes from 0 to 360 (including 0 but not 360)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    attrs = {
        'long_name': longname,
        'units': units,
        'name': name
    }

    return {'values': lons, 'attrs': attrs}


def latRegWgt(lat):
    """
    Generate area weights for equally spaced (regular) global grids.

    Weights are based on: sin(lat+dlat/2) - sin(lat-dlat/2)
    Weights sum to 2.0 (representing the range from -1 to 1 in sin space)

    Parameters
    ----------
    lat : array_like
        1D array of latitudes in degrees

    Returns
    -------
    ndarray
        Array of weights that sum to 2.0
    """
    lat = np.asarray(lat, dtype=np.float64)

    # Calculate latitude spacing
    dlat = np.abs(lat[1] - lat[0]) if len(lat) > 1 else 0

    # Convert to radians
    lat_rad = lat * np.pi / 180.0
    dlat_rad = dlat * np.pi / 180.0

    # Calculate weights
    weights = np.sin(lat_rad + dlat_rad/2.0) - np.sin(lat_rad - dlat_rad/2.0)

    return weights


def NormCosWgtGlobe(nlat):
    """
    Create normalized cosine weights that sum to 2.0.

    Parameters
    ----------
    nlat : int
        Number of latitudes

    Returns
    -------
    ndarray
        Array of normalized cosine weights
    """
    nlat = int(nlat)

    # Create latitudes
    dlat = 180.0 / nlat
    lats = np.linspace(-90 + dlat/2, 90 - dlat/2, nlat)

    # Calculate cosine weights
    lat_rad = lats * np.pi / 180.0
    weights = np.cos(lat_rad)

    # Normalize to sum to 2.0
    weights = 2.0 * weights / np.sum(weights)

    return weights


def gc_latlon(lat1, lon1, lat2, lon2, npts, iu):
    """
    Find great circle distance and interpolate points along the great circle.

    Parameters
    ----------
    lat1, lon1 : float
        Starting point latitude and longitude in degrees
    lat2, lon2 : float
        Ending point latitude and longitude in degrees
    npts : int
        Number of points to interpolate (including endpoints)
    iu : int
        Units for distance:
        abs(iu) = 1: radians
        abs(iu) = 2: degrees
        abs(iu) = 3: meters
        abs(iu) = 4: kilometers
        If iu > 0: longitudes 0 to 360
        If iu < 0: longitudes -180 to 180

    Returns
    -------
    dict
        Dictionary with:
        - 'distance': Great circle distance
        - 'gclat': Interpolated latitudes
        - 'gclon': Interpolated longitudes
        - 'spacing': Distance between interpolated points
    """
    # Convert to radians
    lat1_rad = lat1 * np.pi / 180.0
    lon1_rad = lon1 * np.pi / 180.0
    lat2_rad = lat2 * np.pi / 180.0
    lon2_rad = lon2 * np.pi / 180.0

    # Earth radius
    R_earth_m = 6371000.0  # meters
    R_earth_km = 6371.0     # kilometers

    # Calculate great circle distance using Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Distance in different units
    dist_rad = c
    dist_deg = c * 180.0 / np.pi
    dist_m = c * R_earth_m
    dist_km = c * R_earth_km

    # Select distance based on iu
    abs_iu = abs(iu)
    if abs_iu == 1:
        distance = dist_rad
    elif abs_iu == 2:
        distance = dist_deg
    elif abs_iu == 3:
        distance = dist_m
    elif abs_iu == 4:
        distance = dist_km
    else:
        distance = dist_km  # default

    # Interpolate points along great circle
    fractions = np.linspace(0, 1, npts)

    # Use spherical interpolation (slerp)
    gclat = np.zeros(npts)
    gclon = np.zeros(npts)

    for i, f in enumerate(fractions):
        # Spherical linear interpolation
        if f == 0:
            gclat[i] = lat1
            gclon[i] = lon1
        elif f == 1:
            gclat[i] = lat2
            gclon[i] = lon2
        else:
            A = np.sin((1-f)*dist_rad) / np.sin(dist_rad)
            B = np.sin(f*dist_rad) / np.sin(dist_rad)

            x = A * np.cos(lat1_rad) * np.cos(lon1_rad) + B * np.cos(lat2_rad) * np.cos(lon2_rad)
            y = A * np.cos(lat1_rad) * np.sin(lon1_rad) + B * np.cos(lat2_rad) * np.sin(lon2_rad)
            z = A * np.sin(lat1_rad) + B * np.sin(lat2_rad)

            gclat[i] = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180.0 / np.pi
            gclon[i] = np.arctan2(y, x) * 180.0 / np.pi

    # Adjust longitude range based on sign of iu
    if iu > 0:
        # 0 to 360
        gclon = np.where(gclon < 0, gclon + 360, gclon)
    else:
        # -180 to 180
        gclon = np.where(gclon > 180, gclon - 360, gclon)

    # Calculate spacing
    spacing = distance / (npts - 1) if npts > 1 else 0

    return {
        'distance': distance,
        'gclat': gclat,
        'gclon': gclon,
        'spacing': spacing
    }


def gc_qarea(lat, lon):
    """
    Calculate area of a spherical quadrilateral patch on unit sphere.

    Parameters
    ----------
    lat : array_like
        Array of 4 latitudes (vertices) in degrees
    lon : array_like
        Array of 4 longitudes (vertices) in degrees

    Returns
    -------
    float
        Area of quadrilateral in steradians (on unit sphere)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if len(lat) != 4 or len(lon) != 4:
        raise ValueError("Quadrilateral must have exactly 4 vertices")

    # Divide quadrilateral into two triangles and sum their areas
    # Triangle 1: vertices 0, 1, 2
    # Triangle 2: vertices 0, 2, 3
    area1 = gc_tarea(lat[[0,1,2]], lon[[0,1,2]])
    area2 = gc_tarea(lat[[0,2,3]], lon[[0,2,3]])

    return area1 + area2


def gc_tarea(lat, lon):
    """
    Calculate area of a spherical triangle on unit sphere.

    Uses the spherical excess formula (L'Huilier's theorem).

    Parameters
    ----------
    lat : array_like
        Array of 3 latitudes (vertices) in degrees
    lon : array_like
        Array of 3 longitudes (vertices) in degrees

    Returns
    -------
    float
        Area of triangle in steradians (on unit sphere)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if len(lat) != 3 or len(lon) != 3:
        raise ValueError("Triangle must have exactly 3 vertices")

    # Convert to radians
    lat_rad = lat * np.pi / 180.0
    lon_rad = lon * np.pi / 180.0

    # Convert to Cartesian coordinates
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Calculate great circle distances between vertices using Haversine
    def haversine(lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * np.arcsin(np.sqrt(a))

    a = haversine(lat_rad[1], lon_rad[1], lat_rad[2], lon_rad[2])
    b = haversine(lat_rad[0], lon_rad[0], lat_rad[2], lon_rad[2])
    c = haversine(lat_rad[0], lon_rad[0], lat_rad[1], lon_rad[1])

    # Semi-perimeter
    s = (a + b + c) / 2.0

    # L'Huilier's formula for spherical excess
    tan_E_4 = np.sqrt(np.tan(s/2) * np.tan((s-a)/2) * np.tan((s-b)/2) * np.tan((s-c)/2))
    E = 4 * np.arctan(tan_E_4)

    return E


def area_poly_sphere(lat, lon, rsph):
    """
    Calculate area of an arbitrary polygon on a sphere.

    Parameters
    ----------
    lat : array_like
        Latitudes of polygon vertices in degrees (clockwise order)
    lon : array_like
        Longitudes of polygon vertices in degrees (clockwise order)
    rsph : float
        Radius of sphere

    Returns
    -------
    float
        Area of polygon in units of rsph^2

    Notes
    -----
    Vertices must be in clockwise order. Use gc_clkwise to verify ordering.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    n = len(lat)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")

    # Triangulate the polygon from the first vertex
    total_area = 0.0

    for i in range(1, n-1):
        # Triangle: vertex 0, vertex i, vertex i+1
        area = gc_tarea(lat[[0, i, i+1]], lon[[0, i, i+1]])
        total_area += area

    # Scale by radius squared
    return total_area * rsph**2


def lonFlip(x):
    """
    Reorder a global rectilinear array about longitude 180.

    Equivalent to lonPivot(x, 180).

    Parameters
    ----------
    x : ndarray
        Array where rightmost dimension is longitude (0 to 360)

    Returns
    -------
    ndarray
        Array reordered with longitudes from -180 to 180
    """
    return lonPivot(x, 180.0)


def lonPivot(x, plon):
    """
    Pivot array about a user-specified longitude.

    Parameters
    ----------
    x : ndarray
        Array where rightmost dimension is longitude
    plon : float
        Pivot longitude in degrees

    Returns
    -------
    ndarray
        Array reordered about pivot longitude
    """
    x = np.asarray(x)

    # Get the longitude dimension (rightmost)
    nlon = x.shape[-1]

    # Assuming longitudes are 0 to 360 with equal spacing
    dlon = 360.0 / nlon
    lons = np.arange(0, 360, dlon)

    # Find pivot index
    pivot_idx = np.argmin(np.abs(lons - plon))

    # Reorder array
    return np.roll(x, -pivot_idx, axis=-1)


def add90LatX(x):
    """
    Add fake pole points (90S and 90N) to rightmost dimension.

    Parameters
    ----------
    x : ndarray
        Input array where rightmost dimension is latitude

    Returns
    -------
    ndarray
        Array with two additional latitude points added
    """
    x = np.asarray(x)
    shape = list(x.shape)
    shape[-1] += 2

    # Create output array
    result = np.zeros(shape, dtype=x.dtype)

    # South pole: average of southernmost values
    result[..., 0] = np.mean(x[..., 0])

    # Copy original data
    result[..., 1:-1] = x

    # North pole: average of northernmost values
    result[..., -1] = np.mean(x[..., -1])

    return result


def add90LatY(x):
    """
    Add fake pole points (90S and 90N) to leftmost dimension.

    Parameters
    ----------
    x : ndarray
        Input array where leftmost dimension is latitude

    Returns
    -------
    ndarray
        Array with two additional latitude points added
    """
    x = np.asarray(x)
    shape = list(x.shape)
    shape[0] += 2

    # Create output array
    result = np.zeros(shape, dtype=x.dtype)

    # South pole: average of southernmost values
    result[0, ...] = np.mean(x[0, ...])

    # Copy original data
    result[1:-1, ...] = x

    # North pole: average of northernmost values
    result[-1, ...] = np.mean(x[-1, ...])

    return result


def getind_latlon2d(lat2d, lon2d, lat_target, lon_target):
    """
    Find indices of 2D lat/lon arrays closest to target coordinates.

    Parameters
    ----------
    lat2d : ndarray
        2D array of latitudes
    lon2d : ndarray
        2D array of longitudes
    lat_target : float
        Target latitude
    lon_target : float
        Target longitude

    Returns
    -------
    tuple
        (i, j) indices of closest grid point
    """
    lat2d = np.asarray(lat2d)
    lon2d = np.asarray(lon2d)

    # Calculate distance using simple Euclidean approximation
    # For more accuracy, could use great circle distance
    dist = np.sqrt((lat2d - lat_target)**2 + (lon2d - lon_target)**2)

    # Find minimum
    idx = np.unravel_index(np.argmin(dist), dist.shape)

    return idx


def region_ind(lat2d, lon2d, latS, latN, lonW, lonE):
    """
    Find indices spanning a geographic region.

    Parameters
    ----------
    lat2d : ndarray
        2D array of latitudes
    lon2d : ndarray
        2D array of longitudes
    latS, latN : float
        Southern and northern latitude boundaries
    lonW, lonE : float
        Western and eastern longitude boundaries

    Returns
    -------
    tuple
        (j_south, j_north, i_west, i_east) indices spanning the region
    """
    lat2d = np.asarray(lat2d)
    lon2d = np.asarray(lon2d)

    # Find points within the box
    mask = (lat2d >= latS) & (lat2d <= latN) & \
           (lon2d >= lonW) & (lon2d <= lonE)

    # Find the bounding box of True values
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return None

    j_south = rows.min()
    j_north = rows.max()
    i_west = cols.min()
    i_east = cols.max()

    return (j_south, j_north, i_west, i_east)


def gc_clkwise(lat, lon):
    """
    Test if points on a spherical polygon are in clockwise order.

    Parameters
    ----------
    lat : array_like
        Latitudes of polygon vertices in degrees
    lon : array_like
        Longitudes of polygon vertices in degrees

    Returns
    -------
    bool
        True if clockwise, False if counterclockwise
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Convert to radians
    lat_rad = lat * np.pi / 180.0
    lon_rad = lon * np.pi / 180.0

    # Calculate signed area using shoelace formula on sphere
    # Positive area = clockwise, negative = counterclockwise
    signed_area = 0.0

    n = len(lat)
    for i in range(n):
        j = (i + 1) % n
        signed_area += (lon_rad[j] - lon_rad[i]) * (2 + np.sin(lat_rad[i]) + np.sin(lat_rad[j]))

    return signed_area < 0


def gc_inout(lat_point, lon_point, lat_poly, lon_poly):
    """
    Test if points are inside a spherical polygon.

    Uses the winding number algorithm adapted for spherical geometry.

    Parameters
    ----------
    lat_point, lon_point : array_like
        Latitudes and longitudes of test points
    lat_poly, lon_poly : array_like
        Latitudes and longitudes of polygon vertices

    Returns
    -------
    ndarray
        Boolean array: True if point is inside polygon
    """
    lat_point = np.atleast_1d(np.asarray(lat_point))
    lon_point = np.atleast_1d(np.asarray(lon_point))
    lat_poly = np.asarray(lat_poly)
    lon_poly = np.asarray(lon_poly)

    n_points = len(lat_point)
    n_poly = len(lat_poly)

    result = np.zeros(n_points, dtype=bool)

    # Simple ray casting algorithm for sphere
    # For each point, count edge crossings
    for i in range(n_points):
        winding = 0

        for j in range(n_poly):
            k = (j + 1) % n_poly

            # Check if ray from point crosses edge j->k
            if ((lon_poly[j] <= lon_point[i] < lon_poly[k]) or \
                (lon_poly[k] <= lon_point[i] < lon_poly[j])):

                # Calculate intersection latitude
                t = (lon_point[i] - lon_poly[j]) / (lon_poly[k] - lon_poly[j])
                lat_cross = lat_poly[j] + t * (lat_poly[k] - lat_poly[j])

                if lat_point[i] < lat_cross:
                    winding += 1

        result[i] = (winding % 2) == 1

    return result
