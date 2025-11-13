# NCL-tools

Python implementation of NCL (NCAR Command Language) functions using NumPy.

## Overview

This package provides pure Python implementations of various NCL functions, focusing on heat stress, extreme value statistics, latitude/longitude operations, and meteorological calculations. All functions are implemented using NumPy and SciPy, making them fast and easy to integrate into existing scientific Python workflows.

## Installation

### From source

```bash
git clone https://github.com/fzhao70/NCL-tools.git
cd NCL-tools
pip install -e .
```

### Using pip (once published)

```bash
pip install ncl-tools
```

## Requirements

- Python >= 3.7
- NumPy >= 1.18.0
- SciPy >= 1.5.0

## Features

### Heat Stress Functions

This package currently implements all NCL heat stress functions:

- `fire_index_haines` - Haines fire index (Lower Atmosphere Severity Index)
- `heat_apptemp` - Apparent temperature
- `heat_discoi` - Simplified human discomfort index
- `heat_discoi_stull` - Human discomfort index using Stull wet bulb temperature
- `heat_esidx_moran` - Moran environmental stress index (ESI)
- `heat_humidex` - Canadian humidex
- `heat_index_nws` - NWS heat index
- `heat_swamp_cooleff` - Swamp cooler efficiency temperatures
- `heat_thic_thip` - Thermal humidity comfort and physiology indices
- `heat_wbgt_inout` - Wet-bulb globe temperature (indoor/outdoor)
- `heat_wbgt_simplified` - Simplified WBGT

### Extreme Value Functions

This package implements all NCL extreme value statistics functions:

- `extval_frechet` - Frechet Type II distribution (PDF and CDF)
- `extval_gev` - Generalized Extreme Value (GEV) distribution (PDF and CDF)
- `extval_gumbel` - Gumbel (Type I) distribution (PDF and CDF)
- `extval_weibull` - Weibull Type III distribution (PDF and CDF)
- `extval_pareto` - Pareto distributions (Generalized, Type I, Type II)
- `extval_mlegev` - Maximum Likelihood Estimation for GEV parameters
- `extval_mlegam` - Maximum Likelihood Estimation for Gamma parameters
- `extval_recurrence_table` - Recurrence intervals and probabilities
- `extval_return_period` - Return period calculations
- `extval_return_prob` - Return probability calculations

### Latitude/Longitude Functions

This package implements commonly used NCL latitude/longitude and spherical geometry functions:

**Grid Generation:**
- `gaus` - Compute Gaussian latitudes and weights
- `latGlobeF`, `lonGlobeF` - Generate global fixed grids with metadata
- `latGlobeFo`, `lonGlobeFo` - Generate global fixed offset grids
- `latRegWgt` - Generate area weights for regular grids
- `NormCosWgtGlobe` - Create normalized cosine weights

**Great Circle Functions:**
- `gc_latlon` - Great circle distance and interpolation
- `gc_qarea` - Area of spherical quadrilaterals
- `gc_tarea` - Area of spherical triangles
- `area_poly_sphere` - Area of arbitrary spherical polygons
- `gc_clkwise` - Test clockwise ordering of polygon vertices
- `gc_inout` - Test if points are inside spherical polygons

**Grid Utilities:**
- `lonFlip` - Reorder longitude arrays (0-360 to -180-180)
- `lonPivot` - Pivot arrays about specified longitude
- `add90LatX`, `add90LatY` - Add polar points to arrays
- `getind_latlon2d` - Find nearest grid point indices
- `region_ind` - Find indices spanning a geographic region

## Usage Examples

### Heat Stress Examples

### Haines Fire Index

```python
import numpy as np
from ncl_tools.heat_stress import fire_index_haines

# Atmospheric sounding data
p = np.array([1000, 950, 900, 850, 700, 500, 400, 300])  # hPa
t = np.array([25, 22, 19, 16, 8, -5, -15, -30])  # °C
td = np.array([20, 18, 15, 12, 2, -15, -25, -40])  # °C

# Calculate Haines index for all elevation regimes
haines = fire_index_haines(p, t, td)
print(f"Haines Index - Low: {haines[0]}, Mid: {haines[1]}, High: {haines[2]}")
```

### Apparent Temperature

```python
from ncl_tools.heat_stress import heat_apptemp

# Temperature in Celsius, vapor pressure in hPa, wind speed in m/s
t = 30.0  # °C
vp = 25.0  # hPa
wind = 3.5  # m/s
iounit = [0, 0, 0]  # All in Celsius and hPa

app_temp = heat_apptemp(t, vp, wind, iounit)
print(f"Apparent temperature: {app_temp:.1f}°C")
```

### NWS Heat Index

```python
from ncl_tools.heat_stress import heat_index_nws

# Temperature and relative humidity
t = 95.0  # °F
rh = 65.0  # %
iounit = [2, 2]  # Input/output in Fahrenheit

hi = heat_index_nws(t, rh, iounit)
print(f"Heat Index: {hi:.1f}°F")
```

### Humidex

```python
from ncl_tools.heat_stress import heat_humidex

# Temperature in Celsius, vapor pressure in hPa
t = 30.0  # °C
vp = 28.0  # hPa
iounit = [0, 0]  # Celsius and hPa

humidex = heat_humidex(t, vp, iounit)
print(f"Humidex: {humidex:.1f}")
```

### Environmental Stress Index

```python
from ncl_tools.heat_stress import heat_esidx_moran

# Temperature, relative humidity, solar radiation
t = 35.0  # °C
rh = 70.0  # %
srad = 800.0  # W/m²
iounit = [0, 0]  # Celsius

esi = heat_esidx_moran(t, rh, srad, iounit)
print(f"Environmental Stress Index: {esi:.1f}°C")
```

### WBGT Indoor/Outdoor

```python
from ncl_tools.heat_stress import heat_wbgt_inout

# Wet bulb, globe, and ambient temperatures
tw = 28.0  # °C
tg = 35.0  # °C
ta = 32.0  # °C
iounit = [0, 0]  # Celsius
opt = 2  # Outdoor formula

wbgt = heat_wbgt_inout(tw, tg, ta, iounit, opt)
print(f"WBGT (outdoor): {wbgt:.1f}°C")
```

### THIC and THIP Indices

```python
from ncl_tools.heat_stress import heat_thic_thip

# Temperature and wet bulb temperature
t = 30.25  # °C
twb = 25.5  # °C
iounit = [0]  # Celsius

thic, thip = heat_thic_thip(t, twb, iounit)
print(f"THIC: {thic:.2f}, THIP: {thip:.2f}")
```

### Swamp Cooler Efficiency

```python
from ncl_tools.heat_stress import heat_swamp_cooleff

# Dry bulb and wet bulb temperatures
t = 35.0  # °C
twb = 22.0  # °C
iounit = [0, 0]  # Celsius

temp_80, temp_65 = heat_swamp_cooleff(t, twb, iounit)
print(f"80% efficiency: {temp_80:.1f}°C")
print(f"65% efficiency: {temp_65:.1f}°C")
```

### Working with Arrays

All functions support NumPy arrays for batch processing:

```python
import numpy as np
from ncl_tools.heat_stress import heat_index_nws

# Multiple temperature and humidity values
temperatures = np.array([85, 90, 95, 100])  # °F
humidities = np.array([50, 60, 70, 80])  # %
iounit = [2, 2]  # Fahrenheit

heat_indices = heat_index_nws(temperatures, humidities, iounit)
for t, rh, hi in zip(temperatures, humidities, heat_indices):
    print(f"T={t}°F, RH={rh}% -> HI={hi:.1f}°F")
```

### Extreme Value Examples

#### GEV Distribution

```python
import numpy as np
from ncl_tools.extval import extval_gev

# Define x values and GEV parameters
x = np.linspace(-4, 6, 200)
shape = np.array([-0.5, 0.0, 0.5])  # Weibull, Gumbel, Frechet
scale = np.array([1.0, 1.0, 1.0])
center = np.array([0.0, 0.0, 0.0])

# Calculate PDF and CDF
pdf, cdf = extval_gev(x, shape, scale, center)

# pdf and cdf have shape (3, 200) - three distributions over 200 x values
print(f"PDF shape: {pdf.shape}")
print(f"CDF at x=1.0 for each distribution: {cdf[:, 100]}")
```

#### Gumbel Distribution

```python
from ncl_tools.extval import extval_gumbel

# Temperature data
x = np.linspace(-10, 20, 100)
scale = np.array([2.0, 3.0, 4.0])
center = np.array([0.5, 1.0, 1.5])

pdf, cdf = extval_gumbel(x, scale, center)
print(f"Gumbel PDF: {pdf[0, :5]}")  # First 5 values for first distribution
```

#### Frechet Distribution

```python
from ncl_tools.extval import extval_frechet

x = np.linspace(0.05, 5.5, 100)
shape = np.array([0.5, 1.0, 2.0])
scale = np.array([1.0, 1.0, 1.0])
center = np.array([0.0, 0.0, 0.0])

pdf, cdf = extval_frechet(x, shape, scale, center)
```

#### Weibull Distribution

```python
from ncl_tools.extval import extval_weibull

x = np.linspace(0, 5, 100)
shape = np.array([0.5, 1.0, 1.5, 5.0])
scale = np.array([1.0, 1.0, 1.0, 1.0])
center = np.array([0.0, 0.0, 0.0, 0.0])

pdf, cdf = extval_weibull(x, shape, scale, center)
```

#### Pareto Distribution

```python
from ncl_tools.extval import extval_pareto

x = np.linspace(0.05, 5.5, 100)
shape = np.array([0.5, 1.0, 2.0])
scale = np.array([1.0, 1.0, 1.0])
center = np.array([0.0, 0.0, 0.0])

# Generalized Pareto (ptype=0)
pdf, cdf = extval_pareto(x, shape, scale, center, ptype=0)

# Type I Pareto (ptype=1)
pdf1, cdf1 = extval_pareto(x, shape, scale, center, ptype=1)

# Type II Pareto/Lomax (ptype=2)
pdf2, cdf2 = extval_pareto(x, shape, scale, center, ptype=2)
```

#### Maximum Likelihood Estimation for GEV

```python
from ncl_tools.extval import extval_mlegev

# Flood data (annual maximum river discharge)
flood_data = np.array([487, 732, 784, 965, 1049, 1585, 612, 1156,
                       643, 770, 612, 531, 987, 743, 665])

# Estimate GEV parameters
params = extval_mlegev(flood_data, dims=0)

location, scale, shape, se_loc, se_scale, se_shape = params
print(f"Location: {location:.2f} (SE: {se_loc:.2f})")
print(f"Scale: {scale:.2f} (SE: {se_scale:.2f})")
print(f"Shape: {shape:.4f} (SE: {se_shape:.4f})")
```

#### Maximum Likelihood Estimation for Gamma

```python
from ncl_tools.extval import extval_mlegam

# Sample data
data = np.array([112, 118, 132, 129, 121, 135, 148, 136, 119, 104])

# Estimate Gamma parameters
params = extval_mlegam(data, dims=0)

location, scale, shape, variance, median = params
print(f"Location: {location:.2f}")
print(f"Scale: {scale:.2f}")
print(f"Shape: {shape:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Median: {median:.2f}")
```

#### Recurrence Table

```python
from ncl_tools.extval import extval_recurrence_table

# Annual maximum precipitation data
years = np.arange(1931, 1941)
precipitation = np.array([1306, 1345, 1032, 1580, 1293, 1497,
                         1469, 1392, 1037, 1633])

# Calculate recurrence intervals
table = extval_recurrence_table(years, precipitation, dims=0)

# Table columns: [time, x, cum_prob_rank, cum_prob,
#                 exc_prob_rank, exc_prob, recurrence_interval]
print("Year  Precip  Recurrence")
for i in range(len(table)):
    print(f"{table[i,0]:.0f}  {table[i,1]:.0f}      {table[i,6]:.2f}")
```

#### Return Period and Probability

```python
from ncl_tools.extval import extval_return_period, extval_return_prob

# Calculate return period
# For a 100-year flood with 95% confidence
recurrence_interval = 100  # years
probability = 0.95
period = extval_return_period(recurrence_interval, probability)
print(f"Return period: {period:.1f} years")

# Calculate return probability
# Probability of 10-year event occurring in next 8 years
avg_interval = 10  # years
exceedance_period = 8  # years
prob = extval_return_prob(avg_interval, exceedance_period)
print(f"Probability: {prob:.4f} or {prob*100:.2f}%")
```

### Latitude/Longitude Examples

#### Gaussian Latitudes and Weights

```python
import numpy as np
from ncl_tools.latlon import gaus

# Generate Gaussian latitudes for T42 resolution (64 latitudes)
nlat_per_hem = 32
gau_info = gaus(nlat_per_hem)

latitudes = gau_info[:, 0]  # Gaussian latitudes
weights = gau_info[:, 1]     # Gaussian weights

print(f"Number of latitudes: {len(latitudes)}")
print(f"First 5 latitudes: {latitudes[:5]}")
print(f"Weights sum to: {np.sum(weights):.6f}")  # Should be ~2.0
```

#### Generate Global Grids

```python
from ncl_tools.latlon import latGlobeF, lonGlobeF, latRegWgt

# Create a 2.5° x 2.5° global grid
nlat = 72
nlon = 144

lat_info = latGlobeF(nlat, "lat", "latitude", "degrees_north")
lon_info = lonGlobeF(nlon, "lon", "longitude", "degrees_east")

lats = lat_info['values']  # -90 to 90
lons = lon_info['values']  # 0 to 360

# Calculate area weights
weights = latRegWgt(lats)
print(f"Latitude weights sum to: {np.sum(weights):.6f}")  # Should be 2.0
```

#### Great Circle Distance and Interpolation

```python
from ncl_tools.latlon import gc_latlon

# Calculate distance from New York to London
lat1, lon1 = 40.7128, -74.0060  # New York
lat2, lon2 = 51.5074, -0.1278    # London

result = gc_latlon(lat1, lon1, lat2, lon2, npts=10, iu=4)

print(f"Great circle distance: {result['distance']:.2f} km")
print(f"Interpolated latitudes: {result['gclat']}")
print(f"Interpolated longitudes: {result['gclon']}")
```

#### Calculate Polygon Area on Sphere

```python
from ncl_tools.latlon import area_poly_sphere, gc_clkwise

# Define a polygon (e.g., a rectangular region)
lat_poly = np.array([30, 30, 40, 40])
lon_poly = np.array([-100, -90, -90, -100])

# Check if vertices are in clockwise order
is_clockwise = gc_clkwise(lat_poly, lon_poly)
print(f"Clockwise order: {is_clockwise}")

# Calculate area (Earth radius in km)
R_earth = 6371.0
area = area_poly_sphere(lat_poly, lon_poly, R_earth)
print(f"Polygon area: {area:.2f} km²")
```

#### Longitude Grid Manipulation

```python
from ncl_tools.latlon import lonFlip, lonPivot

# Sample data on 0-360 longitude grid
data = np.random.rand(72, 144)  # (lat, lon)

# Flip to -180 to 180 longitude
data_flipped = lonFlip(data)

# Pivot about 90°E
data_pivoted = lonPivot(data, 90.0)
```

#### Find Nearest Grid Point

```python
from ncl_tools.latlon import getind_latlon2d, region_ind

# 2D lat/lon arrays (e.g., from curvilinear grid)
lat2d = np.random.uniform(-90, 90, (100, 120))
lon2d = np.random.uniform(0, 360, (100, 120))

# Find nearest point to specific location
target_lat, target_lon = 35.0, -105.0
i, j = getind_latlon2d(lat2d, lon2d, target_lat, target_lon)
print(f"Nearest grid point at indices: ({i}, {j})")

# Find region spanning lat/lon box
indices = region_ind(lat2d, lon2d, latS=20, latN=50, lonW=-120, lonE=-80)
if indices:
    j_south, j_north, i_west, i_east = indices
    print(f"Region spans: j={j_south}:{j_north}, i={i_west}:{i_east}")
```

#### Test Point in Polygon

```python
from ncl_tools.latlon import gc_inout

# Define polygon (triangle around equator)
lat_poly = np.array([0, 10, 0])
lon_poly = np.array([0, 5, 10])

# Test points
test_lats = np.array([2, 5, 15])
test_lons = np.array([5, 5, 5])

inside = gc_inout(test_lats, test_lons, lat_poly, lon_poly)
print(f"Points inside polygon: {inside}")  # [True, True, False]
```

## Unit Specifications

Most functions use an `iounit` parameter to specify input/output units:

### Temperature units
- `0` - Celsius (°C)
- `1` - Kelvin (K)
- `2` - Fahrenheit (°F)

### Vapor pressure units
- `0` - Hectopascals/millibars (hPa/mb)
- `1` - Pascals (Pa)
- `2` - Kilopascals (kPa)

## References

All functions are based on NCL implementations and follow the same algorithms and formulas. For more details, see:
- [NCL Heat Stress Functions](https://www.ncl.ucar.edu/Document/Functions/heat_stress.shtml)
- [NCL Extreme Value Functions](https://www.ncl.ucar.edu/Document/Functions/extval.shtml)
- [NCL Latitude/Longitude Functions](https://www.ncl.ucar.edu/Document/Functions/latlon_funcs.shtml)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite the original NCL documentation and relevant scientific papers referenced in the function docstrings.
