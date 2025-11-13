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

### Meteorology Functions

This package implements commonly used NCL meteorology calculation functions:

**Thermodynamic Functions:**
- `satvpr_water_bolton` - Saturation vapor pressure (Bolton's equation)
- `dewtemp_trh` - Dew point from temperature and relative humidity
- `relhum`, `relhum_ttd` - Relative humidity calculations
- `mixhum_ptd`, `mixhum_ptrh` - Mixing ratio calculations
- `temp_virtual` - Virtual temperature
- `pot_temp` - Potential temperature
- `static_stability` - Atmospheric static stability

**Pressure and Vertical Coordinates:**
- `dpres_plevel` - Pressure layer thickness
- `pres_sigma` - Pressure at sigma levels
- `omega_to_w`, `w_to_omega` - Convert vertical velocity units
- `hydro` - Geopotential height from hydrostatic equation
- `prcwater_dp` - Total column precipitable water

**Wind Functions:**
- `wind_speed` - Calculate wind speed from u, v components
- `wind_direction` - Calculate wind direction from u, v
- `wind_component` - Calculate u, v from speed and direction
- `uv2dv_cfd` - Divergence from wind components
- `uv2vr_cfd` - Relative vorticity from wind components

**Other Meteorological Functions:**
- `coriolis_param` - Coriolis parameter
- `wetbulb_stull` - Wet bulb temperature (Stull's method)

### Statistics Functions

This package implements commonly used NCL statistics functions:

**Dimensional Operations:**
- `dim_avg_n`, `dim_stddev_n`, `dim_variance_n` - Statistics over dimensions
- `dim_max_n`, `dim_min_n`, `dim_median_n`, `dim_sum_n` - Min/max/median/sum
- `dim_cumsum_n` - Cumulative sum
- `dim_rmvmean_n`, `dim_rmvmed_n` - Remove mean/median
- `dim_standardize_n` - Standardize/normalize data
- `dim_rmsd_n` - Root-mean-square-difference

**Correlation:**
- `escorc_n` - Pearson correlation at lag 0
- `escovc` - Sample cross-covariance
- `pattern_cor` - Pattern correlation for spatial fields

**Regression:**
- `regline` - Simple linear regression for 1D arrays
- `regCoef_n` - Linear regression coefficients for multi-dimensional arrays

**Trend Analysis:**
- `dtrend_n` - Remove least squares linear trend
- `dtrend_msg_n` - Remove trend (handles missing values)

**Statistical Tests:**
- `ttest` - Student's t-test
- `ftest` - F-test for variances
- `student_t` - Student's t-distribution probability

**Other Functions:**
- `equiv_sample_size` - Effective sample size accounting for autocorrelation
- `taylor_stats` - Taylor diagram statistics
- `bin_avg`, `bin_sum` - Binning operations

### EOF (Empirical Orthogonal Function) Functions

This package implements comprehensive EOF/PCA analysis tools:

**Core EOF Functions:**
- `eofunc` - Compute EOFs (Principal Component Analysis)
- `eofunc_n` - EOF computation with specified time dimension
- `eofunc_ts` - Calculate time series amplitudes (principal components)
- `eofunc_ts_n` - Time series with specified dimension
- `eofcov`, `eofcov_ts` - Covariance-based EOF analysis
- `eofcor`, `eofcor_ts` - Correlation-based EOF analysis

**Reconstruction:**
- `eof2data` - Reconstruct data from EOFs and time series
- `eof2data_n` - Reconstruction with specified dimension

**Rotation and Testing:**
- `eofunc_varimax` - Varimax rotation for localized patterns
- `eofunc_varimax_reorder` - Reorder rotated EOFs by variance
- `eofunc_north` - Test eigenvalue significance (North et al. 1982)

### Printing Functions

This package implements NCL's printing and output functions:

- `printVarSummary` - Display variable metadata (type, dimensions, size, attributes)
- `printMinMax` - Display minimum and maximum values
- `write_matrix` - Output formatted 2D arrays with Fortran-style format strings
- `show_ascii` - Display ASCII character table

### Climatology Functions

This package implements comprehensive climatology analysis functions:

**Monthly Climatology:**
- `clmMonTLL`, `clmMonTLLL`, `clmMonLLT`, `clmMonLLLT` - Compute monthly climatologies

**Daily Climatology:**
- `clmDayTLL`, `clmDayTLLL` - Compute daily climatologies
- `clmDayHourTLL`, `clmDayHourTLLL` - Compute day-hour climatologies
- `clmMon2clmDay` - Convert monthly to daily climatology

**Anomaly Calculations:**
- `calcMonAnomTLL`, `calcMonAnomTLLL`, `calcMonAnomLLT`, `calcMonAnomLLLT` - Monthly anomalies
- `calcDayAnomTLL` - Daily anomalies

**Seasonal Means:**
- `month_to_season` - Compute specific 3-month seasonal mean
- `month_to_season12` - Compute all 12 seasonal means
- `month_to_seasonN` - Compute user-specified list of seasonal means

**Annual Cycle Removal:**
- `rmAnnCycle1D` - Remove annual cycle from 1D time series
- `rmMonAnnCycTLL`, `rmMonAnnCycLLT`, `rmMonAnnCycLLLT` - Remove annual cycle (multiple dimensions)

**Standard Deviations:**
- `stdMonTLL`, `stdMonTLLL`, `stdMonLLT`, `stdMonLLLT` - Monthly standard deviations

**Smoothing:**
- `smthClmDayTLL`, `smthClmDayTLLL` - Smooth daily climatology using FFT

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

### Meteorology Examples

#### Thermodynamic Calculations

```python
import numpy as np
from ncl_tools.meteo import (dewtemp_trh, relhum_ttd, mixhum_ptrh,
                               temp_virtual, pot_temp, satvpr_water_bolton)

# Calculate dew point from temperature and RH
temp = 293.15  # 20°C in Kelvin
rh = 65.0      # 65% relative humidity
dewpoint = dewtemp_trh(temp, rh)
print(f"Dew point: {dewpoint - 273.15:.1f}°C")

# Calculate relative humidity from temperature and dew point
rh_calc = relhum_ttd(temp, dewpoint)
print(f"Relative humidity: {rh_calc:.1f}%")

# Calculate saturation vapor pressure using Bolton's equation
# iounit: [input temp units, output pressure units]
# 1=Kelvin, 1=Pa
es = satvpr_water_bolton(temp, [1, 1])
print(f"Saturation vapor pressure: {es:.1f} Pa")

# Calculate mixing ratio
pres_hpa = 1000.0  # Pressure in hPa
# iswit: 1=mixing ratio in kg/kg, -1=mixing ratio in g/kg
mixr = mixhum_ptrh(pres_hpa, temp, rh, iswit=1)
print(f"Mixing ratio: {mixr*1000:.2f} g/kg")

# Calculate virtual temperature
# iounit: [input temp units, input mixr units, output temp units]
# 1=Kelvin, 0=kg/kg, 1=Kelvin
tv = temp_virtual(temp, mixr, [1, 0, 1])
print(f"Virtual temperature: {tv - 273.15:.2f}°C")

# Calculate potential temperature
pres = 100000.0  # Pressure in Pa
# Parameters: (pressure, temperature, dim, opt)
theta = pot_temp(pres, temp, -1, False)
print(f"Potential temperature: {theta - 273.15:.2f}°C")
```

#### Wind Calculations

```python
from ncl_tools.meteo import wind_speed, wind_direction, wind_component

# Calculate wind speed and direction from components
u = 5.0   # m/s (eastward)
v = 8.66  # m/s (northward)

wspd = wind_speed(u, v)
# opt: 0=return 0 for calm, 1=return NaN for calm
wdir = wind_direction(u, v, opt=0)
print(f"Wind speed: {wspd:.1f} m/s")
print(f"Wind direction: {wdir:.0f}° (from the {['N','NE','E','SE','S','SW','W','NW'][int(wdir/45)]})")

# Calculate components from speed and direction
# opt parameter currently unused, set to 0
u_calc, v_calc = wind_component(wspd, wdir, opt=0)
print(f"U component: {u_calc:.2f} m/s")
print(f"V component: {v_calc:.2f} m/s")
```

#### Vertical Velocity and Pressure

```python
from ncl_tools.meteo import omega_to_w, w_to_omega, dpres_plevel, pres_sigma

# Convert omega to w
omega = -0.5  # Pa/s (negative = upward motion)
pres = 50000.0  # 500 hPa in Pa
temp = 253.15   # -20°C in K
w = omega_to_w(omega, pres, temp)
print(f"Vertical velocity: {w*100:.2f} cm/s")

# Calculate pressure layer thickness
# Parameters: (plev, psfc, ptop, iopt)
plev = np.array([100000, 92500, 85000, 70000, 50000, 30000])  # Pa
psfc = 101325.0  # Surface pressure in Pa
ptop = 10000.0   # Top pressure in Pa
dp = dpres_plevel(plev, psfc, ptop, 0)
print(f"Pressure thickness: {dp}")

# Calculate pressure at sigma levels
# Parameters: (sigma, ps) - note order!
sigma = np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.1])
psfc_2d = np.ones((10, 20)) * 101325.0  # Surface pressure field (lat, lon)
p_sigma = pres_sigma(sigma, psfc_2d)
print(f"Shape: {p_sigma.shape}")  # (6, 10, 20)
print(f"Surface level pressure: {p_sigma[0, 0, 0]/100:.0f} hPa")
```

#### Geopotential Height and Precipitable Water

```python
from ncl_tools.meteo import hydro, prcwater_dp, dpres_plevel

# Calculate geopotential height
# Parameters: (p in mb, tkv in K, zsfc in gpm) - bottom to top order!
p_mb = np.array([1000, 925, 850, 700, 500])  # mb (bottom to top)
tkv = np.array([288, 282, 276, 268, 253])    # Virtual temperature in K
zsfc = 0.0  # Surface height in gpm

z = hydro(p_mb, tkv, zsfc)
print(f"Geopotential heights: {z} gpm")

# Calculate precipitable water
# Need pressure layer thickness
q = np.array([0.012, 0.008, 0.005, 0.002, 0.0005])  # kg/kg
p_pa = p_mb * 100  # Convert to Pa
psfc = p_pa[0]
ptop = p_pa[-1] / 2
dp = dpres_plevel(p_pa, psfc, ptop, 0)
pw = prcwater_dp(q, dp)
print(f"Precipitable water: {pw:.2f} kg/m²")
```

#### Coriolis Parameter and Vorticity

```python
from ncl_tools.meteo import coriolis_param, uv2vr_cfd, uv2dv_cfd

# Calculate Coriolis parameter
latitudes = np.array([0, 30, 45, 60, 90])
f = coriolis_param(latitudes)
print(f"Coriolis parameter at {latitudes}°N: {f*1e5:.2f} x 10^-5 s^-1")

# Calculate vorticity and divergence from wind components
# Example: 2D wind field
nlat, nlon = 50, 100
lats = np.linspace(-90, 90, nlat)
lons = np.linspace(0, 360, nlon, endpoint=False)
u = np.random.randn(nlat, nlon) * 10
v = np.random.randn(nlat, nlon) * 10

# boundOpt: 0=boundaries set to NaN, 1=cyclic in lon,
#           2=one-sided diffs, 3=cyclic lon + one-sided lat
vorticity = uv2vr_cfd(u, v, lats, lons, boundOpt=1)
divergence = uv2dv_cfd(u, v, lats, lons, boundOpt=1)
print(f"Vorticity range: {np.nanmin(vorticity):.2e} to {np.nanmax(vorticity):.2e} s^-1")
print(f"Divergence range: {np.nanmin(divergence):.2e} to {np.nanmax(divergence):.2e} s^-1")
```

#### Wet Bulb Temperature

```python
from ncl_tools.meteo import wetbulb_stull

# Calculate wet bulb temperature using Stull's method
temp_c = np.array([20, 25, 30, 35])  # Celsius
rh = np.array([50, 60, 70, 80])      # %

# iounit: [input temp units, output temp units]
# 0=Celsius, 1=Kelvin, 2=Fahrenheit
tw = wetbulb_stull(temp_c, rh, [0, 0], False)
print("Temp(°C)  RH(%)  Wet Bulb(°C)")
for t, r, w in zip(temp_c, rh, tw):
    print(f"  {t:4.0f}     {r:3.0f}      {w:5.1f}")
```

### Statistics Examples

#### Dimensional Statistics

```python
import numpy as np
from ncl_tools.statistics import dim_avg_n, dim_stddev_n, dim_variance_n, dim_max_n

# Create sample 3D data (time, lat, lon)
data = np.random.randn(100, 50, 100)  # 100 time steps, 50 lats, 100 lons

# Calculate time average at each grid point
time_avg = dim_avg_n(data, 0)  # Average over dimension 0 (time)
print(f"Time average shape: {time_avg.shape}")  # (50, 100)

# Calculate standard deviation over time
time_std = dim_stddev_n(data, 0)
print(f"Time std shape: {time_std.shape}")  # (50, 100)

# Calculate variance over spatial dimensions
spatial_var = dim_variance_n(data, [1, 2])  # Over dims 1 and 2 (lat, lon)
print(f"Spatial variance shape: {spatial_var.shape}")  # (100,)

# Find maximum along time dimension
max_vals = dim_max_n(data, 0)
print(f"Maximum values shape: {max_vals.shape}")  # (50, 100)
```

#### Remove Mean and Standardize

```python
from ncl_tools.statistics import dim_rmvmean_n, dim_standardize_n

# Remove temporal mean at each grid point
anomalies = dim_rmvmean_n(data, 0)
print(f"Mean of anomalies: {np.mean(anomalies):.10f}")  # Close to 0

# Standardize data (remove mean and divide by std)
standardized = dim_standardize_n(data, 1, 0)  # opt=1 uses population std
print(f"Mean: {np.mean(dim_avg_n(standardized, 0)):.10f}")  # ~0
print(f"Std: {np.mean(dim_stddev_n(standardized, 0)):.6f}")  # ~1
```

#### Correlation Analysis

```python
from ncl_tools.statistics import escorc_n, pattern_cor

# Two time series at each grid point
sst = np.random.randn(120, 50, 100)  # SST data
precip = sst + np.random.randn(120, 50, 100) * 0.5  # Correlated precipitation

# Calculate temporal correlation at each grid point
# dims_x=0, dims_y=0 means correlate along dimension 0 (time)
correlation = escorc_n(sst, precip, 0, 0)
print(f"Correlation shape: {correlation.shape}")  # (50, 100)
print(f"Mean correlation: {np.mean(correlation):.3f}")

# Calculate pattern correlation between two spatial fields
field1 = sst[0]  # First time step
field2 = sst[50]  # Middle time step
weights = None  # No weighting
pattern_corr = pattern_cor(field1, field2, weights, opt=0)
print(f"Pattern correlation: {pattern_corr:.3f}")
```

#### Linear Regression

```python
from ncl_tools.statistics import regline, regCoef_n

# Simple 1D regression
years = np.arange(1950, 2024)
temp = 14.5 + 0.02 * (years - 1950) + np.random.randn(74) * 0.3

result = regline(years, temp)
print(f"Trend: {result.value:.4f} °C/year")
print(f"Y-intercept: {result.yintercept:.2f} °C")
print(f"T-statistic: {result.tval:.2f}")
print(f"Standard error: {result.rstd:.6f}")

# Multi-dimensional regression: trend at each grid point
time = np.arange(100)
data_with_trend = 0.01 * time[:, None, None] + np.random.randn(100, 50, 100)

# Calculate trend at each grid point
# dims_x=0, dims_y=0 means regress along dimension 0
trends = regCoef_n(time, data_with_trend, 0, 0)
print(f"Trends shape: {trends.shape}")  # (50, 100)
print(f"Mean trend: {np.mean(trends):.5f}")  # Should be close to 0.01
print(f"T-statistics: {trends.tval.shape}")  # (50, 100)
```

#### Detrend Data

```python
from ncl_tools.statistics import dtrend_n

# Remove linear trend from time series
data_detrended = dtrend_n(data, return_info=True, dim=0)
print(f"Detrended shape: {data_detrended.shape}")  # Same as input
print(f"Slope shape: {data_detrended.slope.shape}")  # (50, 100)
print(f"Mean after detrending: {np.mean(dim_avg_n(data_detrended, 0)):.10f}")

# For data with missing values, use dtrend_msg_n
data_with_nan = data.copy()
data_with_nan[data_with_nan > 2] = np.nan  # Add some missing values

detrended_nan = dtrend_msg_n(None, data_with_nan, True, False, 0)
# x, y, remove_mean, return_info, dim
print(f"Detrended with NaN shape: {detrended_nan.shape}")
```

#### Statistical Tests

```python
from ncl_tools.statistics import ttest, ftest, student_t

# T-test: Compare two sample means
sample1_mean = 10.5
sample1_var = 2.3
sample1_s = 30  # Number of statistically independent observations

sample2_mean = 11.2
sample2_var = 2.8
sample2_s = 35

# Assuming equal variances (iflag=False)
prob = ttest(sample1_mean, sample1_var, sample1_s,
             sample2_mean, sample2_var, sample2_s,
             iflag=False, tval_opt=False)
print(f"T-test probability: {prob:.4f}")
if prob < 0.05:
    print("Difference is statistically significant at 95% level")

# F-test: Compare two sample variances
f_prob = ftest(sample1_var, sample1_n, sample2_var, sample2_n, opt=0)
print(f"F-test probability: {f_prob:.4f}")

# Calculate probability for given t-value
t_value = 2.5
df = 28
p_value = student_t(t_value, df)
print(f"Two-tailed p-value for t={t_value}: {p_value:.4f}")
```

#### Taylor Diagram Statistics

```python
from ncl_tools.statistics import taylor_stats

# Compare model output to observations
observations = np.random.randn(1000)
model_output = 0.8 * observations + np.random.randn(1000) * 0.5

stats = taylor_stats(observations, model_output, opt=0)
print(f"Correlation: {stats[0]:.3f}")
print(f"Std ratio (model/obs): {stats[1]:.3f}")
print(f"Centered RMSD (normalized): {stats[2]:.3f}")
```

#### Binning Operations

```python
from ncl_tools.statistics import bin_avg, bin_sum

# Create some data to bin
x = np.random.randn(1000) * 10 + 50  # Values around 50
y = x * 2 + np.random.randn(1000) * 5  # Correlated values

# Define bins
bins = np.linspace(20, 80, 13)  # 12 bins from 20 to 80

# Calculate bin averages
bin_centers, bin_avgs, bin_counts = bin_avg(x, y, bins)
print(f"Bin centers: {bin_centers}")
print(f"Bin averages: {bin_avgs}")
print(f"Bin counts: {bin_counts}")

# Calculate bin sums
bin_centers, bin_sums, bin_counts = bin_sum(x, y, bins)
print(f"Bin sums: {bin_sums}")
```

#### Effective Sample Size

```python
from ncl_tools.statistics import equiv_sample_size

# Time series with autocorrelation
time_series = np.random.randn(100, 50, 100)

# Estimate effective sample size accounting for temporal autocorrelation
n_eff = equiv_sample_size(time_series, siglvl=0.05, dims=0)
print(f"Effective sample size shape: {n_eff.shape}")  # (50, 100)
print(f"Mean effective n: {np.mean(n_eff):.1f}")
print(f"Actual sample size: {time_series.shape[0]}")
```

### EOF (Empirical Orthogonal Function) Analysis

This module provides comprehensive EOF/PCA analysis tools for climate and meteorological data.

#### Basic EOF Computation

```python
from ncl_tools.eofs import eofunc, eofunc_ts, eof2data
import numpy as np

# Create sample climate data: [lat, lon, time]
nlat, nlon, ntime = 50, 100, 120
data = np.random.randn(nlat, nlon, ntime)

# Add a pattern to make it more realistic
pattern = np.outer(np.sin(np.linspace(0, np.pi, nlat)),
                   np.cos(np.linspace(0, 2*np.pi, nlon)))
time_series = np.sin(np.linspace(0, 4*np.pi, ntime))
data += pattern[:, :, np.newaxis] * time_series[np.newaxis, np.newaxis, :]

# Compute first 3 EOFs
neval = 3
eofs = eofunc(data, neval, optEOF=0)  # 0 for covariance, 1 for correlation

print(f"EOF shape: {eofs.shape}")  # (50, 100, 3)
print(f"Eigenvalues: {eofs.eval}")
print(f"Percent variance: {eofs.pcvar}")
print(f"Matrix type: {eofs.matrix}")

# Calculate time series (principal components)
pcs = eofunc_ts(data, eofs, optETS=0)
print(f"PC shape: {pcs.shape}")  # (3, 120)

# Reconstruct data from EOFs
reconstructed = eof2data(eofs, pcs)

# Add back the mean if needed
if hasattr(pcs, 'ts_mean'):
    # ts_mean was subtracted during calculation
    for i in range(neval):
        reconstructed[..., i] += pcs.ts_mean[i]

print(f"Reconstructed shape: {reconstructed.shape}")  # (50, 100, 120)
```

#### EOF with Non-Standard Time Dimension

```python
from ncl_tools.eofs import eofunc_n, eofunc_ts_n, eof2data_n

# Data with time as first dimension: [time, lat, lon]
data_time_first = np.random.randn(120, 50, 100)

# Compute EOFs specifying time dimension
neval = 3
eofs = eofunc_n(data_time_first, neval, optEOF=0, time_dim=0)
print(f"EOF shape: {eofs.shape}")  # (3, 50, 100) - time dim replaced by neval

# Calculate time series
pcs = eofunc_ts_n(data_time_first, eofs, optETS=0, time_dim=0)
print(f"PC shape: {pcs.shape}")  # (3, 120)

# Reconstruct data
reconstructed = eof2data_n(eofs, pcs, time_dim=0)
print(f"Reconstructed shape: {reconstructed.shape}")  # (120, 50, 100)
```

#### Testing Eigenvalue Significance

```python
from ncl_tools.eofs import eofunc, eofunc_north

# Compute EOFs
data = np.random.randn(50, 100, 200)  # 200 time steps
neval = 5
eofs = eofunc(data, neval, optEOF=0)

# Test if eigenvalues are significantly separated (North et al. 1982)
N = 200  # Number of time steps
is_separated = eofunc_north(eofs.eval, N, prinfo=True)

print("\nSignificant modes:")
for i, sig in enumerate(is_separated):
    if sig:
        print(f"  Mode {i+1}: {eofs.pcvar[i]:.2f}% variance")
```

#### Covariance vs Correlation EOFs

```python
from ncl_tools.eofs import eofcov, eofcor, eofcov_ts, eofcor_ts

# Data with different scales (e.g., temperature and precipitation)
data = np.random.randn(50, 100, 120) * 10  # Large values

# Covariance-based EOF (sensitive to variable magnitude)
eofs_cov = eofcov(data, neval=3)
print(f"Covariance EOF trace: {eofs_cov.trace:.2f}")
print(f"Covariance pcvar: {eofs_cov.pcvar}")

# Correlation-based EOF (normalizes variables first)
eofs_cor = eofcor(data, neval=3)
print(f"Correlation EOF trace: {eofs_cor.trace:.2f}")
print(f"Correlation pcvar: {eofs_cor.pcvar}")

# Calculate corresponding time series
pcs_cov = eofcov_ts(data, eofs_cov)
pcs_cor = eofcor_ts(data, eofs_cor)

print(f"Covariance PC shape: {pcs_cov.shape}")
print(f"Correlation PC shape: {pcs_cor.shape}")
```

#### Varimax Rotation

```python
from ncl_tools.eofs import eofunc, eofunc_varimax, eofunc_varimax_reorder

# Compute standard EOFs
data = np.random.randn(50, 100, 120)
neval = 5
eofs = eofunc(data, neval, optEOF=0)

print("Original EOF variance:")
print(eofs.pcvar)

# Apply varimax rotation for more localized patterns
rotated_eofs = eofunc_varimax(eofs, optEVX=1)
print("\nRotated EOF variance (may not be in order):")
print(rotated_eofs.pcvar_varimax)

# Reorder by descending variance
reordered_eofs = eofunc_varimax_reorder(rotated_eofs)
print("\nReordered rotated EOF variance:")
print(reordered_eofs.pcvar_varimax)
```

#### Real-World Example: Sea Surface Temperature

```python
from ncl_tools.eofs import eofunc, eofunc_ts, eofunc_north
import numpy as np

# Simulate SST anomaly data: [lat, lon, months]
# Real data would come from netCDF files
nlat, nlon, nmonths = 89, 180, 600  # 50 years of monthly data

# Generate synthetic SST with ENSO-like pattern
lat = np.linspace(-88, 88, nlat)
lon = np.linspace(0, 359, nlon)

# Create ENSO pattern in tropical Pacific
enso_pattern = np.zeros((nlat, nlon))
tropical_mask = (lat > -20) & (lat < 20)
pacific_mask = (lon > 120) & (lon < 280)
for i in range(nlat):
    for j in range(nlon):
        if tropical_mask[i] and pacific_mask[j]:
            enso_pattern[i, j] = np.cos((lon[j] - 200) / 40) * \
                                 np.cos(lat[i] / 10)

# Generate time series with ENSO-like variability
enso_ts = np.sin(2 * np.pi * np.arange(nmonths) / 48)  # ~4 year cycle
noise = np.random.randn(nlat, nlon, nmonths) * 0.5

sst_anomalies = enso_pattern[:, :, np.newaxis] * enso_ts + noise

# Compute EOFs
neval = 3
eofs = eofunc(sst_anomalies, neval, optEOF=0)

print(f"SST EOF Analysis Results:")
print(f"  EOF 1: {eofs.pcvar[0]:.1f}% variance (likely ENSO)")
print(f"  EOF 2: {eofs.pcvar[1]:.1f}% variance")
print(f"  EOF 3: {eofs.pcvar[2]:.1f}% variance")

# Get principal components
pcs = eofunc_ts(sst_anomalies, eofs, optETS=0)

# Test significance
is_separated = eofunc_north(eofs.eval, nmonths, prinfo=False)
print(f"\nMode 1 significant: {is_separated[0]}")
print(f"Mode 2 significant: {is_separated[1]}")

# EOF 1 pattern would show ENSO-like structure in tropical Pacific
print(f"\nEOF 1 pattern shape: {eofs[..., 0].shape}")
print(f"PC 1 time series shape: {pcs[0].shape}")
```

### Printing and Output Functions

#### Display Variable Information

```python
from ncl_tools.printing import printVarSummary, printMinMax
import numpy as np

# Create sample climate data
temperature = np.random.randn(12, 50, 100) * 10 + 15  # 12 months, 50x100 grid

# Display variable summary (metadata only, no values)
printVarSummary(temperature, "temperature")

# Output:
# Variable: temperature
# Type: float64
# Total Size: 480000 bytes
#             60000 values
# Number of Dimensions: 3
# Dimensions and sizes:	[12] x [50] x [100]

# Display min/max values
printMinMax(temperature, opt=True, var_name="temperature")

# Output:
# temperature : min=-20.5   max=48.3
```

#### Formatted Matrix Output

```python
from ncl_tools.printing import write_matrix
import numpy as np

# Create a 2D array
data = np.random.randn(5, 7) * 100 + 500

# Output with F format (fixed-point): 7 columns, width 10, 2 decimals
write_matrix(data, "7f10.2", False)

# Output:
#     512.34    487.23    523.45    456.78    534.12    498.56    511.23
#     489.45    521.67    478.90    534.56    501.23    489.78    512.34
#     ...

# With title and row numbers
options = {
    'title': 'Temperature Data (K)',
    'tspace': 5,
    'row': True
}
write_matrix(data, "7f10.2", options)

# Output:
#      Temperature Data (K)
#     0     512.34    487.23    523.45    456.78    534.12    498.56    511.23
#     1     489.45    521.67    478.90    534.56    501.23    489.78    512.34
#     ...

# Different format examples
print("\nInteger format:")
int_data = np.random.randint(100, 1000, (3, 5))
write_matrix(int_data, "5i8", False)

print("\nExponential format:")
sci_data = np.random.randn(3, 4) * 1e-5
write_matrix(sci_data, "4e15.5", False)

print("\nMixed format:")
mixed_data = np.random.randn(4, 6)
write_matrix(mixed_data, "2i6,4f10.3", False)

# Write to file
write_matrix(data, "7f10.2", {'fout': 'output.txt', 'title': 'Data Output'})
print("Data written to output.txt")
```

#### Display ASCII Table

```python
from ncl_tools.printing import show_ascii

# Display complete ASCII character table
show_ascii()

# Output:
# ASCII Character Table
# ================================================================================
#   0: nul    1: soh    2: stx    3: etx    4: eot    5: enq    6: ack    7: bel
#   8:  bs    9:  ht   10:  nl   11:  vt   12:  np   13:  cr   14:  so   15:  si
#  16: dle   17: dc1   18: dc2   19: dc3   20: dc4   21: nak   22: syn   23: etb
#  24: can   25:  em   26: sub   27: esc   28:  fs   29:  gs   30:  rs   31:  us
#  32:  sp   33:   !   34:   "   35:   #   36:   $   37:   %   38:   &   39:   '
#  ...
# ================================================================================
```

#### Combined Usage Example

```python
from ncl_tools.printing import printVarSummary, printMinMax, write_matrix
import numpy as np

# Simulate monthly climate data
months = 12
lat_points = 20
lon_points = 30

# Temperature data
temp = np.random.randn(months, lat_points, lon_points) * 5 + 20

# Analyze the data
print("="*60)
print("CLIMATE DATA ANALYSIS")
print("="*60)

printVarSummary(temp, "monthly_temperature")
printMinMax(temp, opt=True, var_name="Temperature (°C)")

# Calculate monthly means
monthly_means = temp.mean(axis=(1, 2))

print("\nMonthly Mean Temperatures:")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Create output matrix with month index and mean
output_data = np.column_stack([np.arange(1, 13), monthly_means])
print("\nMonth  Mean Temp (°C)")
write_matrix(output_data, "i5,f10.2", False)

# Calculate spatial means for each month
spatial_stats = np.column_stack([
    np.arange(1, 13),
    temp.mean(axis=(1, 2)),
    temp.std(axis=(1, 2)),
    temp.min(axis=(1, 2)),
    temp.max(axis=(1, 2))
])

print("\n" + "="*60)
print("Monthly Statistics")
print("="*60)
print("Month    Mean     Std      Min      Max")
write_matrix(spatial_stats, "i5,4f9.2", False)
```

### Climatology Functions

#### Monthly Climatology and Anomalies

```python
from ncl_tools.climo import clmMonTLL, calcMonAnomTLL, stdMonTLL
import numpy as np

# Generate sample monthly data: 10 years of monthly data
nyears = 10
nmonths = nyears * 12
nlat, nlon = 50, 100

# Simulate temperature data [time, lat, lon]
# Add seasonal cycle + trend + noise
time = np.arange(nmonths)
seasonal_cycle = 10 * np.sin(2 * np.pi * time / 12)
trend = 0.02 * time
noise = np.random.randn(nmonths, nlat, nlon) * 2

temp_data = (15 + seasonal_cycle[:, np.newaxis, np.newaxis] +
             trend[:, np.newaxis, np.newaxis] + noise)

# Compute monthly climatology (12 months)
temp_clim = clmMonTLL(temp_data)
print(f"Climatology shape: {temp_clim.shape}")  # (12, 50, 100)
print(f"January mean: {temp_clim[0].mean():.2f}°C")
print(f"July mean: {temp_clim[6].mean():.2f}°C")

# Calculate anomalies (remove seasonal cycle)
temp_anom = calcMonAnomTLL(temp_data, temp_clim)
print(f"Anomaly shape: {temp_anom.shape}")  # (120, 50, 100)
print(f"Mean anomaly (should be ~0): {temp_anom.mean():.6f}")

# Calculate monthly standard deviations
temp_std = stdMonTLL(temp_data)
print(f"Std shape: {temp_std.shape}")  # (12, 50, 100)
print(f"January std: {temp_std[0].mean():.2f}°C")
```

#### Seasonal Means

```python
from ncl_tools.climo import month_to_season, month_to_season12
import numpy as np

# Monthly data
nmonths = 120  # 10 years
temp_monthly = np.random.randn(nmonths, 50, 100) * 10 + 15

# Compute DJF (Dec-Jan-Feb) seasonal mean
djf_mean = month_to_season(temp_monthly, "DJF")
print(f"DJF shape: {djf_mean.shape}")  # (10, 50, 100) - one per year

# Compute JJA (Jun-Jul-Aug) seasonal mean
jja_mean = month_to_season(temp_monthly, "JJA")
print(f"JJA shape: {jja_mean.shape}")  # (10, 50, 100)

# Compute all 12 seasonal means
all_seasons = month_to_season12(temp_monthly)
print(f"All seasons shape: {all_seasons.shape}")  # (12, 10, 50, 100)

# Available seasons
seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
           'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
print(f"\nSeasonal means for DJF: {djf_mean[:, 25, 50].mean():.2f}°C")
print(f"Seasonal means for JJA: {jja_mean[:, 25, 50].mean():.2f}°C")
```

#### Daily Climatology

```python
from ncl_tools.climo import clmDayTLL, calcDayAnomTLL, smthClmDayTLL
import numpy as np

# Generate daily data for multiple years
nyears = 5
ndays_per_year = 365
ndays_total = nyears * ndays_per_year
nlat, nlon = 30, 60

# Create yyyyddd array (year and day-of-year)
yyyyddd = []
for year in range(2015, 2015 + nyears):
    for doy in range(1, ndays_per_year + 1):
        yyyyddd.append(year * 1000 + doy)
yyyyddd = np.array(yyyyddd)

# Simulate daily temperature with annual cycle
doy_all = yyyyddd % 1000
seasonal_cycle = 15 * np.sin(2 * np.pi * (doy_all - 80) / 365)
daily_temp = (15 + seasonal_cycle[:, np.newaxis, np.newaxis] +
              np.random.randn(ndays_total, nlat, nlon) * 3)

# Compute daily climatology (365 days)
daily_clim = clmDayTLL(daily_temp, yyyyddd)
print(f"Daily climatology shape: {daily_clim.shape}")  # (365, 30, 60)

# Smooth the daily climatology using FFT (retain 2 harmonics)
smooth_clim = smthClmDayTLL(daily_clim, nHarm=2)
print(f"Smoothed climatology shape: {smooth_clim.shape}")  # (365, 30, 60)

# Calculate daily anomalies
daily_anom = calcDayAnomTLL(daily_temp, yyyyddd, daily_clim)
print(f"Daily anomaly shape: {daily_anom.shape}")  # (1825, 30, 60)
print(f"Mean anomaly: {daily_anom.mean():.6f}")

# Compare day 100 climatology: original vs smoothed
print(f"\nDay 100 original clim: {daily_clim[99, 15, 30]:.2f}°C")
print(f"Day 100 smoothed clim: {smooth_clim[99, 15, 30]:.2f}°C")
```

#### Remove Annual Cycle

```python
from ncl_tools.climo import rmAnnCycle1D, rmMonAnnCycTLL
import numpy as np

# 1D time series example
nmonths = 120  # 10 years
time = np.arange(nmonths)

# Create time series with annual cycle + trend + noise
seasonal = 10 * np.sin(2 * np.pi * time / 12)
trend = 0.03 * time
noise = np.random.randn(nmonths) * 2
ts = 15 + seasonal + trend + noise

# Remove annual cycle (leaves trend + noise)
ts_deseasonalized = rmAnnCycle1D(ts)

print("Original time series statistics:")
print(f"  Mean: {ts.mean():.2f}")
print(f"  Std: {ts.std():.2f}")
print(f"  Range: {ts.min():.2f} to {ts.max():.2f}")

print("\nDeseasonalized time series statistics:")
print(f"  Mean: {ts_deseasonalized.mean():.2f}")
print(f"  Std: {ts_deseasonalized.std():.2f}")
print(f"  Range: {ts_deseasonalized.min():.2f} to {ts_deseasonalized.max():.2f}")

# 3D spatial data example
temp_3d = np.random.randn(120, 20, 40) * 5 + 15
for t in range(120):
    month_idx = t % 12
    temp_3d[t] += 10 * np.sin(2 * np.pi * month_idx / 12)

# Remove annual cycle from 3D data
temp_3d_deseason = rmMonAnnCycTLL(temp_3d)
print(f"\n3D deseasonalized shape: {temp_3d_deseason.shape}")
print(f"3D deseasonalized mean: {temp_3d_deseason.mean():.6f}")
```

#### Convert Monthly to Daily Climatology

```python
from ncl_tools.climo import clmMonTLL, clmMon2clmDay
import numpy as np

# Monthly climatology
monthly_clim = np.random.randn(12, 30, 60) * 5 + 15

# Add realistic seasonal cycle to monthly climatology
for month in range(12):
    monthly_clim[month] += 10 * np.sin(2 * np.pi * (month - 2) / 12)

print(f"Monthly climatology shape: {monthly_clim.shape}")  # (12, 30, 60)

# Convert to daily climatology (365 days)
daily_from_monthly = clmMon2clmDay(monthly_clim, nDay=365)
print(f"Daily climatology shape: {daily_from_monthly.shape}")  # (365, 30, 60)

# Compare mid-month values
print("\nTemperature at grid point (15, 30):")
print(f"  January (month 0): {monthly_clim[0, 15, 30]:.2f}°C")
print(f"  Day 15 (mid-Jan): {daily_from_monthly[14, 15, 30]:.2f}°C")
print(f"  July (month 6): {monthly_clim[6, 15, 30]:.2f}°C")
print(f"  Day 196 (mid-Jul): {daily_from_monthly[195, 15, 30]:.2f}°C")
```

#### Complete Climate Analysis Workflow

```python
from ncl_tools.climo import (clmMonTLL, calcMonAnomTLL, stdMonTLL,
                              month_to_season, rmMonAnnCycTLL)
from ncl_tools.printing import printMinMax
import numpy as np

# Simulate 30 years of monthly SST data
nyears = 30
nmonths = nyears * 12
sst = np.random.randn(nmonths, 89, 180) * 2 + 20

# Add ENSO-like variability
enso_pattern = np.outer(np.ones(89), np.sin(np.linspace(0, 2*np.pi, 180)))
enso_ts = np.sin(2 * np.pi * np.arange(nmonths) / 48)  # 4-year cycle
for t in range(nmonths):
    sst[t] += enso_pattern * enso_ts[t] * 2

print("="*60)
print("CLIMATE ANALYSIS WORKFLOW")
print("="*60)

# 1. Compute monthly climatology
print("\n1. Computing monthly climatology...")
sst_clim = clmMonTLL(sst)
printMinMax(sst_clim[0], True, "January SST climatology")
printMinMax(sst_clim[6], False, "July SST climatology")

# 2. Compute anomalies
print("\n2. Computing monthly anomalies...")
sst_anom = calcMonAnomTLL(sst, sst_clim)
printMinMax(sst_anom[0], True, "SST anomalies (month 0)")

# 3. Compute monthly standard deviations
print("\n3. Computing monthly standard deviations...")
sst_std = stdMonTLL(sst)
print(f"January std: {sst_std[0].mean():.2f}°C")
print(f"July std: {sst_std[6].mean():.2f}°C")

# 4. Compute seasonal means
print("\n4. Computing seasonal means...")
djf = month_to_season(sst, "DJF")
jja = month_to_season(sst, "JJA")
print(f"DJF mean SST: {djf.mean():.2f}°C")
print(f"JJA mean SST: {jja.mean():.2f}°C")
print(f"DJF-JJA difference: {(djf.mean() - jja.mean()):.2f}°C")

# 5. Remove annual cycle
print("\n5. Removing annual cycle...")
sst_deseason = rmMonAnnCycTLL(sst)
print(f"Original SST range: {sst.min():.2f} to {sst.max():.2f}°C")
print(f"Deseasonalized range: {sst_deseason.min():.2f} to {sst_deseason.max():.2f}°C")
print(f"Variance reduction: {(1 - sst_deseason.var()/sst.var())*100:.1f}%")

print("="*60)
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
- [NCL Meteorology Functions](https://www.ncl.ucar.edu/Document/Functions/meteo.shtml)
- [NCL Statistics Functions](https://www.ncl.ucar.edu/Document/Functions/statistics.shtml)
- [NCL EOF Functions](https://www.ncl.ucar.edu/Document/Functions/eofs.shtml)
- [NCL Printing Functions](https://www.ncl.ucar.edu/Document/Functions/printing.shtml)
- [NCL Climatology Functions](https://www.ncl.ucar.edu/Document/Functions/climo.shtml)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite the original NCL documentation and relevant scientific papers referenced in the function docstrings.
