# NCL-tools

Python implementation of NCL (NCAR Command Language) functions using NumPy.

## Overview

This package provides pure Python implementations of various NCL functions, focusing on heat stress and meteorological calculations. All functions are implemented using only NumPy, making them fast and easy to integrate into existing scientific Python workflows.

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

## Usage Examples

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

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please cite the original NCL documentation and relevant scientific papers referenced in the function docstrings.
