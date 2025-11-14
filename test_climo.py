"""
Comprehensive test suite for climatology functions.
Tests correctness of calculations, dimension handling, and edge cases.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/NCL-tools')

from ncl_tools.climo import (
    clmMonTLL, clmMonTLLL, clmMonLLT, clmMonLLLT,
    calcMonAnomTLL, calcMonAnomTLLL, calcMonAnomLLT, calcMonAnomLLLT,
    clmDayTLL, clmDayTLLL, calcDayAnomTLL,
    month_to_season, month_to_season12, month_to_seasonN,
    rmAnnCycle1D, rmMonAnnCycTLL, rmMonAnnCycLLT, rmMonAnnCycLLLT,
    stdMonTLL, stdMonTLLL, stdMonLLT, stdMonLLLT,
    smthClmDayTLL, smthClmDayTLLL,
    clmMon2clmDay
)

print("="*70)
print("CLIMATOLOGY FUNCTIONS TEST SUITE")
print("="*70)

# Test 1: Monthly Climatology - TLL dimension ordering
print("\n" + "="*70)
print("TEST 1: Monthly Climatology (clmMonTLL)")
print("="*70)

# Create 3 years of monthly data with known seasonal cycle
nyears = 3
nmonths = nyears * 12
nlat, nlon = 5, 8

# Create data with predictable seasonal cycle
temp_TLL = np.zeros((nmonths, nlat, nlon))
for t in range(nmonths):
    month = t % 12
    # Each month has a specific value: month + 10
    temp_TLL[t, :, :] = month + 10

clim_TLL = clmMonTLL(temp_TLL)

print(f"Input shape: {temp_TLL.shape}")
print(f"Climatology shape: {clim_TLL.shape}")
print(f"Expected shape: (12, {nlat}, {nlon})")

# Check: Each month should have constant value across years
expected_values = np.arange(12) + 10
for month in range(12):
    actual = clim_TLL[month, 0, 0]
    expected = expected_values[month]
    print(f"Month {month+1}: Expected={expected:.1f}, Actual={actual:.1f}, Match={np.isclose(actual, expected)}")

assert clim_TLL.shape == (12, nlat, nlon), "Wrong output shape!"
assert np.allclose(clim_TLL[:, 0, 0], expected_values), "Climatology calculation error!"
print("✓ TEST 1 PASSED")

# Test 2: Monthly Climatology - Different dimension orderings
print("\n" + "="*70)
print("TEST 2: Monthly Climatology - All Dimension Orderings")
print("="*70)

# Create data in different dimension orders
temp_TLLL = np.zeros((nmonths, 3, nlat, nlon))
temp_LLT = np.zeros((nlat, nlon, nmonths))
temp_LLLT = np.zeros((3, nlat, nlon, nmonths))

for t in range(nmonths):
    month = t % 12
    value = month + 10
    temp_TLLL[t, :, :, :] = value
    temp_LLT[:, :, t] = value
    temp_LLLT[:, :, :, t] = value

clim_TLLL = clmMonTLLL(temp_TLLL)
clim_LLT = clmMonLLT(temp_LLT)
clim_LLLT = clmMonLLLT(temp_LLLT)

print(f"clmMonTLLL: Input {temp_TLLL.shape} → Output {clim_TLLL.shape}")
print(f"clmMonLLT:  Input {temp_LLT.shape} → Output {clim_LLT.shape}")
print(f"clmMonLLLT: Input {temp_LLLT.shape} → Output {clim_LLLT.shape}")

# Verify all produce same climatological values
for month in range(12):
    val_TLLL = clim_TLLL[month, 0, 0, 0]
    val_LLT = clim_LLT[0, 0, month]
    val_LLLT = clim_LLLT[0, 0, 0, month]
    expected = month + 10
    print(f"Month {month+1}: TLLL={val_TLLL:.1f}, LLT={val_LLT:.1f}, LLLT={val_LLLT:.1f}, Expected={expected:.1f}")
    assert np.isclose(val_TLLL, expected) and np.isclose(val_LLT, expected) and np.isclose(val_LLLT, expected)

print("✓ TEST 2 PASSED")

# Test 3: Monthly Anomalies
print("\n" + "="*70)
print("TEST 3: Monthly Anomalies")
print("="*70)

# Create data with seasonal cycle + trend
temp_with_trend = np.zeros((nmonths, nlat, nlon))
for t in range(nmonths):
    month = t % 12
    year = t // 12
    temp_with_trend[t, :, :] = (month + 10) + (year * 0.5)  # Seasonal + trend

clim = clmMonTLL(temp_with_trend)
anom = calcMonAnomTLL(temp_with_trend, clim)

print(f"Data shape: {temp_with_trend.shape}")
print(f"Climatology shape: {clim.shape}")
print(f"Anomaly shape: {anom.shape}")

# Anomalies should remove seasonal cycle, leaving trend + small variations
# Mean anomaly should be close to 0
mean_anom = np.mean(anom)
print(f"Mean anomaly: {mean_anom:.6f} (should be ~0)")

# Check specific months
for month in range(12):
    month_indices = [month + 12*y for y in range(nyears)]
    month_data = temp_with_trend[month_indices, 0, 0]
    month_anom = anom[month_indices, 0, 0]

    # Anomalies should show the trend
    expected_anoms = [0.0, 0.5, 1.0]  # Year 0, 1, 2 trends
    actual_anoms = month_anom

    print(f"Month {month+1}: Anomalies = {actual_anoms}, Expected trend = {expected_anoms}")
    assert np.allclose(actual_anoms, expected_anoms, atol=1e-10)

print("✓ TEST 3 PASSED")

# Test 4: Seasonal Means
print("\n" + "="*70)
print("TEST 4: Seasonal Means")
print("="*70)

# Create 2 years of monthly data with clear seasonal cycle
nyears = 2
nmonths = nyears * 12
seasonal_data = np.zeros((nmonths, 10, 10))

for t in range(nmonths):
    month = t % 12
    # Winter (DJF): 0, Spring (MAM): 10, Summer (JJA): 20, Fall (SON): 30
    if month in [11, 0, 1]:  # DJF
        seasonal_data[t] = 0
    elif month in [2, 3, 4]:  # MAM
        seasonal_data[t] = 10
    elif month in [5, 6, 7]:  # JJA
        seasonal_data[t] = 20
    elif month in [8, 9, 10]:  # SON
        seasonal_data[t] = 30

djf = month_to_season(seasonal_data, "DJF")
mam = month_to_season(seasonal_data, "MAM")
jja = month_to_season(seasonal_data, "JJA")
son = month_to_season(seasonal_data, "SON")

print(f"Input shape: {seasonal_data.shape}")
print(f"DJF shape: {djf.shape} (should be [{nyears}, 10, 10])")
print(f"DJF mean: {djf[1, 0, 0]:.1f} (should be 0.0)")
print(f"MAM mean: {mam[0, 0, 0]:.1f} (should be 10.0)")
print(f"JJA mean: {jja[0, 0, 0]:.1f} (should be 20.0)")
print(f"SON mean: {son[0, 0, 0]:.1f} (should be 30.0)")

# Note: DJF first year is only JF (2 months), so it averages Dec(previous)=30, Jan=0, Feb=0
# Actually for first year, there's no previous Dec, so it's just Jan+Feb average = 0
assert np.isclose(djf[0, 0, 0], 0.0, atol=0.1), f"DJF year 0 should be ~0, got {djf[0, 0, 0]}"
assert np.isclose(mam[0, 0, 0], 10.0), "MAM should be 10"
assert np.isclose(jja[0, 0, 0], 20.0), "JJA should be 20"
assert np.isclose(son[0, 0, 0], 30.0), "SON should be 30"

print("✓ TEST 4 PASSED")

# Test 5: Daily Climatology
print("\n" + "="*70)
print("TEST 5: Daily Climatology")
print("="*70)

# Create 2 years of daily data
nyears = 2
ndays_per_year = 365
ndays = nyears * ndays_per_year

# Create yyyyddd array
yyyyddd = []
for year in range(2020, 2020 + nyears):
    for doy in range(1, ndays_per_year + 1):
        yyyyddd.append(year * 1000 + doy)
yyyyddd = np.array(yyyyddd)

# Create data where each day of year has predictable value
daily_data = np.zeros((ndays, 5, 5))
for t in range(ndays):
    doy = yyyyddd[t] % 1000
    daily_data[t] = doy  # Day 1 = 1, Day 2 = 2, etc.

daily_clim = clmDayTLL(daily_data, yyyyddd)

print(f"Input shape: {daily_data.shape}")
print(f"Daily climatology shape: {daily_clim.shape}")
print(f"Expected shape: (365, 5, 5)")

# Check that climatology matches day-of-year
for doy in [1, 100, 200, 365]:
    clim_val = daily_clim[doy-1, 0, 0]
    expected = doy
    print(f"Day {doy}: Climatology={clim_val:.1f}, Expected={expected:.1f}, Match={np.isclose(clim_val, expected)}")
    assert np.isclose(clim_val, expected)

print("✓ TEST 5 PASSED")

# Test 6: Remove Annual Cycle
print("\n" + "="*70)
print("TEST 6: Remove Annual Cycle")
print("="*70)

# Create 1D time series with strong seasonal cycle + weak trend
nmonths = 60  # 5 years
time = np.arange(nmonths)
seasonal_component = 10 * np.sin(2 * np.pi * time / 12)
trend_component = 0.05 * time
noise_component = np.random.randn(nmonths) * 0.5

ts_original = 15 + seasonal_component + trend_component + noise_component

# Remove annual cycle
ts_deseason = rmAnnCycle1D(ts_original)

print(f"Original time series:")
print(f"  Mean: {ts_original.mean():.2f}")
print(f"  Std: {ts_original.std():.2f}")
print(f"  Range: [{ts_original.min():.2f}, {ts_original.max():.2f}]")

print(f"\nDeseasonalized time series:")
print(f"  Mean: {ts_deseason.mean():.2f}")
print(f"  Std: {ts_deseason.std():.2f}")
print(f"  Range: [{ts_deseason.min():.2f}, {ts_deseason.max():.2f}]")

# After removing annual cycle:
# 1. Mean should be close to 0 (we removed the climatological mean)
# 2. Std should be much smaller (seasonal cycle removed)
# 3. Remaining signal should be trend + noise

original_std = np.std(ts_original)
deseason_std = np.std(ts_deseason)
reduction = (1 - deseason_std / original_std) * 100

print(f"\nVariance reduction: {reduction:.1f}%")
print(f"Mean of deseasonalized: {np.abs(ts_deseason.mean()):.6f} (should be close to 0)")

assert deseason_std < original_std, "Deseasonalized std should be smaller!"
assert np.abs(ts_deseason.mean()) < 1.0, "Mean should be close to 0 after removing cycle!"

print("✓ TEST 6 PASSED")

# Test 7: Standard Deviations
print("\n" + "="*70)
print("TEST 7: Monthly Standard Deviations")
print("="*70)

# Create data with varying interannual variability
nyears = 10
nmonths = nyears * 12
data_with_var = np.zeros((nmonths, 5, 5))

for t in range(nmonths):
    month = t % 12
    year = t // 12

    # Add different variability to each month
    # January has high variability, July has low variability
    if month == 0:  # January
        data_with_var[t] = 10 + np.random.randn(5, 5) * 5  # High std
    elif month == 6:  # July
        data_with_var[t] = 20 + np.random.randn(5, 5) * 1  # Low std
    else:
        data_with_var[t] = 15 + np.random.randn(5, 5) * 2

std_monthly = stdMonTLL(data_with_var)

print(f"Input shape: {data_with_var.shape}")
print(f"Monthly std shape: {std_monthly.shape}")

jan_std = std_monthly[0, 2, 2]
jul_std = std_monthly[6, 2, 2]

print(f"January std: {jan_std:.2f} (should be ~5)")
print(f"July std: {jul_std:.2f} (should be ~1)")

# January should have higher std than July
assert jan_std > jul_std, "January should have higher variability!"
assert jan_std > 3.0, f"January std too low: {jan_std}"
assert jul_std < 2.0, f"July std too high: {jul_std}"

print("✓ TEST 7 PASSED")

# Test 8: Smoothing Daily Climatology
print("\n" + "="*70)
print("TEST 8: Smooth Daily Climatology with FFT")
print("="*70)

# Create noisy daily climatology
ndays = 365
daily_clim_noisy = np.zeros((ndays, 3, 3))

for d in range(ndays):
    # Smooth annual cycle
    smooth_cycle = 10 * np.sin(2 * np.pi * d / 365)
    # Add high-frequency noise
    noise = np.random.randn(3, 3) * 2
    daily_clim_noisy[d] = 15 + smooth_cycle + noise

# Smooth with 2 harmonics (keep annual and semi-annual)
smoothed = smthClmDayTLL(daily_clim_noisy, nHarm=2)

print(f"Original climatology std: {np.std(daily_clim_noisy[:, 1, 1]):.2f}")
print(f"Smoothed climatology std: {np.std(smoothed[:, 1, 1]):.2f}")

# After smoothing, high-frequency noise should be reduced
# But mean should be preserved
mean_orig = np.mean(daily_clim_noisy[:, 1, 1])
mean_smooth = np.mean(smoothed[:, 1, 1])

print(f"Original mean: {mean_orig:.2f}")
print(f"Smoothed mean: {mean_smooth:.2f}")
print(f"Mean preservation: {np.isclose(mean_orig, mean_smooth, atol=0.5)}")

assert np.isclose(mean_orig, mean_smooth, atol=1.0), "Mean should be preserved!"

print("✓ TEST 8 PASSED")

# Test 9: Convert Monthly to Daily Climatology
print("\n" + "="*70)
print("TEST 9: Convert Monthly to Daily Climatology")
print("="*70)

# Create monthly climatology with clear pattern
monthly_clim = np.zeros((12, 5, 5))
for month in range(12):
    monthly_clim[month] = month + 1  # Jan=1, Feb=2, ..., Dec=12

# Convert to daily
daily_from_monthly = clmMon2clmDay(monthly_clim, nDay=365)

print(f"Monthly climatology shape: {monthly_clim.shape}")
print(f"Daily climatology shape: {daily_from_monthly.shape}")

# Check that values are interpolated reasonably
# Mid-January (day ~15) should be close to 1
# Mid-July (day ~196) should be close to 7

jan_value = daily_from_monthly[14, 0, 0]  # Day 15
jul_value = daily_from_monthly[195, 0, 0]  # Day 196

print(f"Mid-January value: {jan_value:.2f} (should be close to 1)")
print(f"Mid-July value: {jul_value:.2f} (should be close to 7)")

assert 0.5 < jan_value < 2.0, f"January value out of range: {jan_value}"
assert 6.0 < jul_value < 8.0, f"July value out of range: {jul_value}"

print("✓ TEST 9 PASSED")

# Test 10: Edge Cases and Error Handling
print("\n" + "="*70)
print("TEST 10: Edge Cases and Error Handling")
print("="*70)

# Test 10a: Time dimension not multiple of 12
print("Testing error handling for non-12-multiple time dimension...")
try:
    bad_data = np.random.randn(25, 5, 5)  # 25 months (not divisible by 12)
    clmMonTLL(bad_data)
    print("✗ Should have raised ValueError!")
    assert False
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# Test 10b: Wrong dimension count
print("\nTesting error handling for wrong dimensions...")
try:
    bad_data = np.random.randn(24, 5)  # Only 2D
    clmMonTLL(bad_data)
    print("✗ Should have raised ValueError!")
    assert False
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

# Test 10c: Invalid season name
print("\nTesting error handling for invalid season...")
try:
    data = np.random.randn(24, 5, 5)
    month_to_season(data, "XYZ")  # Invalid season
    print("✗ Should have raised ValueError!")
    assert False
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")

print("\n✓ TEST 10 PASSED")

# Final Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("All 10 test suites passed successfully!")
print("\nTests covered:")
print("  1. ✓ Monthly climatology (TLL)")
print("  2. ✓ All dimension orderings (TLL, TLLL, LLT, LLLT)")
print("  3. ✓ Monthly anomalies")
print("  4. ✓ Seasonal means")
print("  5. ✓ Daily climatology")
print("  6. ✓ Remove annual cycle")
print("  7. ✓ Monthly standard deviations")
print("  8. ✓ FFT smoothing")
print("  9. ✓ Monthly to daily conversion")
print(" 10. ✓ Edge cases and error handling")
print("\n" + "="*70)
print("CLIMATOLOGY LIBRARY VERIFIED CORRECT!")
print("="*70)
