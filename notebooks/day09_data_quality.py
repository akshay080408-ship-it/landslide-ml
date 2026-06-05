# ============================================
# DAY 9 — Data Quality Handling
# Date: 31 May 2026
# Author: Akshay
# Goal: Detect and fix all data quality
#       issues before ML training
#       Real ESP32 data will have these!
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*50)
print("   DAY 9 — DATA QUALITY HANDLING")
print("="*50)

# ── Load clean labeled dataset ────────────────
df_clean = pd.read_csv(
    'data/processed/sensor_labeled.csv'
)
df_clean['timestamp'] = pd.to_datetime(
    df_clean['timestamp']
)
print(f"Loaded: {len(df_clean)} rows, "
      f"{len(df_clean.columns)} columns")

# ── Create dirty version ──────────────────────
# Simulate real ESP32 data problems
df = df_clean.copy()
np.random.seed(42)

print("\n⚙️  Injecting real-world data problems...")

# Problem 1: WiFi dropout (missing values)
wifi_drops = np.random.choice(
    len(df), 40, replace=False
)
df.loc[wifi_drops[:20], 'moisture'] = np.nan
df.loc[wifi_drops[20:], 'accel_x'] = np.nan
print(f"✅ Injected WiFi drops: "
      f"40 missing values")

# Problem 2: Sensor malfunction (impossible values)
bad_sensors = np.random.choice(
    len(df), 15, replace=False
)
df.loc[bad_sensors[:8], 'moisture'] = 9999
df.loc[bad_sensors[8:], 'accel_x'] = -999
print(f"✅ Injected sensor malfunctions: "
      f"15 impossible values")

# Problem 3: Stuck sensor
stuck_start = 300
df.loc[stuck_start:stuck_start+12,
       'moisture'] = 2048
print(f"✅ Injected stuck sensor: "
      f"rows {stuck_start}-{stuck_start+12}")

# Problem 4: Physical disturbance spike
spike_idx = 500
df.loc[spike_idx, 'accel_x'] = 4090
df.loc[spike_idx, 'accel_y'] = 4085
df.loc[spike_idx, 'accel_z'] = 4080
print(f"✅ Injected disturbance spike: row {spike_idx}")

# Problem 5: Gradual sensor drift
drift_start = 700
for i in range(50):
    df.loc[drift_start+i, 'moisture'] = \
        df.loc[drift_start+i, 'moisture'] + i*10
print(f"✅ Injected sensor drift: "
      f"rows {drift_start}-{drift_start+50}")

print(f"\nTotal problems injected: 5 types")

# ════════════════════════════════════════════
# PART 1: MISSING VALUES DETECTION
# WHY: WiFi drops cause NaN which
#      crash ML model completely ❌
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 1: Missing Values Detection")
print("="*50)

missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()
missing_pct = total_missing / \
              (len(df) * len(df.columns)) * 100

print(f"\nMissing values per column:")
for col, count in missing_counts.items():
    if count > 0:
        pct = count/len(df)*100
        print(f"  {col:20s}: "
              f"{count:4d} ({pct:.1f}%)")

print(f"\nTotal missing: {total_missing} "
      f"({missing_pct:.2f}% of all data)")

if total_missing == 0:
    print("✅ No missing values!")
else:
    print("⚠️  Missing values found — fixing...")

    # Fix: Forward fill (use last known value)
    # WHY: In time series, last known sensor
    #      value is best estimate for next reading
    df['moisture'] = df['moisture'].ffill()
    df['accel_x'] = df['accel_x'].ffill()

    # Verify fix
    remaining = df.isnull().sum().sum()
    print(f"✅ After forward fill: "
          f"{remaining} missing values")

# ════════════════════════════════════════════
# PART 2: OUTLIER DETECTION AND REMOVAL
# WHY: Impossible values (9999, -999)
#      destroy ML model accuracy ❌
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 2: Outlier Detection")
print("="*50)

# ESP32 ADC valid range
VALID_MIN = 0
VALID_MAX = 4095

sensor_cols = ['accel_x', 'accel_y',
               'accel_z', 'moisture']

print(f"\nValid sensor range: "
      f"{VALID_MIN} to {VALID_MAX}")
print(f"\nOut-of-range values:")

total_outliers = 0
for col in sensor_cols:
    outliers = df[
        (df[col] < VALID_MIN) |
        (df[col] > VALID_MAX)
    ]
    if len(outliers) > 0:
        print(f"  {col:15s}: "
              f"{len(outliers)} outliers "
              f"(min:{outliers[col].min():.0f} "
              f"max:{outliers[col].max():.0f})")
        total_outliers += len(outliers)

print(f"\nTotal outliers: {total_outliers}")

# Fix: Clip to valid range
# WHY: Clipping preserves data points
#      but removes impossible extremes
print("\nFixing outliers by clipping...")
for col in sensor_cols:
    df[col] = df[col].clip(VALID_MIN, VALID_MAX)

# Verify
remaining_outliers = 0
for col in sensor_cols:
    bad = df[
        (df[col] < VALID_MIN) |
        (df[col] > VALID_MAX)
    ]
    remaining_outliers += len(bad)

print(f"✅ After clipping: "
      f"{remaining_outliers} outliers")

# Z-score method for statistical outliers
print(f"\nZ-score outlier detection:")
for col in sensor_cols:
    mean = df[col].mean()
    std = df[col].std()
    z_scores = np.abs((df[col] - mean) / std)
    statistical_outliers = (z_scores > 3).sum()
    if statistical_outliers > 0:
        print(f"  {col:15s}: "
              f"{statistical_outliers} "
              f"statistical outliers (z>3)")

# ════════════════════════════════════════════
# PART 3: STUCK SENSOR DETECTION
# WHY: Stuck sensor = no warning during
#      actual landslide = lives lost ❌
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 3: Stuck Sensor Detection")
print("="*50)

def detect_stuck_sensor(series,
                         window=10,
                         threshold=5.0):
    """
    Detects if sensor is stuck
    WHY: Rolling std near zero means
         sensor not changing = stuck!
    Returns: boolean Series
    True = stuck at that reading
    """
    rolling_std = series.rolling(
        window=window, min_periods=1
    ).std()
    return rolling_std < threshold

print(f"\nChecking for stuck sensors "
      f"(window=10, threshold=5.0):")

for col in sensor_cols:
    stuck_mask = detect_stuck_sensor(df[col])
    stuck_count = stuck_mask.sum()
    window_count = 10
    if stuck_count > window_count:
        print(f"  ⚠️  {col:15s}: "
              f"{stuck_count} readings "
              f"appear stuck!")

        # Find stuck periods
        stuck_periods = df[stuck_mask].index
        if len(stuck_periods) > 0:
            print(f"     First stuck at row: "
                  f"{stuck_periods[0]}")
    else:
        print(f"  ✅ {col:15s}: "
              f"No stuck sensor detected")

# ════════════════════════════════════════════
# PART 4: DATA SMOOTHING
# WHY: Raw sensor noise causes false alarms
#      Smoothing gives reliable readings ✅
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 4: Data Smoothing")
print("="*50)

# Exponential smoothing
# WHY: Recent readings get more weight
#      Better for real-time detection
alpha = 0.3  # smoothing factor
# α=0.3: 30% current, 70% history

df['moisture_exp'] = df['moisture'].ewm(
    alpha=alpha, adjust=False
).mean()

df['magnitude_exp'] = df[
    'accel_magnitude'
].ewm(alpha=alpha, adjust=False).mean()

print(f"Exponential smoothing applied (α={alpha})")
print(f"→ Recent readings: {alpha*100:.0f}% weight")
print(f"→ Historical data: {(1-alpha)*100:.0f}% weight")

# Compare noise levels
raw_std = df['moisture'].std()
smooth_std = df['moisture_exp'].std()
noise_reduction = (1 - smooth_std/raw_std)*100

print(f"\nNoise reduction:")
print(f"  Raw moisture std:      "
      f"{raw_std:.2f}")
print(f"  Smoothed moisture std: "
      f"{smooth_std:.2f}")
print(f"  Noise reduced by:      "
      f"{noise_reduction:.1f}% ✅")

# ════════════════════════════════════════════
# PART 5: DATA VALIDATION RULES
# WHY: Every ESP32 reading must pass
#      validation before reaching ML model
#      Prevents production crashes ✅
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 5: Data Validation Rules")
print("="*50)

def validate_reading(row):
    """
    Validates single sensor reading
    Returns: (is_valid, error_message)
    WHY: Gates bad data before ML model
    """
    errors = []

    # Rule 1: Moisture range
    if not (0 <= row['moisture'] <= 4095):
        errors.append(
            f"moisture out of range: "
            f"{row['moisture']}"
        )

    # Rule 2: Acceleration range
    for axis in ['accel_x', 'accel_y', 'accel_z']:
        if not (0 <= row[axis] <= 4095):
            errors.append(
                f"{axis} out of range: "
                f"{row[axis]}"
            )

    # Rule 3: Vibration binary
    if row['vibration'] not in [0, 1]:
        errors.append(
            f"vibration invalid: "
            f"{row['vibration']}"
        )

    # Rule 4: Magnitude physically possible
    if row['accel_magnitude'] <= 0:
        errors.append("magnitude must be positive")

    # Rule 5: No NaN values
    if row.isnull().any():
        errors.append("contains NaN values")

    is_valid = len(errors) == 0
    return is_valid, errors

# Validate all readings
print("\nValidating all 2000 readings...")
valid_count = 0
invalid_count = 0
error_types = {}

for idx, row in df.iterrows():
    is_valid, errors = validate_reading(row)
    if is_valid:
        valid_count += 1
    else:
        invalid_count += 1
        for err in errors:
            err_type = err.split(':')[0]
            error_types[err_type] = \
                error_types.get(err_type, 0) + 1

print(f"\nValidation Results:")
print(f"  ✅ Valid readings:   {valid_count}")
print(f"  ❌ Invalid readings: {invalid_count}")

if error_types:
    print(f"\nError breakdown:")
    for err, count in error_types.items():
        print(f"  {err}: {count}")
else:
    print(f"  All readings valid! ✅")

# ════════════════════════════════════════════
# PART 6: DATA QUALITY REPORT
# WHY: Documents data reliability for
#      IEEE paper and government review ✅
# ════════════════════════════════════════════
print("\n" + "="*50)
print("📌 PART 6: Data Quality Report")
print("="*50)

total_readings = len(df)
completeness = (1 - total_missing/
                (total_readings *
                 len(sensor_cols))) * 100
outlier_rate = total_outliers / \
               total_readings * 100

print(f"""
╔══════════════════════════════════════╗
║     LANDSENSE DATA QUALITY REPORT   ║
╠══════════════════════════════════════╣
║ Total readings      : {total_readings:6d}         ║
║ Missing values      : {total_missing:6d}         ║
║ Outliers detected   : {total_outliers:6d}         ║
║ Outlier rate        : {outlier_rate:6.2f}%        ║
║ Data completeness   : {completeness:6.2f}%        ║
║ Validation passed   : {valid_count:6d}         ║
║ Validation failed   : {invalid_count:6d}         ║
╠══════════════════════════════════════╣
║ FIXES APPLIED:                       ║
║ ✅ Forward fill for missing values   ║
║ ✅ Clipping for impossible values    ║
║ ✅ Exponential smoothing for noise   ║
╠══════════════════════════════════════╣
║ DATA QUALITY SCORE: {min(completeness,100):.1f}%           ║
╚══════════════════════════════════════╝
""")

# ════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════
print("Creating quality report graph...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    'Data Quality Analysis — Day 9\n'
    'Before and After Cleaning',
    fontsize=13, fontweight='bold'
)

# Show first 200 readings
sample_clean = df_clean.iloc[:200]
sample_dirty = df.iloc[:200]

# Graph 1: Raw vs cleaned moisture
axes[0,0].plot(
    sample_clean['moisture'],
    color='green', alpha=0.7,
    label='Clean data',
    linewidth=1
)
axes[0,0].plot(
    sample_dirty['moisture'],
    color='red', alpha=0.5,
    label='After fixes',
    linewidth=1
)
axes[0,0].set_title('Moisture: Clean vs Fixed')
axes[0,0].legend()
axes[0,0].set_ylabel('Moisture Value')

# Graph 2: Smoothing comparison
axes[0,1].plot(
    sample_dirty['moisture'],
    color='lightblue',
    alpha=0.7,
    label='Raw',
    linewidth=0.8
)
axes[0,1].plot(
    sample_dirty['moisture_exp'],
    color='darkblue',
    linewidth=2,
    label='Exponential smooth'
)
axes[0,1].plot(
    sample_dirty['moisture_avg5'],
    color='orange',
    linewidth=2,
    label='Rolling avg (Day 3)'
)
axes[0,1].set_title(
    'Smoothing Methods Comparison'
)
axes[0,1].legend()
axes[0,1].set_ylabel('Moisture Value')

# Graph 3: Quality metrics bar chart
metrics = ['Valid\nReadings',
           'Missing\nFixed',
           'Outliers\nFixed']
values = [valid_count/total_readings*100,
          (total_missing/
           total_readings*100),
          (total_outliers/
           total_readings*100)]
colors_bar = ['green', 'orange', 'red']

bars = axes[1,0].bar(
    metrics, values,
    color=colors_bar,
    edgecolor='black',
    alpha=0.8
)
for bar, val in zip(bars, values):
    axes[1,0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%',
        ha='center',
        fontweight='bold'
    )
axes[1,0].set_title('Data Quality Metrics')
axes[1,0].set_ylabel('Percentage (%)')

# Graph 4: Z-score distribution
z_scores = np.abs(
    (df['moisture'] -
     df['moisture'].mean()) /
    df['moisture'].std()
)
axes[1,1].hist(
    z_scores,
    bins=50,
    color='steelblue',
    edgecolor='black',
    alpha=0.7
)
axes[1,1].axvline(
    x=3, color='red',
    linestyle='--',
    label='Outlier threshold (z=3)'
)
axes[1,1].set_title(
    'Z-Score Distribution\n'
    '(outliers have z > 3)'
)
axes[1,1].set_xlabel('Z-Score')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(
    'plots/day09_data_quality.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✅ Graph saved: "
      "plots/day09_data_quality.png")

# ── Save clean dataset ────────────────────────
df.to_csv(
    'data/processed/sensor_clean.csv',
    index=False
)
print("✅ Clean dataset saved: "
      "data/processed/sensor_clean.csv")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*50)
print("   DAY 9 COMPLETE — DATA QUALITY")
print("="*50)
print("\n6 data quality concepts learned:")
print("  1. Missing values detection + ffill")
print("     → WiFi drops handled ✅")
print("  2. Outlier detection + clipping")
print("     → Impossible values removed ✅")
print("  3. Stuck sensor detection")
print("     → Rolling std method ✅")
print("  4. Data smoothing")
print("     → Exponential smoothing ✅")
print("  5. Validation rules")
print("     → Gates bad data from ML ✅")
print("  6. Quality report")
print("     → IEEE paper ready ✅")
print(f"\n📊 Final data quality score: "
      f"{min(completeness,100):.1f}%")
print("\n⚠️  PHASE 1 COMPLETE!")
