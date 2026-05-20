# ============================================
# DAY 3 — Feature Engineering
# Date: 20 May 2026
# Author: Akshay
# Goal: Create new meaningful features
#       from raw sensor data
# ============================================
# How many columns after features?   13
# What is average accel_magnitude?   
# What is danger row magnitude_change?
# What is safe row magnitude_change?
# Why is moisture_change important?
import pandas as pd
import numpy as np

# ── Load cleaned data ─────────────────────────
df = pd.read_csv('data/raw/sensor_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("="*50)
print("   DAY 3 — FEATURE ENGINEERING")
print("="*50)

print(f"\nBefore: {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# ── Feature 1: Acceleration Magnitude ─────────
# Physics: total movement in 3D space
# √(x² + y² + z²)
df['accel_magnitude'] = np.sqrt(
    df['accel_x']**2 +
    df['accel_y']**2 +
    df['accel_z']**2
)

print("\n✅ Feature 1 created: accel_magnitude")
print(f"   Normal range : "
      f"{df['accel_magnitude'].mean():.0f} avg")
print(f"   Max value    : "
      f"{df['accel_magnitude'].max():.0f}")

# ── Feature 2: Rolling Averages ───────────────
# Smooth out sensor noise
# Look at last 5 readings
df['moisture_avg5'] = df['moisture'].rolling(
    window=5
).mean()

df['magnitude_avg5'] = df['accel_magnitude'].rolling(
    window=5
).mean()

print("\n✅ Feature 2 created: rolling averages")
print(f"   moisture_avg5 — smoothed moisture")
print(f"   magnitude_avg5 — smoothed magnitude")

# ── Feature 3: Rate of Change ─────────────────
# How fast are values changing?
# Positive = rising, Negative = falling
df['moisture_change'] = df['moisture'].diff()
df['magnitude_change'] = df['accel_magnitude'].diff()

print("\n✅ Feature 3 created: rate of change")
print(f"   moisture_change — how fast moisture rises")
print(f"   magnitude_change — how fast slope moves")

# ── Feature 4: Time Features ──────────────────
# What hour did this reading happen?
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

print("\n✅ Feature 4 created: time features")
print(f"   hour — hour of day (0-23)")
print(f"   minute — minute of hour")

# ── Fill empty values ─────────────────────────
# Rolling and diff create NaN for first few rows
df.bfill(inplace=True)

print("\n✅ Empty values filled")

# ── Save enhanced dataset ──────────────────────
df.to_csv('data/processed/sensor_features.csv',
           index=False)

# ── Print Summary ─────────────────────────────
print("\n" + "="*50)
print("   FEATURE ENGINEERING SUMMARY")
print("="*50)
print(f"\nBefore : 6 columns")
print(f"After  : {len(df.columns)} columns ✅")
print(f"\nNew columns added:")
new_cols = ['accel_magnitude', 'moisture_avg5',
            'magnitude_avg5', 'moisture_change',
            'magnitude_change', 'hour', 'minute']
for col in new_cols:
    print(f"   → {col}")

print(f"\nSample — Danger row features:")
danger = df[df['moisture'] > 3500].head(1)
print(f"   moisture        : "
      f"{danger['moisture'].values[0]:.0f}")
print(f"   accel_magnitude : "
      f"{danger['accel_magnitude'].values[0]:.0f}")
print(f"   moisture_change : "
      f"{danger['moisture_change'].values[0]:.0f}")

print(f"\nSample — Safe row features:")
safe = df[df['moisture'] <= 2800].head(1)
print(f"   moisture        : "
      f"{safe['moisture'].values[0]:.0f}")
print(f"   accel_magnitude : "
      f"{safe['accel_magnitude'].values[0]:.0f}")
print(f"   moisture_change : "
      f"{safe['moisture_change'].values[0]:.0f}")

print(f"\nFile saved: data/processed/sensor_features.csv")
print("="*50)
print("\n✅ Day 3 Complete — Features Created!")
print("ML now has more meaningful data to learn from")