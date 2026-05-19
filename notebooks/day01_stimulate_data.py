# DAY 1 — Simulate Landslide Sensor Data
# This creates fake ESP32 readings for ML training

import pandas as pd      # For creating data table
import numpy as np       # For random number generation

# Set random seed — makes sure we get
# same "random" numbers every time we run
# Important for reproducible results
np.random.seed(42)

# How many sensor readings to create
n = 2000  # 2000 readings = ~16 hours of data

print("Creating sensor dataset...")

# Create all sensor columns
data = {
    # timestamp: one reading every 30 seconds
    'timestamp': pd.date_range(
        start='2026-05-19',
        periods=n,
        freq='30s'
    ),

    # Accelerometer X axis
    # normal(centre_value, variation, count)
    'accel_x': np.random.normal(2048, 40, n),

    # Accelerometer Y axis
    'accel_y': np.random.normal(2048, 40, n),

    # Accelerometer Z axis
    'accel_z': np.random.normal(2048, 40, n),

    # Soil moisture
    # randint(min, max, count)
    'moisture': np.random.randint(1500, 2800, n),

    # Vibration sensor
    # choice([options], count, probability)
    # 97% chance = 0 (no vibration)
    # 3% chance = 1 (vibration detected)
    'vibration': np.random.choice(
        [0, 1], n, p=[0.97, 0.03]
    )
}

# Create DataFrame (table) from data
df = pd.DataFrame(data)

print(f"Basic dataset created: {df.shape[0]} rows")

# ── Inject 30 Danger Events ───────────────────
# Pick 30 random row positions
event_indices = np.random.choice(n, 30, replace=False)

print(f"Injecting 30 danger events...")

for i in event_indices:
    # Spike acceleration — slope is moving!
    df.loc[i, 'accel_x'] += np.random.randint(400, 700)
    df.loc[i, 'accel_y'] += np.random.randint(400, 700)

    # Max out moisture — soil is saturated!
    df.loc[i, 'moisture'] = np.random.randint(3500, 4095)

    # Trigger vibration
    df.loc[i, 'vibration'] = 1

# ── Save to CSV ───────────────────────────────
df.to_csv('data/raw/sensor_data.csv', index=False)

# ── Print Summary ─────────────────────────────
print("\n" + "="*45)
print("   DAY 1 COMPLETE — DATASET SUMMARY")
print("="*45)
print(f"Total readings    : {len(df)}")
print(f"Time span         : {df['timestamp'].iloc[0]}")
print(f"                    to {df['timestamp'].iloc[-1]}")
print(f"\nSensor ranges:")
print(f"accel_x  : {df['accel_x'].min():.0f} to {df['accel_x'].max():.0f}")
print(f"accel_y  : {df['accel_y'].min():.0f} to {df['accel_y'].max():.0f}")
print(f"accel_z  : {df['accel_z'].min():.0f} to {df['accel_z'].max():.0f}")
print(f"moisture : {df['moisture'].min()} to {df['moisture'].max()}")
print(f"vibration: {df['vibration'].sum()} events detected")
print(f"\nDanger events injected: 30")
print(f"File saved: data/raw/sensor_data.csv")
print("="*45)
print("\n✅ Ready for Day 2!")
#DAY 1 COMPLETE 
#DATE 19 MAY 2026
#Author AKSHAY

# DANGER EVENTS FOUND 
#TOTAL DANGER ROWS: 30
# ROW MOISTURE VALUE
# 7 3649
#101 3669
#229 3683
#250 3581
#.......... like that 30 danger rows .......... with vibration detection and high moisture values........