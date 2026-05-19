# ============================================
# DAY 1 — Simulate Landslide Sensor Data
# Date: 19 May 2026
# Author: Akshay
#
# DANGER EVENTS FOUND:
# Total danger rows : 30
# Row 7   → moisture: 3649
# Row 101 → moisture: 3669
# Row 229 → moisture: 3683
# Row 250 → moisture: 3581
#
# UPDATE: Added 60 WARNING events26   2200.960257  2242.020447      3219          0
#120  2353.641278  2333.504617      3417          0
#128  2331.986055  2219.615253      2929          0
#151  2309.857928  2377.458807      3079          0
#190  2285.139402  2237.075026      3372          0

import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

print("Creating sensor dataset...")

data = {
    'timestamp': pd.date_range(
        start='2026-05-19',
        periods=n,
        freq='30s'
    ),
    'accel_x': np.random.normal(2048, 40, n),
    'accel_y': np.random.normal(2048, 40, n),
    'accel_z': np.random.normal(2048, 40, n),
    'moisture': np.random.randint(1500, 2800, n),
    'vibration': np.random.choice(
        [0, 1], n, p=[0.97, 0.03]
    )
}

df = pd.DataFrame(data)
print(f"Basic dataset created: {df.shape[0]} rows")

# ── Inject 30 DANGER Events ───────────────────
danger_indices = np.random.choice(n, 30, replace=False)
print("Injecting 30 danger events...")

for i in danger_indices:
    df.loc[i, 'accel_x'] += np.random.randint(400, 700)
    df.loc[i, 'accel_y'] += np.random.randint(400, 700)
    df.loc[i, 'moisture'] = np.random.randint(3500, 4095)
    df.loc[i, 'vibration'] = 1

# ── Inject 60 WARNING Events ──────────────────
print("Injecting 60 warning events...")

remaining = list(set(range(n)) - set(danger_indices))
warning_indices = np.random.choice(
    remaining, 60, replace=False
)

for i in warning_indices:
    df.loc[i, 'accel_x'] += np.random.randint(100, 300)
    df.loc[i, 'accel_y'] += np.random.randint(100, 300)
    df.loc[i, 'moisture'] = np.random.randint(2800, 3500)
    df.loc[i, 'vibration'] = 0

# ── Save to CSV ───────────────────────────────
df.to_csv('data/raw/sensor_data.csv', index=False)

# ── Print Summary ─────────────────────────────
safe    = len(df[df['moisture'] <= 2800])
warning = len(df[(df['moisture'] > 2800) &
                  (df['moisture'] <= 3500)])
danger  = len(df[df['moisture'] > 3500])

print("\n" + "="*45)
print("   DAY 1 UPDATED — DATASET SUMMARY")
print("="*45)
print(f"Total readings    : {len(df)}")
print(f"\nRisk distribution:")
print(f"🟢 Safe    : {safe}")
print(f"🟡 Warning : {warning}")
print(f"🔴 Danger  : {danger}")
print(f"\nSensor ranges:")
print(f"accel_x  : {df['accel_x'].min():.0f} to "
      f"{df['accel_x'].max():.0f}")
print(f"accel_y  : {df['accel_y'].min():.0f} to "
      f"{df['accel_y'].max():.0f}")
print(f"accel_z  : {df['accel_z'].min():.0f} to "
      f"{df['accel_z'].max():.0f}")
print(f"moisture : {df['moisture'].min()} to "
      f"{df['moisture'].max()}")
print(f"vibration: {int(df['vibration'].sum())} events")
print(f"\nFile saved: data/raw/sensor_data.csv")
print("="*45)
print("\n✅ Ready for Day 2!")