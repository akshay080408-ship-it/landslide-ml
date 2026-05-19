# ============================================
# DAY 2 — Explore Sensor Data
# Date: 20 May 2026
# Author: Akshay
# Goal: Understand our dataset before ML
# ============================================

import pandas as pd

# ── Load your CSV file ────────────────────────
# pd.read_csv reads the file into a table
df = pd.read_csv('data/raw/sensor_data.csv')

print("="*50)
print("   DAY 2 — DATA EXPLORATION")
print("="*50)

# ── 1. Shape ──────────────────────────────────
# How many rows and columns?
print("\n1️⃣  SHAPE OF DATA:")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
# Expected: 2000 rows, 6 columns

# ── 2. Column Names ───────────────────────────
print("\n2️⃣  COLUMN NAMES:")
for col in df.columns:
    print(f"   → {col}")

# ── 3. Data Types ─────────────────────────────
print("\n3️⃣  DATA TYPES:")
print(df.dtypes)

# ── 4. First 5 Rows ───────────────────────────
print("\n4️⃣  FIRST 5 ROWS:")
print(df.head())

# ── 5. Last 5 Rows ────────────────────────────
print("\n5️⃣  LAST 5 ROWS:")
print(df.tail())

# ── 6. Statistics ─────────────────────────────
print("\n6️⃣  STATISTICS:")
print(df.describe().round(2))

# ── 7. Missing Values ─────────────────────────
print("\n7️⃣  MISSING VALUES:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values!")
else:
    print(missing)

# ── 8. Danger Analysis ────────────────────────
print("\n8️⃣  DANGER ANALYSIS:")
safe    = len(df[df['moisture'] <= 2800])
warning = len(df[(df['moisture'] > 2800) & 
                  (df['moisture'] <= 3500)])
danger  = len(df[df['moisture'] > 3500])

print(f"   🟢 Safe readings    : {safe}")
print(f"   🟡 Warning readings : {warning}")
print(f"   🔴 Danger readings  : {danger}")
print(f"   Total               : {len(df)}")

# ── 9. Vibration Count ────────────────────────
print("\n9️⃣  VIBRATION EVENTS:")
vib_count = df['vibration'].sum()
print(f"   Total vibration events : {int(vib_count)}")
print(f"   % of total readings    : "
      f"{vib_count/len(df)*100:.1f}%")

# ── 10. Key Observations ──────────────────────
print("\n🔟  KEY OBSERVATIONS:")
print(f"   Average moisture  : {df['moisture'].mean():.0f}")
print(f"   Average accel_x   : {df['accel_x'].mean():.0f}")
print(f"   Average accel_y   : {df['accel_y'].mean():.0f}")
print(f"   Average accel_z   : {df['accel_z'].mean():.0f}")
print(f"   Max moisture      : {df['moisture'].max()}")
print(f"   Max accel_x       : {df['accel_x'].max():.0f}")

print("\n" + "="*50)
print("✅ Day 2 Complete — Data Understood!")
print("="*50)