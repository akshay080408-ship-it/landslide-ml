# ============================================
# DAY 8 — Pandas Deep Dive
# Date: 30 May 2026
# Author: Akshay
# Goal: Master Pandas for sensor data
#       Every ML operation needs Pandas
# ============================================
# ============================================

# DAY 8 OBSERVATIONS:
# 1. DataFrame shape for ML: (2000, 6)
# 2. Series shape for labels: (2000,)
# 3. Train size (80%): 1600 rows
# 4. Test size (20%): 400 rows
# 5. Danger score avg (danger): 79.61
# 6. Danger score avg (safe): 12.32
# 7. Forward fill fixes WiFi dropout gaps
# 8. Time ordering critical for LSTM
# ============================================

import pandas as pd
import numpy as npgii
import matplotlib.pyplot as plt

print("="*50)
print("   DAY 8 — PANDAS DEEP DIVE")
print("="*50)

# ── Load your labeled dataset ─────────────────
df = pd.read_csv(
    'data/processed/sensor_labeled.csv'
)
df['timestamp'] = pd.to_datetime(
    df['timestamp']
)
print(f"Loaded: {df.shape[0]} rows, "
      f"{df.shape[1]} columns")

# ════════════════════════════════════════════
# PART 1: DataFrame vs Series
# WHY: Know what type you're working with
# ════════════════════════════════════════════
print("\n📌 PART 1: DataFrame vs Series")

# Full table = DataFrame
print(f"df type: {type(df)}")
print(f"df shape: {df.shape}")

# Single column = Series
moisture_series = df['moisture']
print(f"\nmoisture type: {type(moisture_series)}")
print(f"moisture shape: {moisture_series.shape}")

# Multiple columns = DataFrame
features_df = df[['moisture',
                    'accel_magnitude',
                    'vibration']]
print(f"\nfeatures type: {type(features_df)}")
print(f"features shape: {features_df.shape}")

# WHY THIS MATTERS:
# ML model needs DataFrame for X (features)
# and Series for y (labels)
X = df[['accel_x', 'accel_y', 'accel_z',
        'moisture', 'vibration',
        'accel_magnitude']]
y = df['risk_label']
print(f"\nML ready:")
print(f"X (features): {X.shape} ← DataFrame ✅")
print(f"y (labels):   {y.shape} ← Series ✅")

# ════════════════════════════════════════════
# PART 2: Selecting Data
# WHY: Extract exactly what ML needs
# ════════════════════════════════════════════
print("\n📌 PART 2: Selecting Data")

# Select by row number (iloc)
print("First reading:")
print(df.iloc[0][['moisture',
                   'accel_magnitude',
                   'risk_label']])

print("\nLast reading:")
print(df.iloc[-1][['moisture',
                    'accel_magnitude',
                    'risk_label']])

print("\nFirst 3 readings:")
print(df.iloc[0:3][['moisture',
                     'vibration',
                     'risk_label']])

# Select by condition (loc)
danger_rows = df.loc[df['risk_label'] == 2]
print(f"\nDanger rows (risk_label==2): "
      f"{len(danger_rows)}")

# WHY THIS MATTERS:
# Train/test split uses iloc
train_size = int(len(df) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
print(f"\nTrain/test split preview:")
print(f"Training rows: {len(X_train)} (80%)")
print(f"Testing rows:  {len(X_test)} (20%)")

# ════════════════════════════════════════════
# PART 3: Filtering
# WHY: Remove noise before ML training
# ════════════════════════════════════════════
print("\n📌 PART 3: Filtering")

# Single condition
high_moisture = df[df['moisture'] > 3000]
print(f"High moisture readings: "
      f"{len(high_moisture)}")

# AND condition
dangerous = df[
    (df['moisture'] > 3500) &
    (df['vibration'] == 1)
]
print(f"High moisture AND vibration: "
      f"{len(dangerous)}")

# OR condition
any_alert = df[
    (df['moisture'] > 3500) |
    (df['accel_magnitude'] > 4200)
]
print(f"Any alert condition: "
      f"{len(any_alert)}")

# WHY THIS MATTERS:
# Filter out impossible sensor values
# before training ML model
valid_data = df[
    (df['moisture'] >= 0) &
    (df['moisture'] <= 4095) &
    (df['accel_x'] >= 0) &
    (df['accel_x'] <= 4095)
]
print(f"\nValid readings after filtering: "
      f"{len(valid_data)}/{len(df)} ✅")

# ════════════════════════════════════════════
# PART 4: GroupBy
# WHY: Understand patterns per risk level
# ════════════════════════════════════════════
print("\n📌 PART 4: GroupBy")

# Average sensor values per risk level
stats = df.groupby('risk_label').agg({
    'moisture': 'mean',
    'accel_magnitude': 'mean',
    'moisture_change': 'mean',
    'vibration': 'sum'
}).round(2)

stats.index = ['🟢 SAFE',
               '🟡 WARNING',
               '🔴 DANGER']
print("Average sensor values per risk level:")
print(stats.to_string())

# Count per risk level
counts = df.groupby('risk_label').size()
print(f"\nCount per risk level:")
print(f"🟢 Safe:    {counts[0]}")
print(f"🟡 Warning: {counts[1]}")
print(f"🔴 Danger:  {counts[2]}")

# WHY THIS MATTERS:
# Confirm labels make sense
# Safe should have lowest moisture
# Danger should have highest ✅
print(f"\n✅ Label sanity check:")
safe_avg = stats.loc['🟢 SAFE', 'moisture']
danger_avg = stats.loc['🔴 DANGER', 'moisture']
print(f"Safe avg moisture:   {safe_avg:.0f}")
print(f"Danger avg moisture: {danger_avg:.0f}")
if danger_avg > safe_avg:
    print("✅ Correct! Danger > Safe moisture")

# ════════════════════════════════════════════
# PART 5: Apply Function
# WHY: Create new features for ML
# ════════════════════════════════════════════
print("\n📌 PART 5: Apply Function")

# Create new feature using apply
def danger_score(row):
    """
    Custom danger score 0-100
    Combines multiple sensors
    WHY: Single number summarizing
         overall danger level
    """
    score = 0

    # Moisture contribution (0-40 points)
    moisture_score = min(
        (row['moisture'] - 1500) /
        (4095 - 1500) * 40, 40
    )
    score += moisture_score

    # Magnitude contribution (0-30 points)
    mag_score = min(
        (row['accel_magnitude'] - 3400) /
        (4500 - 3400) * 30, 30
    )
    score += max(0, mag_score)

    # Vibration contribution (0-20 points)
    if row['vibration'] == 1:
        score += 20

    # Rate of change (0-10 points)
    if row['moisture_change'] > 500:
        score += 10

    return round(score, 2)

print("Calculating danger scores...")
df['danger_score'] = df.apply(
    danger_score, axis=1
)

print(f"\nDanger scores by risk level:")
score_stats = df.groupby(
    'risk_label'
)['danger_score'].agg(['mean', 'min', 'max'])
score_stats.index = ['🟢 SAFE',
                     '🟡 WARNING',
                     '🔴 DANGER']
print(score_stats.round(2).to_string())
print("\n✅ Higher risk = Higher score! "
      "Apply working correctly")

# ════════════════════════════════════════════
# PART 6: Missing Values
# WHY: Real ESP32 data will have gaps
# ════════════════════════════════════════════
print("\n📌 PART 6: Missing Values")

# Current data check
print("Current missing values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values in clean data")
else:
    print(missing[missing > 0])

# Simulate missing data (like real WiFi drops)
df_dirty = df.copy()
missing_indices = np.random.choice(
    len(df), 30, replace=False
)
df_dirty.loc[missing_indices, 'moisture'] = np.nan
df_dirty.loc[missing_indices[:10],
             'accel_x'] = np.nan

print(f"\nSimulated missing values:")
print(f"Missing moisture: "
      f"{df_dirty['moisture'].isnull().sum()}")
print(f"Missing accel_x:  "
      f"{df_dirty['accel_x'].isnull().sum()}")

# Fix 1: Forward fill
# WHY: Use last known good sensor value
df_fixed = df_dirty.copy()
df_fixed['moisture'] = df_dirty['moisture'].ffill()


# Fix 2: Mean fill for acceleration
# WHY: Acceleration rarely has
#      long sustained changes
df_fixed['accel_x'] = df_dirty['accel_x'].ffill()

print(f"\nAfter fixing:")
print(f"Missing moisture: "
      f"{df_fixed['moisture'].isnull().sum()} ✅")
print(f"Missing accel_x:  "
      f"{df_fixed['accel_x'].isnull().sum()} ✅")

# ════════════════════════════════════════════
# PART 7: Sorting
# WHY: Find worst danger events
#      for model validation
# ════════════════════════════════════════════
print("\n📌 PART 7: Sorting")

# Top 5 most dangerous readings
print("Top 5 most dangerous readings:")
top_danger = df.sort_values(
    'danger_score', ascending=False
).head(5)
print(top_danger[['moisture',
                   'accel_magnitude',
                   'vibration',
                   'risk_label',
                   'danger_score']].to_string())

# Sort by multiple columns
print("\nTop 5 by moisture AND magnitude:")
top_combined = df.sort_values(
    ['moisture', 'accel_magnitude'],
    ascending=[False, False]
).head(5)
print(top_combined[['moisture',
                     'accel_magnitude',
                     'risk_label']].to_string())

# ════════════════════════════════════════════
# PART 8: Time Series
# WHY: Sensor data is time-ordered
#      Order matters for LSTM ✅
# ════════════════════════════════════════════
print("\n📌 PART 8: Time Series")

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second

# Average risk by hour
hourly = df.groupby('hour').agg({
    'risk_label': 'mean',
    'moisture': 'mean'
}).round(3)

print("Hourly patterns (first 6 hours):")
print(hourly.head(6).to_string())

# WHY THIS MATTERS:
# LSTM reads data in TIME ORDER
# If data is shuffled LSTM fails!
# Pandas timestamp ensures order ✅
print(f"\n✅ Data is time-ordered:")
print(f"First reading: {df['timestamp'].iloc[0]}")
print(f"Last reading:  {df['timestamp'].iloc[-1]}")

# ════════════════════════════════════════════
# PART 9: Merging (Preview for Month 2)
# WHY: Real deployment needs
#      multiple data sources
# ════════════════════════════════════════════
print("\n📌 PART 9: Merging Preview")

# Simulate rainfall data
# (will come from real rain gauge in Month 2)
rainfall_data = pd.DataFrame({
    'hour': range(24),
    'rainfall_mm': np.random.exponential(
        2, 24
    ).round(2)
})

# Merge sensor + rainfall by hour
sensor_hourly = df.groupby('hour').agg({
    'moisture': 'mean',
    'risk_label': 'mean'
}).reset_index()

combined = pd.merge(
    sensor_hourly,
    rainfall_data,
    on='hour',
    how='left'
)

print("Sensor + Rainfall combined (first 5 hours):")
print(combined.head().to_string())
print("\n✅ Merge successful!")
print("Month 2: Real rainfall data merges here")

# ════════════════════════════════════════════
# PART 10: Save Enhanced Dataset
# WHY: Save with danger_score
#      for Day 12 ML training
# ════════════════════════════════════════════
print("\n📌 PART 10: Save Enhanced Dataset")

# Save with new danger_score column
df.to_csv(
    'data/processed/sensor_labeled.csv',
    index=False
)
print(f"✅ Saved with danger_score column")
print(f"   Columns now: {len(df.columns)}")
print(f"   New column: danger_score ✅")

# Verify save worked
df_check = pd.read_csv(
    'data/processed/sensor_labeled.csv'
)
assert len(df_check) == 2000, "Data lost!"
assert 'danger_score' in df_check.columns, \
    "danger_score missing!"
assert 'risk_label' in df_check.columns, \
    "risk_label missing!"
print(f"✅ Save verified successfully!")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*50)
print("   DAY 8 COMPLETE — PANDAS MASTERED")
print("="*50)
print("\n10 Pandas concepts learned:")
print("  1. DataFrame vs Series")
print("     → X needs DataFrame, y needs Series")
print("  2. Selecting data (iloc vs loc)")
print("     → Train/test split uses iloc")
print("  3. Filtering data")
print("     → Remove bad sensor readings")
print("  4. GroupBy")
print("     → Confirm labels are correct")
print("  5. Apply function")
print("     → Created danger_score feature")
print("  6. Missing values")
print("     → Forward fill for real ESP32 data")
print("  7. Sorting")
print("     → Find worst danger events")
print("  8. Time series")
print("     → LSTM needs time-ordered data")
print("  9. Merging")
print("     → Month 2: Add rainfall data")
print("  10. Save/Load")
print("      → Always verify after saving")
print(f"\nKey insight:")
print(f"  Danger score avg: "
      f"{df[df['risk_label']==2]['danger_score'].mean():.1f}")
print(f"  Safe score avg:   "
      f"{df[df['risk_label']==0]['danger_score'].mean():.1f}")
print(f"  → Clear separation confirms")
print(f"    labels are correct ✅")
print("\n✅ Ready for Day 9!")
print("="*50)