# ============================================
# DAY 7 — NumPy Deep Dive
# Date: 30 May 2026
# Author: Akshay
# Goal: Master NumPy for ML operations
#       Every ML library uses NumPy
# ============================================
# ============================================
# DAY 7 OBSERVATIONS:
# 1. NumPy is 100x faster than Python list
# 2. My ML data shape: (2000, 6)
# 3. Safe moisture average: 2038
# 4. Danger moisture average: 3703
# 5. Difference between safe and danger: 1665
# 6. After normalization moisture range: 0-1
# 7. LSTM will need shape: (3, 10, 6)
# ============================================

import numpy as np
import pandas as pd

print("="*50)
print("   DAY 7 — NUMPY DEEP DIVE")
print("="*50)

# ════════════════════════════════════════════
# PART 1: ARRAYS vs LISTS
# ════════════════════════════════════════════
print("\n📌 PART 1: Arrays vs Lists")

# Python list
py_list = [2048, 2051, 2045, 1823, 3649]
print(f"Python list: {py_list}")
print(f"Type: {type(py_list)}")

# NumPy array
np_array = np.array([2048, 2051, 2045,
                      1823, 3649])
print(f"\nNumPy array: {np_array}")
print(f"Type: {type(np_array)}")
print(f"dtype: {np_array.dtype}")

# Speed comparison
import time

big_list = list(range(1000000))
big_array = np.array(big_list)

# List operation timing
start = time.time()
list_result = [x * 2 for x in big_list]
list_time = time.time() - start

# NumPy operation timing
start = time.time()
array_result = big_array * 2
array_time = time.time() - start

print(f"\nSpeed comparison (1 million operations):")
print(f"Python list: {list_time*1000:.2f}ms")
print(f"NumPy array: {array_time*1000:.2f}ms")
print(f"NumPy is {list_time/array_time:.0f}x faster! ✅")

# ════════════════════════════════════════════
# PART 2: ARRAY SHAPES
# ════════════════════════════════════════════
print("\n📌 PART 2: Array Shapes")

# 1D array — one sensor reading
one_reading = np.array([2048, 2051, 2045,
                         1823, 0])
print(f"1D array (one reading): {one_reading}")
print(f"Shape: {one_reading.shape}")
print(f"Dimensions: {one_reading.ndim}")

# 2D array — multiple readings
multiple_readings = np.array([
    [2048, 2051, 2045, 1823, 0],
    [2052, 2048, 2049, 1901, 0],
    [2789, 3102, 2045, 3876, 1]
])
print(f"\n2D array (multiple readings):")
print(multiple_readings)
print(f"Shape: {multiple_readings.shape}")
print(f"→ {multiple_readings.shape[0]} readings, "
      f"{multiple_readings.shape[1]} features")

# Your actual ML data shape
df = pd.read_csv(
    'data/processed/sensor_labeled.csv'
)
features = ['accel_x', 'accel_y', 'accel_z',
            'moisture', 'vibration',
            'accel_magnitude']
X = df[features].values  # .values converts to NumPy
y = df['risk_label'].values

print(f"\nYour ML data:")
print(f"X shape: {X.shape} "
      f"({X.shape[0]} readings, "
      f"{X.shape[1]} features)")
print(f"y shape: {y.shape} "
      f"({y.shape[0]} labels)")

# ════════════════════════════════════════════
# PART 3: ARRAY OPERATIONS
# ════════════════════════════════════════════
print("\n📌 PART 3: Array Operations")

moisture = np.array([1800, 2200, 3600,
                     2100, 3900, 1950])
print(f"Moisture array: {moisture}")

print(f"\nMath operations:")
print(f"moisture + 100 = {moisture + 100}")
print(f"moisture * 2   = {moisture * 2}")
print(f"moisture / 100 = {(moisture/100).round(2)}")
print(f"moisture ** 2  = {moisture ** 2}")

print(f"\nMath functions:")
print(f"np.sqrt(moisture) = "
      f"{np.sqrt(moisture).round(2)}")

# Your actual formula from Day 3!
accel_x = np.array([2048, 2100, 2600])
accel_y = np.array([2048, 2090, 2700])
accel_z = np.array([2048, 2045, 2045])

magnitude = np.sqrt(
    accel_x**2 + accel_y**2 + accel_z**2
)
print(f"\nYour accel_magnitude formula:")
print(f"accel_x: {accel_x}")
print(f"accel_y: {accel_y}")
print(f"accel_z: {accel_z}")
print(f"magnitude: {magnitude.round(2)}")

# ════════════════════════════════════════════
# PART 4: NORMALIZATION
# ════════════════════════════════════════════
print("\n📌 PART 4: Normalization (CRITICAL)")

moisture_raw = np.array([1800, 2200,
                          3600, 2100,
                          3900, 1950])
print(f"Raw moisture: {moisture_raw}")
print(f"Range: {moisture_raw.min()} "
      f"to {moisture_raw.max()}")

# Min-Max normalization
def minmax_normalize(arr):
    return (arr - arr.min()) / \
           (arr.max() - arr.min())

moisture_norm = minmax_normalize(moisture_raw)
print(f"\nAfter normalization:")
print(f"{moisture_norm.round(3)}")
print(f"Range: {moisture_norm.min():.3f} "
      f"to {moisture_norm.max():.3f}")
print("✅ All values now between 0 and 1!")

# Normalize your actual sensor data
print(f"\nNormalizing your actual sensor data:")
X_norm = np.zeros_like(X, dtype=float)
for i in range(X.shape[1]):
    col = X[:, i]
    if col.max() != col.min():
        X_norm[:, i] = minmax_normalize(col)
    else:
        X_norm[:, i] = col

print(f"Before normalization:")
print(f"  moisture range: "
      f"{X[:, 3].min():.0f} - "
      f"{X[:, 3].max():.0f}")
print(f"After normalization:")
print(f"  moisture range: "
      f"{X_norm[:, 3].min():.3f} - "
      f"{X_norm[:, 3].max():.3f} ✅")

# ════════════════════════════════════════════
# PART 5: BOOLEAN MASKING
# ════════════════════════════════════════════
print("\n📌 PART 5: Boolean Masking")

moisture = np.array([1800, 2200, 3600,
                     2100, 3900, 1950])

# Create mask
danger_mask = moisture > 3500
print(f"Moisture: {moisture}")
print(f"Mask (>3500): {danger_mask}")
print(f"Dangerous: {moisture[danger_mask]}")

# Multiple conditions
warning_mask = (moisture > 2800) & \
               (moisture <= 3500)
print(f"\nWarning (2800-3500): "
      f"{moisture[warning_mask]}")

# Apply to your actual data
danger_readings = X[y == 2]
print(f"\nYour actual danger readings:")
print(f"Total: {len(danger_readings)}")
print(f"Avg moisture: "
      f"{danger_readings[:, 3].mean():.0f}")
print(f"Avg magnitude: "
      f"{danger_readings[:, 5].mean():.0f}")

# ════════════════════════════════════════════
# PART 6: STATISTICAL FUNCTIONS
# ════════════════════════════════════════════
print("\n📌 PART 6: Statistical Functions")

moisture_col = X[:, 3]  # moisture column

print(f"Your moisture column statistics:")
print(f"Mean   : {np.mean(moisture_col):.2f}")
print(f"Std    : {np.std(moisture_col):.2f}")
print(f"Min    : {np.min(moisture_col):.0f}")
print(f"Max    : {np.max(moisture_col):.0f}")
print(f"Median : {np.median(moisture_col):.0f}")
print(f"25th % : "
      f"{np.percentile(moisture_col, 25):.0f}")
print(f"75th % : "
      f"{np.percentile(moisture_col, 75):.0f}")

# Compare safe vs danger moisture
safe_moisture = X[y==0][:, 3]
danger_moisture = X[y==2][:, 3]

print(f"\nSafe moisture mean:   "
      f"{np.mean(safe_moisture):.0f}")
print(f"Danger moisture mean: "
      f"{np.mean(danger_moisture):.0f}")
print(f"Difference: "
      f"{np.mean(danger_moisture)-np.mean(safe_moisture):.0f}")
print(f"→ Danger readings have "
      f"{np.mean(danger_moisture)-np.mean(safe_moisture):.0f}"
      f" higher moisture on average ✅")

# ════════════════════════════════════════════
# PART 7: RANDOM FUNCTIONS RECAP
# ════════════════════════════════════════════
print("\n📌 PART 7: Random Functions")

np.random.seed(42)

# What you used in Day 1
normal_sample = np.random.normal(2048, 40, 5)
print(f"Normal(2048, 40): {normal_sample.round(1)}")
print(f"→ Values around 2048 ✅")

randint_sample = np.random.randint(1500, 2800, 5)
print(f"\nRandint(1500, 2800): {randint_sample}")
print(f"→ Whole numbers 1500-2800 ✅")

choice_sample = np.random.choice(
    [0, 1], 10, p=[0.97, 0.03]
)
print(f"\nChoice([0,1], p=[0.97,0.03]):")
print(f"{choice_sample}")
print(f"→ Mostly 0s, rare 1s ✅")

# ════════════════════════════════════════════
# PART 8: RESHAPING
# ════════════════════════════════════════════
print("\n📌 PART 8: Reshaping")

arr = np.arange(12)
print(f"Original: {arr}")
print(f"Shape: {arr.shape}")

arr_2d = arr.reshape(3, 4)
print(f"\nReshaped to (3,4):")
print(arr_2d)
print(f"Shape: {arr_2d.shape}")

# LSTM reshaping preview
print(f"\nLSTM reshaping preview:")
print(f"Your data: {X.shape}")
sample_seq = X[:30].reshape(3, 10, 6)
print(f"After reshape for LSTM: {sample_seq.shape}")
print(f"→ 3 sequences of 10 readings "
      f"with 6 features ✅")
print(f"(This is what Day 24 will do!)")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*50)
print("   DAY 7 COMPLETE — NUMPY MASTERED")
print("="*50)
print("\n8 NumPy concepts learned:")
print("  1. Arrays vs Lists (100x faster)")
print("  2. Array shapes (1D, 2D, 3D)")
print("  3. Array operations (math on arrays)")
print("  4. Normalization (0 to 1 scaling)")
print("  5. Boolean masking (filter arrays)")
print("  6. Statistical functions")
print("  7. Random functions (Day 1 review)")
print("  8. Reshaping (LSTM preparation)")
print(f"\nKey numbers from your data:")
print(f"  Safe moisture avg:   "
      f"{np.mean(safe_moisture):.0f}")
print(f"  Danger moisture avg: "
      f"{np.mean(danger_moisture):.0f}")
print(f"  NumPy speedup: ~100x vs Python list")
print("\n✅ Ready for Day 8 — Pandas Deep Dive!")
print("="*50)