# ============================================
# DAY 6 — Risk Labeling
# Date: 29 May 2026
# Author: Akshay
# Goal: Label every sensor reading as
#       SAFE(0) WARNING(1) DANGER(2)
#       This is the ML training target
# ============================================
# ============================================

# DAY 6 OBSERVATIONS:

# 1. Total safe readings: 1384

# 2. Total warning readings: 579

# 3. Total danger readings: 37

# 4. Borderline cases: less(0.5)

# 5. Smart labeling catches more warnings

#    than simple threshold method

# 6. Confidence scores help ML learn

#    borderline cases carefully

# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load feature engineered data ─────────────
df = pd.read_csv('data/processed/sensor_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("="*50)
print("   DAY 6 — RISK LABELING")
print("="*50)
print(f"Loaded: {len(df)} rows, "
      f"{len(df.columns)} columns")

# ════════════════════════════════════════════
# SMART LABELING FUNCTION
# Uses multiple sensor conditions
# Not just single threshold
# ════════════════════════════════════════════

def label_risk(row):
    """
    Labels each reading as:
    0 = SAFE    (no danger signs)
    1 = WARNING (some concerning signs)
    2 = DANGER  (multiple danger signs)

    Uses multiple sensor conditions
    More accurate than single threshold
    """

    # ── DANGER CONDITIONS ─────────────────────
    # Any ONE of these = DANGER

    # High moisture + vibration together
    if row['moisture'] > 3500 and \
       row['vibration'] == 1:
        return 2

    # Extremely high moisture alone
    if row['moisture'] > 3800:
        return 2

    # Very high acceleration
    if row['accel_magnitude'] > 4200:
        return 2

    # Rapid moisture rise
    if row['moisture_change'] > 1500:
        return 2

    # High magnitude + high moisture
    if row['accel_magnitude'] > 4000 and \
       row['moisture'] > 3000:
        return 2

    # ── WARNING CONDITIONS ────────────────────
    # Any ONE of these = WARNING

    # Moisture above warning level
    if row['moisture'] > 2800:
        return 1

    # Fast moisture rise
    if row['moisture_change'] > 400:
        return 1

    # Elevated acceleration
    if row['accel_magnitude'] > 3800:
        return 1

    # Vibration detected
    if row['vibration'] == 1:
        return 1

    # Rolling average rising (sustained rise)
    if row['moisture_avg5'] > 2600:
        return 1

    # ── SAFE ──────────────────────────────────
    return 0

# ── Apply labels to all rows ──────────────────
print("\nLabeling 2000 readings...")
df['risk_label'] = df.apply(label_risk, axis=1)
print("✅ Labeling complete!")

# ════════════════════════════════════════════
# LABEL CONFIDENCE SCORE
# How sure are we about each label?
# ════════════════════════════════════════════

def label_confidence(row):
    """
    Returns confidence score (0.0 to 1.0)
    High = very sure about this label
    Low = borderline case
    """
    label = row['risk_label']

    if label == 2:  # DANGER
        # More conditions met = higher confidence
        conditions_met = 0
        if row['moisture'] > 3500: conditions_met += 1
        if row['vibration'] == 1: conditions_met += 1
        if row['accel_magnitude'] > 4000: conditions_met += 1
        if row['moisture_change'] > 1000: conditions_met += 1
        return min(0.6 + conditions_met * 0.1, 1.0)

    elif label == 1:  # WARNING
        conditions_met = 0
        if row['moisture'] > 3000: conditions_met += 1
        if row['moisture_change'] > 200: conditions_met += 1
        if row['accel_magnitude'] > 3700: conditions_met += 1
        return min(0.5 + conditions_met * 0.1, 1.0)

    else:  # SAFE
        # How far from warning threshold?
        moisture_safety = (2800 - row['moisture']) / 2800
        return min(0.7 + moisture_safety * 0.3, 1.0)

print("\nCalculating confidence scores...")
df['confidence'] = df.apply(
    label_confidence, axis=1
)
print("✅ Confidence scores added!")

# ════════════════════════════════════════════
# LABEL QUALITY VERIFICATION
# Check everything is correct
# ════════════════════════════════════════════
print("\n" + "="*50)
print("   LABEL QUALITY CHECK")
print("="*50)

# Count each label
counts = df['risk_label'].value_counts().sort_index()
total = len(df)

print(f"\n🟢 SAFE (0)    : "
      f"{counts.get(0,0):4d} readings "
      f"({counts.get(0,0)/total*100:.1f}%)")
print(f"🟡 WARNING (1) : "
      f"{counts.get(1,0):4d} readings "
      f"({counts.get(1,0)/total*100:.1f}%)")
print(f"🔴 DANGER (2)  : "
      f"{counts.get(2,0):4d} readings "
      f"({counts.get(2,0)/total*100:.1f}%)")
print(f"Total          : {total}")

# Quality assertions
print("\n🔍 Running quality checks...")

# Check 1: All 3 labels present
assert df['risk_label'].nunique() == 3, \
    "Not all 3 labels present!"
print("✅ All 3 labels present (0, 1, 2)")

# Check 2: No missing labels
assert df['risk_label'].isnull().sum() == 0, \
    "Missing labels found!"
print("✅ No missing labels")

# Check 3: Realistic distribution
# Safe should be majority
assert counts.get(0,0) > counts.get(1,0), \
    "More warnings than safe readings!"
assert counts.get(1,0) > counts.get(2,0), \
    "More danger than warning readings!"
print("✅ Realistic distribution "
      "(safe > warning > danger)")

# Check 4: Danger rows have high moisture
danger_moisture = df[
    df['risk_label'] == 2
]['moisture'].min()
assert danger_moisture > 2500, \
    "Danger row has suspiciously low moisture!"
print(f"✅ All danger rows have "
      f"moisture > {danger_moisture:.0f}")

# Check 5: Confidence scores valid
assert df['confidence'].between(0, 1).all(), \
    "Invalid confidence scores!"
print("✅ All confidence scores between 0-1")

print("\n🎯 ALL QUALITY CHECKS PASSED!")

# ════════════════════════════════════════════
# LABEL ANALYSIS
# ════════════════════════════════════════════
print("\n" + "="*50)
print("   LABEL ANALYSIS")
print("="*50)

# Average sensor values per risk level
print("\nAverage sensor values by risk level:")
analysis = df.groupby('risk_label').agg({
    'moisture': 'mean',
    'accel_magnitude': 'mean',
    'moisture_change': 'mean',
    'vibration': 'sum',
    'confidence': 'mean'
}).round(2)

analysis.index = ['🟢 SAFE', '🟡 WARNING',
                   '🔴 DANGER']
print(analysis.to_string())

# Borderline cases
borderline = df[df['confidence'] < 0.65]
print(f"\n⚠️  Borderline cases "
      f"(confidence < 65%): {len(borderline)}")
print("These are the hardest readings to classify")
print("ML will need to learn these carefully")

# ════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════
print("\nCreating label distribution graph...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Risk Label Analysis\n'
    'Day 6 — Label Distribution and Quality',
    fontsize=13, fontweight='bold'
)

# Graph 1: Label distribution
axes[0].bar(
    ['🟢 Safe', '🟡 Warning', '🔴 Danger'],
    [counts.get(0,0), counts.get(1,0),
     counts.get(2,0)],
    color=['green', 'orange', 'red'],
    edgecolor='black', alpha=0.8
)
for i, (count, pct) in enumerate([
    (counts.get(0,0),
     counts.get(0,0)/total*100),
    (counts.get(1,0),
     counts.get(1,0)/total*100),
    (counts.get(2,0),
     counts.get(2,0)/total*100)
]):
    axes[0].text(
        i, count + 10,
        f'{count}\n({pct:.1f}%)',
        ha='center', fontweight='bold'
    )
axes[0].set_title('Label Distribution')
axes[0].set_ylabel('Count')

# Graph 2: Confidence distribution
axes[1].hist(
    df['confidence'],
    bins=30,
    color='steelblue',
    edgecolor='black',
    alpha=0.7
)
axes[1].axvline(
    x=0.65, color='red',
    linestyle='--',
    label='Borderline threshold'
)
axes[1].set_title('Label Confidence Distribution')
axes[1].set_xlabel('Confidence Score')
axes[1].set_ylabel('Count')
axes[1].legend()

# Graph 3: Moisture by label
safe_m = df[df['risk_label']==0]['moisture']
warn_m = df[df['risk_label']==1]['moisture']
dang_m = df[df['risk_label']==2]['moisture']

axes[2].boxplot(
    [safe_m, warn_m, dang_m],
    labels=['Safe', 'Warning', 'Danger'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue'),
)
axes[2].set_title(
    'Moisture Distribution by Risk Level'
)
axes[2].set_ylabel('Moisture Value')
axes[2].axhline(
    y=3500, color='red',
    linestyle='--', alpha=0.5
)

plt.tight_layout()
plt.savefig('plots/day06_labels.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graph saved: plots/day06_labels.png")

# ── Save final labeled dataset ─────────────
df.to_csv(
    'data/processed/sensor_labeled.csv',
    index=False
)

print("\n" + "="*50)
print("   DAY 6 COMPLETE — LABELS READY")
print("="*50)
print(f"\n✅ Labeled dataset saved:")
print(f"   data/processed/sensor_labeled.csv")
print(f"\n📊 Final dataset:")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {len(df.columns)}")
print(f"   Labels  : 0(safe) 1(warning) 2(danger)")
print(f"\n🤖 This file is your ML training data!")
print(f"   Day 12 → Random Forest trains on this")
print(f"   Day 24 → LSTM trains on this")
print("="*50)
print("\n✅ Ready for Day 7!")