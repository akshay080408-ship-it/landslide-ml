# ============================================
# DAY 13 — Feature Importance Deep Dive
# Date: 01 June 2026
# Author: Akshay
# Goal: Understand WHY model makes decisions
#       Remove useless features
#       Retrain improved model
# ============================================
# ============================================
# DAY 13 OBSERVATIONS:
# 1. Most important: moisture_change (0.389)
#    Both methods agree → very reliable ✅
# 2. Features removed: magnitude_avg5, hour
# 3. v1 accuracy: 99.75%
# 4. v2 accuracy: 99.75% (same, cleaner model)
# 5. DANGER recall maintained: 100% ✅
# 6. Hardware priority: moisture sensor first
#    62.7% of model power from moisture features
# 7. Permutation importance more honest than
#    built-in for real importance measurement
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 13 — FEATURE IMPORTANCE DEEP DIVE")
print("="*55)

# ════════════════════════════════════════════
# LOAD DATA AND ORIGINAL MODEL
# ════════════════════════════════════════════
df = pd.read_csv('data/processed/sensor_labeled.csv')

features_v1 = [
    'accel_x', 'accel_y', 'moisture',
    'vibration', 'accel_magnitude',
    'moisture_avg5', 'magnitude_avg5',
    'moisture_change', 'magnitude_change', 'hour'
]

X = df[features_v1]
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Load Day 12 model
model_v1 = joblib.load('models/rf_model.pkl')
y_pred_v1 = model_v1.predict(X_test)
acc_v1 = accuracy_score(y_test, y_pred_v1)

print(f"\n📊 Day 12 Model (v1):")
print(f"Features: {len(features_v1)}")
print(f"Accuracy: {acc_v1*100:.2f}%")

# ════════════════════════════════════════════
# PART 1: BUILT-IN IMPORTANCE (Day 12 recap)
# ════════════════════════════════════════════
print("\n📌 PART 1: Built-in Feature Importance")
print("(How much each feature helps trees split)")

builtin_imp = pd.Series(
    model_v1.feature_importances_,
    index=features_v1
).sort_values(ascending=False)

print("\nRanking:")
for rank, (feat, imp) in enumerate(
    builtin_imp.items(), 1
):
    bar = '█' * int(imp * 60)
    status = '✅' if imp > 0.05 else '⚠️ ' if imp > 0.02 else '❌'
    print(f"  {rank:2d}. {feat:20s}: "
          f"{imp:.4f} {status} {bar}")

# ════════════════════════════════════════════
# PART 2: PERMUTATION IMPORTANCE
# ════════════════════════════════════════════
print("\n📌 PART 2: Permutation Importance")
print("(Shuffle each feature, measure accuracy drop)")
print("Calculating... (takes ~30 seconds)")

perm_imp = permutation_importance(
    model_v1, X_test, y_test,
    n_repeats=10,   # Shuffle 10 times each
    random_state=42,
    n_jobs=-1
)

perm_series = pd.Series(
    perm_imp.importances_mean,
    index=features_v1
).sort_values(ascending=False)

print("\nAccuracy drop when feature is shuffled:")
for feat, drop in perm_series.items():
    bar = '█' * int(abs(drop) * 200)
    status = '✅' if drop > 0.01 else '⚠️ ' if drop > 0.001 else '❌'
    print(f"  {feat:20s}: "
          f"{drop:.4f} accuracy drop {status} {bar}")

# ════════════════════════════════════════════
# PART 3: COMPARE BOTH METHODS
# ════════════════════════════════════════════
print("\n📌 PART 3: Comparing Both Methods")

comparison = pd.DataFrame({
    'Built-in': builtin_imp,
    'Permutation': perm_series
}).round(4)

comparison['Agreement'] = comparison.apply(
    lambda row: '✅ Agree' if (
        (row['Built-in'] > 0.05) ==
        (row['Permutation'] > 0.005)
    ) else '⚠️  Disagree', axis=1
)

print(f"\n{'Feature':<20} {'Built-in':>10} "
      f"{'Permutation':>13} {'Agreement':>12}")
print("-" * 58)
for feat in comparison.index:
    print(f"{feat:<20} "
          f"{comparison.loc[feat,'Built-in']:>10.4f} "
          f"{comparison.loc[feat,'Permutation']:>13.4f} "
          f"{comparison.loc[feat,'Agreement']:>12}")

# ════════════════════════════════════════════
# PART 4: VISUALIZATION
# ════════════════════════════════════════════
print("\n📌 PART 4: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Day 13 — Feature Importance Deep Dive\n'
    'LandSense Random Forest Analysis',
    fontsize=14, fontweight='bold'
)

# Graph 1: Built-in importance
colors1 = [
    'red' if i > 0.2
    else 'orange' if i > 0.05
    else 'steelblue' if i > 0.02
    else 'lightgray'
    for i in builtin_imp.values
]
builtin_imp.sort_values().plot(
    kind='barh', ax=axes[0,0],
    color=colors1[::-1],
    edgecolor='black', alpha=0.8
)
axes[0,0].set_title(
    'Built-in Feature Importance\n'
    'Red=Critical Orange=Important Blue=Minor Gray=Remove'
)
axes[0,0].axvline(x=0.05, color='orange',
                   linestyle='--', alpha=0.7,
                   label='5% threshold')
axes[0,0].axvline(x=0.02, color='gray',
                   linestyle='--', alpha=0.7,
                   label='2% threshold (remove)')
axes[0,0].legend(fontsize=8)
axes[0,0].set_xlabel('Importance Score')

# Graph 2: Permutation importance
perm_sorted = perm_series.sort_values()
colors2 = [
    'red' if i > 0.01
    else 'orange' if i > 0.003
    else 'lightgray'
    for i in perm_sorted.values
]
perm_sorted.plot(
    kind='barh', ax=axes[0,1],
    color=colors2[::-1],
    edgecolor='black', alpha=0.8
)
axes[0,1].set_title(
    'Permutation Importance\n'
    'How much accuracy drops when feature shuffled'
)
axes[0,1].axvline(x=0.003, color='gray',
                   linestyle='--', alpha=0.7,
                   label='Remove threshold')
axes[0,1].legend(fontsize=8)
axes[0,1].set_xlabel('Accuracy Drop')

# Graph 3: Side by side comparison
x = np.arange(len(features_v1))
width = 0.35
builtin_vals = [builtin_imp[f] for f in features_v1]
perm_vals = [max(0, perm_series[f])
             for f in features_v1]

# Normalize for comparison
builtin_norm = np.array(builtin_vals) / \
               max(builtin_vals)
perm_norm = np.array(perm_vals) / \
            max(perm_vals) if max(perm_vals) > 0 else perm_vals

axes[1,0].bar(x - width/2, builtin_norm,
              width, label='Built-in',
              color='steelblue', alpha=0.8,
              edgecolor='black')
axes[1,0].bar(x + width/2, perm_norm,
              width, label='Permutation',
              color='orange', alpha=0.8,
              edgecolor='black')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(
    [f.replace('_','\n') for f in features_v1],
    fontsize=7
)
axes[1,0].set_title(
    'Both Methods Side by Side\n'
    '(Normalized to 0-1 scale)'
)
axes[1,0].legend()
axes[1,0].set_ylabel('Normalized Importance')

# Graph 4: Features to keep vs remove
keep_features = [f for f in features_v1
                 if builtin_imp[f] > 0.02]
remove_features = [f for f in features_v1
                   if builtin_imp[f] <= 0.02]

wedge_sizes = [
    sum(builtin_imp[f] for f in keep_features),
    sum(builtin_imp[f] for f in remove_features)
]
axes[1,1].pie(
    wedge_sizes,
    labels=[f'Keep\n({len(keep_features)} features)',
            f'Remove\n({len(remove_features)} features)'],
    colors=['#2ecc71', '#e74c3c'],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
axes[1,1].set_title(
    'Feature Retention Decision\n'
    'Keep vs Remove'
)

plt.tight_layout()
plt.savefig('plots/day13_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualization saved")

# ════════════════════════════════════════════
# PART 5: REMOVE LOW IMPORTANCE FEATURES
# ════════════════════════════════════════════
print("\n📌 PART 5: Removing Low Importance Features")

# Features to remove (importance < 2%)
features_to_remove = [
    f for f in features_v1
    if builtin_imp[f] <= 0.02
]

features_v2 = [
    f for f in features_v1
    if f not in features_to_remove
]

print(f"\nRemoving features (importance < 2%):")
for f in features_to_remove:
    print(f"  ❌ {f}: {builtin_imp[f]:.4f}")

print(f"\nKeeping features:")
for f in features_v2:
    print(f"  ✅ {f}: {builtin_imp[f]:.4f}")

print(f"\nFeatures: {len(features_v1)} → {len(features_v2)}")

# ════════════════════════════════════════════
# PART 6: RETRAIN IMPROVED MODEL
# ════════════════════════════════════════════
print("\n📌 PART 6: Retraining Improved Model (v2)")

X_v2 = df[features_v2]
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = \
    train_test_split(
        X_v2, y, test_size=0.2,
        random_state=42, stratify=y
    )

model_v2 = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model_v2.fit(X_train_v2, y_train_v2)

y_pred_v2 = model_v2.predict(X_test_v2)
acc_v2 = accuracy_score(y_test_v2, y_pred_v2)

print(f"\n📊 Model Comparison:")
print(f"{'':25s} {'v1 (10 features)':>18} "
      f"{'v2 (8 features)':>16}")
print("-" * 62)
print(f"{'Overall Accuracy':25s} "
      f"{acc_v1*100:>17.2f}% "
      f"{acc_v2*100:>15.2f}%")

report_v1 = classification_report(
    y_test, y_pred_v1,
    output_dict=True
)
report_v2 = classification_report(
    y_test_v2, y_pred_v2,
    output_dict=True
)

for label, name in [(0,'SAFE'), (1,'WARNING'), (2,'DANGER')]:
    r1 = report_v1[str(label)]['recall']
    r2 = report_v2[str(label)]['recall']
    change = '✅' if r2 >= r1 else '⚠️'
    print(f"{name+' Recall':25s} "
          f"{r1*100:>17.1f}% "
          f"{r2*100:>15.1f}% {change}")
# Save v2 model
joblib.dump(model_v2, 'models/rf_model_v2.pkl')
joblib.dump(features_v2, 'models/feature_list_v2.pkl')
print(f"\n✅ Improved model saved: models/rf_model_v2.pkl")

# ════════════════════════════════════════════
# PART 7: HARDWARE INSIGHTS
# ════════════════════════════════════════════
print("\n📌 PART 7: Hardware Decision Insights")
print("\nBased on feature importance, your next")
print("hardware upgrade priority should be:\n")

hardware_insights = [
    (1, "Moisture sensor upgrade",
     "moisture_change (0.389) + moisture (0.237) = 62.7% of importance",
     "Piezometric water pressure sensor → ₹500",
     "More accurate moisture reading = much better model"),
    (2, "Rain gauge addition",
     "Rainfall causes moisture rise (currently captured by moisture_change)",
     "Tipping bucket rain gauge → ₹500",
     "Direct rainfall data = better early warning"),
    (3, "Better accelerometer",
     "accel_x + accel_y = 16% importance",
     "MPU6050 (digital, I2C) → ₹150",
     "More precise than ADXL335, less noise"),
    (4, "Vibration sensor OK",
     "vibration = 5.4% importance",
     "Current SW-420 is fine",
     "No upgrade needed yet"),
]

for priority, name, reason, cost, benefit in hardware_insights:
    print(f"Priority {priority}: {name}")
    print(f"  Why: {reason}")
    print(f"  Cost: {cost}")
    print(f"  Benefit: {benefit}\n")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("="*55)
print("   DAY 13 COMPLETE — FEATURE ANALYSIS DONE")
print("="*55)
print(f"\n🔍 Key Discoveries:")
print(f"  Most important: moisture_change (0.389)")
print(f"  Why: Rate of rise predicts danger")
print(f"       BEFORE threshold is crossed!")
print(f"\n  Features removed: {features_to_remove}")
print(f"  Model improved: {len(features_v1)} → "
      f"{len(features_v2)} features")
print(f"\n  v1 accuracy: {acc_v1*100:.2f}%")
print(f"  v2 accuracy: {acc_v2*100:.2f}%")
print(f"\n💡 Hardware insight:")
print(f"  Upgrade moisture sensor FIRST!")
print(f"  62.7% of model power comes from moisture")
print(f"\n🚀 Next: Day 14 — Model Improvement")
print("="*55)