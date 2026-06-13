# ============================================
# DAY 12 — Random Forest Classifier
# Date: 01 June 2026
# Author: Akshay
# Goal: Train first ML model on sensor data
#       Predict SAFE/WARNING/DANGER
# ============================================


#Overall Accuracy : 99.75% 🔥
#DANGER Recall    : 100.0% ✅ Perfect!
#WARNING Recall   : 99.1%  ✅ Excellent!
Zero danger events missed! ✅

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                              accuracy_score,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 12 — RANDOM FOREST CLASSIFIER")
print("="*55)

# ════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════
print("\n📂 STEP 1: Loading training data...")

df = pd.read_csv('data/processed/sensor_labeled.csv')
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

# Check label distribution
counts = df['risk_label'].value_counts().sort_index()
print(f"\nLabel distribution:")
print(f"🟢 Safe    : {counts[0]:4d} ({counts[0]/len(df)*100:.1f}%)")
print(f"🟡 Warning : {counts[1]:4d} ({counts[1]/len(df)*100:.1f}%)")
print(f"🔴 Danger  : {counts[2]:4d} ({counts[2]/len(df)*100:.1f}%)")

# ════════════════════════════════════════════
# STEP 2: SELECT FEATURES
# ════════════════════════════════════════════
print("\n📌 STEP 2: Selecting features...")

# All engineered features
# Dropping accel_z (correlation = -0.03, useless!)
features = [
    'accel_x',          # Accelerometer X axis
    'accel_y',          # Accelerometer Y axis
    # accel_z dropped! (-0.03 correlation)
    'moisture',         # Soil moisture
    'vibration',        # Vibration sensor
    'accel_magnitude',  # Most important! (0.89 correlation)
    'moisture_avg5',    # Smoothed moisture
    'magnitude_avg5',   # Smoothed magnitude
    'moisture_change',  # Rate of moisture rise
    'magnitude_change', # Rate of slope movement
    'hour',             # Time of day
]

X = df[features]
y = df['risk_label']

print(f"Features selected: {len(features)}")
print(f"accel_z DROPPED (correlation=-0.03)")
print(f"\nFeature list:")
for i, f in enumerate(features, 1):
    print(f"  {i:2d}. {f}")

# ════════════════════════════════════════════
# STEP 3: TRAIN/TEST SPLIT
# ════════════════════════════════════════════
print("\n✂️  STEP 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducible split
    stratify=y          # Keep label proportions
)

print(f"Training set : {len(X_train)} rows (80%)")
print(f"Testing set  : {len(X_test)} rows (20%)")
print(f"\nTraining label distribution:")
train_counts = y_train.value_counts().sort_index()
print(f"  Safe: {train_counts[0]}, "
      f"Warning: {train_counts[1]}, "
      f"Danger: {train_counts[2]}")
print(f"Testing label distribution:")
test_counts = y_test.value_counts().sort_index()
print(f"  Safe: {test_counts[0]}, "
      f"Warning: {test_counts[1]}, "
      f"Danger: {test_counts[2]}")

# ════════════════════════════════════════════
# STEP 4: TRAIN RANDOM FOREST
# ════════════════════════════════════════════
print("\n🌲 STEP 4: Training Random Forest...")
print("Building 100 decision trees...")

model = RandomForestClassifier(
    n_estimators=100,       # 100 trees voting
    max_depth=10,           # Limit tree depth
    random_state=42,        # Reproducible
    class_weight='balanced',# Handle imbalanced labels
    n_jobs=-1               # Use all CPU cores
)

model.fit(X_train, y_train)
print("✅ Training complete!")
print(f"Trees built: {len(model.estimators_)}")

# ════════════════════════════════════════════
# STEP 5: EVALUATE MODEL
# ════════════════════════════════════════════
print("\n📊 STEP 5: Evaluating model...")

# Predictions on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# Detailed report
print("\nDetailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['SAFE', 'WARNING', 'DANGER']
))

# ════════════════════════════════════════════
# STEP 6: CONFUSION MATRIX
# ════════════════════════════════════════════
print("\n🔲 STEP 6: Confusion Matrix Analysis...")

cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Day 12 — Random Forest Results\n'
    'Confusion Matrix Analysis',
    fontsize=13, fontweight='bold'
)

# Raw count confusion matrix
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues',
    xticklabels=['SAFE','WARNING','DANGER'],
    yticklabels=['SAFE','WARNING','DANGER'],
    ax=axes[0], linewidths=0.5
)
axes[0].set_title(f'Confusion Matrix\n'
                   f'(Accuracy: {accuracy*100:.1f}%)')
axes[0].set_ylabel('Actual Label')
axes[0].set_xlabel('Predicted Label')

# Normalized confusion matrix
cm_norm = cm.astype('float') / \
          cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_norm, annot=True, fmt='.1%',
    cmap='Greens',
    xticklabels=['SAFE','WARNING','DANGER'],
    yticklabels=['SAFE','WARNING','DANGER'],
    ax=axes[1], linewidths=0.5
)
axes[1].set_title('Normalized Confusion Matrix\n'
                   '(% correctly classified per class)')
axes[1].set_ylabel('Actual Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('plots/day12_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# Critical analysis
print("\n🚨 CRITICAL ERROR ANALYSIS:")
danger_as_safe = cm[2][0]
danger_as_warning = cm[2][1]
danger_correct = cm[2][2]

print(f"Danger → predicted SAFE:    {danger_as_safe}"
      f" ← CRITICAL (missed warning!)")
print(f"Danger → predicted WARNING: {danger_as_warning}"
      f" ← Acceptable")
print(f"Danger → predicted DANGER:  {danger_correct}"
      f" ← Correct ✅")

if danger_as_safe == 0:
    print("\n✅ PERFECT! No danger events missed!")
else:
    print(f"\n⚠️  {danger_as_safe} danger events "
          f"classified as SAFE — need improvement!")

# ════════════════════════════════════════════
# STEP 7: FEATURE IMPORTANCE
# ════════════════════════════════════════════
print("\n⭐ STEP 7: Feature Importance...")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=True)

print("\nFeature importance ranking:")
for feat, imp in importance.sort_values(
    ascending=False
).items():
    bar = '█' * int(imp * 50)
    print(f"  {feat:20s}: {imp:.4f} {bar}")

# Plot feature importance
plt.figure(figsize=(10, 6))
colors_bar = [
    'red' if imp > 0.2
    else 'orange' if imp > 0.1
    else 'steelblue'
    for imp in importance.values
]
importance.plot(
    kind='barh',
    color=colors_bar,
    edgecolor='black',
    alpha=0.8
)
plt.title(
    'Feature Importance — Random Forest\n'
    'Red = Critical  Orange = Important  Blue = Minor',
    fontsize=12, fontweight='bold'
)
plt.xlabel('Importance Score')
plt.axvline(x=0.1, color='gray',
            linestyle='--', alpha=0.5,
            label='10% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('plots/day12_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance saved")

# ════════════════════════════════════════════
# STEP 8: SAVE MODEL
# ════════════════════════════════════════════
print("\n💾 STEP 8: Saving model...")

joblib.dump(model, 'models/rf_model.pkl')
joblib.dump(features, 'models/feature_list.pkl')
print("✅ Model saved: models/rf_model.pkl")
print("✅ Features saved: models/feature_list.pkl")

# ════════════════════════════════════════════
# STEP 9: TEST SINGLE PREDICTION
# ════════════════════════════════════════════
print("\n🎯 STEP 9: Testing single predictions...")

# Load saved model
loaded_model = joblib.load('models/rf_model.pkl')
loaded_features = joblib.load('models/feature_list.pkl')

def predict_single(ax, ay, moisture, vibration,
                   magnitude, m_avg5, mag_avg5,
                   m_change, mag_change, hour):
    """Predict risk for single ESP32 reading"""
    reading = pd.DataFrame([[
        ax, ay, moisture, vibration,
        magnitude, m_avg5, mag_avg5,
        m_change, mag_change, hour
    ]], columns=loaded_features)

    prediction = loaded_model.predict(reading)[0]
    probabilities = loaded_model.predict_proba(reading)[0]

    labels = {0:'🟢 SAFE', 1:'🟡 WARNING', 2:'🔴 DANGER'}
    return {
        'risk': labels[prediction],
        'risk_level': int(prediction),
        'confidence': round(probabilities[prediction]*100, 1),
        'safe_prob':    round(probabilities[0]*100, 1),
        'warning_prob': round(probabilities[1]*100, 1),
        'danger_prob':  round(probabilities[2]*100, 1)
    }

# Test 1: Normal safe reading
r1 = predict_single(
    ax=2100, ay=2050,
    moisture=1800, vibration=0,
    magnitude=3565, m_avg5=1850,
    mag_avg5=3560, m_change=50,
    mag_change=20, hour=10
)
print(f"\nTest 1 (Normal reading):")
print(f"  Result: {r1['risk']} "
      f"(confidence: {r1['confidence']}%)")

# Test 2: Warning reading
r2 = predict_single(
    ax=2200, ay=2250,
    moisture=3100, vibration=0,
    magnitude=3750, m_avg5=2900,
    mag_avg5=3700, m_change=400,
    mag_change=150, hour=16
)
print(f"\nTest 2 (Warning reading):")
print(f"  Result: {r2['risk']} "
      f"(confidence: {r2['confidence']}%)")

# Test 3: Danger reading
r3 = predict_single(
    ax=2700, ay=2800,
    moisture=3900, vibration=1,
    magnitude=4300, m_avg5=3500,
    mag_avg5=4100, m_change=1200,
    mag_change=600, hour=16
)
print(f"\nTest 3 (Danger reading):")
print(f"  Result: {r3['risk']} "
      f"(confidence: {r3['confidence']}%)")
print(f"  Probabilities: "
      f"Safe={r3['safe_prob']}% "
      f"Warning={r3['warning_prob']}% "
      f"Danger={r3['danger_prob']}%")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 12 COMPLETE — RANDOM FOREST TRAINED!")
print("="*55)
print(f"\n📊 Model Performance:")
print(f"  Overall Accuracy : {accuracy*100:.2f}%")

report = classification_report(
    y_test, y_pred,
    target_names=['SAFE','WARNING','DANGER'],
    output_dict=True
)
print(f"  SAFE Recall    : "
      f"{report['SAFE']['recall']*100:.1f}%")
print(f"  WARNING Recall : "
      f"{report['WARNING']['recall']*100:.1f}%")
print(f"  DANGER Recall  : "
      f"{report['DANGER']['recall']*100:.1f}% "
      f"← Most Important!")

print(f"\n⭐ Top 3 Features:")
top3 = importance.sort_values(
    ascending=False
).head(3)
for i, (feat, imp) in enumerate(top3.items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print(f"\n💾 Saved:")
print(f"  models/rf_model.pkl")
print(f"  models/feature_list.pkl")
print(f"  plots/day12_confusion_matrix.png")
print(f"  plots/day12_feature_importance.png")

print(f"\n🚀 Next: Day 13 — Feature Importance Deep Dive")
print("="*55)