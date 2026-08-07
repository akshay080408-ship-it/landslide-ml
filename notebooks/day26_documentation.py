# ============================================
# DAY 26 — RF vs LSTM Documentation
#           + Confidence Scoring System
# Date: 22 July 2026
# Author: Akshay Kumar
# Goal: Document model comparison findings
#       Build production confidence scorer
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 26 — RF vs LSTM + CONFIDENCE SCORING")
print("="*55)

# ════════════════════════════════════════════
# PART 1: LOAD ALL RESULTS FOR COMPARISON
# ════════════════════════════════════════════
print("\n📌 PART 1: Loading all model results...")

# Load RF model and data
rf_package = joblib.load('models/production_model.pkl')
rf_model = rf_package['rf_model']
iso_forest = rf_package['iso_forest']
scaler = rf_package['scaler']
features = rf_package['features']

df = pd.read_csv('data/processed/sensor_labeled.csv')
X = df[features]
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# RF predictions
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"RF Test Accuracy: {rf_accuracy*100:.2f}%")

# LSTM results (from Day 24-25)
lstm_v1_accuracy = 0.7060
lstm_v2_accuracy = 0.6055
lstm_v1_danger_recall = 0.0
lstm_v2_danger_recall = 0.091

# ════════════════════════════════════════════
# PART 2: COMPREHENSIVE COMPARISON
# ════════════════════════════════════════════
print("\n📌 PART 2: Model Comparison Analysis...")

rf_report = classification_report(
    y_test, rf_pred,
    target_names=['SAFE', 'WARNING', 'DANGER'],
    output_dict=True
)

print("\nRandom Forest Detailed Results:")
print(classification_report(
    y_test, rf_pred,
    target_names=['SAFE', 'WARNING', 'DANGER']
))

# Key metrics comparison table
print("\n" + "="*65)
print(f"{'Metric':<25} {'RF':>10} {'LSTM v1':>10} {'LSTM v2':>10}")
print("="*65)
print(f"{'Overall Accuracy':<25} "
      f"{rf_accuracy*100:>9.2f}% "
      f"{lstm_v1_accuracy*100:>9.2f}% "
      f"{lstm_v2_accuracy*100:>9.2f}%")
print(f"{'DANGER Recall':<25} "
      f"{rf_report['DANGER']['recall']*100:>9.1f}% "
      f"{lstm_v1_danger_recall*100:>9.1f}% "
      f"{lstm_v2_danger_recall*100:>9.1f}%")
print(f"{'WARNING Recall':<25} "
      f"{rf_report['WARNING']['recall']*100:>9.1f}% "
      f"{'N/A':>10} "
      f"{'75.5%':>10}")
print(f"{'Training samples':<25} "
      f"{'1600':>10} "
      f"{'1592':>10} "
      f"{'1858':>10}")
print(f"{'Danger train samples':<25} "
      f"{'30':>10} "
      f"{'30':>10} "
      f"{'300':>10}")
print("="*65)

# ════════════════════════════════════════════
# PART 3: WHY RF WINS — TECHNICAL ANALYSIS
# ════════════════════════════════════════════
print("\n📌 PART 3: Technical Analysis...")

print("""
WHY RANDOM FOREST OUTPERFORMS LSTM:

1. DATASET SIZE
   RF: Works well with 1600 samples ✅
   LSTM: Typically needs 10,000+ samples
   Your dataset: Only 1600 sequences

2. CLASS IMBALANCE
   DANGER: only 2% of data (30 samples)
   RF handles with class_weight='balanced'
   LSTM struggles even with oversampling
   (300 synthetic from 30 real = artificial)

3. FEATURE TYPE
   RF: Works with any feature distribution
   LSTM: Optimized for natural time sequences
   Your features (engineered in Day 3)
   already capture temporal patterns!
   moisture_change ALREADY encodes rate
   so LSTM's sequence advantage is reduced

4. INTERPRETABILITY
   RF: Feature importance shows WHY ✅
   LSTM: Black box, hard to explain
   For disaster management systems,
   explainability is critical

WHEN LSTM WOULD WIN:
→ 6+ months of real ESP32 data
→ Natural sequence patterns without
  pre-engineered features
→ Multiple sensor nodes (spatial patterns)
→ Data augmentation from real events
""")

# ════════════════════════════════════════════
# PART 4: CONFIDENCE SCORING SYSTEM
# ════════════════════════════════════════════
print("\n📌 PART 4: Building Confidence Scoring...")

def get_confidence_band(confidence, risk_level):
    """
    Converts raw probability into
    actionable confidence band

    WHY: 51% DANGER and 99% DANGER
         need DIFFERENT responses!
    """
    if risk_level == 2:  # DANGER
        if confidence >= 90:
            return {
                'band': 'CRITICAL',
                'action': 'EVACUATE IMMEDIATELY',
                'priority': 1,
                'color': '#FF0000'
            }
        elif confidence >= 70:
            return {
                'band': 'HIGH',
                'action': 'PREPARE FOR EVACUATION',
                'priority': 2,
                'color': '#FF4500'
            }
        else:
            return {
                'band': 'MODERATE',
                'action': 'MONITOR CLOSELY',
                'priority': 3,
                'color': '#FF8C00'
            }
    elif risk_level == 1:  # WARNING
        if confidence >= 80:
            return {
                'band': 'HIGH WARNING',
                'action': 'INCREASE MONITORING',
                'priority': 4,
                'color': '#FFA500'
            }
        else:
            return {
                'band': 'LOW WARNING',
                'action': 'STAY ALERT',
                'priority': 5,
                'color': '#FFD700'
            }
    else:  # SAFE
        if confidence >= 90:
            return {
                'band': 'VERY SAFE',
                'action': 'NORMAL OPERATIONS',
                'priority': 6,
                'color': '#00CC00'
            }
        else:
            return {
                'band': 'SAFE',
                'action': 'CONTINUE MONITORING',
                'priority': 6,
                'color': '#90EE90'
            }

def enhanced_predict(reading_dict, rf_model,
                      iso_forest, scaler,
                      features):
    """
    Full prediction with confidence scoring
    Returns rich response for Flask API
    """
    reading_df = pd.DataFrame([reading_dict])[features]

    # RF prediction
    risk_pred = rf_model.predict(reading_df)[0]
    risk_proba = rf_model.predict_proba(reading_df)[0]
    confidence = float(risk_proba[risk_pred]) * 100

    # Anomaly detection
    reading_scaled = scaler.transform(reading_df)
    is_anomaly = (
        iso_forest.predict(reading_scaled)[0] == -1
    )

    # Model agreement
    # RF says danger AND Isolation Forest
    # flags anomaly = high trust
    model_agreement = (
        (risk_pred == 2 and is_anomaly) or
        (risk_pred == 0 and not is_anomaly)
    )

    # Confidence band
    conf_band = get_confidence_band(
        confidence, risk_pred
    )

    labels = {0: 'SAFE', 1: 'WARNING', 2: 'DANGER'}

    return {
        'risk_level': labels[risk_pred],
        'risk_code': int(risk_pred),
        'confidence': round(confidence, 1),
        'confidence_band': conf_band['band'],
        'action_required': conf_band['action'],
        'alert_priority': conf_band['priority'],
        'probabilities': {
            'safe': round(float(risk_proba[0])*100, 1),
            'warning': round(float(risk_proba[1])*100, 1),
            'danger': round(float(risk_proba[2])*100, 1)
        },
        'is_anomaly': bool(is_anomaly),
        'model_agreement': bool(model_agreement),
        'deployment_model': 'Random Forest v2',
        'lstm_status': 'Experimental (needs real data)'
    }

# Test confidence scoring
print("\nTesting enhanced predictions:")
print("-"*50)

test_cases = [
    {
        'name': 'Clear SAFE reading',
        'data': {
            'accel_x': 2050, 'accel_y': 2048,
            'moisture': 1800, 'vibration': 0,
            'accel_magnitude': 3550,
            'moisture_avg5': 1850,
            'moisture_change': 20,
            'magnitude_change': 10
        }
    },
    {
        'name': 'Borderline WARNING',
        'data': {
            'accel_x': 2200, 'accel_y': 2250,
            'moisture': 2900, 'vibration': 0,
            'accel_magnitude': 3700,
            'moisture_avg5': 2800,
            'moisture_change': 300,
            'magnitude_change': 100
        }
    },
    {
        'name': 'Clear DANGER reading',
        'data': {
            'accel_x': 2700, 'accel_y': 2800,
            'moisture': 3900, 'vibration': 1,
            'accel_magnitude': 4300,
            'moisture_avg5': 3500,
            'moisture_change': 1200,
            'magnitude_change': 600
        }
    }
]

for case in test_cases:
    result = enhanced_predict(
        case['data'], rf_model,
        iso_forest, scaler, features
    )
    print(f"\n{case['name']}:")
    print(f"  Risk:       {result['risk_level']}"
          f" ({result['confidence']}%)")
    print(f"  Band:       {result['confidence_band']}")
    print(f"  Action:     {result['action_required']}")
    print(f"  Anomaly:    {result['is_anomaly']}")
    print(f"  Agreement:  {result['model_agreement']}")

# ════════════════════════════════════════════
# PART 5: VISUALIZATION
# ════════════════════════════════════════════
print("\n📌 PART 5: Creating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    'Day 26 — RF vs LSTM Analysis\n'
    'LandSense ML Model Comparison',
    fontsize=14, fontweight='bold'
)

# Graph 1: Accuracy comparison
model_names = [
    'Random\nForest', 'LSTM v1\n(Day 24)',
    'LSTM v2\n(Day 25)'
]
accuracies = [
    rf_accuracy*100,
    lstm_v1_accuracy*100,
    lstm_v2_accuracy*100
]
colors = ['#2ecc71', '#e74c3c', '#e67e22']
bars = axes[0,0].bar(
    model_names, accuracies,
    color=colors, edgecolor='black',
    alpha=0.85, width=0.5
)
for bar, acc in zip(bars, accuracies):
    axes[0,0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', fontweight='bold',
        fontsize=11
    )
axes[0,0].set_title(
    'Overall Accuracy Comparison',
    fontsize=12, fontweight='bold'
)
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim(0, 110)
axes[0,0].axhline(
    y=90, color='gray',
    linestyle='--', alpha=0.5,
    label='90% threshold'
)
axes[0,0].legend()

# Graph 2: DANGER Recall comparison
danger_recalls = [
    rf_report['DANGER']['recall']*100,
    lstm_v1_danger_recall*100,
    lstm_v2_danger_recall*100
]
bars2 = axes[0,1].bar(
    model_names, danger_recalls,
    color=colors, edgecolor='black',
    alpha=0.85, width=0.5
)
for bar, recall in zip(bars2, danger_recalls):
    axes[0,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{recall:.1f}%',
        ha='center', fontweight='bold',
        fontsize=11
    )
axes[0,1].set_title(
    '🚨 DANGER Recall Comparison\n'
    '(Most Critical Metric)',
    fontsize=12, fontweight='bold'
)
axes[0,1].set_ylabel('DANGER Recall (%)')
axes[0,1].set_ylim(0, 115)
axes[0,1].axhline(
    y=90, color='green',
    linestyle='--', alpha=0.7,
    label='Target: 90%'
)
axes[0,1].legend()

# Graph 3: RF Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Greens', ax=axes[1,0],
    xticklabels=['SAFE','WARNING','DANGER'],
    yticklabels=['SAFE','WARNING','DANGER'],
    linewidths=0.5
)
axes[1,0].set_title(
    f'RF Confusion Matrix\n'
    f'(Accuracy: {rf_accuracy*100:.2f}%)',
    fontsize=12, fontweight='bold'
)
axes[1,0].set_ylabel('Actual')
axes[1,0].set_xlabel('Predicted')

# Graph 4: Confidence distribution
rf_max_proba = rf_proba.max(axis=1) * 100
axes[1,1].hist(
    rf_max_proba,
    bins=30, color='steelblue',
    edgecolor='black', alpha=0.7
)
axes[1,1].axvline(
    x=90, color='red',
    linestyle='--', label='90% threshold'
)
axes[1,1].axvline(
    x=70, color='orange',
    linestyle='--', label='70% threshold'
)
axes[1,1].set_title(
    'RF Prediction Confidence Distribution',
    fontsize=12, fontweight='bold'
)
axes[1,1].set_xlabel('Confidence (%)')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()

high_conf = (rf_max_proba >= 90).sum()
print(f"\nConfidence analysis:")
print(f"  90%+ confidence: "
      f"{high_conf}/{len(rf_max_proba)} "
      f"({high_conf/len(rf_max_proba)*100:.1f}%)")

plt.tight_layout()
plt.savefig(
    'plots/day26_model_comparison.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✅ Comparison visualization saved")

# ════════════════════════════════════════════
# PART 6: SAVE FOR FLASK API UPDATE
# ════════════════════════════════════════════
print("\n📌 PART 6: Saving updated configs...")

model_decision = {
    'production_model': 'Random Forest',
    'reason': [
        '99.75% accuracy vs LSTM 60.55%',
        '100% DANGER recall vs LSTM 9.1%',
        'Works with small datasets (1600 samples)',
        'Interpretable via feature importance',
        'Faster inference (<1ms per prediction)'
    ],
    'lstm_status': 'experimental',
    'lstm_future': (
        'Retrain with 6+ months real ESP32 data'
    ),
    'rf_version': 'v2 (8 features, 150 trees)',
    'confidence_bands': {
        'CRITICAL': 'danger >= 90%',
        'HIGH': 'danger 70-90%',
        'MODERATE': 'danger 50-70%',
        'HIGH WARNING': 'warning >= 80%',
        'LOW WARNING': 'warning < 80%',
        'VERY SAFE': 'safe >= 90%',
        'SAFE': 'safe < 90%'
    }
}

with open('models/model_decision.json', 'w') as f:
    json.dump(model_decision, f, indent=2)
print("✅ Saved: models/model_decision.json")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 26 COMPLETE!")
print("="*55)
print(f"\n📊 Final Model Decision:")
print(f"  Production: Random Forest ✅")
print(f"  Accuracy:   {rf_accuracy*100:.2f}%")
print(f"  DANGER Recall: "
      f"{rf_report['DANGER']['recall']*100:.1f}%")
print(f"\n📝 IEEE Paper Finding:")
print(f"  'RF outperforms LSTM on small")
print(f"   imbalanced IoT datasets'")
print(f"\n🎯 Confidence Scoring:")
print(f"  7 confidence bands defined")
print(f"  From CRITICAL to VERY SAFE")
print(f"\n🚀 Next: Day 27 — HTML Dashboard!")
print("="*55)