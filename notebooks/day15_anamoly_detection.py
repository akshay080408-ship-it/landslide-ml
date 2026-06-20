# ============================================
# DAY 15 — Anomaly Detection
# Date: 03 June 2026
# Author: Akshay
# Goal: Detect unusual sensor patterns
#       WITHOUT needing labels
#       Catches dangers our labels missed!
# ============================================
# ============================================
# DAY 15 OBSERVATIONS:
# 1. Total anomalies detected: 60 (3.0%)
# 2. % of danger events caught: 94.6% (35/37)
# 3. New SAFE anomalies found: 1 (row 1249)
#    → moisture dropped -2097 suddenly
#    → possible sensor issue or drainage event
# 4. Isolation Forest needs NO labels (unsupervised)
# 5. Independently confirms Day 6 labeling was correct
# 6. Single feature extremes don't guarantee
#    anomaly flag — model considers ALL features together
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 15 — ANOMALY DETECTION")
print("   (Isolation Forest — Unsupervised Learning)")
print("="*55)

# ── Load data ──────────────────────────────────
df = pd.read_csv('data/processed/sensor_labeled.csv')

features = [
    'accel_x', 'accel_y', 'moisture',
    'vibration', 'accel_magnitude',
    'moisture_avg5', 'moisture_change',
    'magnitude_change'
]

X = df[features]
print(f"\nUsing {len(features)} features")
print(f"NO LABELS USED — this is unsupervised!")

# ════════════════════════════════════════════
# PART 1: SCALE THE DATA
# WHY: Isolation Forest works better when
#      all features are similar scale
# ════════════════════════════════════════════
print("\n📌 PART 1: Scaling features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Before scaling - moisture range: "
      f"{X['moisture'].min():.0f} to "
      f"{X['moisture'].max():.0f}")
print(f"After scaling  - moisture range: "
      f"{X_scaled[:,2].min():.2f} to "
      f"{X_scaled[:,2].max():.2f}")
print("✅ All features now similar scale")

# ════════════════════════════════════════════
# PART 2: TRAIN ISOLATION FOREST
# ════════════════════════════════════════════
print("\n📌 PART 2: Training Isolation Forest")

iso_forest = IsolationForest(
    n_estimators=100,      # 100 random trees
    contamination=0.03,    # Expect 3% anomalies
    random_state=42,
    n_jobs=-1
)

# fit_predict: trains AND predicts in one step
# Returns: 1 = normal, -1 = anomaly
predictions = iso_forest.fit_predict(X_scaled)

# Anomaly scores (more negative = more anomalous)
anomaly_scores = iso_forest.score_samples(X_scaled)

df['anomaly'] = predictions
df['anomaly_score'] = anomaly_scores

n_anomalies = (predictions == -1).sum()
n_normal = (predictions == 1).sum()

print(f"✅ Training complete!")
print(f"\nResults:")
print(f"  Normal readings  : {n_normal} "
      f"({n_normal/len(df)*100:.1f}%)")
print(f"  Anomaly readings : {n_anomalies} "
      f"({n_anomalies/len(df)*100:.1f}%)")

# ════════════════════════════════════════════
# PART 3: COMPARE WITH KNOWN LABELS
# Did unsupervised method find
# the same dangers we labeled?
# ════════════════════════════════════════════
print("\n📌 PART 3: Comparing with Known Labels")
print("(Even though we didn't use labels to train!)")

comparison = pd.crosstab(
    df['risk_label'].map({0:'SAFE',1:'WARNING',2:'DANGER'}),
    df['anomaly'].map({1:'Normal',-1:'Anomaly'})
)
print(f"\nCross-tabulation:")
print(comparison)

# How many DANGER readings were caught as anomalies?
danger_rows = df[df['risk_label'] == 2]
danger_caught = (danger_rows['anomaly'] == -1).sum()
danger_total = len(danger_rows)

print(f"\n🎯 Critical Check:")
print(f"  Danger readings caught as anomaly: "
      f"{danger_caught}/{danger_total} "
      f"({danger_caught/danger_total*100:.1f}%)")

if danger_caught/danger_total > 0.7:
    print(f"  ✅ EXCELLENT! Isolation Forest")
    print(f"     independently confirms most dangers!")
else:
    print(f"  ⚠️  Some danger events look statistically")
    print(f"     'normal' — interesting finding!")

# Were there NEW anomalies labeled SAFE?
safe_anomalies = df[
    (df['risk_label'] == 0) &
    (df['anomaly'] == -1)
]
print(f"\n🔍 NEW DISCOVERY:")
print(f"  Readings labeled SAFE but flagged as ANOMALY: "
      f"{len(safe_anomalies)}")
print(f"  These might be readings our manual")
print(f"  labeling rules MISSED!")

if len(safe_anomalies) > 0:
    print(f"\n  Sample anomalous 'safe' readings:")
    print(safe_anomalies[
        ['moisture','accel_magnitude',
         'moisture_change','anomaly_score']
    ].head(3).to_string())

# ════════════════════════════════════════════
# PART 4: VISUALIZATION
# ════════════════════════════════════════════
print("\n📌 PART 4: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Day 15 — Anomaly Detection Analysis\n'
    'Isolation Forest (Unsupervised)',
    fontsize=14, fontweight='bold'
)

# Graph 1: Anomaly score distribution
axes[0,0].hist(
    df[df['anomaly']==1]['anomaly_score'],
    bins=40, alpha=0.7, color='green',
    label='Normal', edgecolor='black'
)
axes[0,0].hist(
    df[df['anomaly']==-1]['anomaly_score'],
    bins=40, alpha=0.7, color='red',
    label='Anomaly', edgecolor='black'
)
axes[0,0].set_title('Anomaly Score Distribution')
axes[0,0].set_xlabel('Anomaly Score '
                      '(lower = more anomalous)')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()

# Graph 2: Scatter - moisture vs magnitude
colors_map = {1: 'green', -1: 'red'}
for val in [1, -1]:
    mask = df['anomaly'] == val
    label = 'Normal' if val == 1 else 'Anomaly'
    axes[0,1].scatter(
        df[mask]['accel_magnitude'],
        df[mask]['moisture'],
        c=colors_map[val], label=label,
        alpha=0.5, s=15
    )
axes[0,1].set_title(
    'Anomalies in Feature Space\n'
    '(Red = flagged as unusual)'
)
axes[0,1].set_xlabel('Acceleration Magnitude')
axes[0,1].set_ylabel('Moisture')
axes[0,1].legend()

# Graph 3: Comparison with risk labels
risk_anomaly = df.groupby('risk_label').apply(
    lambda x: (x['anomaly']==-1).mean()*100
)
bars = axes[1,0].bar(
    ['SAFE','WARNING','DANGER'],
    risk_anomaly.values,
    color=['green','orange','red'],
    edgecolor='black', alpha=0.8
)
axes[1,0].set_title(
    '% of Each Risk Level Flagged as Anomaly\n'
    '(Validates unsupervised vs supervised)'
)
axes[1,0].set_ylabel('% Flagged as Anomaly')
for bar, val in zip(bars, risk_anomaly.values):
    axes[1,0].text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+1,
        f'{val:.1f}%', ha='center',
        fontweight='bold'
    )

# Graph 4: Timeline of anomalies
axes[1,1].scatter(
    range(len(df)),
    df['anomaly_score'],
    c=df['anomaly'].map({1:'green',-1:'red'}),
    alpha=0.5, s=10
)
axes[1,1].axhline(
    y=df[df['anomaly']==-1]['anomaly_score'].max(),
    color='red', linestyle='--',
    label='Anomaly threshold'
)
axes[1,1].set_title('Anomaly Score Timeline')
axes[1,1].set_xlabel('Reading Number')
axes[1,1].set_ylabel('Anomaly Score')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('plots/day15_anomaly_detection.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualization saved")

# ════════════════════════════════════════════
# PART 5: SAVE MODEL
# ════════════════════════════════════════════
print("\n📌 PART 5: Saving Isolation Forest model")

joblib.dump(iso_forest, 'models/isolation_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Saved: models/isolation_forest.pkl")
print("✅ Saved: models/scaler.pkl")

# ════════════════════════════════════════════
# PART 6: TEST SINGLE PREDICTION
# ════════════════════════════════════════════
print("\n📌 PART 6: Testing on new readings")

def check_anomaly(ax, ay, moisture, vibration,
                   magnitude, m_avg5, m_change,
                   mag_change):
    reading = pd.DataFrame([[
        ax, ay, moisture, vibration,
        magnitude, m_avg5, m_change, mag_change
    ]], columns=features)
    reading_scaled = scaler.transform(reading)
    pred = iso_forest.predict(reading_scaled)[0]
    score = iso_forest.score_samples(reading_scaled)[0]
    return ('ANOMALY' if pred == -1 else 'NORMAL'), score

# Test: Completely weird reading (not in any
# label category but statistically strange)
result, score = check_anomaly(
    ax=2048, ay=2048, moisture=2000,
    vibration=1,           # Vibration but normal moisture!
    magnitude=3550, m_avg5=2000,
    m_change=5000,          # HUGE change, unusual!
    mag_change=10
)
print(f"\nWeird test reading "
      f"(vibration+normal moisture+huge change):")
print(f"  Result: {result} (score: {score:.3f})")
print(f"  This pattern wasn't in our DANGER rules")
print(f"  but Isolation Forest still flags it!")

# ════════════════════════════════════════════
# SAVE ENHANCED DATASET
# ════════════════════════════════════════════
df.to_csv('data/processed/sensor_with_anomaly.csv',
          index=False)
print(f"\n✅ Enhanced dataset saved")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 15 COMPLETE — ANOMALY DETECTION DONE")
print("="*55)
print(f"\n🔍 Key Results:")
print(f"  Total anomalies found    : {n_anomalies} "
      f"({n_anomalies/len(df)*100:.1f}%)")
print(f"  Danger events caught     : "
      f"{danger_caught}/{danger_total} "
      f"({danger_caught/danger_total*100:.1f}%)")
print(f"  New SAFE anomalies found : {len(safe_anomalies)}")
print(f"\n💡 Why This Matters:")
print(f"  Random Forest catches KNOWN danger patterns")
print(f"  Isolation Forest catches UNKNOWN weird patterns")
print(f"  Together = Complete safety system ✅")
print(f"\n💾 Saved:")
print(f"  models/isolation_forest.pkl")
print(f"  models/scaler.pkl")
print(f"  data/processed/sensor_with_anomaly.csv")
print(f"\n🚀 Next: Day 16 — Save and Load Models Properly")
print("="*55)