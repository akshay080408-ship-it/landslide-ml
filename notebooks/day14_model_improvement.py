# ============================================
# DAY 14 — Model Improvement
# Date: 02 June 2026
# Author: Akshay
# Goal: Validate model honestly using
#       cross-validation and tuning
#       Prevent overfitting concerns
# ============================================
# ============================================
# DAY 14 OBSERVATIONS:
# 1. CV mean accuracy: 99.70% ± 0.19%
# 2. Best hyperparameters: n_estimators=150, max_depth=10
# 3. Learning curve: PLATEAUED at ~1600 samples
#    → More simulated data won't help
#    → Need REAL ESP32 data next (Month 2)
# 4. Best model type: Gradient Boosting (99.85%)
#    → Slightly better than Random Forest
#    → Keeping RF for simplicity, noting GB in paper
# 5. Day 12's 99.75% CONFIRMED not lucky ✅
#    Low std deviation proves stability
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, learning_curve, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 14 — MODEL IMPROVEMENT & VALIDATION")
print("="*55)

# ── Load data with v2 features ───────────────
df = pd.read_csv('data/processed/sensor_labeled.csv')

features = [
    'accel_x', 'accel_y', 'moisture',
    'vibration', 'accel_magnitude',
    'moisture_avg5', 'moisture_change',
    'magnitude_change'
]

X = df[features]
y = df['risk_label']

print(f"\nUsing {len(features)} features from Day 13")

# ════════════════════════════════════════════
# PART 1: 5-FOLD CROSS VALIDATION
# ════════════════════════════════════════════
print("\n" + "="*55)
print("📌 PART 1: 5-Fold Cross Validation")
print("="*55)

model_base = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    random_state=42, class_weight='balanced',
    n_jobs=-1
)

# StratifiedKFold keeps class proportions in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True,
                       random_state=42)

cv_scores = cross_val_score(
    model_base, X, y, cv=skf, scoring='accuracy'
)

print(f"\nIndividual fold scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score*100:.2f}%")

print(f"\n📊 Cross-Validation Results:")
print(f"  Mean Accuracy : {cv_scores.mean()*100:.2f}%")
print(f"  Std Deviation : {cv_scores.std()*100:.2f}%")
print(f"  95% CI        : {cv_scores.mean()*100:.2f}% "
      f"± {1.96*cv_scores.std()*100:.2f}%")

if cv_scores.std() < 0.02:
    print(f"\n✅ LOW variance — model is STABLE!")
    print(f"   Day 12's 99.75% was NOT just luck ✅")
else:
    print(f"\n⚠️  HIGH variance — model may be unstable")

# ════════════════════════════════════════════
# PART 2: HYPERPARAMETER TUNING
# ════════════════════════════════════════════
print("\n" + "="*55)
print("📌 PART 2: Hyperparameter Tuning")
print("="*55)
print("\nTesting 20 combinations... (~1-2 mins)")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

grid_search.fit(X_train, y_train)

print(f"\n🏆 Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest CV score: "
      f"{grid_search.best_score_*100:.2f}%")

# Top 5 combinations
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = results_df.nlargest(5, 'mean_test_score')[
    ['param_n_estimators', 'param_max_depth',
     'mean_test_score', 'std_test_score']
]
print(f"\nTop 5 combinations:")
print(top5.to_string(index=False))

# ════════════════════════════════════════════
# PART 3: LEARNING CURVE
# ════════════════════════════════════════════
print("\n" + "="*55)
print("📌 PART 3: Learning Curve Analysis")
print("="*55)
print("\nTesting different training sizes...")

train_sizes, train_scores, val_scores = learning_curve(
    grid_search.best_estimator_,
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

print(f"\nTraining size vs Accuracy:")
for size, val_acc in zip(train_sizes, val_mean):
    print(f"  {int(size):4d} samples: {val_acc*100:.2f}%")

# Check if curve has plateaued
last_3_improvement = val_mean[-1] - val_mean[-3]
if last_3_improvement < 0.005:
    print(f"\n✅ Curve has PLATEAUED")
    print(f"   Current data size is SUFFICIENT")
    print(f"   More simulated data won't help much")
else:
    print(f"\n📈 Curve still RISING")
    print(f"   More data could improve accuracy")

# ════════════════════════════════════════════
# PART 4: MODEL COMPARISON
# ════════════════════════════════════════════
print("\n" + "="*55)
print("📌 PART 4: Comparing Different Models")
print("="*55)

models_to_test = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight='balanced'
    )
}

print(f"\n{'Model':<22} {'CV Accuracy':>14} {'Std Dev':>10}")
print("-" * 48)

comparison_results = {}
for name, mdl in models_to_test.items():
    scores = cross_val_score(
        mdl, X, y, cv=5, scoring='accuracy'
    )
    comparison_results[name] = scores
    print(f"{name:<22} "
          f"{scores.mean()*100:>13.2f}% "
          f"{scores.std()*100:>9.2f}%")

best_model_name = max(
    comparison_results,
    key=lambda k: comparison_results[k].mean()
)
print(f"\n🏆 Best model: {best_model_name}")

# ════════════════════════════════════════════
# PART 5: VISUALIZATION
# ════════════════════════════════════════════
print("\n📌 PART 5: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Day 14 — Model Improvement & Validation',
    fontsize=14, fontweight='bold'
)

# Graph 1: Cross-validation fold scores
axes[0,0].bar(
    range(1, 6), cv_scores*100,
    color='steelblue', edgecolor='black', alpha=0.8
)
axes[0,0].axhline(
    y=cv_scores.mean()*100, color='red',
    linestyle='--',
    label=f'Mean: {cv_scores.mean()*100:.2f}%'
)
axes[0,0].set_title('5-Fold Cross Validation Scores')
axes[0,0].set_xlabel('Fold Number')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].set_ylim(95, 100)
axes[0,0].legend()

# Graph 2: Learning curve
axes[0,1].plot(
    train_sizes, train_mean*100,
    'o-', color='steelblue', label='Training score'
)
axes[0,1].plot(
    train_sizes, val_mean*100,
    'o-', color='orange', label='Validation score'
)
axes[0,1].set_title('Learning Curve')
axes[0,1].set_xlabel('Training Set Size')
axes[0,1].set_ylabel('Accuracy (%)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Graph 3: Hyperparameter heatmap
pivot = results_df.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators'
)
import seaborn as sns
sns.heatmap(
    pivot, annot=True, fmt='.4f',
    cmap='YlGnBu', ax=axes[1,0]
)
axes[1,0].set_title('Hyperparameter Grid Search\n'
                     '(max_depth vs n_estimators)')

# Graph 4: Model comparison boxplot
box_data = [scores*100 for scores in
            comparison_results.values()]
axes[1,1].boxplot(
    box_data,
    labels=list(comparison_results.keys())
)
axes[1,1].set_title('Model Comparison\n'
                     '(5-Fold CV Distribution)')
axes[1,1].set_ylabel('Accuracy (%)')
plt.setp(axes[1,1].get_xticklabels(),
          rotation=20, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig('plots/day14_model_improvement.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✅ Visualization saved")

# ════════════════════════════════════════════
# PART 6: SAVE FINAL MODEL
# ════════════════════════════════════════════
print("\n📌 PART 6: Saving Final Production Model")

final_model = grid_search.best_estimator_
final_model.fit(X, y)  # Train on ALL data for production

joblib.dump(final_model, 'models/rf_model_final.pkl')
joblib.dump(features, 'models/feature_list_final.pkl')

print(f"✅ Final model saved: models/rf_model_final.pkl")
print(f"   Best params: {grid_search.best_params_}")
print(f"   Trained on ALL {len(df)} samples")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 14 COMPLETE — MODEL VALIDATED")
print("="*55)
print(f"\n📊 Honest Performance Metrics:")
print(f"  Day 12 single split : 99.75%")
print(f"  Day 14 CV average   : "
      f"{cv_scores.mean()*100:.2f}% "
      f"± {cv_scores.std()*100:.2f}%")
print(f"\n🏆 Best hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n📈 Learning curve: "
      f"{'Plateaued ✅' if last_3_improvement < 0.005 else 'Still rising 📈'}")
print(f"\n🥇 Best model type: {best_model_name}")
print(f"\n💾 Saved: models/rf_model_final.pkl")
print(f"   (Production-ready model)")
print(f"\n🚀 Next: Day 15 — Anomaly Detection")
print("="*55)
