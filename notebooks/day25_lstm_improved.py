# ============================================
# DAY 25 — Improved LSTM Model
# Date: 22 July 2026
# Author: Akshay
# Goal: Fix Day 24's 0% DANGER recall
#       Using oversampling + better tuning
# ============================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   DAY 25 — IMPROVED LSTM MODEL")
print("="*55)

# ════════════════════════════════════════════
# STEP 1: LOAD SEQUENCES
# ════════════════════════════════════════════
print("\n📂 STEP 1: Loading sequences...")

X_train = np.load(
    'data/processed/X_train_seq.npy'
)
X_test = np.load(
    'data/processed/X_test_seq.npy'
)
y_train = np.load(
    'data/processed/y_train_seq.npy'
)
y_test = np.load(
    'data/processed/y_test_seq.npy'
)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

unique, counts = np.unique(
    y_train, return_counts=True
)
labels = ['SAFE', 'WARNING', 'DANGER']
print(f"\nOriginal training distribution:")
for u, c in zip(unique, counts):
    print(f"  {labels[int(u)]}: {c}")

# ════════════════════════════════════════════
# STEP 2: OVERSAMPLE DANGER SEQUENCES
# WHY: Only 32 danger sequences is far too few
#      LSTM needs hundreds to learn the pattern
# ════════════════════════════════════════════
print("\n📌 STEP 2: Oversampling DANGER sequences...")

def oversample_danger(X, y, target_count=300):
    """
    Creates copies of danger sequences
    with small random noise added.

    WHY noise?
    Exact duplicates make model memorize
    Slight variations make model generalize ✅
    """
    danger_mask = (y == 2)
    X_danger = X[danger_mask]
    y_danger = y[danger_mask]

    current_count = len(X_danger)
    copies_needed = target_count - current_count

    print(f"  Original danger sequences: "
          f"{current_count}")
    print(f"  Target danger sequences:   "
          f"{target_count}")
    print(f"  Creating {copies_needed} "
          f"synthetic copies...")

    synthetic_X = []
    synthetic_y = []

    for i in range(copies_needed):
        # Pick random danger sequence
        idx = np.random.randint(0, current_count)
        sequence = X_danger[idx].copy()

        # Add tiny random noise (0.5% of range)
        # WHY: Prevents exact memorization
        noise = np.random.normal(
            0, 0.005,
            sequence.shape
        )
        sequence = np.clip(
            sequence + noise, 0, 1
        )

        synthetic_X.append(sequence)
        synthetic_y.append(2)

    # Combine original + synthetic
    X_new = np.vstack([
        X,
        np.array(synthetic_X)
    ])
    y_new = np.concatenate([
        y,
        np.array(synthetic_y)
    ])

    # Shuffle to mix synthetic with real
    shuffle_idx = np.random.permutation(len(X_new))
    return X_new[shuffle_idx], y_new[shuffle_idx]

np.random.seed(42)
X_train_bal, y_train_bal = oversample_danger(
    X_train, y_train, target_count=300
)

print(f"\nBalanced training distribution:")
unique_bal, counts_bal = np.unique(
    y_train_bal, return_counts=True
)
for u, c in zip(unique_bal, counts_bal):
    print(f"  {labels[int(u)]}: {c}")

print(f"\nTotal training sequences: "
      f"{len(X_train_bal)} "
      f"(was {len(X_train)})")

# ════════════════════════════════════════════
# STEP 3: BUILD IMPROVED LSTM MODEL
# WHY changes from Day 24:
# - Simpler architecture (1 LSTM layer)
#   Less complex = less overfitting
#   on small dataset
# - More units in dense layer
# - BatchNormalization added
# ════════════════════════════════════════════
print("\n📌 STEP 3: Building improved LSTM...")

n_timesteps = X_train.shape[1]  # 10
n_features  = X_train.shape[2]  # 8
n_classes   = 3

model = models.Sequential([
    layers.Input(shape=(n_timesteps, n_features)),

    # Single LSTM layer (simpler than Day 24)
    # WHY: Two layers was too complex for
    #      our small dataset → overfitting
    layers.LSTM(
        128,                # More units
        return_sequences=False
    ),

    # Batch normalization
    # WHY: Normalizes layer outputs
    #      stabilizes training
    #      prevents oscillating accuracy
    layers.BatchNormalization(),

    # Dropout
    layers.Dropout(0.3),

    # Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),

    # Output
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ════════════════════════════════════════════
# STEP 4: MANUAL CLASS WEIGHTS
# WHY: Auto-balanced wasn't aggressive enough
#      Manually set DANGER weight very high
# ════════════════════════════════════════════
print("\n📌 STEP 4: Setting manual class weights...")

# DANGER gets 10x more weight than SAFE
# WARNING gets 3x more weight than SAFE
manual_weights = {
    0: 1.0,   # SAFE
    1: 3.0,   # WARNING
    2: 10.0   # DANGER ← very high weight
}

print("Manual class weights:")
for i, w in manual_weights.items():
    print(f"  {labels[i]:8s}: {w:.1f}x")
print("DANGER mistakes penalized 10x more!")

# ════════════════════════════════════════════
# STEP 5: CALLBACKS
# ════════════════════════════════════════════
print("\n📌 STEP 5: Setting up callbacks...")

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,         # More patience than Day 24
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'models/lstm_v2_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Reduce learning rate when stuck
# WHY: If model stops improving,
#      try smaller learning steps
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=0.00001
)

# ════════════════════════════════════════════
# STEP 6: TRAIN IMPROVED MODEL
# ════════════════════════════════════════════
print("\n📌 STEP 6: Training improved LSTM...")
print("(3-7 minutes — more data + epochs)")
print("-"*55)

history = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_test, y_test),
    epochs=100,          # More epochs
    batch_size=16,       # Smaller batch
                         # WHY: more updates per epoch
                         # helps with small dataset
    class_weight=manual_weights,
    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],
    verbose=1
)

print("-"*55)
print("✅ Training complete!")
print(f"Trained for "
      f"{len(history.history['accuracy'])} "
      f"epochs")

# ════════════════════════════════════════════
# STEP 7: EVALUATE
# ════════════════════════════════════════════
print("\n📌 STEP 7: Evaluating improved model...")

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

test_loss, test_acc = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"(Day 24 was: 70.60%)")

print("\nDetailed Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['SAFE', 'WARNING', 'DANGER']
))

# DANGER recall specifically
danger_idx = np.where(y_test == 2)[0]
if len(danger_idx) > 0:
    danger_correct = np.sum(
        y_pred[danger_idx] == 2
    )
    danger_recall = danger_correct / len(danger_idx)
    print(f"🚨 DANGER Recall: "
          f"{danger_recall*100:.1f}%")
    print(f"   ({danger_correct}/{len(danger_idx)} "
          f"danger sequences caught)")

    if danger_recall >= 0.80:
        print("✅ GOOD improvement from Day 24!")
    elif danger_recall >= 0.50:
        print("⚠️  Better but still needs work")
    else:
        print("❌ Still poor — need more data")

# ════════════════════════════════════════════
# STEP 8: VISUALIZE IMPROVEMENTS
# ════════════════════════════════════════════
print("\n📌 STEP 8: Visualizing results...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Day 25 — Improved LSTM Results\n'
    'Oversampling + Better Architecture',
    fontsize=13, fontweight='bold'
)

# Training curves
axes[0,0].plot(
    history.history['accuracy'],
    label='Training', color='steelblue'
)
axes[0,0].plot(
    history.history['val_accuracy'],
    label='Validation', color='orange'
)
axes[0,0].set_title('Accuracy (Improved Model)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(
    history.history['loss'],
    label='Training', color='steelblue'
)
axes[0,1].plot(
    history.history['val_loss'],
    label='Validation', color='orange'
)
axes[0,1].set_title('Loss (Improved Model)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues', ax=axes[1,0],
    xticklabels=['SAFE','WARNING','DANGER'],
    yticklabels=['SAFE','WARNING','DANGER']
)
axes[1,0].set_title(
    f'Confusion Matrix v2\n'
    f'(Accuracy: {test_acc*100:.1f}%)'
)
axes[1,0].set_ylabel('Actual')
axes[1,0].set_xlabel('Predicted')

# Before vs After comparison
models_list = ['RF', 'LSTM v1', 'LSTM v2']
accs = [99.75, 70.60, test_acc*100]
colors = ['green', 'red', 'orange']
bars = axes[1,1].bar(
    models_list, accs,
    color=colors, edgecolor='black',
    alpha=0.8
)
for bar, acc in zip(bars, accs):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{acc:.1f}%',
        ha='center', fontweight='bold'
    )
axes[1,1].set_title(
    'Model Comparison\nRF vs LSTM v1 vs LSTM v2'
)
axes[1,1].set_ylabel('Accuracy (%)')
axes[1,1].set_ylim(60, 105)
axes[1,1].axhline(
    y=99.75, color='green',
    linestyle='--', alpha=0.5,
    label='RF baseline'
)
axes[1,1].legend()

plt.tight_layout()
plt.savefig(
    'plots/day25_lstm_improved.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("✅ Visualization saved")

# ════════════════════════════════════════════
# STEP 9: SAVE IMPROVED MODEL
# ════════════════════════════════════════════
model.save('models/lstm_model_v2.keras')
print("✅ Saved: models/lstm_model_v2.keras")

# ════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════
print("\n" + "="*55)
print("   DAY 25 COMPLETE — LSTM IMPROVED!")
print("="*55)
print(f"\n📊 Progress:")
print(f"  Day 24 LSTM v1: 70.60% (DANGER: 0%)")
print(f"  Day 25 LSTM v2: {test_acc*100:.2f}%")
if len(danger_idx) > 0:
    print(f"  DANGER Recall:  "
          f"{danger_recall*100:.1f}%")
print(f"\n💡 Key improvements made:")
print(f"  ✅ Oversampled danger: "
      f"32 → 300 sequences")
print(f"  ✅ Simpler architecture "
      f"(1 LSTM layer)")
print(f"  ✅ BatchNormalization added")
print(f"  ✅ Manual class weights "
      f"(DANGER = 10x)")
print(f"  ✅ ReduceLROnPlateau added")
print(f"\n🚀 Next: Day 26 — Class imbalance "
      f"deep dive")
print("="*55)